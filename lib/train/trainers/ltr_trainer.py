import os
import datetime
from collections import OrderedDict
from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from lib.utils.misc import get_world_size


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard and wandb
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval
                self.wandb_writer = WandbWriter(
                    settings.project_path[6:], {}, tensorboard_writer_dir, cur_train_samples, interval
                )

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _is_main_process(self) -> bool:
        """True if this process should print/write logs (rank 0 or non-DDP)."""
        # Prefer env vars since they exist even if dist isn't initialized yet.
        try:
            rank_env = os.environ.get("RANK", None)
            if rank_env is not None:
                return int(rank_env) == 0
            local_rank_env = os.environ.get("LOCAL_RANK", None)
            if local_rank_env is not None:
                return int(local_rank_env) == 0
        except Exception:
            pass

        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank() == 0
        except Exception:
            pass

        return getattr(self.settings, "local_rank", -1) in (-1, 0)

    # [MOD] DDP-safe boolean sync:
    # return True only if `flag` is True on all ranks.
    def _ddp_all_true(self, flag: bool, device: torch.device) -> bool:
        t = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
        except Exception:
            pass
        return bool(t.item())

    def _set_default_settings(self):
        # Dict of all default values
        default = {
            'print_interval': 10,
            'print_stats': None,
            'description': ''
        }

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()

            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['iter'] = i   # [MOD] expose iter id for debugging bad batches
            data['settings'] = self.settings

            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad(set_to_none=True)

                # [MOD] DDP-safe finite-loss check BEFORE backward.
                # If any rank sees non-finite loss, all ranks skip this iter.
                loss_is_finite_local = bool(torch.isfinite(loss).all().item())
                loss_is_finite_global = self._ddp_all_true(loss_is_finite_local, loss.device)

                if not loss_is_finite_global:
                    if self._is_main_process():
                        print(f"[WARN] Non-finite loss at epoch {self.epoch}, iter {i}. Skip this iteration.")
                    continue

                if not self.use_amp:
                    loss.backward()

                    grad_is_finite_local = True
                    if self.settings.grad_clip_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.actor.net.parameters(),
                            self.settings.grad_clip_norm,
                            error_if_nonfinite=False,
                        )
                        grad_tensor = torch.as_tensor(grad_norm, device=loss.device)
                        grad_is_finite_local = bool(torch.isfinite(grad_tensor).all().item())

                    # [MOD] DDP-safe finite-grad check AFTER backward.
                    # If any rank gets non-finite grad norm, all ranks skip optimizer.step().
                    grad_is_finite_global = self._ddp_all_true(grad_is_finite_local, loss.device)
                    if not grad_is_finite_global:
                        if self._is_main_process():
                            print(f"[WARN] Non-finite grad norm at epoch {self.epoch}, iter {i}. Skip optimizer.step().")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    self.optimizer.step()

                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)

                    grad_is_finite_local = True
                    if self.settings.grad_clip_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.actor.net.parameters(),
                            self.settings.grad_clip_norm,
                            error_if_nonfinite=False,
                        )
                        grad_tensor = torch.as_tensor(grad_norm, device=loss.device)
                        grad_is_finite_local = bool(torch.isfinite(grad_tensor).all().item())

                    grad_is_finite_global = self._ddp_all_true(grad_is_finite_local, loss.device)
                    if not grad_is_finite_global:
                        if self._is_main_process():
                            print(f"[WARN] Non-finite grad norm at epoch {self.epoch}, iter {i}. Skip optimizer.step().")
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

            # update wandb status
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                if self._is_main_process():
                    self.wandb_writer.write_log(self.stats, self.epoch)

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        if self._is_main_process():
            print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
            print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
            print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
            print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self._is_main_process():
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i_lr, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i_lr)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()

            # Avoid poisoning meters with NaN/Inf.
            try:
                if isinstance(val, torch.Tensor):
                    if not torch.isfinite(val).all():
                        continue
                    val = val.item()
                else:
                    if not (val == val) or val is None:  # NaN check for python floats
                        continue
            except Exception:
                pass

            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            if not self._is_main_process():
                return

            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (
                self.avg_date_time / self.num_frames * batch_size,
                self.avg_gpu_trans_time / self.num_frames * batch_size
            )
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except Exception:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i_lr, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i_lr)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)