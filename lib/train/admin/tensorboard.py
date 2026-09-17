import os
from collections import OrderedDict

# tensorboard / tensorboardX are optional for training.
# Some environments have incompatible protobuf versions which can make imports crash.
_disable_tb = str(os.environ.get("MMTRACK_DISABLE_TENSORBOARD", "0")).lower() in ("1", "true", "yes")
_summary_writer = None
if not _disable_tb:
    try:
        from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
        _summary_writer = _TorchSummaryWriter
    except Exception:
        try:
            from tensorboardX import SummaryWriter as _TensorboardXSummaryWriter
            _summary_writer = _TensorboardXSummaryWriter
        except Exception as e:
            print(f"WARNING: Tensorboard disabled due to import failure: {type(e).__name__}: {e}")
else:
    print("WARNING: Tensorboard disabled by MMTRACK_DISABLE_TENSORBOARD.")


class TensorboardWriter:
    def __init__(self, directory, loader_names):
        self.directory = directory
        if _summary_writer is None:
            # No-op writer to avoid crashing training when tensorboard deps are broken.
            self.writer = None
            return
        self.writer = OrderedDict({name: _summary_writer(os.path.join(self.directory, name)) for name in loader_names})

    def write_info(self, script_name, description):
        if self.writer is None:
            return
        tb_info_writer = _summary_writer(os.path.join(self.directory, 'info'))
        tb_info_writer.add_text('Script_name', script_name)
        tb_info_writer.add_text('Description', description)
        tb_info_writer.close()

    def write_epoch(self, stats: OrderedDict, epoch: int, ind=-1):
        if self.writer is None:
            return
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue
            for var_name, val in loader_stats.items():
                if hasattr(val, 'history') and getattr(val, 'has_new_data', True):
                    self.writer[loader_name].add_scalar(var_name, val.history[ind], epoch)