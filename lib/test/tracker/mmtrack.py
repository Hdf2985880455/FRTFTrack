import math
import time

from lib.models.mmtrack import build_mmtrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box, box_xyxy_to_cxcywh
from lib.utils.ce_utils import generate_mask_cond


class MMTrack(BaseTracker):
    def __init__(self, params):
        super(MMTrack, self).__init__(params)
        network = build_mmtrack(params.cfg, is_training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.prev_reliability = None
        self.rel_drop_query_suppress_next = False
        self.rel_change_search_next = False
        self.last_rel_drop = None
        self.last_rel_delta = None
        self.last_rel_change_state_abnormal = None
        self.last_query_suppressed = False
        self.rel_drop_query_suppress_enabled = os.environ.get("MMTRACK_REL_DROP_QUERY_SUPPRESS", "0") == "1"
        self.rel_change_adaptive_search_enabled = os.environ.get("MMTRACK_REL_CHANGE_ADAPTIVE_SEARCH", "0") == "1"
        self.rel_drop_thr = float(os.environ.get("MMTRACK_REL_DROP_THR", "0.04"))
        self.rel_rise_thr = float(os.environ.get("MMTRACK_REL_RISE_THR", "0.04"))
        self.rel_change_search_factor = float(os.environ.get("MMTRACK_REL_CHANGE_SEARCH_FACTOR", "5.15"))
        self.rel_update_interval = int(os.environ.get("MMTRACK_REL_UPDATE_INTERVAL", "5"))
        self.rel_center_shift_thr = float(os.environ.get("MMTRACK_REL_CENTER_SHIFT_THR", "0.15"))
        self.rel_scale_change_thr = float(os.environ.get("MMTRACK_REL_SCALE_CHANGE_THR", "0.2"))
        self.rel_query_warmup_frames = int(os.environ.get("MMTRACK_REL_QUERY_WARMUP_FRAMES", "5"))
        self.rel_query_clamp_min = float(os.environ.get("MMTRACK_REL_QUERY_CLAMP_MIN", "0.3"))
        self.rel_query_clamp_max = float(os.environ.get("MMTRACK_REL_QUERY_CLAMP_MAX", "0.7"))
        self.adaptive_search_enabled = os.environ.get("MMTRACK_ADAPTIVE_SEARCH", "0") == "1"
        base_search_factor = float(self.params.search_factor)
        self.search_factor_high = float(os.environ.get("MMTRACK_SEARCH_FACTOR_HIGH", str(base_search_factor)))
        self.search_factor_mid = float(os.environ.get("MMTRACK_SEARCH_FACTOR_MID", str(base_search_factor + 0.1)))
        self.search_factor_low = float(os.environ.get("MMTRACK_SEARCH_FACTOR_LOW", str(base_search_factor + 0.2)))
        self.search_rel_low = float(os.environ.get("MMTRACK_SEARCH_REL_LOW", "0.35"))
        self.search_rel_high = float(os.environ.get("MMTRACK_SEARCH_REL_HIGH", "0.70"))
        self.search_state_gate_enabled = os.environ.get("MMTRACK_SEARCH_STATE_GATE", "0") == "1"
        self.last_search_center_shift_ratio = None
        self.last_search_scale_change_ratio = None
        self.last_search_state_abnormal = None
        self.prev_track_state = None
        self.profile_enabled = os.environ.get("MMTRACK_TRACKER_PROFILE", "0") == "1"
        self.rel_debug_enabled = os.environ.get("MMTRACK_REL_DEBUG", "0") == "1"
        self.rel_debug_dir = os.environ.get("MMTRACK_REL_DEBUG_DIR", "output/debug/reliability_trace")
        self.rel_debug_file = None

        # CI2P changes the effective token grid size (overall downsample becomes 2 * STRIDE)
        use_ci2p = getattr(self.cfg.MODEL.BACKBONE, 'USE_CI2P', False)
        compression_ratio = getattr(self.cfg.MODEL.BACKBONE, 'COMPRESSION_RATIO', 4)
        stride_for_feat_sz = self.cfg.MODEL.BACKBONE.STRIDE
        if use_ci2p and compression_ratio == 4:
            stride_for_feat_sz = self.cfg.MODEL.BACKBONE.STRIDE * 2
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // stride_for_feat_sz
        # motion constrain
        # self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def _profile_sync(self):
        if self.profile_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _profile_time_block(self, fn):
        if not self.profile_enabled:
            return fn(), None

        self._profile_sync()
        start = time.perf_counter()
        result = fn()
        self._profile_sync()
        return result, time.perf_counter() - start

    def _profile_log(self, timings):
        if not self.profile_enabled:
            return

        ordered_keys = [
            "sample_target",
            "preprocess",
            "network_track",
            "reliability_update",
            "box_postprocess",
            "state_update",
            "tracker_total",
        ]
        parts = []
        for key in ordered_keys:
            value = timings.get(key)
            if value is not None:
                parts.append(f"{key}={value * 1000:.2f}ms")
        print(f"[MMTrackTrackerProfile][frame {self.frame_id}] " + " ".join(parts))

    def _get_adaptive_search_factor(self):
        self.last_search_center_shift_ratio = None
        self.last_search_scale_change_ratio = None
        self.last_search_state_abnormal = None

        if (
            self.rel_change_adaptive_search_enabled
            and self.rel_change_search_next
            and self.frame_id > self.rel_query_warmup_frames
        ):
            return self.rel_change_search_factor

        if not self.adaptive_search_enabled:
            return self.params.search_factor

        if self.prev_reliability is None or self.frame_id <= self.rel_query_warmup_frames:
            return self.params.search_factor

        if isinstance(self.prev_reliability, torch.Tensor):
            reliability = float(self.prev_reliability.detach().mean().item())
        else:
            reliability = float(self.prev_reliability)

        if self.search_state_gate_enabled:
            if self.prev_track_state is None or self.state is None:
                return self.search_factor_high

            prev_x, prev_y, prev_w, prev_h = self.prev_track_state
            curr_x, curr_y, curr_w, curr_h = self.state

            prev_cx = prev_x + 0.5 * prev_w
            prev_cy = prev_y + 0.5 * prev_h
            curr_cx = curr_x + 0.5 * curr_w
            curr_cy = curr_y + 0.5 * curr_h

            center_shift = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
            curr_diag = max(math.hypot(curr_w, curr_h), 1e-6)
            center_shift_ratio = center_shift / curr_diag

            prev_area = max(prev_w * prev_h, 1e-6)
            curr_area = max(curr_w * curr_h, 1e-6)
            scale_change_ratio = abs(math.log(curr_area / prev_area))

            state_abnormal = (
                center_shift_ratio > self.rel_center_shift_thr
                or scale_change_ratio > self.rel_scale_change_thr
            )
            self.last_search_center_shift_ratio = center_shift_ratio
            self.last_search_scale_change_ratio = scale_change_ratio
            self.last_search_state_abnormal = state_abnormal

            if not state_abnormal:
                return self.search_factor_high

        if reliability < self.search_rel_low:
            return self.search_factor_low
        if reliability < self.search_rel_high:
            return self.search_factor_mid
        return self.search_factor_high

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # Run Language Network
        self.text_features, self.text_sentence_features = self.network.forward_text_encoder(
                                                            [info['init_text_description']], template.tensors.device)

        # save states
        self.state = info['init_bbox']
        self.text_description = list(info['init_text_description'])
        self.prev_reliability = None
        self.rel_drop_query_suppress_next = False
        self.rel_change_search_next = False
        self.last_rel_drop = None
        self.last_rel_delta = None
        self.last_rel_change_state_abnormal = None
        self.last_query_suppressed = False
        self.prev_track_state = None
        self.frame_id = 0
        if self.rel_debug_enabled:
            os.makedirs(self.rel_debug_dir, exist_ok=True)
            seq_name = info.get('sequence_name', info.get('seq_name', 'unknown'))
            safe_seq_name = str(seq_name).replace('/', '_').replace(' ', '_')
            self.rel_debug_file = os.path.join(self.rel_debug_dir, f"{safe_seq_name}_{int(time.time() * 1000)}.tsv")
            with open(self.rel_debug_file, 'w') as f:
                f.write('frame_id\tshould_update_rel\tpred_rel\tprev_rel\ttracker_prev_rel\tstate_update_rel\tsearch_factor\tsearch_center_shift_ratio\tsearch_scale_change_ratio\tsearch_state_abnormal\trel_drop\trel_delta\trel_change_state_abnormal\trel_change_search_next\tquery_suppressed\n')
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def _should_update_reliability(self):
        if self.prev_reliability is None:
            return True

        if self.frame_id % self.rel_update_interval == 0:
            return True

        if self.prev_track_state is None or self.state is None:
            return False

        prev_x, prev_y, prev_w, prev_h = self.prev_track_state
        curr_x, curr_y, curr_w, curr_h = self.state

        prev_cx = prev_x + 0.5 * prev_w
        prev_cy = prev_y + 0.5 * prev_h
        curr_cx = curr_x + 0.5 * curr_w
        curr_cy = curr_y + 0.5 * curr_h

        center_shift = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
        curr_diag = max(math.hypot(curr_w, curr_h), 1e-6)
        center_shift_ratio = center_shift / curr_diag

        prev_area = max(prev_w * prev_h, 1e-6)
        curr_area = max(curr_w * curr_h, 1e-6)
        scale_change_ratio = abs(math.log(curr_area / prev_area))

        return (
            center_shift_ratio > self.rel_center_shift_thr
            or scale_change_ratio > self.rel_scale_change_thr
        )

    def track(self, image, info: dict = None):
        timings = {}
        self._profile_sync()
        track_start = time.perf_counter() if self.profile_enabled else None
        H, W, _ = image.shape
        self.frame_id += 1
        prev_state = list(self.state) if self.state is not None else None
        search_factor = self._get_adaptive_search_factor()
        (x_patch_arr, resize_factor, x_amask_arr), timings["sample_target"] = self._profile_time_block(
            lambda: sample_target(image, self.state, search_factor,
                                  output_sz=self.params.search_size)
        )  # (x1, y1, w, h)
        search, timings["preprocess"] = self._profile_time_block(
            lambda: self.preprocessor.process(x_patch_arr, x_amask_arr)
        )

        with torch.no_grad():
            x_dict = search
            should_update_rel = self._should_update_reliability()
            use_query_gate = getattr(self.cfg.MODEL.DYNAMIC_QUERY, "USE_GATE", False)
            use_scale_modulation = getattr(
                self.cfg.MODEL.DYNAMIC_QUERY, "USE_SCALE_MODULATION", False
            )
            use_temporal_feedback = getattr(
                self.cfg.MODEL.DYNAMIC_QUERY, "USE_TEMPORAL_FEEDBACK", False
            )
            test_gate_proxy = float(
                getattr(self.cfg.MODEL.DYNAMIC_QUERY, "TEST_GATE_PROXY", 0.5)
            )
            test_scale_proxy = float(
                getattr(self.cfg.MODEL.DYNAMIC_QUERY, "TEST_SCALE_PROXY", 0.5)
            )
            if not use_query_gate and not use_scale_modulation:
                tracker_prev_reliability = None
            elif use_temporal_feedback:
                # Keep early frames on the safer non-feedback path until the
                # reliability estimate has had a few chances to stabilize.
                if (
                    self.prev_reliability is None
                    or self.frame_id <= self.rel_query_warmup_frames
                ):
                    tracker_prev_reliability = None
                else:
                    tracker_prev_reliability = self.prev_reliability.clamp(
                        self.rel_query_clamp_min, self.rel_query_clamp_max
                    )
            elif use_scale_modulation:
                tracker_prev_reliability = test_scale_proxy
            else:
                tracker_prev_reliability = test_gate_proxy
            suppress_dynamic_query = (
                self.rel_drop_query_suppress_enabled
                and self.frame_id > self.rel_query_warmup_frames
                and self.rel_drop_query_suppress_next
            )
            self.last_query_suppressed = bool(suppress_dynamic_query)

            # run visual encoder and decoder
            out_dict, timings["network_track"] = self._profile_time_block(
                lambda: self.network.track(
                    template=self.z_dict1.tensors,
                    search=x_dict,
                    ce_template_mask=self.box_mask_z,
                    text_features=self.text_features,
                    text_sentence_features=self.text_sentence_features,
                    prev_reliability=tracker_prev_reliability,
                    suppress_dynamic_query=suppress_dynamic_query,
                    with_bbox=True, with_mask=False,
                    update_reliability=should_update_rel)
            )
        
        pred_boxes = out_dict.pop('pred_bboxes')  # (x1, y1, x2, y2)
        pred_masks = out_dict.pop('pred_masks')
        pred_reliability = out_dict.pop('pred_reliability', None) if should_update_rel else None

        rel_change_info = {"delta": None}

        def _update_reliability():
            if pred_reliability is None:
                return
            if isinstance(pred_reliability, torch.Tensor):
                current_rel = pred_reliability.mean().detach()
            else:
                current_rel = pred_boxes.new_tensor(float(pred_reliability))

            current_rel = current_rel.clamp_(0.05, 0.95)

            prev_rel_before_update = self.prev_reliability.detach().clone() if self.prev_reliability is not None else None
            if self.prev_reliability is None:
                self.prev_reliability = current_rel
                self.last_rel_drop = 0.0
                self.last_rel_delta = 0.0
                self.rel_drop_query_suppress_next = False
                self.rel_change_search_next = False
            else:
                rel_delta = float((current_rel - prev_rel_before_update).detach().mean().item())
                rel_drop = -rel_delta
                rel_change_info["delta"] = rel_delta
                self.last_rel_delta = rel_delta
                self.last_rel_drop = rel_drop
                self.rel_drop_query_suppress_next = (
                    self.rel_drop_query_suppress_enabled
                    and rel_drop > self.rel_drop_thr
                )
                momentum = 0.8
                self.prev_reliability = momentum * self.prev_reliability + (1.0 - momentum) * current_rel

        _, timings["reliability_update"] = self._profile_time_block(_update_reliability)

        def _get_state_update_reliability():
            if pred_reliability is not None:
                if isinstance(pred_reliability, torch.Tensor):
                    rel = pred_reliability.mean().detach()
                else:
                    rel = pred_boxes.new_tensor(float(pred_reliability))
                return rel.clamp_(0.05, 0.95)

            if self.prev_reliability is not None:
                return self.prev_reliability.detach().clamp(0.05, 0.95)

            return None

        def _box_postprocess():
            post_boxes = box_xyxy_to_cxcywh(pred_boxes) # (x1, y1, x2, y2) to (cx, cy, w, h)
            post_box = (post_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
            return post_boxes, post_box

        (pred_boxes, pred_box), timings["box_postprocess"] = self._profile_time_block(_box_postprocess)

        self.rel_change_search_next = False
        self.last_rel_change_state_abnormal = None
        rel_delta = rel_change_info["delta"]
        if (
            self.rel_change_adaptive_search_enabled
            and rel_delta is not None
            and self.frame_id > self.rel_query_warmup_frames
        ):
            state_abnormal = False
            if prev_state is not None:
                curr_state_for_risk = self.map_box_back(pred_box, resize_factor)
                prev_x, prev_y, prev_w, prev_h = prev_state
                curr_x, curr_y, curr_w, curr_h = curr_state_for_risk

                prev_cx = prev_x + 0.5 * prev_w
                prev_cy = prev_y + 0.5 * prev_h
                curr_cx = curr_x + 0.5 * curr_w
                curr_cy = curr_y + 0.5 * curr_h

                center_shift = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
                curr_diag = max(math.hypot(curr_w, curr_h), 1e-6)
                center_shift_ratio = center_shift / curr_diag

                prev_area = max(prev_w * prev_h, 1e-6)
                curr_area = max(curr_w * curr_h, 1e-6)
                scale_change_ratio = abs(math.log(curr_area / prev_area))

                state_abnormal = (
                    center_shift_ratio > self.rel_center_shift_thr
                    or scale_change_ratio > self.rel_scale_change_thr
                )
            self.last_rel_change_state_abnormal = state_abnormal

            drop_risk = rel_delta < -self.rel_drop_thr
            rise_risk = rel_delta > self.rel_rise_thr and state_abnormal
            self.rel_change_search_next = bool(drop_risk or rise_risk)

        state_update_reliability = _get_state_update_reliability()

        def _rel_to_float(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return float(value.detach().mean().item())
            return float(value)

        def _write_reliability_debug():
            if not self.rel_debug_enabled or self.rel_debug_file is None:
                return
            values = [
                self.frame_id,
                int(bool(should_update_rel)),
                _rel_to_float(pred_reliability),
                _rel_to_float(self.prev_reliability),
                _rel_to_float(tracker_prev_reliability),
                _rel_to_float(state_update_reliability),
                search_factor,
                self.last_search_center_shift_ratio,
                self.last_search_scale_change_ratio,
                None if self.last_search_state_abnormal is None else int(bool(self.last_search_state_abnormal)),
                self.last_rel_drop,
                self.last_rel_delta,
                None if self.last_rel_change_state_abnormal is None else int(bool(self.last_rel_change_state_abnormal)),
                int(bool(self.rel_change_search_next)),
                int(bool(self.last_query_suppressed)),
            ]
            with open(self.rel_debug_file, 'a') as f:
                f.write('\t'.join('None' if v is None else str(v) for v in values) + '\n')

        _write_reliability_debug()

        def _state_update():
            curr_state = self.map_box_back(pred_box, resize_factor)

            rel_update_cfg = getattr(self.cfg.MODEL, "RELIABILITY_UPDATE", None)
            use_rel_update = (
                rel_update_cfg is not None
                and getattr(rel_update_cfg, "ENABLE", False)
            )
            use_fixed_ema = (
                rel_update_cfg is not None
                and getattr(rel_update_cfg, "FIXED_EMA", False)
            )

            if prev_state is None:
                self.state = clip_box(curr_state, H, W, margin=10)
                return

            # Parameter-matched fixed-smoothing control. This branch
            # deliberately ignores both reliability and motion anomaly.
            if use_fixed_ema:
                beta = float(getattr(rel_update_cfg, "MOMENTUM_LOW", 0.85))
                if not 0.0 <= beta <= 1.0:
                    raise ValueError(f"Invalid Fixed EMA beta: {beta}")
                blended_state = [
                    beta * curr_state[i] + (1.0 - beta) * prev_state[i]
                    for i in range(4)
                ]
                self.state = clip_box(blended_state, H, W, margin=10)
                return

            if (not use_rel_update) or state_update_reliability is None:
                self.state = clip_box(curr_state, H, W, margin=10)
                return

            low_thr = float(getattr(rel_update_cfg, "LOW_THR", 0.3))
            high_thr = float(getattr(rel_update_cfg, "HIGH_THR", 0.5))
            alpha_low = float(getattr(rel_update_cfg, "MOMENTUM_LOW", 0.85))
            alpha_high = float(getattr(rel_update_cfg, "MOMENTUM_HIGH", 1.0))

            r = float(state_update_reliability.item())

            if r >= high_thr:
                self.state = clip_box(curr_state, H, W, margin=10)
                return

            prev_x, prev_y, prev_w, prev_h = prev_state
            curr_x, curr_y, curr_w, curr_h = curr_state
            prev_cx = prev_x + 0.5 * prev_w
            prev_cy = prev_y + 0.5 * prev_h
            curr_cx = curr_x + 0.5 * curr_w
            curr_cy = curr_y + 0.5 * curr_h

            center_shift = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
            curr_diag = max(math.hypot(curr_w, curr_h), 1e-6)
            center_shift_ratio = center_shift / curr_diag

            prev_area = max(prev_w * prev_h, 1e-6)
            curr_area = max(curr_w * curr_h, 1e-6)
            scale_change_ratio = abs(math.log(curr_area / prev_area))
            abnormal_update = (
                center_shift_ratio > self.rel_center_shift_thr
                or scale_change_ratio > self.rel_scale_change_thr
            )

            if r > low_thr or not abnormal_update:
                self.state = clip_box(curr_state, H, W, margin=10)
                return

            blended_state = [
                alpha_low * curr_state[i] + (1.0 - alpha_low) * prev_state[i]
                for i in range(4)
            ]
            self.state = clip_box(blended_state, H, W, margin=10)

        _, timings["state_update"] = self._profile_time_block(_state_update)
        self.prev_track_state = prev_state
        if self.profile_enabled:
            self._profile_sync()
            timings["tracker_total"] = time.perf_counter() - track_start
            self._profile_log(timings)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return MMTrack
