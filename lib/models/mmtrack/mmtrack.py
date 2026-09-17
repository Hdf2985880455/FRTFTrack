
import math
import random
import numpy as np
import os
# torch 1.9 environments require protobuf's Python implementation when wandb
# is imported transitively through timm.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import time
from typing import List
import pycocotools.mask as maskUtils
from mmdet.core import BitmapMasks
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones
from torchvision.ops import roi_align
from lib.models.mmtrack.vit import vit_base_patch16_224
from lib.models.mmtrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xywh_to_xyxy, box_xywh_to_cxcywh
from lib.models.transformers import build_decoder, VisionLanguageFusionModule, PositionEmbeddingSine1D
from lib.models.predictor import build_predictor
from lib.models.mmtrack.reliability import ReliabilityHead
from lib.models.mmtrack.region_reliability import RegionGroundedReliabilityHead
from lib.models.mmtrack.query_modulation import ReliabilityAwareQueryModulator
from lib.models.mmtrack.fine_grained_query_modulation import (
    FineGrainedTemporalQueryModulator,
)
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast
from lib.utils.misc import NestedTensor
from einops import repeat


# =========================================================
# [MOD] Helper: interpolate positional embeddings
# =========================================================
def _resize_pos_embed_2d(pos_embed_ckpt: torch.Tensor, target_hw):
    """
    Resize 2D positional embedding by bicubic interpolation.

    Args:
        pos_embed_ckpt: [1, N, C] or [1, 1+N, C]
        target_hw: (H_new, W_new)

    Returns:
        resized positional embedding tensor with the same cls-token style
        as the checkpoint tensor.
    """
    assert pos_embed_ckpt.ndim == 3 and pos_embed_ckpt.shape[0] == 1, \
        f"Unexpected pos_embed shape: {tuple(pos_embed_ckpt.shape)}"

    H_new, W_new = target_hw
    N = pos_embed_ckpt.shape[1]
    C = pos_embed_ckpt.shape[2]

    has_cls = False
    if int(math.sqrt(N - 1)) ** 2 == (N - 1):
        # has cls token
        has_cls = True
        cls_tok = pos_embed_ckpt[:, :1, :]      # [1,1,C]
        img_tok = pos_embed_ckpt[:, 1:, :]      # [1,N-1,C]
        num_img_tok = N - 1
    elif int(math.sqrt(N)) ** 2 == N:
        cls_tok = None
        img_tok = pos_embed_ckpt
        num_img_tok = N
    else:
        raise ValueError(f"Cannot infer 2D grid from pos_embed shape {tuple(pos_embed_ckpt.shape)}")

    H_old = W_old = int(math.sqrt(num_img_tok))
    assert H_old * W_old == num_img_tok, f"Old pos_embed is not square: {num_img_tok}"

    # [1, N, C] -> [1, C, H, W]
    img_tok = img_tok.reshape(1, H_old, W_old, C).permute(0, 3, 1, 2).contiguous()

    # bicubic interpolation
    img_tok = F.interpolate(img_tok, size=(H_new, W_new), mode='bicubic', align_corners=False)

    # [1, C, H, W] -> [1, H*W, C]
    img_tok = img_tok.permute(0, 2, 3, 1).reshape(1, H_new * W_new, C).contiguous()

    if has_cls:
        return torch.cat([cls_tok, img_tok], dim=1)
    return img_tok


# =========================================================
# [MOD] Helper: infer target grid from model pos_embed shape
# =========================================================
def _infer_target_hw_from_pos_embed_shape(pos_embed_target: torch.Tensor):
    """
    Infer target 2D grid from model pos_embed tensor shape.

    Supports:
      - [1, 1+H*W, C]
      - [1, H*W, C]
    """
    assert pos_embed_target.ndim == 3 and pos_embed_target.shape[0] == 1, \
        f"Unexpected target pos_embed shape: {tuple(pos_embed_target.shape)}"

    N = pos_embed_target.shape[1]

    if int(math.sqrt(N - 1)) ** 2 == (N - 1):
        S = int(math.sqrt(N - 1))
        return S, S
    elif int(math.sqrt(N)) ** 2 == N:
        S = int(math.sqrt(N))
        return S, S
    else:
        raise ValueError(f"Cannot infer target grid from shape {tuple(pos_embed_target.shape)}")


# =========================================================
# [MOD] Helper: fetch ckpt tensor by multiple candidate keys
# =========================================================
def _get_ckpt_tensor(state_dict, candidate_keys):
    for k in candidate_keys:
        if k in state_dict:
            return state_dict[k], k
    return None, None


def _init_ci2p_from_patch_embed(model, state_dict):
    backbone = model.backbone
    patch_embed = getattr(backbone, "patch_embed", None)

    if patch_embed is None:
        return
    if not getattr(patch_embed, "init_from_patch_embed", False):
        return
    if not getattr(patch_embed, "use_compression", False):
        return

    ckpt_w, ckpt_w_key = _get_ckpt_tensor(
        state_dict,
        ["backbone.patch_embed.proj.weight", "patch_embed.proj.weight"],
    )
    ckpt_b, ckpt_b_key = _get_ckpt_tensor(
        state_dict,
        ["backbone.patch_embed.proj.bias", "patch_embed.proj.bias"],
    )

    if ckpt_w is None:
        print("[CI2P init] skip proj.weight: no baseline patch_embed weight found")
        return

    with torch.no_grad():
        target_w = patch_embed.proj.weight
        if ckpt_w.shape == target_w.shape:
            target_w.copy_(ckpt_w)
            print(f"[CI2P init] copied {ckpt_w_key} -> CI2P proj.weight")
        elif (
            ckpt_w.ndim == 4
            and target_w.ndim == 4
            and ckpt_w.shape[0] == target_w.shape[0]
            and ckpt_w.shape[1] == target_w.shape[1]
            and ckpt_w.shape[2] == target_w.shape[2] * 2
            and ckpt_w.shape[3] == target_w.shape[3] * 2
        ):
            resized = F.avg_pool2d(ckpt_w, kernel_size=2, stride=2)
            target_w.copy_(resized)
            print(
                f"[CI2P init] avg-pooled {ckpt_w_key} {tuple(ckpt_w.shape)} -> "
                f"CI2P proj.weight {tuple(target_w.shape)}"
            )
        else:
            print(
                f"[CI2P init] skip proj.weight: ckpt {tuple(ckpt_w.shape)} "
                f"target {tuple(target_w.shape)}"
            )

        if ckpt_b is not None and patch_embed.proj.bias is not None and ckpt_b.shape == patch_embed.proj.bias.shape:
            patch_embed.proj.bias.copy_(ckpt_b)
            print(f"[CI2P init] copied {ckpt_b_key} -> CI2P proj.bias")


class MMTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, encoder, decoder, predictor,
                tokenizer=None, text_encoder=None,
                feat_sz=20, num_bin=1000, shuffle_fraction=-1,
                top_p=-1, input_size=256,
                bbox_task=False, language_task=False,
                num_vlfusion_layers=0, vl_input_type='separate',
                bbox_type='xyxy', aux_loss=False, cfg=None):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.bbox_task = bbox_task
        self.top_p = top_p
        self.num_bin = num_bin
        self.shuffle_fraction = shuffle_fraction
        self.bbox_type = bbox_type
        self.cfg = cfg
        self.backbone = encoder
        self.bottleneck = nn.Sequential(
                                nn.Linear(encoder.embed_dim, predictor.input_dim),
                                nn.GELU(),
                            )  # the bottleneck layer
        
        # text encoder
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        self.decoder = decoder
        self.profile_enabled = os.environ.get("MMTRACK_PROFILE", "0") == "1"
        self.profile_frame_idx = 0

        # Auxiliary head
        self.aux_loss = aux_loss
        if aux_loss:
            self.predictor = _get_clones(predictor, decoder.num_decoder_layers)
        else:
            self.predictor = predictor
        
        text_feat_size = self.text_encoder.config.hidden_size
        
        self.text_adj = nn.Sequential(
            nn.Linear(text_feat_size, predictor.input_dim, bias=True),
            nn.LayerNorm(predictor.input_dim, eps=1e-12),
            nn.Dropout(0.1),
        )

        self.use_reliability = (
            cfg is not None
            and hasattr(cfg, "MODEL")
            and hasattr(cfg.MODEL, "RELIABILITY")
            and getattr(cfg.MODEL.RELIABILITY, "ENABLE", False)
        )
        if self.use_reliability:
            self.reliability_type = getattr(cfg.MODEL.RELIABILITY, "TYPE", "lite")
            rel_hidden_dim = getattr(cfg.MODEL.RELIABILITY, "HIDDEN_DIM", predictor.input_dim)
            rel_dropout = getattr(cfg.MODEL.RELIABILITY, "DROPOUT", 0.1)
            if self.reliability_type == "region_grounded":
                rel_num_heads = getattr(cfg.MODEL.RELIABILITY, "CROSS_ATTN_HEADS", 4)
                self.reliability_head = RegionGroundedReliabilityHead(
                    hidden_dim=rel_hidden_dim,
                    num_heads=rel_num_heads,
                    dropout=rel_dropout,
                )
            elif self.reliability_type == "lite":
                self.reliability_head = ReliabilityHead(
                    hidden_dim=rel_hidden_dim,
                    dropout=rel_dropout,
                )
            else:
                raise ValueError(f"Unsupported reliability type: {self.reliability_type}")
        else:
            self.reliability_type = "disabled"
            self.reliability_head = None

        self.use_dynamic_query = (
            cfg is not None
            and hasattr(cfg, "MODEL")
            and hasattr(cfg.MODEL, "DYNAMIC_QUERY")
            and getattr(cfg.MODEL.DYNAMIC_QUERY, "ENABLE", False)
        )
        if self.use_dynamic_query:
            dq_type = getattr(cfg.MODEL.DYNAMIC_QUERY, "TYPE", "lite")
            dq_hidden_dim = getattr(cfg.MODEL.DYNAMIC_QUERY, "HIDDEN_DIM", predictor.input_dim)
            dq_dropout = getattr(cfg.MODEL.DYNAMIC_QUERY, "DROPOUT", 0.1)
            dq_residual_scale = getattr(cfg.MODEL.DYNAMIC_QUERY, "RESIDUAL_SCALE", 0.5)
            dq_use_gate = getattr(cfg.MODEL.DYNAMIC_QUERY, "USE_GATE", True)
            dq_use_scale_modulation = getattr(
                cfg.MODEL.DYNAMIC_QUERY, "USE_SCALE_MODULATION", False
            )
            dq_scale_min = getattr(cfg.MODEL.DYNAMIC_QUERY, "SCALE_MIN", 0.9)
            dq_scale_max = getattr(cfg.MODEL.DYNAMIC_QUERY, "SCALE_MAX", 1.1)
            if dq_type == "fine_grained_temporal":
                self.query_modulator = FineGrainedTemporalQueryModulator(
                    hidden_dim=dq_hidden_dim,
                    residual_scale=dq_residual_scale,
                    scale_min=dq_scale_min,
                    scale_max=dq_scale_max,
                    dropout=dq_dropout,
                )
            elif dq_type == "lite":
                self.query_modulator = ReliabilityAwareQueryModulator(
                    hidden_dim=dq_hidden_dim,
                    residual_scale=dq_residual_scale,
                    use_gate=dq_use_gate,
                    use_scale_modulation=dq_use_scale_modulation,
                    scale_min=dq_scale_min,
                    scale_max=dq_scale_max,
                )
            else:
                raise ValueError(f"Unsupported dynamic query type: {dq_type}")
        else:
            self.query_modulator = None

        # multi-modal vision language fusion
        self.vl_fusion = VisionLanguageFusionModule(dim=predictor.input_dim, num_heads=8, attn_drop=0.1, proj_drop=0.1,
                                                    num_vlfusion_layers=num_vlfusion_layers, vl_input_type=vl_input_type)
        self.text_pos = PositionEmbeddingSine1D(predictor.input_dim, normalize=True)
        
        self.input_size = input_size
        self.feat_sz_s = int(feat_sz)
        self.feat_len_s = int(feat_sz ** 2)

        if bbox_task or language_task:
            # bbox_token, x1, y1, x2, y2, language_token, token1, token2
            self.task_embedding = nn.Embedding(1, predictor.input_dim)

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

        self.profile_frame_idx += 1
        ordered_keys = [
            "forward_encoder",
            "vl_fusion",
            "query_modulation",
            "generate_sequence",
            "generate",
            "reliability_head",
            "forward_decoder",
            "track_total",
        ]
        parts = []
        for key in ordered_keys:
            value = timings.get(key)
            if value is not None:
                parts.append(f"{key}={value * 1000:.2f}ms")
        print(f"[MMTrackProfile][frame {self.profile_frame_idx}] " + " ".join(parts))

    def quantize(self, seq):
        return (seq * self.num_bin).long()

    def dequantize(self, seq):
        return seq / self.num_bin

    def sequentialize(self,
                      gt_bbox=None,
                      text_embed=None,
                      bbox_type='xyxy',
                      ):
        """Args:
            gt_bbox (list[tensor]): [4, ]. (x1,y1,w,h)
            text_embed: (B, C)
        """
        with_bbox = gt_bbox is not None

        # text_embed repeat
        if with_bbox and not text_embed is None:
            text_embed = repeat(text_embed, 'b c -> b n c', n = 1) # (B, 256) to (B, n, 256)

        batch_size = gt_bbox.size(1)           # (1, b, 4)
        
        # quantize bbox coords
        if bbox_type == 'xyxy':
            seq_in_bbox = box_xywh_to_xyxy(gt_bbox)[0]    # (x1,y1,w,h) to (x1, y1, x2, y2): (B,4)
        elif bbox_type == 'cxcywh':
            seq_in_bbox = box_xywh_to_cxcywh(gt_bbox)[0]  # (x1,y1,w,h) to (cx, cy, w, h): (B,4)

        seq_in_bbox = self.quantize(seq_in_bbox)    # normalized coord -> quantize coord
        seq_in_bbox = seq_in_bbox.clamp(min=0, max=self.num_bin-1)

        seq_in = seq_in_bbox

        seq_in_embeds_bbox = self.decoder.query_embedding(seq_in)  # generate coordinates-based querys: (B,4,256)
        
        if not text_embed is None:
            task_language = self.task_embedding.weight[0].unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            seq_in_embeds_text = text_embed * task_language
            seq_in_embeds = torch.cat([seq_in_embeds_text, seq_in_embeds_bbox], dim=1)  # language + vision

        else:
            seq_in_embeds = torch.cat(
                [seq_in_embeds_bbox.new_zeros((batch_size, 1, self.decoder.d_model)), seq_in_embeds_bbox], dim=1)
        
        targets = torch.cat(
            [seq_in, seq_in.new_full((batch_size, 1), self.num_bin)], dim=-1)
        
        return seq_in_embeds, targets

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)

            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            text_features = encoded_text.last_hidden_state
            text_features = self.text_adj(text_features)
            text_features = NestedTensor(text_features, text_attention_mask)

            text_sentence_features = encoded_text.pooler_output
            text_sentence_features = self.text_adj(text_sentence_features)
        else:
            raise ValueError("Please make sure the caption is a list of string")
        return text_features, text_sentence_features

    def extract_reliability_feature(self, decoder_output, text_sentence_features):
        """Legacy lightweight reliability path kept for old configurations."""
        if not self.use_reliability or self.reliability_head is None:
            return None

        if decoder_output is None or text_sentence_features is None:
            return None

        if decoder_output.ndim == 3:
            if decoder_output.size(1) >= 5:
                decoder_feat = decoder_output[:, 1:5, :].mean(dim=1)
            else:
                decoder_feat = decoder_output.mean(dim=1)
        elif decoder_output.ndim == 2:
            decoder_feat = decoder_output
        else:
            raise ValueError(f"Unsupported decoder_output shape: {tuple(decoder_output.shape)}")

        return self.reliability_head(decoder_feat, text_sentence_features)

    def extract_roi_tokens(self, vl_features, pred_bboxes, roi_size=3):
        """Extract fixed-size ROI tokens from normalized predicted boxes."""
        batch_size, num_tokens, channels = vl_features.shape
        feat_h = self.feat_sz_s
        feat_w = self.feat_sz_s
        if num_tokens != feat_h * feat_w:
            raise ValueError(
                f"Unexpected VL token count: {num_tokens}, expected {feat_h * feat_w}"
            )
        if pred_bboxes is None or pred_bboxes.shape != (batch_size, 4):
            raise ValueError(
                f"Expected pred_bboxes shape {(batch_size, 4)}, "
                f"got {None if pred_bboxes is None else tuple(pred_bboxes.shape)}"
            )

        feat_map = (
            vl_features.transpose(1, 2)
            .reshape(batch_size, channels, feat_h, feat_w)
            .contiguous()
        )
        boxes = pred_bboxes.detach().to(
            device=vl_features.device,
            dtype=vl_features.dtype,
        ).clamp(0.0, 1.0)
        raw_x1, raw_y1, raw_x2, raw_y2 = boxes.unbind(dim=-1)
        x1 = torch.minimum(raw_x1, raw_x2)
        y1 = torch.minimum(raw_y1, raw_y2)
        x2 = torch.maximum(raw_x1, raw_x2)
        y2 = torch.maximum(raw_y1, raw_y2)

        min_w = 1.0 / feat_w
        min_h = 1.0 / feat_h
        x1 = x1.clamp(max=1.0 - min_w)
        y1 = y1.clamp(max=1.0 - min_h)
        x2 = torch.maximum(x2, x1 + min_w).clamp(max=1.0)
        y2 = torch.maximum(y2, y1 + min_h).clamp(max=1.0)

        batch_index = torch.arange(
            batch_size,
            device=boxes.device,
            dtype=boxes.dtype,
        )
        rois = torch.stack(
            [
                batch_index,
                x1 * feat_w,
                y1 * feat_h,
                x2 * feat_w,
                y2 * feat_h,
            ],
            dim=-1,
        )
        roi_features = roi_align(
            feat_map,
            rois,
            output_size=(roi_size, roi_size),
            spatial_scale=1.0,
            aligned=True,
        )
        return roi_features.flatten(2).transpose(1, 2).contiguous()

    def extract_region_reliability(
        self,
        vl_features,
        pred_bboxes,
        text_features,
        text_sentence_features,
    ):
        """Shared region-grounded reliability path for training and inference."""
        if not self.use_reliability or self.reliability_head is None:
            return None, None
        if self.reliability_type != "region_grounded":
            raise ValueError(
                "extract_region_reliability requires MODEL.RELIABILITY.TYPE=region_grounded"
            )
        if text_features is None or text_sentence_features is None:
            return None, None

        roi_size = getattr(self.cfg.MODEL.RELIABILITY, "ROI_SIZE", 3)
        detach_vl_features = getattr(
            self.cfg.MODEL.RELIABILITY,
            "DETACH_VL_FEATURES",
            False,
        )
        roi_source_features = (
            vl_features.detach()
            if detach_vl_features
            else vl_features
        )
        roi_tokens = self.extract_roi_tokens(
            vl_features=roi_source_features,
            pred_bboxes=pred_bboxes,
            roi_size=roi_size,
        )
        return self.reliability_head(
            roi_tokens=roi_tokens,
            text_tokens=text_features.tensors,
            text_padding_mask=text_features.mask,
            text_sentence_feature=text_sentence_features,
        )

    def apply_dynamic_query_modulation(
        self,
        seq_in_embeds,
        text_features=None,
        text_sentence_features=None,
        reliability=None,
    ):
        if not self.use_dynamic_query or self.query_modulator is None:
            return seq_in_embeds

        if text_sentence_features is None:
            return seq_in_embeds

        if (
            reliability is not None
            and self.cfg is not None
            and getattr(self.cfg.MODEL.DYNAMIC_QUERY, "DETACH_RELIABILITY", True)
            and isinstance(reliability, torch.Tensor)
        ):
            reliability = reliability.detach()

        dq_type = "lite"
        if self.cfg is not None:
            dq_type = getattr(self.cfg.MODEL.DYNAMIC_QUERY, "TYPE", "lite")

        if dq_type == "fine_grained_temporal":
            if text_features is None:
                return seq_in_embeds
            return self.query_modulator(
                base_query=seq_in_embeds,
                text_tokens=text_features.tensors,
                text_padding_mask=text_features.mask,
                text_sentence_feature=text_sentence_features,
                reliability=reliability,
            )

        return self.query_modulator(
            base_query=seq_in_embeds,
            text_sentence_feature=text_sentence_features,
            reliability=reliability,
        )

    def prepare_temporal_reliability(
        self,
        pred_reliability_prev,
        clamp_min=0.3,
        clamp_max=0.7,
        detach=True,
    ):
        if pred_reliability_prev is None:
            return None

        if not isinstance(pred_reliability_prev, torch.Tensor):
            raise TypeError("pred_reliability_prev must be a torch.Tensor")

        rel = pred_reliability_prev
        if rel.ndim > 1:
            rel = rel.view(rel.size(0), -1).mean(dim=1, keepdim=True)
        elif rel.ndim == 1:
            rel = rel.unsqueeze(-1)

        rel = torch.sigmoid(rel)
        rel = rel.clamp(clamp_min, clamp_max)

        if detach:
            rel = rel.detach()

        return rel

    def _forward_train_step(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        search_attn_mask: torch.Tensor,
        search_anno=None,
        search_segmask_vertices=None,
        exp_str=None,
        ce_template_mask=None,
        ce_keep_rate=None,
        return_last_attn=False,
        prev_reliability_for_query=None,
    ):
        """
        Single-step training forward used by both the default path and the
        temporal Stage C path.
        """
        with_bbox = search_anno is not None
        with_mask = search_segmask_vertices is not None
        text_features, text_sentence_features = None, None

        if exp_str:
            text_features, text_sentence_features = self.forward_text(exp_str, device=search.device)

        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn)

        enc_opt = x[:, -self.feat_len_s:]
        enc_opt = self.bottleneck(enc_opt)

        enc_mask, enc_pos_embeds = self.decoder.memory_mask_pos_enc(search_attn_mask, self.feat_sz_s)
        text_pos = self.text_pos(text_features)

        vl_features = self.vl_fusion(enc_opt,
                                    text_features.tensors,
                                    query_pos=enc_pos_embeds,
                                    memory_pos=text_pos,
                                    query_key_padding_mask=enc_mask,
                                    memory_key_padding_mask=text_features.mask,
                                    need_weights=False)

        seq_in_embeds, targets = self.sequentialize(gt_bbox=search_anno,
                                                    text_embed=text_sentence_features,
                                                    bbox_type=self.bbox_type)

        reliability_for_query = prev_reliability_for_query
        if reliability_for_query is None:
            reliability_proxy = None
            use_query_gate = False
            use_scale_modulation = False
            if self.cfg is not None:
                use_query_gate = getattr(self.cfg.MODEL.DYNAMIC_QUERY, "USE_GATE", False)
                use_scale_modulation = getattr(
                    self.cfg.MODEL.DYNAMIC_QUERY, "USE_SCALE_MODULATION", False
                )
            if (
                self.use_dynamic_query
                and text_sentence_features is not None
                and self.cfg is not None
                and (use_query_gate or use_scale_modulation)
            ):
                reliability_proxy = seq_in_embeds.new_full((seq_in_embeds.size(0), 1), 0.5)
            reliability_for_query = reliability_proxy

        if self.use_dynamic_query and text_sentence_features is not None:
            seq_in_embeds = self.apply_dynamic_query_modulation(
                seq_in_embeds=seq_in_embeds,
                text_features=text_features,
                text_sentence_features=text_sentence_features,
                reliability=reliability_for_query,
            )

        logits, aux_logits = self.decoder(seq_in_embeds, vl_features, enc_mask, enc_pos_embeds,
                                        return_intermediate_output=self.aux_loss)

        aux_outputs = []
        if self.aux_loss:
            for i in range(len(aux_outputs)):
                if aux_logits[i].size(1) > 5:
                    aux_logits[i] = aux_logits[i][:, :5]
                aux_logits[i] = self.predictor[i](aux_logits[i])

                aux_preds = self.train_statis(aux_logits[i], with_bbox=with_bbox, with_mask=with_mask)
                aux_outputs.append({
                                'logits': aux_logits[i],
                                'pred_bboxes': aux_preds['pred_bboxes'],
                                'pred_masks': aux_preds['pred_masks'],
                            })

        if logits.size(1) > 5:
            logits = logits[:, :5]
        decoder_output = logits
        logits = self.predictor(logits)
        preds = self.train_statis(logits, with_bbox=with_bbox, with_mask=with_mask)

        pred_reliability = None
        if exp_str and self.use_reliability:
            if self.reliability_type == "region_grounded":
                if preds["pred_bboxes"] is not None:
                    pred_reliability, _ = self.extract_region_reliability(
                        vl_features=vl_features,
                        pred_bboxes=preds["pred_bboxes"],
                        text_features=text_features,
                        text_sentence_features=text_sentence_features,
                    )
            else:
                pred_reliability = self.extract_reliability_feature(
                    decoder_output,
                    text_sentence_features,
                )

        return {
                'logits': logits,
                'aux_outputs': aux_outputs,
                'targets': targets,
                'pred_bboxes': preds['pred_bboxes'],
                'pred_masks': preds['pred_masks'],
                'pred_reliability': pred_reliability,
            }

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                search_attn_mask: torch.Tensor,
                search_anno=None,
                search_segmask_vertices=None,
                exp_str=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        """
        search_anno: (x1,y1,w,h)
        exp_str List[str]: language descriptions
        """
        return self._forward_train_step(
            template=template,
            search=search,
            search_attn_mask=search_attn_mask,
            search_anno=search_anno,
            search_segmask_vertices=search_segmask_vertices,
            exp_str=exp_str,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
            prev_reliability_for_query=None,
        )

    def forward_temporal(
        self,
        template: torch.Tensor,
        search_prev: torch.Tensor,
        search_curr: torch.Tensor,
        search_prev_attn_mask: torch.Tensor,
        search_curr_attn_mask: torch.Tensor,
        search_prev_anno=None,
        search_curr_anno=None,
        exp_str=None,
        ce_template_mask=None,
        ce_keep_rate=None,
        return_last_attn=False,
    ):
        with torch.no_grad():
            prev_out = self._forward_train_step(
                template=template,
                search=search_prev,
                search_attn_mask=search_prev_attn_mask,
                search_anno=search_prev_anno,
                search_segmask_vertices=None,
                exp_str=exp_str,
                ce_template_mask=ce_template_mask,
                ce_keep_rate=ce_keep_rate,
                return_last_attn=return_last_attn,
                prev_reliability_for_query=None,
            )

        prev_rel = self.prepare_temporal_reliability(
            prev_out.get("pred_reliability", None),
            clamp_min=0.3,
            clamp_max=0.7,
            detach=True,
        )

        curr_out = self._forward_train_step(
            template=template,
            search=search_curr,
            search_attn_mask=search_curr_attn_mask,
            search_anno=search_curr_anno,
            search_segmask_vertices=None,
            exp_str=exp_str,
            ce_template_mask=ce_template_mask,
            ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn,
            prev_reliability_for_query=prev_rel,
        )
        curr_out["temporal_prev_reliability"] = prev_rel
        return curr_out

    def train_statis(self, logits, with_bbox=False, with_mask=False, inference=False):
        # training statistics
        with torch.no_grad():
            if with_mask and with_bbox:
                logits_bbox = logits[:, :4, :-1]
                scores_bbox = F.softmax(logits_bbox, dim=-1)
                _, seq_out_bbox = scores_bbox.max(dim=-1, keepdim=False)

                logits_mask = logits[:, 5:-1, :-1]  # (B, bbox+mask, 1001) to (B, mask, 1000)
                scores_mask = F.softmax(logits_mask, dim=-1)
                _, seq_out_mask = scores_mask.max(dim=-1, keepdim=False)

                seq_out_dict = dict(seq_out_bbox=seq_out_bbox.detach(), seq_out_mask=seq_out_mask.detach())
            else:
                if with_bbox:
                    logits = logits[:, :-1, :-1]
                elif with_mask:
                    logits = logits[:, :-1, :-1]

                scores = F.softmax(logits, dim=-1)
                _, seq_out = scores.max(dim=-1, keepdim=False)

                if with_bbox:
                    seq_out_dict = dict(seq_out_bbox=seq_out.detach())
                elif with_mask:
                    seq_out_dict = dict(seq_out_mask=seq_out.detach())

            return self.get_predictions(seq_out_dict, inference=inference)

    def get_predictions(self, seq_out_dict, inference=False, rescale=False):
        """Args:
            seq_out_dict (dict[tensor]): [batch_size, 4/2*num_ray+1].

            rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
                back to `ori_shape`.
        """
        pred_bboxes, pred_masks = None, None
        with_bbox = 'seq_out_bbox' in seq_out_dict
        with_mask = 'seq_out_mask' in seq_out_dict

        if with_mask:
            seq_out_mask = seq_out_dict['seq_out_mask']
            seq_out_mask = seq_out_mask.cpu().numpy()
            pred_masks = []
            for pred_mask in seq_out_mask:
                if len(pred_mask) % 2 != 0:  # must be even
                    pred_mask = pred_mask[:-1]

                pred_mask = self.dequantize(pred_mask)
                pred_mask *= self.input_size  # norm coords --> input image coords

                if len(pred_mask) < 4:  # at least three points to make a mask
                    pred_mask = np.array([0, 0, 0, 0, 0, 0], order='F', dtype=np.uint8)
                    pred_mask = [pred_mask]
                elif len(pred_mask) == 4:
                    pred_mask = pred_mask[None]  # as a bbox
                else:
                    pred_mask = [pred_mask]  # as a polygon

                # [MOD] fix numpy alias bug: use np instead of numpy
                pred_rles = maskUtils.frPyObjects(pred_mask, self.input_size, self.input_size)  # list[rle]
                pred_rle = maskUtils.merge(pred_rles)

                if inference:
                    pred_mask = BitmapMasks(maskUtils.decode(pred_rle)[None], self.input_size, self.input_size)
                    pred_mask = pred_mask.masks[0]
                    pred_masks.append(pred_mask)
                else:
                    pred_masks.append(pred_rle)

        if with_bbox:
            seq_out_bbox = seq_out_dict['seq_out_bbox']
            pred_bboxes = self.dequantize(seq_out_bbox)
            pred_bboxes = pred_bboxes.double()

        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks)

    def forward_encoder(self,
                        template: torch.Tensor,
                        search: torch.Tensor,
                        ce_template_mask=None,
                        ce_keep_rate=None,
                        return_last_attn=False,):
        x, _ = self.backbone(z=template, x=search,
                            ce_template_mask=ce_template_mask,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=return_last_attn, )
        
        # encoder output for the search region (B, HW, C)
        enc_opt = x[:, -self.feat_len_s:]
        
        # adjust dims
        enc_opt = self.bottleneck(enc_opt)

        return enc_opt
    
    def forward_decoder(self, 
                        enc_opt: torch.Tensor,
                        search_attn_mask: torch.Tensor,
                        gt_mask_vertices=None,
                        text_features=None,
                        text_sentence_features=None,
                        prev_reliability=None,
                        suppress_dynamic_query=False,
                        with_bbox=False,
                        with_mask=False,
                        inference=False,
                        update_reliability=True):
        timings = {}
        
        h, w = search_attn_mask.shape[-2:] # search shape

        # generate position encoding for the encoder output
        enc_mask, enc_pos_embeds = self.decoder.memory_mask_pos_enc(search_attn_mask, self.feat_sz_s)
        text_pos = self.text_pos(text_features)  # [batch_size, length, c]

        # vision language early-fusion
        vl_features, timings["vl_fusion"] = self._profile_time_block(
            lambda: self.vl_fusion(
                enc_opt,
                text_features.tensors,
                query_pos=enc_pos_embeds,
                memory_pos=text_pos,
                memory_key_padding_mask=text_features.mask,
            )
        )

        # decode coordinates-based querys
        seq_out_dict, seq_timings = self.generate_sequence(
            vl_features,
            enc_mask,
            enc_pos_embeds,
            text_features=text_features,
            text_embed=text_sentence_features,
            prev_reliability=prev_reliability,
            suppress_dynamic_query=suppress_dynamic_query,
            with_bbox=with_bbox,
            with_mask=with_mask,
        )
        timings.update(seq_timings)
        
        preds = self.get_predictions(seq_out_dict, inference=inference)

        pred_reliability = None
        if (
            update_reliability
            and self.use_reliability
            and with_bbox
            and text_sentence_features is not None
        ):
            if self.reliability_type == "region_grounded":
                reliability_output, timings["reliability_head"] = self._profile_time_block(
                    lambda: self.extract_region_reliability(
                        vl_features=vl_features,
                        pred_bboxes=preds["pred_bboxes"],
                        text_features=text_features,
                        text_sentence_features=text_sentence_features,
                    )
                )
                pred_reliability, _ = reliability_output
            else:
                decoder_like_feat = vl_features.mean(dim=1)
                pred_reliability, timings["reliability_head"] = self._profile_time_block(
                    lambda: self.extract_reliability_feature(
                        decoder_like_feat,
                        text_sentence_features,
                    )
                )
            preds["pred_reliability"] = torch.sigmoid(pred_reliability)

        return preds, timings

    def generate_sequence(
        self,
        memory,
        memory_mask,
        memory_pos_embeds,
        text_features=None,
        text_embed=None,
        prev_reliability=None,
        suppress_dynamic_query=False,
        with_bbox=False,
        with_mask=False,
    ):
        """Args:
            memory (tensor): encoder's output, [batch_size, h*w, d_model].

            x_mask (tensor): [batch_size, h*w], dtype is torch.bool, True means
                ignored position.

            x_pos_embeds (tensor): [batch_size, h*w, d_model].
        """
        with_language = text_embed is not None
        timings = {}

        batch_size = memory.size(0)
        
        text_sentence_feature = text_embed

        if with_language:
            task_language = self.task_embedding.weight[0].unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            text_embed = text_embed.unsqueeze(0).expand(batch_size, -1, -1)
            seq_in_embeds = text_embed * task_language
        else:
            seq_in_embeds = memory.new_zeros((batch_size, 1, self.decoder.d_model))  # (B, 1, 256)

        if self.use_dynamic_query and text_sentence_feature is not None:
            reliability_tensor = None
            if prev_reliability is not None:
                if isinstance(prev_reliability, torch.Tensor):
                    reliability_tensor = prev_reliability.to(seq_in_embeds.device)
                    if reliability_tensor.ndim == 0:
                        reliability_tensor = reliability_tensor.view(1, 1).expand(batch_size, 1)
                    elif reliability_tensor.ndim == 1:
                        reliability_tensor = reliability_tensor.view(-1, 1)
                else:
                    reliability_tensor = seq_in_embeds.new_full((batch_size, 1), float(prev_reliability))

            suppress_low_rel_query = bool(suppress_dynamic_query)
            suppress_low_rel_query = suppress_low_rel_query or (
                os.environ.get("MMTRACK_QUERY_SUPPRESS_LOW_REL", "0") == "1"
                and reliability_tensor is not None
            )
            if suppress_low_rel_query and reliability_tensor is not None:
                suppress_thr = float(os.environ.get("MMTRACK_QUERY_SUPPRESS_REL_THR", "0.45"))
                suppress_low_rel_query = bool(suppress_dynamic_query) or float(reliability_tensor.detach().mean().item()) < suppress_thr

            if not suppress_low_rel_query:
                seq_in_embeds, timings["query_modulation"] = self._profile_time_block(
                    lambda: self.apply_dynamic_query_modulation(
                        seq_in_embeds=seq_in_embeds,
                        text_features=text_features,
                        text_sentence_features=text_sentence_feature,
                        reliability=reliability_tensor,
                    )
                )
        
        if with_mask:
            decode_steps = self.num_ray * 2 + 1
        elif with_bbox:
            decode_steps = 4
        
        seq_out, timings["generate"] = self._profile_time_block(
            lambda: self.generate(seq_in_embeds, memory, memory_pos_embeds, memory_mask, decode_steps)
        )
        timings["generate_sequence"] = timings["generate"]
        
        if with_bbox:
            return dict(seq_out_bbox=seq_out), timings
        elif with_mask:
            return dict(seq_out_mask=seq_out), timings

    def generate(self, seq_in_embeds, memory, memory_pos_embeds, memory_mask, decode_steps):
        seq_out = []
        for step in range(decode_steps):
            # Forward decoder
            out = self.decoder(seq_in_embeds, memory, memory_mask, memory_pos_embeds, is_inference=True)  # (B, 1, 256)
            logits = out[:, -1, :]

            # Forward predictor
            logits = self.predictor(logits)  # (B, 1001)
            logits = logits[:, :-1]

            probs = F.softmax(logits, dim=-1)
            if self.top_p > 0.:
                sorted_score, sorted_idx = torch.sort(probs, descending=True)
                cum_score = sorted_score.cumsum(dim=-1)
                sorted_idx_to_remove = cum_score > self.top_p
                sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                sorted_idx_to_remove[..., 0] = 0
                idx_to_remove = sorted_idx_to_remove.scatter(
                    1, sorted_idx, sorted_idx_to_remove)
                probs = probs.masked_fill(idx_to_remove, 0.)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = probs.max(dim=-1, keepdim=True)

            seq_in_embeds = torch.cat(
                [seq_in_embeds, self.decoder.query_embedding(next_token)], dim=1)

            seq_out.append(next_token)

        seq_out = torch.cat(seq_out, dim=-1)
        return seq_out

    def forward_text_encoder(self, exp_str, device):
        text_features, text_sentence_features = self.forward_text(exp_str, device=device)
        return text_features, text_sentence_features

    def track(self, template: torch.Tensor,
                search,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                gt_mask_vertices=None,
                text_features=None,
                text_sentence_features=None,
                prev_reliability=None,
                suppress_dynamic_query=False,
                with_bbox=False,
                with_mask=False,
                inference=False,
                update_reliability=True,
                ):
        timings = {}
        self._profile_sync()
        track_start = time.perf_counter() if self.profile_enabled else None

        enc_opt, timings["forward_encoder"] = self._profile_time_block(
            lambda: self.forward_encoder(template=template,
                                         search=search.tensors,
                                         ce_template_mask=ce_template_mask,
                                         ce_keep_rate=ce_keep_rate,
                                         return_last_attn=return_last_attn)
        )
        decoder_result, timings["forward_decoder"] = self._profile_time_block(
            lambda: self.forward_decoder(enc_opt=enc_opt,
                                         search_attn_mask=search.mask,
                                         gt_mask_vertices=gt_mask_vertices,
                                         text_features=text_features,
                                         text_sentence_features=text_sentence_features,
                                         prev_reliability=prev_reliability,
                                         suppress_dynamic_query=suppress_dynamic_query,
                                         with_bbox=with_bbox,
                                         with_mask=with_mask,
                                         inference=inference,
                                         update_reliability=update_reliability)
        )
        preds, decoder_timings = decoder_result
        timings.update(decoder_timings)
        if self.profile_enabled:
            self._profile_sync()
            timings["track_total"] = time.perf_counter() - track_start
            self._profile_log(timings)
        return preds


def build_mmtrack(cfg, is_training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    compression_ratio = getattr(cfg.MODEL.BACKBONE, 'COMPRESSION_RATIO', 4)
    init_from_patch_embed = getattr(cfg.MODEL.BACKBONE, 'INIT_FROM_PATCH_EMBED', False)
    
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        use_ci2p = getattr(cfg.MODEL.BACKBONE, 'USE_CI2P', False)
        compression_dim = getattr(cfg.MODEL.BACKBONE, 'COMPRESSION_DIM', 64)
        backbone = vit_base_patch16_224(
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            use_ci2p=use_ci2p,
            compression_dim=compression_dim,
            compression_ratio=compression_ratio,
            init_from_patch_embed=init_from_patch_embed,
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        use_ci2p = getattr(cfg.MODEL.BACKBONE, 'USE_CI2P', False)
        compression_dim = getattr(cfg.MODEL.BACKBONE, 'COMPRESSION_DIM', 64)
        backbone = vit_base_patch16_224_ce(drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           use_ci2p=use_ci2p,
                                           compression_dim=compression_dim,
                                           compression_ratio=compression_ratio,
                                           init_from_patch_embed=init_from_patch_embed,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        use_ci2p = getattr(cfg.MODEL.BACKBONE, 'USE_CI2P', False)
        compression_dim = getattr(cfg.MODEL.BACKBONE, 'COMPRESSION_DIM', 64)
        backbone = vit_large_patch16_224_ce(drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            use_ci2p=use_ci2p,
                                            compression_dim=compression_dim,
                                            compression_ratio=compression_ratio,
                                            init_from_patch_embed=init_from_patch_embed,
                                            )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # Build Text Encoder
    tokenizer, text_encoder = None, None
    if cfg.MODEL.TEXT_ENCODER == 'roberta-base':
        tokenizer = RobertaTokenizerFast.from_pretrained('pretrained_networks/roberta-base')
        try:
            text_encoder = RobertaModel.from_pretrained(
                'pretrained_networks/roberta-base',
                use_safetensors=False,
            )
        except TypeError:
            # Older transformers releases do not expose use_safetensors.
            text_encoder = RobertaModel.from_pretrained('pretrained_networks/roberta-base')
    elif cfg.MODEL.TEXT_ENCODER == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('pretrained_networks/bert-base-cased')
        text_encoder = BertModel.from_pretrained('bert-base-cased')
    
    decoder = build_decoder(cfg)
    predictor = build_predictor(cfg)

    stride = cfg.MODEL.BACKBONE.STRIDE

    use_ci2p = getattr(cfg.MODEL.BACKBONE, 'USE_CI2P', False)
    stride_for_feat_sz = stride
    if use_ci2p and compression_ratio == 4:
        stride_for_feat_sz = stride * 2
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride_for_feat_sz)

    language_task = getattr(cfg.TRAIN, "LANGUAGE_TASK", False)

    model = MMTrack(
        backbone,
        decoder,
        predictor,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        feat_sz=feat_sz,
        input_size=cfg.DATA.SEARCH.SIZE,
        bbox_task=cfg.TRAIN.BBOX_TASK,
        language_task=language_task,
        num_vlfusion_layers=cfg.MODEL.VLFUSION_LAYERS,
        vl_input_type=cfg.MODEL.VL_INPUT_TYPE,
        bbox_type=cfg.MODEL.DECODER.BBOX_TYPE,
        aux_loss=cfg.TRAIN.AUX_LOSS,
        cfg=cfg,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and cfg.TRAIN.LANGUAGE_TASK and is_training:
        model_path = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        try:
            checkpoint = torch.load(model_path, map_location="cpu")

            # Support both {"net": state_dict} and plain state_dict checkpoints.
            if isinstance(checkpoint, dict) and "net" in checkpoint:
                state_dict = checkpoint["net"]
            else:
                state_dict = checkpoint

            # [MOD] 1) load only shape-matched keys first
            model_state = model.state_dict()
            loadable_state_dict = {}
            skipped = []

            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    loadable_state_dict[k] = v
                else:
                    skipped.append((k, tuple(v.shape), tuple(model_state[k].shape) if k in model_state else None))

            missing_keys, unexpected_keys = model.load_state_dict(loadable_state_dict, strict=False)

            print("Load OSTrack model from:", model_path)
            print("Loaded keys:", len(loadable_state_dict))
            print("Skipped keys:", skipped[:20])
            print("missing keys:", missing_keys[:20])
            print("unexpected keys:", unexpected_keys[:20])

            # =========================================================
            # [MOD] 2) CI2P positional embeddings:
            #         interpolate from OSTrack checkpoint instead of
            #         leaving them randomly initialized.
            # =========================================================
            if use_ci2p:
                with torch.no_grad():
                    backbone_module = model.backbone

                    # backbone.pos_embed
                    ckpt_tensor, ckpt_key = _get_ckpt_tensor(
                        state_dict,
                        ["backbone.pos_embed", "pos_embed"]
                    )
                    if ckpt_tensor is not None and hasattr(backbone_module, "pos_embed"):
                        target = backbone_module.pos_embed
                        target_hw = _infer_target_hw_from_pos_embed_shape(target)
                        resized = _resize_pos_embed_2d(ckpt_tensor, target_hw)
                        if resized.shape == target.shape:
                            target.copy_(resized)
                            print(f"[CI2P init] interpolated {ckpt_key} -> backbone.pos_embed {tuple(target.shape)}")
                        else:
                            print(f"[CI2P init] skip backbone.pos_embed, resized {tuple(resized.shape)} != target {tuple(target.shape)}")

                    # backbone.pos_embed_z
                    ckpt_tensor, ckpt_key = _get_ckpt_tensor(
                        state_dict,
                        ["backbone.pos_embed_z", "pos_embed_z"]
                    )
                    if ckpt_tensor is not None and hasattr(backbone_module, "pos_embed_z"):
                        target = backbone_module.pos_embed_z
                        target_hw = _infer_target_hw_from_pos_embed_shape(target)
                        resized = _resize_pos_embed_2d(ckpt_tensor, target_hw)
                        if resized.shape == target.shape:
                            target.copy_(resized)
                            print(f"[CI2P init] interpolated {ckpt_key} -> backbone.pos_embed_z {tuple(target.shape)}")
                        else:
                            print(f"[CI2P init] skip backbone.pos_embed_z, resized {tuple(resized.shape)} != target {tuple(target.shape)}")

                    # backbone.pos_embed_x
                    ckpt_tensor, ckpt_key = _get_ckpt_tensor(
                        state_dict,
                        ["backbone.pos_embed_x", "pos_embed_x"]
                    )
                    if ckpt_tensor is not None and hasattr(backbone_module, "pos_embed_x"):
                        target = backbone_module.pos_embed_x
                        target_hw = _infer_target_hw_from_pos_embed_shape(target)
                        resized = _resize_pos_embed_2d(ckpt_tensor, target_hw)
                        if resized.shape == target.shape:
                            target.copy_(resized)
                            print(f"[CI2P init] interpolated {ckpt_key} -> backbone.pos_embed_x {tuple(target.shape)}")
                        else:
                            print(f"[CI2P init] skip backbone.pos_embed_x, resized {tuple(resized.shape)} != target {tuple(target.shape)}")

                    _init_ci2p_from_patch_embed(model, state_dict)

        except Exception as e:
            print(f"Warning: OSTrack model weights are not loaded ! error={type(e).__name__}: {e}")
  
    return model
