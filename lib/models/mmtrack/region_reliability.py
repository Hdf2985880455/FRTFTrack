import torch
import torch.nn as nn


class RegionGroundedReliabilityHead(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.roi_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        roi_tokens,
        text_tokens,
        text_padding_mask,
        text_sentence_feature,
    ):
        roi_query = self.roi_norm(roi_tokens)
        aligned_tokens, _ = self.cross_attn(
            query=roi_query,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )
        aligned_tokens = self.fusion_norm(roi_tokens + aligned_tokens)
        roi_semantic_feature = aligned_tokens.mean(dim=1)
        reliability_input = torch.cat(
            [roi_semantic_feature, text_sentence_feature],
            dim=-1,
        )
        reliability_logit = self.mlp(reliability_input)
        return reliability_logit, roi_semantic_feature