import torch
import torch.nn as nn


class ReliabilityHead(nn.Module):
    """
    A lightweight reliability estimator for MMTrack.

    Inputs:
        decoder_feat: [B, C]
        text_feat:    [B, C]

    Output:
        reliability_logit: [B, 1]
    """

    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, decoder_feat, text_feat):
        """
        Args:
            decoder_feat: [B, C]
            text_feat: [B, C]
        Returns:
            reliability_logit: [B, 1]
        """
        x = torch.cat([decoder_feat, text_feat], dim=-1)
        return self.mlp(x)