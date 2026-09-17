import torch
import torch.nn as nn


class ReliabilityAwareQueryModulator(nn.Module):
    """
    B1: text-guided query residual + optional reliability gate.

    Inputs:
        base_query: [B, N, C]
        text_sentence_feature: [B, C]
        reliability: [B], [B, 1], or None
    Returns:
        modulated_query: [B, N, C]
    """

    def __init__(
        self,
        hidden_dim=256,
        residual_scale=0.5,
        use_gate=True,
        use_scale_modulation=False,
        scale_min=0.9,
        scale_max=1.1,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.use_gate = use_gate
        self.use_scale_modulation = use_scale_modulation
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.text_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, base_query, text_sentence_feature, reliability=None):
        """
        Args:
            base_query: [B, N, C]
            text_sentence_feature: [B, C]
            reliability: [B], [B, 1], or None
        """
        delta_q = self.text_mlp(text_sentence_feature)  # [B, C]
        scale = base_query.new_full((base_query.size(0), 1, 1), self.residual_scale)

        if self.use_scale_modulation and reliability is not None:
            if reliability.ndim == 1:
                reliability = reliability.unsqueeze(-1)  # [B, 1]
            scale_factor = self.scale_min + (self.scale_max - self.scale_min) * reliability
            scale = scale * scale_factor.unsqueeze(1)
        elif self.use_gate and reliability is not None:
            if reliability.ndim == 1:
                reliability = reliability.unsqueeze(-1)  # [B, 1]
            # Keep the gate conservative so Stage B tests whether light
            # reliability-aware scaling helps without destabilizing queries.
            gate = 0.8 + 0.2 * self.gate_mlp(reliability)  # [B, 1]
            delta_q = delta_q * gate

        modulated_query = base_query.clone()
        delta_q = delta_q.unsqueeze(1)  # [B, 1, C]
        modulated_query[:, :1, :] = modulated_query[:, :1, :] + scale * delta_q
        return modulated_query
