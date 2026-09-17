import torch
import torch.nn as nn


class FineGrainedTemporalQueryModulator(nn.Module):
    """
    Reliability-conditioned word-level query modulation.

    The module keeps the original conservative RQM behavior by updating only
    the first autoregressive query token.
    """

    def __init__(
        self,
        hidden_dim=256,
        residual_scale=0.08,
        scale_min=0.9,
        scale_max=1.1,
        dropout=0.1,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.rel_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.token_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        base_query,
        text_tokens,
        text_padding_mask,
        text_sentence_feature,
        reliability=None,
    ):
        if reliability is None:
            reliability = base_query.new_full((base_query.size(0), 1), 0.5)

        if reliability.ndim == 1:
            reliability = reliability.unsqueeze(-1)

        reliability = reliability.clamp(0.0, 1.0)
        rel_feature = self.rel_mlp(reliability)

        rel_tokens = rel_feature.unsqueeze(1).expand(-1, text_tokens.size(1), -1)
        token_logits = self.token_score(
            torch.cat([text_tokens, rel_tokens], dim=-1)
        ).squeeze(-1)
        token_logits = token_logits.masked_fill(
            text_padding_mask,
            torch.finfo(token_logits.dtype).min,
        )

        token_weights = torch.softmax(token_logits, dim=-1)
        token_context = torch.sum(token_weights.unsqueeze(-1) * text_tokens, dim=1)

        delta_q = self.query_mlp(
            torch.cat([token_context, text_sentence_feature, rel_feature], dim=-1)
        )
        delta_q = self.out_norm(delta_q)

        scale_factor = self.scale_min + (self.scale_max - self.scale_min) * reliability

        modulated_query = base_query.clone()
        modulated_query[:, :1, :] = (
            modulated_query[:, :1, :]
            + self.residual_scale * scale_factor.unsqueeze(1) * delta_q.unsqueeze(1)
        )
        return modulated_query
