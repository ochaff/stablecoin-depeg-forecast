import math
import argparse
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import RevIN, Baseclass_forecast


# ============================================================
# Sparse probability maps
# ============================================================

def _make_ix_like(x: torch.Tensor, dim: int) -> torch.Tensor:
    d = x.size(dim)
    rho = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    view = [1] * x.dim()
    view[dim] = d
    return rho.view(view)


def sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    zs = torch.sort(x, dim=dim, descending=True).values
    rhos = _make_ix_like(zs, dim)
    cssv = zs.cumsum(dim) - 1
    support = zs > cssv / rhos
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = cssv.gather(dim, k - 1) / k
    return torch.clamp(x - tau, min=0.0)


def entmax_bisect(
    x: torch.Tensor,
    alpha: float = 1.5,
    dim: int = -1,
    n_iter: int = 50,
) -> torch.Tensor:
    if not (1.0 < alpha <= 2.0):
        raise ValueError("alpha must be in (1, 2].")

    x = x - x.max(dim=dim, keepdim=True).values
    power = 1.0 / (alpha - 1.0)
    x_scaled = (alpha - 1.0) * x

    tau_lo = x_scaled.min(dim=dim, keepdim=True).values - 1.0
    tau_hi = x_scaled.max(dim=dim, keepdim=True).values

    for _ in range(n_iter):
        tau_m = 0.5 * (tau_lo + tau_hi)
        p_m = torch.clamp(x_scaled - tau_m, min=0.0) ** power
        s_m = p_m.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(s_m > 1.0, tau_m, tau_lo)
        tau_hi = torch.where(s_m <= 1.0, tau_m, tau_hi)

    p = torch.clamp(x_scaled - tau_hi, min=0.0) ** power
    return p / (p.sum(dim=dim, keepdim=True) + 1e-12)


def sparse_probability_map(x: torch.Tensor, kind: str = "sparsemax", dim: int = -1) -> torch.Tensor:
    if kind == "softmax":
        return torch.softmax(x, dim=dim)
    if kind == "sparsemax":
        return sparsemax(x, dim=dim)
    if kind in {"entmax", "entmax15", "entmax1.5"}:
        return entmax_bisect(x, alpha=1.5, dim=dim)
    raise ValueError(f"Unknown attention map: {kind}")


# ============================================================
# Hard-concrete gate
# ============================================================

class HardConcreteGate(nn.Module):
    def __init__(
        self,
        temperature: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.zeta = zeta
        self.eps = eps

    def forward(self, log_alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            u = torch.rand_like(log_alpha).clamp(self.eps, 1 - self.eps)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / self.temperature)
        else:
            s = torch.sigmoid(log_alpha)

        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, 0.0, 1.0)

        expected_open = torch.sigmoid(
            log_alpha - self.temperature * math.log(-self.gamma / self.zeta)
        )
        return z, expected_open


# ============================================================
# Basic layers
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ctx = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim is not None else None
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.fc1(x)
        if context is not None and self.ctx is not None:
            h = h + self.ctx(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        g = torch.sigmoid(self.gate(h))
        out = self.skip(x) + g * h
        return self.norm(out)


# ============================================================
# Sparse multi-head attention
# ============================================================

class MultiHeadSparseAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        attn_activation: str = "sparsemax",
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.attn_activation = attn_activation

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x = x.view(b, l, self.n_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, L, Dh]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if v is None:
            v = k

        qh = self._reshape(self.q_proj(q))
        kh = self._reshape(self.k_proj(k))
        vh = self._reshape(self.v_proj(v))

        scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # [B, heads, Lq, Lk]
        attn = sparse_probability_map(scores, kind=self.attn_activation, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, vh)
        out = out.transpose(1, 2).contiguous().view(q.shape[0], q.shape[1], self.d_model)
        out = self.o_proj(out)

        weights = attn.mean(dim=1) if need_weights else None
        return out, weights


class SparseTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_activation: str = "sparsemax",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSparseAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attn_activation=attn_activation,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class CrossFusionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_activation: str = "sparsemax",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSparseAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attn_activation=attn_activation,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadSparseAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attn_activation=attn_activation,
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        target_tokens: torch.Tensor,
        cov_tokens: Optional[torch.Tensor],
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h, _ = self.self_attn(self.norm1(target_tokens), self.norm1(target_tokens), self.norm1(target_tokens))
        x = target_tokens + h

        attn_weights = None
        if cov_tokens is not None and cov_tokens.size(1) > 0:
            h, attn_weights = self.cross_attn(
                self.norm2(x), cov_tokens, cov_tokens, need_weights=need_weights
            )
            x = x + h

        x = x + self.ff(self.norm3(x))
        return x, attn_weights


# ============================================================
# NHITS target branch
# ============================================================

class NHiTSLatentBlock(nn.Module):
    """
    NHITS-like residual block:
      residual target -> pooled summary -> backcast + latent horizon representation
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        pooled_size: int,
        hidden_size: int,
        d_model: int,
        n_knots: int,
        n_mlp_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_knots = n_knots

        self.avg_pool = nn.AdaptiveAvgPool1d(pooled_size)
        self.max_pool = nn.AdaptiveMaxPool1d(pooled_size)

        layers = []
        in_dim = 2 * pooled_size
        for i in range(n_mlp_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_size, hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        self.backcast_proj = nn.Linear(hidden_size, seq_len)
        self.forecast_knots_proj = nn.Linear(hidden_size, d_model * n_knots)

    def forward(self, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # residual: [B, L]
        x = residual.unsqueeze(1)  # [B,1,L]
        pooled = torch.cat(
            [self.avg_pool(x).squeeze(1), self.max_pool(x).squeeze(1)],
            dim=-1,
        )  # [B, 2 * pooled_size]

        hidden = self.mlp(pooled)
        backcast = self.backcast_proj(hidden)  # [B, L]

        knots = self.forecast_knots_proj(hidden).view(-1, self.d_model, self.n_knots)  # [B,D,K]
        horizon_latent = F.interpolate(
            knots,
            size=self.pred_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)  # [B,H,D]

        new_residual = residual - backcast
        return new_residual, horizon_latent


class NHiTSTargetEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        pooled_sizes,
        hidden_size: int,
        n_knots: int,
        n_mlp_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            NHiTSLatentBlock(
                seq_len=seq_len,
                pred_len=pred_len,
                pooled_size=p,
                hidden_size=hidden_size,
                d_model=d_model,
                n_knots=n_knots,
                n_mlp_layers=n_mlp_layers,
                dropout=dropout,
            )
            for p in pooled_sizes
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, y_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # y_target: [B, L]
        residual = y_target
        latent_sum = None

        for block in self.blocks:
            residual, latent = block(residual)
            latent_sum = latent if latent_sum is None else latent_sum + latent

        latent_sum = self.norm(latent_sum)
        return latent_sum, residual  # [B,H,D], [B,L]


# ============================================================
# Covariate branch: variates as tokens + sparse selection
# ============================================================

class VariableSelectionGate(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        dropout: float = 0.1,
        selector_activation: str = "sparsemax",
        use_hard_concrete: bool = True,
    ):
        super().__init__()
        self.selector_activation = selector_activation
        self.use_hard_concrete = use_hard_concrete

        self.var_grn = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=hidden_size,
            output_dim=d_model,
            context_dim=d_model,
            dropout=dropout,
        )
        self.score_grn = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=hidden_size,
            output_dim=d_model,
            context_dim=d_model,
            dropout=dropout,
        )
        self.score_proj = nn.Linear(d_model, 1)

        if use_hard_concrete:
            self.log_alpha_net = nn.Sequential(
                nn.Linear(2 * d_model, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
            self.hard_concrete = HardConcreteGate()

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # tokens: [B, V, D], context: [B, D]
        b, v, d = tokens.shape
        ctx = context.unsqueeze(1).expand(b, v, d)

        transformed = self.var_grn(tokens, context=ctx)
        score_hidden = self.score_grn(tokens, context=ctx)
        logits = self.score_proj(score_hidden).squeeze(-1)  # [B,V]
        weights = sparse_probability_map(logits, kind=self.selector_activation, dim=1)  # [B,V]

        if self.use_hard_concrete:
            log_alpha = self.log_alpha_net(torch.cat([tokens, ctx], dim=-1))  # [B,V,1]
            gates, expected_open = self.hard_concrete(log_alpha)
        else:
            gates = torch.ones(b, v, 1, device=tokens.device, dtype=tokens.dtype)
            expected_open = gates

        out = transformed * weights.unsqueeze(-1) * gates

        aux = {
            "selection_logits": logits,
            "selection_weights": weights,
            "hard_gates": gates.squeeze(-1),
            "expected_open": expected_open.squeeze(-1),
            "l0_penalty": expected_open.mean(),
        }
        return out, aux


class VariateTokenEncoder(nn.Module):
    """
    x_cov: [B, L, V]
    transpose -> [B, V, L]
    each variable trajectory becomes one token
    """

    def __init__(
        self,
        seq_len: int,
        num_vars: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        selector_hidden: int,
        dropout: float,
        attn_activation: str,
        selector_activation: str,
        use_hard_concrete: bool,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_vars = num_vars

        self.time_norm = nn.LayerNorm(seq_len)
        self.time_proj = nn.Linear(seq_len, d_model)
        self.var_embedding = nn.Embedding(num_vars, d_model)

        self.selector = VariableSelectionGate(
            d_model=d_model,
            hidden_size=selector_hidden,
            dropout=dropout,
            selector_activation=selector_activation,
            use_hard_concrete=use_hard_concrete,
        )

        self.layers = nn.ModuleList([
            SparseTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attn_activation=attn_activation,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_cov: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x_cov: [B, L, V]
        x = x_cov.transpose(1, 2)  # [B,V,L]
        x = self.time_norm(x)
        tokens = self.time_proj(x)  # [B,V,D]

        var_ids = torch.arange(self.num_vars, device=x_cov.device)
        tokens = tokens + self.var_embedding(var_ids).unsqueeze(0)

        tokens, aux = self.selector(tokens, context=context)

        for layer in self.layers:
            tokens = layer(tokens)

        tokens = self.norm(tokens)
        return tokens, aux


# ============================================================
# Horizon output head
# ============================================================

class HorizonOutputHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,D] -> [B,H,out_dim]
        return self.net(x)


# ============================================================
# Main forecasting model
# ============================================================

class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        d_model,
        dropout,
        method,
        forecast_task=None,
        dist_side=None,
        enc_in=None,
        affine=True,
        revin_type='revin',
        n_cheb=2,
        tail_model='none',
        # target / NHITS
        target_hidden_size=256,
        target_pooled_sizes=(8, 16, 32),
        target_n_knots=8,
        target_n_mlp_layers=2,
        # covariate branch
        cov_n_layers=2,
        cov_n_heads=4,
        cov_d_ff=256,
        selector_hidden=128,
        selector_activation='sparsemax',
        use_hard_concrete=1,
        # fusion
        fusion_n_layers=2,
        fusion_n_heads=4,
        fusion_d_ff=256,
        attn_activation='sparsemax',
    ):
        super().__init__()

        if enc_in is None or enc_in < 1:
            raise ValueError("enc_in must be >= 1")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.method = method
        self.forecast_task = forecast_task
        self.dist_side = dist_side
        self.n_cheb = n_cheb
        self.tail_model = tail_model
        self.d_model = d_model
        self.num_covariates = enc_in - 1
        self.use_hard_concrete = bool(use_hard_concrete)

        self.std_activ = nn.Softplus()

        # Separate RevIN modules:
        # - target: 1 feature
        # - covariates: M-1 features
        self.revin_target = RevIN(1, affine=affine, mode=revin_type)
        self.revin_cov = RevIN(self.num_covariates, affine=affine, mode=revin_type) if self.num_covariates > 0 else None

        # Target branch
        self.target_encoder = NHiTSTargetEncoder(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            pooled_sizes=target_pooled_sizes,
            hidden_size=target_hidden_size,
            n_knots=target_n_knots,
            n_mlp_layers=target_n_mlp_layers,
            dropout=dropout,
        )

        # Covariate branch
        if self.num_covariates > 0:
            self.covariate_encoder = VariateTokenEncoder(
                seq_len=seq_len,
                num_vars=self.num_covariates,
                d_model=d_model,
                n_heads=cov_n_heads,
                n_layers=cov_n_layers,
                d_ff=cov_d_ff,
                selector_hidden=selector_hidden,
                dropout=dropout,
                attn_activation=attn_activation,
                selector_activation=selector_activation,
                use_hard_concrete=self.use_hard_concrete,
            )
        else:
            self.covariate_encoder = None

        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            CrossFusionLayer(
                d_model=d_model,
                n_heads=fusion_n_heads,
                d_ff=fusion_d_ff,
                dropout=dropout,
                attn_activation=attn_activation,
            )
            for _ in range(fusion_n_layers)
        ])
        self.fusion_norm = nn.LayerNorm(d_model)

        # Output head
        self.output_dim = self._get_output_dim()
        self.head = HorizonOutputHead(d_model=d_model, out_dim=self.output_dim, dropout=dropout)

        # Diagnostics
        self.latest_aux = {}

    def _get_output_dim(self) -> int:
        if self.forecast_task in ["point", None]:
            return 1
        if self.forecast_task in ["quantile", "expectile"]:
            return 2 if self.dist_side == "both" else 1
        if self.forecast_task == "gaussian":
            return 2
        if self.forecast_task == "distribution":
            return 4 + self.n_cheb if self.tail_model == "gpd" else 2 + self.n_cheb
        raise ValueError(f"Unsupported forecast_task: {self.forecast_task}")

    def _split_input(self, x_enc: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x_enc: [B,L,M], last variate is target
        y_target = x_enc[:, :, -1:]               # [B,L,1]
        x_cov = x_enc[:, :, :-1] if self.num_covariates > 0 else None
        return y_target, x_cov

    def _denorm_target(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,C] -> denorm each channel using target statistics
        b, h, c = x.shape
        x = x.reshape(b, h * c, 1)
        x = self.revin_target(x, 'denorm')
        return x.reshape(b, h, c)

    def _denorm_target_scale(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,C] -> denorm scale using target statistics
        b, h, c = x.shape
        x = x.reshape(b, h * c, 1)
        x = self.revin_target(x, 'denorm_scale')
        return x.reshape(b, h, c)

    def _format_output(self, raw_out: torch.Tensor) -> torch.Tensor:
        # raw_out: [B,H,C]
        if self.forecast_task == "distribution":
            mean = self._denorm_target(raw_out[:, :, 0:1])
            std = self._denorm_target_scale(raw_out[:, :, 1:2])
            std = self.std_activ(std)
            extras = raw_out[:, :, 2:]
            out = torch.cat([mean, std, extras], dim=-1)
            return out

        if self.forecast_task == "gaussian":
            mean = self._denorm_target(raw_out[:, :, 0:1])
            std = self._denorm_target_scale(raw_out[:, :, 1:2])
            std = self.std_activ(std)
            out = torch.cat([mean, std], dim=-1)
            return out

        # point / quantile / expectile
        out = self._denorm_target(raw_out)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)  # [B,H]
        return out

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        # x_enc: [B,L,M]
        y_target, x_cov = self._split_input(x_enc)

        # Normalize separately
        y_norm = self.revin_target(y_target, 'norm').squeeze(-1)  # [B,L]
        x_cov_norm = self.revin_cov(x_cov, 'norm') if x_cov is not None else None  # [B,L,M-1]

        # 1) NHITS target branch
        target_tokens, target_residual = self.target_encoder(y_norm)  # [B,H,D], [B,L]

        # use pooled target information as context for variable selection
        target_context = target_tokens.mean(dim=1)  # [B,D]

        # 2) Covariate branch
        if self.covariate_encoder is not None:
            cov_tokens, cov_aux = self.covariate_encoder(x_cov_norm, target_context)  # [B,V,D]
        else:
            cov_tokens = None
            cov_aux = {
                "selection_logits": None,
                "selection_weights": None,
                "hard_gates": None,
                "expected_open": None,
                "l0_penalty": torch.tensor(0.0, device=x_enc.device),
            }

        # 3) Fusion: horizon tokens attend to covariate tokens
        fused = target_tokens
        cross_attn_maps = []
        for layer in self.fusion_layers:
            fused, attn_map = layer(fused, cov_tokens, need_weights=True)
            cross_attn_maps.append(attn_map)

        fused = self.fusion_norm(fused)

        # 4) Per-horizon output head
        raw_out = self.head(fused)  # [B,H,C]
        out = self._format_output(raw_out)

        # Save diagnostics for later logging/inspection
        self.latest_aux = {
            "target_residual": target_residual,
            "target_tokens": target_tokens,
            "target_context": target_context,
            "cov_tokens": cov_tokens,
            "selection_logits": cov_aux["selection_logits"],
            "selection_weights": cov_aux["selection_weights"],
            "hard_gates": cov_aux["hard_gates"],
            "expected_open": cov_aux["expected_open"],
            "l0_penalty": cov_aux["l0_penalty"],
            "cross_attn_maps": cross_attn_maps,
        }
        return out

    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        # Not used by Baseclass_forecast automatically, but available if you want it.
        if "l0_penalty" in self.latest_aux:
            return {"l0_penalty": self.latest_aux["l0_penalty"]}
        return {"l0_penalty": torch.tensor(0.0, device=next(self.parameters()).device)}

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.method != 'forecast':
            raise NotImplementedError("This implementation currently supports method='forecast' only.")
        return self.forecast(x_enc)


# ============================================================
# Lightning wrapper
# ============================================================

class SparseNHITSiTransformer_forecast(Baseclass_forecast):
    def __init__(
        self,
        seq_len,
        pred_len,
        d_model,
        dropout,
        enc_in,
        method,
        batch_size,
        test_batch_size,
        affine,
        revin_type,
        forecast_task,
        dist_side,
        tau_pinball,
        n_cheb,
        twcrps_threshold_low,
        twcrps_threshold_high,
        twcrps_side,
        twcrps_smooth_h,
        u_grid_size,
        dist_loss,
        grid_density,
        quantile_decomp,
        spline_degree,
        knot_kind,
        knot_p,
        tail_model,
        gpd_u_low,
        gpd_u_high,
        gpd_xi_min,
        gpd_xi_max,
        # model-specific
        target_hidden_size,
        target_pooled_sizes,
        target_n_knots,
        target_n_mlp_layers,
        cov_n_layers,
        cov_n_heads,
        cov_d_ff,
        selector_hidden,
        selector_activation,
        use_hard_concrete,
        fusion_n_layers,
        fusion_n_heads,
        fusion_d_ff,
        attn_activation,
        l0_lambda,
        save_test_diagnostics,
        diag_top_k_vars,
        diag_max_plot_samples,
        use_log_price,
        **kwargs
    ):
        super().__init__(
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            learning_rate=0.0001,
            method=method,
            forecast_task=forecast_task,
            dist_side=dist_side,
            tau_pinball=tau_pinball,
            n_cheb=n_cheb,
            twcrps_threshold_low=twcrps_threshold_low,
            twcrps_threshold_high=twcrps_threshold_high,
            twcrps_side=twcrps_side,
            twcrps_smooth_h=twcrps_smooth_h,
            u_grid_size=u_grid_size,
            grid_density=grid_density,
            dist_loss=dist_loss,
            revin_type=revin_type,
            quantile_decomp=quantile_decomp,
            spline_degree=spline_degree,
            knot_kind=knot_kind,
            knot_p=knot_p,
            tail_model=tail_model,
            gpd_u_low=gpd_u_low,
            gpd_u_high=gpd_u_high,
            gpd_xi_min=gpd_xi_min,
            gpd_xi_max=gpd_xi_max,
            l0_lambda=(l0_lambda if use_hard_concrete else 0.0),
            save_test_diagnostics=save_test_diagnostics,
            diag_top_k_vars=diag_top_k_vars,
            diag_max_plot_samples=diag_max_plot_samples,
            use_log_price=use_log_price,

        )

        self.model = Model(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            dropout=dropout,
            method=method,
            forecast_task=forecast_task,
            dist_side=dist_side,
            enc_in=enc_in,
            affine=bool(affine),
            revin_type=revin_type,
            n_cheb=n_cheb,
            tail_model=tail_model,
            target_hidden_size=target_hidden_size,
            target_pooled_sizes=target_pooled_sizes,
            target_n_knots=target_n_knots,
            target_n_mlp_layers=target_n_mlp_layers,
            cov_n_layers=cov_n_layers,
            cov_n_heads=cov_n_heads,
            cov_d_ff=cov_d_ff,
            selector_hidden=selector_hidden,
            selector_activation=selector_activation,
            use_hard_concrete=use_hard_concrete,
            fusion_n_layers=fusion_n_layers,
            fusion_n_heads=fusion_n_heads,
            fusion_d_ff=fusion_d_ff,
            attn_activation=attn_activation,
        )
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('Model-specific arguments')

        # Core
        model_parser.add_argument('--d_model', type=int, default=128)
        model_parser.add_argument('--dropout', type=float, default=0.1)
        model_parser.add_argument('--revin_type', type=str, choices=['revin', 'robust'], default='revin')
        model_parser.add_argument('--affine', type=int, choices=[0, 1], default=1)

        # Target / NHITS
        model_parser.add_argument('--target_hidden_size', type=int, default=256)
        model_parser.add_argument('--target_pooled_sizes', type=int, nargs='+', default=[8, 16, 32])
        model_parser.add_argument('--target_n_knots', type=int, default=8)
        model_parser.add_argument('--target_n_mlp_layers', type=int, default=2)

        # Covariate encoder
        model_parser.add_argument('--cov_n_layers', type=int, default=2)
        model_parser.add_argument('--cov_n_heads', type=int, default=4)
        model_parser.add_argument('--cov_d_ff', type=int, default=256)
        model_parser.add_argument('--selector_hidden', type=int, default=128)
        model_parser.add_argument(
            '--selector_activation',
            type=str,
            choices=['softmax', 'sparsemax', 'entmax15', 'entmax', 'entmax1.5'],
            default='sparsemax'
        )
        model_parser.add_argument('--use_hard_concrete', type=int, choices=[0, 1], default=1)

        # Fusion
        model_parser.add_argument('--fusion_n_layers', type=int, default=2)
        model_parser.add_argument('--fusion_n_heads', type=int, default=4)
        model_parser.add_argument('--fusion_d_ff', type=int, default=256)
        model_parser.add_argument(
            '--attn_activation',
            type=str,
            choices=['softmax', 'sparsemax', 'entmax15', 'entmax', 'entmax1.5'],
            default='sparsemax'
        )

        Baseclass_forecast.add_task_specific_args(parent_parser)
        return parent_parser