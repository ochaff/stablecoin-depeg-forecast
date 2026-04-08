import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import numpy as np



class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = None if alpha is None else float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

        if pos_weight is not None:
            # register so it moves with .to(device)
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # logits: (B,) or (B,1), y: {0,1} same shape
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        if y.ndim > 1:
            y = y.squeeze(-1)

        y = y.float()

        # BCE term (stable)
        bce = F.binary_cross_entropy_with_logits(
            logits, y, reduction="none", pos_weight=self.pos_weight
        )

        # p_t = p if y=1 else (1-p)
        p = torch.sigmoid(logits)
        pt = y * p + (1.0 - y) * (1.0 - p)

        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha_t = y * self.alpha + (1.0 - y) * (1.0 - self.alpha)

        loss = alpha_t * focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
    

class CRPSFromQuantiles(nn.Module):
    def __init__(self, u: torch.Tensor, wu: torch.Tensor, crps_convention: bool = True):
        super().__init__()
        self.register_buffer("u", u)    # (J,)
        self.register_buffer("wu", wu)  # (J,)
        self.crps_convention = crps_convention

    def forward(self, Q: torch.Tensor, q:torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Q: (B,H,J), y: (B,H)
        u = self.u.view(1, 1, -1)                 # (1,1,J)
        y_ = y.unsqueeze(-1).repeat(1,1,u.shape[-1])                      # (B,H,J)

        e = y_ - Q    
        pinball = torch.maximum(u*e, (u-1)*e)                            # (B,H,J)

        loss_bh = torch.sum(pinball * self.wu.view(1, 1, -1), dim=-1)  # (B,H)

        loss = loss_bh.mean()
        if self.crps_convention:
            loss = 2.0 * loss                     # so loss equals CRPS (common convention)

        return loss


class GaussianCRPS(nn.Module):
    """
    Gaussian CRPS computed via the same quantile-grid approximation
    used for the nonparametric forecast.

    This makes the Gaussian and nonparametric CRPS more directly comparable.

    Expects output[..., 0] = mu
            output[..., 1] = scale parameter

    If scale_is_raw=True:
        sigma = softplus(output[...,1]) + eps
    else:
        sigma = clamp_min(output[...,1], eps)
    """
    def __init__(
        self,
        u_grid: torch.Tensor,
        crps_convention: bool = True,
    ):
        super().__init__()

        # For Gaussian quantiles, avoid exact 0 and 1
        self.register_buffer("u", u_grid)  # (J,)

        # trapezoidal weights, same as your CRPSFromQuantiles
        du = u_grid[1:] - u_grid[:-1]
        wu = torch.zeros_like(u_grid)
        wu[0] = du[0] / 2
        wu[-1] = du[-1] / 2
        wu[1:-1] = (du[:-1] + du[1:]) / 2
        self.register_buffer("wu", wu)

        # Standard normal quantiles z(u) = Phi^{-1}(u)
        normal = Normal(
            loc=torch.tensor(0.0, dtype=u_grid.dtype, device=u_grid.device),
            scale=torch.tensor(1.0, dtype=u_grid.dtype, device=u_grid.device),
        )
        z_u = normal.icdf(u_grid)  # (J,)
        self.register_buffer("z_u", z_u)

        # Standard normal pdf evaluated at z_u
        phi_z = torch.exp(-0.5 * z_u**2) / math.sqrt(2.0 * math.pi)
        self.register_buffer("phi_z", phi_z)

        self.base_loss = CRPSFromQuantiles(
            u=self.u,
            wu=self.wu,
            crps_convention=crps_convention,
        )

    def params_to_quantiles(self, output: torch.Tensor):
        """
        output: (B,H,2)

        returns:
          Q: (B,H,J)
          q: (B,H,J) with q(u)=dQ/du
        """
        mu = output[..., 0]  # (B,H)

        sigma = output[..., 1]

        Q = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * self.z_u.view(1, 1, -1)

        q = sigma.unsqueeze(-1) / self.phi_z.view(1, 1, -1)

        return Q, q

    def forward(self, output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Q, q = self.params_to_quantiles(output)
        return self.base_loss(Q, q, y)

class GaussianTWCRPS(nn.Module):
    def __init__(
        self,
        u_grid: torch.Tensor,
        eps: float = 1e-6,
        crps_convention: bool = True,
        threshold_low: float = None,
        threshold_high: float = None,
        side: str = "two_sided",
        smooth_h: float = 0.0,
    ):
        super().__init__()
        self.eps = float(eps)

        u_grid = u_grid.float().clamp(min=self.eps, max=1.0 - self.eps).contiguous()
        self.register_buffer("u", u_grid)

        normal = Normal(
            loc=torch.tensor(0.0, dtype=u_grid.dtype, device=u_grid.device),
            scale=torch.tensor(1.0, dtype=u_grid.dtype, device=u_grid.device),
        )
        self.register_buffer("z_u", normal.icdf(u_grid))

        du = u_grid[1:] - u_grid[:-1]
        wu = torch.zeros_like(u_grid)
        wu[0] = du[0] / 2
        wu[-1] = du[-1] / 2
        wu[1:-1] = (du[:-1] + du[1:]) / 2

        self.base_loss = ThresholdWeightedCRPSFromQuantiles(
            u=u_grid,
            wu=wu,
            crps_convention=crps_convention,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            side=side,
            smooth_h=smooth_h,
        )

    def forward(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu = params[..., 0]
        sigma = params[..., 1]
        Q = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * self.z_u.view(1, 1, -1)
        q_dummy = torch.ones_like(Q)
        return self.base_loss(Q, q_dummy, y)

class ThresholdWeightedCRPSFromQuantiles(nn.Module):
    def __init__(self, u: torch.Tensor, wu: torch.Tensor, crps_convention: bool = True, 
                threshold_low: float = None, threshold_high: float = None, 
                side: str = "two_sided", smooth_h: float = 0.0):
        super().__init__()
        self.register_buffer("u", u)    # (J,)
        self.register_buffer("wu", wu)  # (J,)
        self.crps_convention = crps_convention
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.side = side
        self.smooth_h = smooth_h
        self.chain = ChainingFunction(threshold_low=threshold_low, 
                                      threshold_high=threshold_high, 
                                      side=side, 
                                      smooth_h=smooth_h
                                      )

    def forward(self, Q: torch.Tensor, q:torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Q: (B,H,J), y: (B,H)
        u = self.u.view(1, 1, -1)                 # (1,1,J)
        y_ = y.unsqueeze(-1).repeat(1,1,u.shape[-1])                      # (B,H,J)

        e = self.chain(y_) - self.chain(Q)    
        pinball = torch.maximum(u*e, (u-1)*e)                            # (B,H,J)

        loss_bh = torch.sum(pinball * self.wu.view(1, 1, -1), dim=-1)  # (B,H)

        loss = loss_bh.mean()
        if self.crps_convention:
            loss = 2.0 * loss                     # so loss equals CRPS (common convention)
        return loss


class ChainingFunction(nn.Module):
    """
    v(x) =
      two_sided:  (x-thr_low) for x<=thr_low  +  (x-thr_high) for x>=thr_high
                = -ReLU(thr_low - x) + ReLU(x - thr_high)
      below:     (x-thr_low) for x<=thr_low  = -ReLU(thr_low - x)
      above:     (x-thr_high) for x>=thr_high = ReLU(x - thr_high)

    If smooth_h > 0, uses Softplus as a smooth approximation:
      ReLU(z) ~ Softplus(h*z)/h  (for large h)
    """
    def __init__(self, threshold_low=None, threshold_high=None,
                 side: str = "two_sided", smooth_h: float = 0.0):
        super().__init__()
        assert side in {"two_sided", "below", "above"}
        self.side = side
        self.smooth_h = float(smooth_h)

        if threshold_low is None:
            self.register_buffer("threshold_low", torch.tensor(0.0))
            self.has_low = False
        else:
            self.register_buffer("threshold_low", torch.tensor(float(threshold_low)))
            self.has_low = True

        if threshold_high is None:
            self.register_buffer("threshold_high", torch.tensor(0.0))
            self.has_high = False
        else:
            self.register_buffer("threshold_high", torch.tensor(float(threshold_high)))
            self.has_high = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.smooth_h

        # Pick ReLU-like primitive (exact or smoothed)
        if h and h > 0:
            def relu_like(z):  # smooth ReLU(z)
                return F.softplus(h * z) / h
        else:
            relu_like = F.relu

        v = 0.0

        if self.side in {"two_sided", "below"}:
            if not self.has_low:
                raise ValueError("threshold_low must be provided for side='below' or 'two_sided'")
            v = v - relu_like(self.threshold_low - x)  # (x-thr_low) for x<=thr_low

        if self.side in {"two_sided", "above"}:
            if not self.has_high:
                raise ValueError("threshold_high must be provided for side='above' or 'two_sided'")
            v = v + relu_like(x - self.threshold_high)  # (x-thr_high) for x>=thr_high

        return v
