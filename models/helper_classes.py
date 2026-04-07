import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import numpy as np

from models.utils import (
    make_open_nonuniform_knots, chebyshev_basis, 
    cdf_from_quantile_on_grid, _interp_idx_w,
    make_open_nonuniform_knots, mspline_ispline_on_grid
    )


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, mode = 'revin'):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine == 1:
            self._init_params()
        self.type = mode

    def forward(self, x, mode:str):
            if mode == 'norm':
                self._get_statistics(x)
                x = self._normalize(x)
            elif mode == 'denorm':
                x = self._denormalize(x)
            elif mode == 'denorm_scale':
                x = self._denormalize_scale(x)
            return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        if self.type == 'revin':
            dim2reduce = tuple(range(1, x.ndim-1))
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            if self.affine :    
                self.mean = self.mean + self.affine_bias
                self.stdev = self.stdev * (torch.relu(self.affine_weight) + self.eps)
        elif self.type == 'robust':
            dim2reduce = tuple(range(1, x.ndim-1))
            self.mean = torch.quantile(x, 0.5, dim=1, keepdim=True).detach()
            self.stdev = (torch.quantile(x, 0.75, dim=1, keepdim=True).detach() - torch.quantile(x, 0.25, dim=1, keepdim=True).detach())
            self.stdev = self.stdev + self.eps
            if self.affine :    
                self.mean = self.mean + self.affine_bias
                self.stdev = self.stdev * (torch.relu(self.affine_weight) + self.eps)
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x
    
    def _denormalize_scale(self, x, eps = 1e-5):  
        x = x * self.stdev
        return x
    def robust_statistics(self, x, dim=-1, eps=1e-6):
        return None


    
class ShapProbWrapper(nn.Module):
    """Wraps a model that outputs probabilities so SHAP sees an nn.Module."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_model(x.float())  # already probabilities
        if out.ndim == 1:
            out = out.unsqueeze(1)        # (B, 1)
        elif out.ndim == 2 and out.shape[1] != 1:
            out = out[:, :1]
        return out


class ChebyshevQuantile(nn.Module):
    """
    params -> (Q(u_j), q(u_j)) on a fixed u-grid, monotone by construction.
    params shape: (B, H, 2+K) = [b, log_s, a_0..a_{K-1}]
    """
    def __init__(self, K: int, u_grid: torch.Tensor, eps: float = 1e-6, normalize: bool = True, u0: float = 0.5, revin_type: str = 'revin'):
        super().__init__()
        self.K = K
        self.eps = eps
        self.normalize = normalize
        self.u0 = u0
        self.revin_type = revin_type
        
        u_grid = u_grid.contiguous()
        # buffers move with .to(device)
        self.register_buffer("u", u_grid)                       # (J,)
        self.register_buffer("du", u_grid[1:] - u_grid[:-1])    # (J-1,)
        self.register_buffer("T", chebyshev_basis(u_grid, K))   # (J,K)

        # trapezoid weights on u for ∫_0^1 f(u) du
        du = self.du
        wu = torch.zeros_like(u_grid)
        wu[0] = du[0] / 2
        wu[-1] = du[-1] / 2
        wu[1:-1] = (du[:-1] + du[1:]) / 2
        self.register_buffer("wu", wu)                          # (J,)


        i0,  w0  = _interp_idx_w(self.u, 0.5)
        self.register_buffer("i0",  i0)
        self.register_buffer("w0",  w0)
        
        if self.revin_type == 'robust':
            i25, w25 = _interp_idx_w(self.u, 0.25)
            i75, w75 = _interp_idx_w(self.u, 0.75)
            self.register_buffer("i25", i25)
            self.register_buffer("w25", w25)
            self.register_buffer("i75", i75)
            self.register_buffer("w75", w75)


    def forward(self, params: torch.Tensor):
        """
        returns:
          Q: (B,H,J)
          q: (B,H,J)  (quantile density)
        """
        b = params[..., 0]                           # (B,H)
        s = F.softplus(params[..., 1]) + self.eps     # (B,H)
        a = params[..., 2:]                          # (B,H,K)


        g = torch.einsum("bhk,jk->bhj", a, self.T)    # (B,H,J)


        q = F.softplus(g) + self.eps                 # (B,H,J)

        q_mid = 0.5 * (q[..., 1:] + q[..., :-1])     # (B,H,J-1)
        integ = torch.cumsum(q_mid * self.du, dim=-1) # (B,H,J-1)
        integ = torch.cat([torch.zeros_like(b)[..., None], integ], dim=-1)  # (B,H,J)

        if self.normalize:
            def interp(integ, i, w):
                return (1 - w) * integ[..., i] + w * integ[..., i + 1]  # (B,H)

            i0  = self.i0.item();  w0  = self.w0.item()
            if self.revin_type == "robust":
                i25 = self.i25.item(); w25 = self.w25.item()
                i75 = self.i75.item(); w75 = self.w75.item()
           
            Q_u0 = interp(integ, i0, w0)                 # (B,H)
            Qc = integ - Q_u0[..., None]                 # (B,H,J)

            if self.revin_type == "revin":
                mu = torch.sum(Qc * self.wu, dim=-1)     # (B,H)
                m2 = torch.sum((Qc ** 2) * self.wu, dim=-1)
                var = (m2 - mu ** 2).clamp_min(0.0)
                alpha = s / torch.sqrt(var + self.eps)   # (B,H)

            elif self.revin_type == "robust":
                Q25 = interp(integ, i25, w25) - Q_u0     # == interp(Qc, ...)
                Q75 = interp(integ, i75, w75) - Q_u0
                iqr = (Q75 - Q25).clamp_min(self.eps)    # (B,H)
                alpha = s / iqr
            else:
                raise ValueError(f"Unknown revin_type: {self.revin_type}")

            Q = b[..., None] + alpha[..., None] * Qc
            q = alpha[..., None] * q
        else:
            Q = b[..., None] + integ
        return Q, q
 
class ISplineQuantile(nn.Module):
    """
    params -> (Q(u_j), q(u_j)) on a fixed u-grid, monotone by construction via I-splines.

    params shape: (B, H, 2+K) = [b, log_s, a_0..a_{K-1}]
      b: location
      s: scale target (positive after softplus)
      a_k: nonnegative I-spline coefficients (softplus), guarantee monotone Q(u)

    Outputs:
      Q: (B,H,J) quantile function on u-grid
      q: (B,H,J) quantile density dQ/du on u-grid (>= 0)
    """
    def __init__(
        self,
        K: int,
        u_grid: torch.Tensor,
        eps: float = 1e-6,
        normalize: bool = True,
        u0: float = 0.5,
        revin_type: str = "revin",
        # spline params
        degree: int = 3,
        knot_kind: str = "power_tails",
        knot_p: float = 3.0,
    ):
        super().__init__()
        self.K = int(K)
        self.degree = int(degree)
        self.eps = float(eps)
        self.normalize = bool(normalize)
        self.u0 = float(u0)
        self.revin_type = revin_type

        u_grid = u_grid.contiguous()
        self.register_buffer("u", u_grid)                       # (J,)
        self.register_buffer("du", u_grid[1:] - u_grid[:-1])    # (J-1,)

        # trapezoid weights on u for ∫_0^1 f(u) du (matches your ChebyshevQuantile)
        du = self.du
        wu = torch.zeros_like(u_grid)
        wu[0] = du[0] / 2
        wu[-1] = du[-1] / 2
        wu[1:-1] = (du[:-1] + du[1:]) / 2
        self.register_buffer("wu", wu)                          # (J,)

        # interp indices for anchoring at u0 (and IQR if robust)
        i0, w0 = _interp_idx_w(self.u, self.u0)
        self.register_buffer("i0", i0)
        self.register_buffer("w0", w0)

        if self.revin_type == "robust":
            i25, w25 = _interp_idx_w(self.u, 0.25)
            i75, w75 = _interp_idx_w(self.u, 0.75)
            self.register_buffer("i25", i25); self.register_buffer("w25", w25)
            self.register_buffer("i75", i75); self.register_buffer("w75", w75)

        # knots + basis sampled on the u-grid
        
        knots = make_open_nonuniform_knots(
                n_basis=self.K,
                degree=self.degree,
                a=0.0, b=1.0,
                kind=knot_kind,
                p=knot_p,
                device=u_grid.device,
            ).to(dtype=u_grid.dtype, device=u_grid.device)
       
        self.register_buffer("knots", knots)

        M, I = mspline_ispline_on_grid(self.u, self.knots, degree=self.degree)  # (J,K), (J,K)
        self.register_buffer("M", M)  # M-spline basis sampled on grid (for q)
        self.register_buffer("I", I)  # I-spline basis sampled on grid (for Q)

    def forward(self, params: torch.Tensor):
        """
        returns:
          Q: (B,H,J)
          q: (B,H,J)  quantile density dQ/du >= 0
        """
        b = params[..., 0]                           # (B,H)
        s = F.softplus(params[..., 1]) + self.eps     # (B,H)
        a_raw = params[..., 2:]                       # (B,H,K)

        # nonnegative coefficients => monotone Q(u)
        a = F.softplus(a_raw) + self.eps              # (B,H,K)

        # Q_base(u) = sum_k a_k I_k(u)
        Q_base = torch.einsum("bhk,jk->bhj", a, self.I)  # (B,H,J)

        # q(u) = dQ/du = sum_k a_k M_k(u)
        q = torch.einsum("bhk,jk->bhj", a, self.M).clamp_min(self.eps)  # (B,H,J)

        if self.normalize:
            def interp(x, i, w):
                i = int(i.item()); w = float(w.item())
                return (1 - w) * x[..., i] + w * x[..., i + 1]  # (B,H)

            Q_u0 = interp(Q_base, self.i0, self.w0)        # (B,H)
            Qc = Q_base - Q_u0[..., None]                  # centered at u0

            if self.revin_type == "revin":
                mu = torch.sum(Qc * self.wu, dim=-1)       # (B,H)
                m2 = torch.sum((Qc ** 2) * self.wu, dim=-1)
                var = (m2 - mu ** 2).clamp_min(0.0)
                alpha = s / torch.sqrt(var + self.eps)     # (B,H)

            elif self.revin_type == "robust":
                Q25 = interp(Q_base, self.i25, self.w25) - Q_u0
                Q75 = interp(Q_base, self.i75, self.w75) - Q_u0
                iqr = (Q75 - Q25).clamp_min(self.eps)
                alpha = s / iqr

            else:
                raise ValueError(f"Unknown revin_type: {self.revin_type}")

            Q = b[..., None] + alpha[..., None] * Qc
            q = alpha[..., None] * q
        else:
            Q = b[..., None] + Q_base

        return Q, q

class SplicedGPDQuantile(nn.Module):
    """
    Splice GPD tails onto an existing quantile body model.

    Expected params shape:
        (B, H, body_param_dim + 2)

    where the last 2 entries are:
        raw_xi_low, raw_xi_high

    The body model must return:
        Q_body: (B,H,J)
        q_body: (B,H,J)

    on the same u-grid.

    Splice:
      - lower tail for u < u_low
      - body for u_low <= u <= u_high
      - upper tail for u > u_high

    Tail scales are slope-matched to the body:
      beta_low  = u_low * q_body(u_low)
      beta_high = (1-u_high) * q_body(u_high)

    xi is constrained to [xi_min, xi_max] using a sigmoid transform.
    If you want guaranteed unbounded tails, set xi_min=0.0.
    """
    def __init__(
        self,
        body_quantile: nn.Module,
        body_param_dim: int,
        u_low: float = 0.01,
        u_high: float = 0.99,
        xi_min: float = -0.25,
        xi_max: float = 0.50,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert 0.0 < u_low < u_high < 1.0, "Require 0 < u_low < u_high < 1"

        self.body = body_quantile
        self.body_param_dim = int(body_param_dim)
        self.u_low = float(u_low)
        self.u_high = float(u_high)
        self.xi_min = float(xi_min)
        self.xi_max = float(xi_max)
        self.eps = float(eps)

        # Reuse the body's grid/weights
        u = body_quantile.u.float().clamp(min=eps, max=1.0 - eps).contiguous()
        self.register_buffer("u", u)
        self.register_buffer("wu", body_quantile.wu)
        self.register_buffer("du", body_quantile.du)

        # interpolation indices at splice points
        iL, wL = _interp_idx_w(self.u, self.u_low)
        iU, wU = _interp_idx_w(self.u, self.u_high)
        self.register_buffer("iL", iL)
        self.register_buffer("wL", wL)
        self.register_buffer("iU", iU)
        self.register_buffer("wU", wU)

        # masks on the global u-grid
        self.register_buffer("mask_low", self.u < self.u_low)
        self.register_buffer("mask_mid", (self.u >= self.u_low) & (self.u <= self.u_high))
        self.register_buffer("mask_high", self.u > self.u_high)

    def _interp_on_u(self, X: torch.Tensor, i: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        X: (B,H,J)
        returns interpolated values at a scalar u-location: (B,H)
        """
        ii = int(i.item())
        ww = float(w.item())
        return (1.0 - ww) * X[..., ii] + ww * X[..., ii + 1]

    def _map_xi(self, raw_xi: torch.Tensor) -> torch.Tensor:
        # map to [xi_min, xi_max]
        return self.xi_min + (self.xi_max - self.xi_min) * torch.sigmoid(raw_xi)

    def _upper_tail_quantile(self, u_tail: torch.Tensor, xU: torch.Tensor, betaU: torch.Tensor, xiU: torch.Tensor):
        """
        u_tail: (Jt,) values in (u_high, 1)
        xU, betaU, xiU: (B,H)

        returns:
          Q_up: (B,H,Jt)
          q_up: (B,H,Jt)
        """
        r = ((1.0 - u_tail) / (1.0 - self.u_high)).view(1, 1, -1).clamp_min(self.eps)  # (1,1,Jt)

        xU = xU.unsqueeze(-1)
        betaU = betaU.unsqueeze(-1).clamp_min(self.eps)
        xiU = xiU.unsqueeze(-1)

        near_zero = xiU.abs() < 1e-6

        # Q(u)
        Q_exp = xU - betaU * torch.log(r)
        Q_gpd = xU + (betaU / xiU) * (torch.pow(r, -xiU) - 1.0)
        Q_up = torch.where(near_zero, Q_exp, Q_gpd)

        # q(u) = dQ/du
        q_exp = betaU / ((1.0 - self.u_high) * r)
        q_gpd = (betaU / (1.0 - self.u_high)) * torch.pow(r, -xiU - 1.0)
        q_up = torch.where(near_zero, q_exp, q_gpd)

        return Q_up, q_up

    def _lower_tail_quantile(self, u_tail: torch.Tensor, xL: torch.Tensor, betaL: torch.Tensor, xiL: torch.Tensor):
        """
        u_tail: (Jt,) values in (0, u_low)
        xL, betaL, xiL: (B,H)

        returns:
          Q_lo: (B,H,Jt)
          q_lo: (B,H,Jt)
        """
        r = (u_tail / self.u_low).view(1, 1, -1).clamp_min(self.eps)  # (1,1,Jt)

        xL = xL.unsqueeze(-1)
        betaL = betaL.unsqueeze(-1).clamp_min(self.eps)
        xiL = xiL.unsqueeze(-1)

        near_zero = xiL.abs() < 1e-6

        # Q(u)
        Q_exp = xL + betaL * torch.log(r)
        Q_gpd = xL - (betaL / xiL) * (torch.pow(r, -xiL) - 1.0)
        Q_lo = torch.where(near_zero, Q_exp, Q_gpd)

        # q(u) = dQ/du
        q_exp = betaL / (self.u_low * r)
        q_gpd = (betaL / self.u_low) * torch.pow(r, -xiL - 1.0)
        q_lo = torch.where(near_zero, q_exp, q_gpd)

        return Q_lo, q_lo

    def _tail_params_from_body(self, body_params: torch.Tensor, raw_xi_low: torch.Tensor, raw_xi_high: torch.Tensor):
        """
        Returns body outputs and splice/tail params.
        """
        Q_body, q_body = self.body(body_params)  # (B,H,J), (B,H,J)

        xL = self._interp_on_u(Q_body, self.iL, self.wL)               # (B,H)
        qL = self._interp_on_u(q_body, self.iL, self.wL).clamp_min(self.eps)

        xU = self._interp_on_u(Q_body, self.iU, self.wU)               # (B,H)
        qU = self._interp_on_u(q_body, self.iU, self.wU).clamp_min(self.eps)

        xiL = self._map_xi(raw_xi_low)                                 # (B,H)
        xiU = self._map_xi(raw_xi_high)                                # (B,H)

        betaL = (self.u_low * qL).clamp_min(self.eps)                  # slope match
        betaU = ((1.0 - self.u_high) * qU).clamp_min(self.eps)         # slope match

        return Q_body, q_body, xL, qL, xU, qU, xiL, xiU, betaL, betaU

    def forward(self, params: torch.Tensor):
        """
        params: (B,H,body_param_dim + 2)

        returns:
          Q: (B,H,J)
          q: (B,H,J)
        """
        body_params = params[..., :self.body_param_dim]
        raw_xi_low = params[..., self.body_param_dim]
        raw_xi_high = params[..., self.body_param_dim + 1]

        Q_body, q_body, xL, qL, xU, qU, xiL, xiU, betaL, betaU = self._tail_params_from_body(
            body_params, raw_xi_low, raw_xi_high
        )

        Q = Q_body.clone()
        q = q_body.clone()

        if self.mask_low.any():
            u_low_grid = self.u[self.mask_low]
            Q_lo, q_lo = self._lower_tail_quantile(u_low_grid, xL, betaL, xiL)
            Q[..., self.mask_low] = Q_lo
            q[..., self.mask_low] = q_lo

        if self.mask_high.any():
            u_high_grid = self.u[self.mask_high]
            Q_hi, q_hi = self._upper_tail_quantile(u_high_grid, xU, betaU, xiU)
            Q[..., self.mask_high] = Q_hi
            q[..., self.mask_high] = q_hi

        return Q, q

    def cdf_on_grid(self, params: torch.Tensor, z_grid: torch.Tensor):
        """
        Analytic tails + interpolation-based body CDF.

        params: (B,H,body_param_dim + 2)
        z_grid: (Z,)

        returns:
          F: (B,H,Z)
        """
        body_params = params[..., :self.body_param_dim]
        raw_xi_low = params[..., self.body_param_dim]
        raw_xi_high = params[..., self.body_param_dim + 1]

        Q, q = self.forward(params)
        F = cdf_from_quantile_on_grid(Q, self.u, z_grid)  # interior approximation

        Q_body, q_body, xL, qL, xU, qU, xiL, xiU, betaL, betaU = self._tail_params_from_body(
            body_params, raw_xi_low, raw_xi_high
        )

        z = z_grid.view(1, 1, -1)  # (1,1,Z)

        # lower tail: F(z) = u_low * S_GPD(xL-z)
        dL = (xL.unsqueeze(-1) - z).clamp_min(0.0)
        xiL_ = xiL.unsqueeze(-1)
        betaL_ = betaL.unsqueeze(-1)

        near_zero_L = xiL_.abs() < 1e-6
        baseL = 1.0 + xiL_ * dL / betaL_
        S_exp_L = torch.exp(-dL / betaL_)
        S_gpd_L = torch.where(baseL > 0, torch.pow(baseL, -1.0 / xiL_), torch.zeros_like(baseL))
        S_L = torch.where(near_zero_L, S_exp_L, S_gpd_L)
        F_low = self.u_low * S_L

        # upper tail: 1 - F(z) = (1-u_high) * S_GPD(z-xU)
        dU = (z - xU.unsqueeze(-1)).clamp_min(0.0)
        xiU_ = xiU.unsqueeze(-1)
        betaU_ = betaU.unsqueeze(-1)

        near_zero_U = xiU_.abs() < 1e-6
        baseU = 1.0 + xiU_ * dU / betaU_
        S_exp_U = torch.exp(-dU / betaU_)
        S_gpd_U = torch.where(baseU > 0, torch.pow(baseU, -1.0 / xiU_), torch.zeros_like(baseU))
        S_U = torch.where(near_zero_U, S_exp_U, S_gpd_U)
        F_high = 1.0 - (1.0 - self.u_high) * S_U

        # overwrite the tail regions analytically
        F = torch.where(z < xL.unsqueeze(-1), F_low, F)
        F = torch.where(z > xU.unsqueeze(-1), F_high, F)

        return F.clamp(0.0, 1.0)

    def pdf_on_grid(self, params: torch.Tensor, z_grid: torch.Tensor):
        """
        Analytic tails + approximate body PDF from quantile grid:
          f(z) = 1 / q(u(z))

        params: (B,H,body_param_dim + 2)
        z_grid: (Z,)

        returns:
          f: (B,H,Z)
        """
        body_params = params[..., :self.body_param_dim]
        raw_xi_low = params[..., self.body_param_dim]
        raw_xi_high = params[..., self.body_param_dim + 1]

        Q, q = self.forward(params)
        F = cdf_from_quantile_on_grid(Q, self.u, z_grid)  # (B,H,Z)

        # approximate body pdf by mapping F(z) -> u(z), then interpolate q(u)
        # q(u) = dQ/du, so f(z) = 1/q(u(z))
        u_of_z = F.clamp(min=float(self.u[0]), max=float(self.u[-1]))

        # interpolate q(u(z)) from q over self.u
        q_np = q.detach().cpu().numpy()
        u_np = self.u.detach().cpu().numpy()
        uz_np = u_of_z.detach().cpu().numpy()

        B, H, J = q_np.shape
        Z = uz_np.shape[-1]
        pdf_body = np.empty((B, H, Z), dtype=np.float32)

        for b in range(B):
            for h in range(H):
                qi = np.maximum(q_np[b, h], self.eps)
                pdf_body[b, h] = 1.0 / np.interp(uz_np[b, h], u_np, qi)

        pdf_body = torch.from_numpy(pdf_body).to(q.device)

        Q_body, q_body, xL, qL, xU, qU, xiL, xiU, betaL, betaU = self._tail_params_from_body(
            body_params, raw_xi_low, raw_xi_high
        )

        z = z_grid.view(1, 1, -1)

        # lower tail density
        dL = (xL.unsqueeze(-1) - z).clamp_min(0.0)
        xiL_ = xiL.unsqueeze(-1)
        betaL_ = betaL.unsqueeze(-1)
        near_zero_L = xiL_.abs() < 1e-6
        baseL = 1.0 + xiL_ * dL / betaL_
        f_exp_L = (self.u_low / betaL_) * torch.exp(-dL / betaL_)
        f_gpd_L = torch.where(baseL > 0, (self.u_low / betaL_) * torch.pow(baseL, -1.0 / xiL_ - 1.0), torch.zeros_like(baseL))
        f_low = torch.where(near_zero_L, f_exp_L, f_gpd_L)

        # upper tail density
        dU = (z - xU.unsqueeze(-1)).clamp_min(0.0)
        xiU_ = xiU.unsqueeze(-1)
        betaU_ = betaU.unsqueeze(-1)
        near_zero_U = xiU_.abs() < 1e-6
        baseU = 1.0 + xiU_ * dU / betaU_
        f_exp_U = ((1.0 - self.u_high) / betaU_) * torch.exp(-dU / betaU_)
        f_gpd_U = torch.where(baseU > 0, ((1.0 - self.u_high) / betaU_) * torch.pow(baseU, -1.0 / xiU_ - 1.0), torch.zeros_like(baseU))
        f_high = torch.where(near_zero_U, f_exp_U, f_gpd_U)

        pdf = pdf_body
        pdf = torch.where(z < xL.unsqueeze(-1), f_low, pdf)
        pdf = torch.where(z > xU.unsqueeze(-1), f_high, pdf)

        return pdf.clamp_min(0.0)
    
    def _cdf_from_quantiles_at_y(self, Q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Invert monotone Q(u) numerically to get F(y) on the current u-grid.

        Q: (B,H,J)
        y: (B,H)

        returns:
          F_mid: (B,H)
        """
        B, H, J = Q.shape
        N = B * H

        Qf = Q.reshape(N, J)
        yf = y.reshape(N)

        # guard against tiny numerical non-monotonicity
        Qf = torch.cummax(Qf, dim=-1).values

        # searchsorted per row
        idx = torch.searchsorted(Qf, yf.unsqueeze(-1), right=False).squeeze(-1)  # (N,)

        u = self.u
        eps = self.eps

        Ff = torch.empty_like(yf)

        # below grid
        m_lo = idx <= 0
        Ff[m_lo] = u[0]

        # above grid
        m_hi = idx >= J
        Ff[m_hi] = u[-1]

        # interior
        m_mid = (~m_lo) & (~m_hi)
        if m_mid.any():
            idxm = idx[m_mid]
            rows = torch.arange(N, device=Q.device)[m_mid]

            ql = Qf[rows, idxm - 1]
            qr = Qf[rows, idxm]
            yl = yf[m_mid]

            ul = u[idxm - 1]
            ur = u[idxm]

            w = (yl - ql) / (qr - ql).clamp_min(eps)
            Ff[m_mid] = ul + w * (ur - ul)

        return Ff.reshape(B, H)

    def cdf_at_y(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Tail-aware CDF evaluated exactly at realized y.

        params: (B,H,P)
        y:      (B,H)

        returns:
          F(y): (B,H)
        """
        body_params = params[..., :self.body_param_dim]
        raw_xi_low = params[..., self.body_param_dim]
        raw_xi_high = params[..., self.body_param_dim + 1]

        Q, q = self.forward(params)
        F_mid = self._cdf_from_quantiles_at_y(Q, y)  # (B,H)

        _, _, xL, qL, xU, qU, xiL, xiU, betaL, betaU = self._tail_params_from_body(
            body_params, raw_xi_low, raw_xi_high
        )

        # lower tail
        dL = (xL - y).clamp_min(0.0)
        near_zero_L = xiL.abs() < 1e-6

        baseL = 1.0 + xiL * dL / betaL
        S_exp_L = torch.exp(-dL / betaL)
        S_gpd_L = torch.where(
            baseL > 0,
            torch.pow(baseL, -1.0 / xiL),
            torch.zeros_like(baseL)
        )
        S_L = torch.where(near_zero_L, S_exp_L, S_gpd_L)
        F_low = self.u_low * S_L

        # upper tail
        dU = (y - xU).clamp_min(0.0)
        near_zero_U = xiU.abs() < 1e-6

        baseU = 1.0 + xiU * dU / betaU
        S_exp_U = torch.exp(-dU / betaU)
        S_gpd_U = torch.where(
            baseU > 0,
            torch.pow(baseU, -1.0 / xiU),
            torch.zeros_like(baseU)
        )
        S_U = torch.where(near_zero_U, S_exp_U, S_gpd_U)
        F_high = 1.0 - (1.0 - self.u_high) * S_U

        F = F_mid
        F = torch.where(y < xL, F_low, F)
        F = torch.where(y > xU, F_high, F)

        return F.clamp(0.0, 1.0)

    def pit(self, params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Alias for tail-aware PIT = F(y).
        """
        return self.cdf_at_y(params, y)
    
    def quantile_at_levels(self, params: torch.Tensor, u_eval: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the spliced quantile function at arbitrary probability levels.

        params: (B,H,P)
        u_eval: (A,)

        returns:
          Q_eval: (B,H,A)
        """
        if not torch.is_tensor(u_eval):
            u_eval = torch.tensor(u_eval, device=params.device, dtype=params.dtype)

        u_eval = u_eval.to(device=params.device, dtype=params.dtype).flatten()
        u_eval = u_eval.clamp(min=self.eps, max=1.0 - self.eps)

        body_params = params[..., :self.body_param_dim]
        raw_xi_low = params[..., self.body_param_dim]
        raw_xi_high = params[..., self.body_param_dim + 1]

        Q_body, q_body, xL, qL, xU, qU, xiL, xiU, betaL, betaU = self._tail_params_from_body(
            body_params, raw_xi_low, raw_xi_high
        )

        B, H, _ = Q_body.shape
        A = u_eval.numel()
        Q_eval = torch.empty(B, H, A, device=params.device, dtype=params.dtype)

        m_low = u_eval < self.u_low
        m_mid = (u_eval >= self.u_low) & (u_eval <= self.u_high)
        m_high = u_eval > self.u_high

        # lower tail
        if m_low.any():
            Q_lo, _ = self._lower_tail_quantile(u_eval[m_low], xL, betaL, xiL)
            Q_eval[..., m_low] = Q_lo

        # middle via interpolation on body grid
        if m_mid.any():
            mids = []
            for ue in u_eval[m_mid]:
                idx = int(torch.searchsorted(self.u, ue, right=False).item())
                if idx <= 0:
                    qv = Q_body[..., 0]
                elif idx >= self.u.numel():
                    qv = Q_body[..., -1]
                else:
                    ul = self.u[idx - 1]
                    ur = self.u[idx]
                    w = (ue - ul) / (ur - ul).clamp_min(self.eps)
                    qv = (1.0 - w) * Q_body[..., idx - 1] + w * Q_body[..., idx]
                mids.append(qv)
            Q_eval[..., m_mid] = torch.stack(mids, dim=-1)

        # upper tail
        if m_high.any():
            Q_hi, _ = self._upper_tail_quantile(u_eval[m_high], xU, betaU, xiU)
            Q_eval[..., m_high] = Q_hi

        return Q_eval

    def expected_shortfall(
        self,
        params: torch.Tensor,
        alphas: torch.Tensor,
        side: str = "lower",
        n_int: int = 256,
        log_grid: bool = True,
    ) -> torch.Tensor:
        """
        Numerical ES from the exact spliced quantile function.

        For lower tail:
          ES_alpha = (1/alpha) ∫_0^alpha Q(u) du

        For upper tail:
          ES_alpha = (1/alpha) ∫_{1-alpha}^1 Q(u) du

        params: (B,H,P)
        alphas: (A,)
        returns:
          ES: (B,H,A)
        """
        if not torch.is_tensor(alphas):
            alphas = torch.tensor(alphas, device=params.device, dtype=params.dtype)

        alphas = alphas.to(device=params.device, dtype=params.dtype).flatten()
        out = []

        for a in alphas:
            a_val = float(a.item())
            a_val = max(a_val, self.eps)

            if side == "lower":
                if log_grid and a_val < 0.2:
                    u_eval = torch.logspace(
                        math.log10(self.eps),
                        math.log10(a_val),
                        n_int,
                        device=params.device,
                        dtype=params.dtype,
                    )
                else:
                    u_eval = torch.linspace(
                        self.eps,
                        a_val,
                        n_int,
                        device=params.device,
                        dtype=params.dtype,
                    )

                Q_eval = self.quantile_at_levels(params, u_eval)   # (B,H,n_int)
                es = torch.trapz(Q_eval, u_eval, dim=-1) / a_val   # (B,H)

            elif side == "upper":
                if log_grid and a_val < 0.2:
                    t = torch.logspace(
                        math.log10(self.eps),
                        math.log10(a_val),
                        n_int,
                        device=params.device,
                        dtype=params.dtype,
                    )
                    u_eval = 1.0 - torch.flip(t, dims=[0])  # increasing: 1-a ... 1-eps
                else:
                    u_eval = torch.linspace(
                        1.0 - a_val,
                        1.0 - self.eps,
                        n_int,
                        device=params.device,
                        dtype=params.dtype,
                    )

                Q_eval = self.quantile_at_levels(params, u_eval)
                es = torch.trapz(Q_eval, u_eval, dim=-1) / a_val

            else:
                raise ValueError("side must be 'lower' or 'upper'")

            out.append(es)

        return torch.stack(out, dim=-1)  # (B,H,A)