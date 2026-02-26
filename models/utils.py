import torch
import matplotlib.pyplot as plt
import os 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def make_open_knots_from_internal(internal: torch.Tensor, degree: int, a=0.0, b=1.0):
    """
    Open/clamped knot vector on [a,b] from user internal knots.
    """
    internal = internal.to(dtype=torch.float64)  # stable knot math
    if internal.numel() > 0:
        if not torch.all(internal[1:] > internal[:-1]):
            raise ValueError("internal knots must be strictly increasing")
        if not (internal.min() > a and internal.max() < b):
            raise ValueError("internal knots must lie strictly inside (a,b)")

    k = int(degree)
    left = torch.full((k + 1,), float(a), dtype=internal.dtype, device=internal.device)
    right = torch.full((k + 1,), float(b), dtype=internal.dtype, device=internal.device)
    return torch.cat([left, internal, right], dim=0)

def make_open_nonuniform_knots(n_basis: int, degree: int, a=0.0, b=1.0, kind="power_tails", p=3.0, device="cpu"):
    """
    Convenience knot generator.
    n_basis = number of spline basis functions.
    """
    k = int(degree)
    n_internal = n_basis - (k + 1)
    if n_internal < 0:
        raise ValueError("n_basis must be >= degree+1")

    if n_internal == 0:
        internal = torch.empty(0, dtype=torch.float64, device=device)
        return make_open_knots_from_internal(internal, degree=degree, a=a, b=b)

    if kind == "uniform":
        internal = torch.linspace(a, b, n_internal + 2, dtype=torch.float64, device=device)[1:-1]
    elif kind == "power_tails":
        # p>1 => denser near 0 and 1
        t = torch.linspace(0.0, 1.0, n_internal + 2, dtype=torch.float64, device=device)[1:-1]
        u = 0.5 + 0.5 * torch.sign(t - 0.5) * torch.abs(2 * t - 1) ** float(p)
        internal = float(a) + (float(b) - float(a)) * u
    else:
        raise ValueError(f"Unknown kind: {kind}")

    eps = 1e-12
    internal = internal.clamp(a + eps, b - eps)
    # ensure strict increase (numerical safety)
    internal, _ = torch.sort(torch.unique(internal))
    return make_open_knots_from_internal(internal, degree=degree, a=a, b=b)


def bspline_basis(u: torch.Tensor, knots: torch.Tensor, degree: int):
    """
    Stable Cox–de Boor recursion for B-spline basis evaluation on a 1D grid.

    u:     (J,) increasing (can be non-uniform)
    knots: (n_knots,) = n_basis + degree + 1 (open/clamped recommended)
    returns:
      B: (J, n_basis)
    """
    u = u.to(dtype=torch.float64)
    t = knots.to(dtype=torch.float64, device=u.device)
    k = int(degree)

    # number of basis functions
    n_basis = t.numel() - (k + 1)
    if n_basis <= 0:
        raise ValueError("Invalid knots/degree: need len(knots) >= degree+2")

    # p=0 basis: N_{i,0}(u) = 1 if t_i <= u < t_{i+1}
    uu = u[:, None]  # (J,1)
    t0 = t[:-1][None, :]  # (1, n_knots-1)
    t1 = t[1:][None, :]   # (1, n_knots-1)
    N = ((uu >= t0) & (uu < t1)).to(u.dtype)  # (J, n_knots-1)

    # include the right endpoint u == t[-1] in the last basis
    N[u == t[-1], -1] = 1.0

    # recursion for d = 1..k
    for d in range(1, k + 1):
        # after each step, number of columns decreases by 1
        n_cols = N.shape[1]
        m = n_cols - 1
        if m <= 0:
            break

        t_i      = t[0:m]             # (m,)
        t_i_d    = t[d:d + m]         # (m,)
        t_i1     = t[1:m + 1]         # (m,)
        t_i_d1   = t[d + 1:d + 1 + m] # (m,)

        den1 = t_i_d  - t_i           # (m,)
        den2 = t_i_d1 - t_i1          # (m,)

        # safe divisions (avoid NaNs when denom = 0 due to repeated knots)
        den1_safe = torch.where(den1 > 0, den1, torch.ones_like(den1))
        den2_safe = torch.where(den2 > 0, den2, torch.ones_like(den2))

        w1 = (uu - t_i[None, :]) / den1_safe[None, :]
        w2 = (t_i_d1[None, :] - uu) / den2_safe[None, :]

        term1 = w1 * N[:, :m] * (den1 > 0).to(u.dtype)[None, :]
        term2 = w2 * N[:, 1:m + 1] * (den2 > 0).to(u.dtype)[None, :]

        N = term1 + term2

    # after k steps, N has (n_knots-1-k) = n_basis columns
    B = N[:, :n_basis]
    return B.to(dtype=u.dtype)  # float64

def mspline_ispline_on_grid(u: torch.Tensor, knots: torch.Tensor, degree: int, eps=1e-12):
    """
    Build M-spline and I-spline basis sampled on u-grid.
    Returns:
      M: (J, K) nonnegative
      I: (J, K) monotone in u, with I(0)=0 (on the grid)
    """
    u64 = u.to(dtype=torch.float64)
    B = bspline_basis(u64, knots, degree=degree)  # (J,K)

    k = int(degree)
    t = knots.to(dtype=torch.float64, device=u.device)
    K = B.shape[1]

    denom = (t[k + 1:k + 1 + K] - t[0:K]).clamp_min(eps)  # (K,)
    scale = (k + 1) / denom
    M = (B * scale[None, :]).clamp_min(0.0)

    # integrate M from u[0] to u[j] via trapezoid rule (precompute once)
    du = (u64[1:] - u64[:-1]).clamp_min(0.0)                   # (J-1,)
    M_mid = 0.5 * (M[1:, :] + M[:-1, :])                       # (J-1,K)
    I = torch.cumsum(M_mid * du[:, None], dim=0)               # (J-1,K)
    I = torch.cat([torch.zeros((1, K), dtype=torch.float64, device=u.device), I], dim=0)
    return M.to(u.dtype), I.to(u.dtype)


def chebyshev_lobatto_u(J: int, eps: float = 1e-5, device=None):
    # x_j = cos(pi*j/(J-1)) in [-1,1], dense at endpoints
    j = torch.arange(J, device=device, dtype=torch.float32)
    x = torch.cos(torch.pi * j / (J - 1))              # 1..-1
    u = (x + 1.0) / 2.0                                # in [0,1]
    u = torch.flip(u, dims=[0])                        # increasing
    u = u.clamp(eps, 1 - eps)                          # avoid exact 0/1
    return u

def uniform_u(J: int, eps: float = 1e-5, device=None, dtype=torch.float32):
    # J points centered in each of J equal bins -> strictly inside (0,1)
    u = (torch.arange(J, device=device, dtype=dtype) + 0.5) / J
    # optional extra safety clamp
    u = u.clamp(eps, 1.0 - eps)
    return u

def chebyshev_basis(u: torch.Tensor, K: int):
    """
    u: (J,) in (0,1)
    returns T: (J,K) with T_k(2u-1)
    """
    x = 2*u - 1
    T0 = torch.ones_like(x)
    if K == 1:
        return T0.unsqueeze(-1)
    T1 = x
    Ts = [T0, T1]
    for k in range(2, K):
        Ts.append(2*x*Ts[-1] - Ts[-2])
    return torch.stack(Ts[:K], dim=-1)

def _interp_idx_w(u_grid: torch.Tensor, u0: float, eps: float = 1e-12):
    # returns (i, w) s.t. value(u0) ≈ (1-w)*value[i] + w*value[i+1]
    i = torch.searchsorted(u_grid, torch.tensor(u0, device=u_grid.device)) - 1
    i = i.clamp(0, u_grid.numel() - 2)
    uL, uR = u_grid[i], u_grid[i+1]
    w = (u0 - uL) / (uR - uL + eps)
    return i, w

      
def cdf_from_quantile_on_grid(Q: torch.Tensor, u: torch.Tensor, z_grid: torch.Tensor, eps: float = 1e-8):
    """
    Q:      (B,H,J) monotone increasing in J
    u:      (J,)
    z_grid: (Z,)
    returns:
      F: (B,H,Z) with F(z_grid)
    """
    assert Q.ndim == 3
    B, H, J = Q.shape
    Z = z_grid.numel()

    # Expand u so we can gather with batched indices
    u_expand = u.view(1, 1, J).expand(B, H, J)  # (B,H,J)

    # idx_raw in [0..J], shape (B,H,Z)
    idx_raw = torch.searchsorted(Q, z_grid.view(1, 1, Z).expand(B, H, Z), right=True)

    # Masks for out-of-range z
    below = (idx_raw == 0)
    above = (idx_raw >= J)

    # Clamp to [1, J-1] so we can take idx-1 and idx safely
    idx = idx_raw.clamp(1, J - 1)

    idx0 = idx - 1
    idx1 = idx

    Q0 = Q.gather(-1, idx0)
    Q1 = Q.gather(-1, idx1)
    u0 = u_expand.gather(-1, idx0)
    u1 = u_expand.gather(-1, idx1)

    z = z_grid.view(1, 1, Z).expand(B, H, Z)
    t = (z - Q0) / (Q1 - Q0 + eps)
    F = u0 + t * (u1 - u0)

    # Set exact tails
    F = torch.where(below, torch.zeros_like(F), F)
    F = torch.where(above, torch.ones_like(F), F)
    return F

def _batched_range(n, bs):
    for i in range(0, n, bs):
        yield i, min(i + bs, n)
        
  
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_quantile_cdf_pdf(u_grid, Q_i_h, q_i_h, z_grid, cdf_i_h,
                          thr_low=None, thr_high=None, side="two_sided",
                          title_prefix="", out_path_prefix="dist"):
    """
    u_grid: (J,)
    Q_i_h:  (J,)
    q_i_h:  (J,)
    z_grid: (Z,)
    cdf_i_h:(Z,)
    """
    # 1) Quantile function
    plt.figure(figsize=(7,4))
    plt.plot(u_grid, Q_i_h, lw=2)
    plt.xlabel("u")
    plt.ylabel("Q(u)  [depeg bps]")
    plt.title(f"{title_prefix} Quantile function")
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_quantile.png", dpi=200, transparent=True, bbox_inches="tight")
    plt.close()

    # 2) CDF
    plt.figure(figsize=(7,4))
    plt.plot(z_grid, cdf_i_h, lw=2)
    if side in ["two_sided", "below"] and thr_low is not None:
        plt.axvline(thr_low, color="black", ls="--", lw=1, label="thr_low")
    if side in ["two_sided", "above"] and thr_high is not None:
        plt.axvline(thr_high, color="black", ls="--", lw=1, label="thr_high")
    plt.xlabel("z  [depeg bps]")
    plt.ylabel("F(z)")
    plt.title(f"{title_prefix} CDF")
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_cdf.png", dpi=200, transparent=True, bbox_inches="tight")
    plt.close()

    # 3) Implied PDF at quantile grid: f(Q(u)) ≈ 1 / q(u)
    pdf_at_Q = 1.0 / np.maximum(q_i_h, 1e-8)
    plt.figure(figsize=(7,4))
    plt.plot(Q_i_h, pdf_at_Q, lw=2)
    if side in ["two_sided", "below"] and thr_low is not None:
        plt.axvline(thr_low, color="black", ls="--", lw=1)
    if side in ["two_sided", "above"] and thr_high is not None:
        plt.axvline(thr_high, color="black", ls="--", lw=1)
    plt.xlabel("z = Q(u)  [depeg bps]")
    plt.ylabel("pdf(z)  (approx)")
    plt.title(f"{title_prefix} Implied PDF")
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_pdf.png", dpi=200, transparent=True, bbox_inches="tight")
    plt.close()


def plot_fan_chart(u_grid, Q_i_allH, y_true_i_allH,
                   qs=(0.01, 0.05, 0.5, 0.95, 0.99),
                   thr_low=None, thr_high=None, side="two_sided",
                   title_prefix="", out_path="fan.png"):
    """
    Q_i_allH: (H,J)
    y_true_i_allH: (H,)
    """
    H, J = Q_i_allH.shape
    u_grid = np.asarray(u_grid)

    def Q_at_p(Q_h, p):
        return np.interp(p, u_grid, Q_h)

    bands = {p: np.array([Q_at_p(Q_i_allH[h], p) for h in range(H)]) for p in qs}
    x = np.arange(H)

    plt.figure(figsize=(9,4))
    # widest band first
    if 0.01 in bands and 0.99 in bands:
        plt.fill_between(x, bands[0.01], bands[0.99], alpha=0.12, label="98% PI")
    if 0.05 in bands and 0.95 in bands:
        plt.fill_between(x, bands[0.05], bands[0.95], alpha=0.20, label="90% PI")

    if 0.5 in bands:
        plt.plot(x, bands[0.5], lw=2, label="median")

    plt.plot(x, y_true_i_allH, lw=1.5, color="red", label="true")

    if side in ["two_sided", "below"] and thr_low is not None:
        plt.axhline(thr_low, color="black", ls="--", lw=1)
    if side in ["two_sided", "above"] and thr_high is not None:
        plt.axhline(thr_high, color="black", ls="--", lw=1)

    plt.xlabel("horizon step")
    plt.ylabel("depeg bps")
    plt.title(f"{title_prefix} Fan chart over horizon")
    plt.legend(ncols =2, bbox_to_anchor=(0.66, -0.12), frameon = False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, transparent=True, bbox_inches="tight")
    plt.close()


def plot_pit_hist(u_grid, Q_all, y_true, horizon=0, bins=20,
                  title_prefix="", out_path="pit.png"):
    """
    PIT for a fixed horizon: PIT = F(y). We approximate via inverting Q:
      PIT = interp(y, Q(u), u)
    Q_all:  (B,H,J)
    y_true: (B,H)
    """
    u_grid = np.asarray(u_grid)
    Q_h = Q_all[:, horizon, :]      # (B,J)
    y_h = y_true[:, horizon]        # (B,)

    pits = []
    for i in range(Q_h.shape[0]):
        Qi = Q_h[i]
        yi = y_h[i]
        # clip to support to avoid NaNs
        if yi <= Qi[0]:
            pit = 0.0
        elif yi >= Qi[-1]:
            pit = 1.0
        else:
            pit = float(np.interp(yi, Qi, u_grid))
        pits.append(pit)

    pits = np.asarray(pits)

    plt.figure(figsize=(7,4))
    plt.hist(pits, bins=bins, range=(0,1), density=True, alpha=0.7, facecolor="cornflowerblue", edgecolor="black")
    plt.axhline(1.0, color="crimson", lw=1.5, label="uniform density")
    plt.xlabel("PIT = F(y)")
    plt.ylabel("density")
    plt.title(f"{title_prefix} PIT histogram (h={horizon})")
    plt.legend(ncols = 2, bbox_to_anchor=(0.66, -0.12), frameon = False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, transparent=True, bbox_inches="tight")
    plt.close()




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

# def chaining_function(x, threshold_high, threshold_low, side="two_sided", smooth_h=0.0):
#     if side == "two_sided":
#         if smooth_h and smooth_h > 0:
#             v_low = -torch.nn.functional.softplus((threshold_low - x) * smooth_h)  
#             v_high = torch.nn.functional.softplus((x - threshold_high) * smooth_h) 
#             v = v_low + v_high
#         else:
#             v = (x-threshold_low)*(x <= threshold_low).float() + (x-threshold_high)*(x >= threshold_high).float()
#     elif side == "below":
#         if smooth_h and smooth_h > 0:
#             v = -torch.nn.functional.softplus((threshold_low - x) * smooth_h)  
#         else:
#             v = (x-threshold_low)*(x <= threshold_low).float()
#     elif side == "above":
#         if smooth_h and smooth_h > 0:
#             v = torch.nn.functional.softplus((x - threshold_high) * smooth_h) 
#         else:
#             v = (x-threshold_high)*(x >= threshold_high).float()
#     return v

if __name__ == "__main__":
    pass