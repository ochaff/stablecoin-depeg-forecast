import torch
import matplotlib.pyplot as plt
import os 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kstest, cramervonmises


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

def logit_u(J: int, eps: float = 1e-6, device=None, dtype=torch.float32):
    # uniform spacing in logit space -> many points near 0 and 1
    lo = torch.logit(torch.tensor(eps, device=device, dtype=dtype))
    hi = torch.logit(torch.tensor(1.0 - eps, device=device, dtype=dtype))
    t = torch.linspace(lo, hi, J, device=device, dtype=dtype)
    u = torch.sigmoid(t)
    return u

def power_tails_u(J: int, p: float = 1/3, eps: float = 1e-6, device=None, dtype=torch.float32):
    # start from uniform in (0,1), then warp toward tails
    u0 = (torch.arange(J, device=device, dtype=dtype) + 0.5) / J  # (0,1)
    # symmetric power warp around 0.5
    u = 0.5 + 0.5 * torch.sign(u0 - 0.5) * (2.0 * torch.abs(u0 - 0.5))**p
    return u.clamp(eps, 1.0 - eps)

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

def plot_pit_ecdf(
    u_grid,
    Q_all,
    y_true,
    horizon=0,
    title_prefix="",
    out_path="pit_ecdf.png",
):
    """
    Plot PIT ECDF against Uniform(0,1) reference.

    Parameters
    ----------
    u_grid : (J,)
    Q_all  : (B,H,J)
    y_true : (B,H)
    horizon : int
    title_prefix : str
    out_path : str

    Returns
    -------
    pits : np.ndarray
    stats : dict
    """

    u_grid = np.asarray(u_grid, dtype=np.float64)
    Q_all = np.asarray(Q_all, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    Q_h = Q_all[:, horizon, :]
    y_h = y_true[:, horizon]

    pits = []
    for i in range(Q_h.shape[0]):
        Qi = np.maximum.accumulate(Q_h[i])
        yi = float(y_h[i])

        if yi <= Qi[0]:
            pit = float(u_grid[0])
        elif yi >= Qi[-1]:
            pit = float(u_grid[-1])
        else:
            pit = float(np.interp(yi, Qi, u_grid))
        pits.append(pit)

    pits = np.asarray(pits, dtype=np.float64)
    pits = pits[np.isfinite(pits)]
    pits = np.clip(pits, 0.0, 1.0)

    N = len(pits)
    stats = {"n": int(N), "ks_stat": np.nan, "ks_pvalue": np.nan}

    if N == 0:
        return pits, stats

    pits_sorted = np.sort(pits)
    ecdf = np.arange(1, N + 1) / N


    try:
        ks = kstest(pits, "uniform")
        stats["ks_stat"] = float(ks.statistic)
        stats["ks_pvalue"] = float(ks.pvalue)
    except Exception:
        pass

    # Approx 95% KS band
    band = 1.36 / np.sqrt(N)
    uu = np.linspace(0, 1, 500)
    lo = np.clip(uu - band, 0, 1)
    hi = np.clip(uu + band, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)

    ax.plot(uu, uu, "k--", lw=1.5, label="uniform")
    ax.fill_between(uu, lo, hi, color="gray", alpha=0.18, label="95% KS band")
    ax.step(pits_sorted, ecdf, where="post", color="royalblue", lw=2, label="PIT ECDF")

    title = f"{title_prefix} PIT ECDF (h={horizon})"
    if np.isfinite(stats["ks_stat"]):
        title += f"\nKS={stats['ks_stat']:.3f}, p={stats['ks_pvalue']:.3g}"

    ax.set_title(title)
    ax.set_xlabel("u")
    ax.set_ylabel("ECDF of PIT")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close(fig)

    return pits, stats

def plot_tail_exceedance_calibration(
    u_grid,
    Q_all,
    y_true,
    horizon=0,
    alphas=None,
    title_prefix="",
    out_path="tail_exceedance_calibration.png",
    log_x=True,
):
    """
    Tail calibration plot using exceedance frequencies.

    For each alpha:
      lower tail target: P(Y <= Q(alpha)) = alpha
      upper tail target: P(Y >= Q(1-alpha)) = alpha

    Parameters
    ----------
    u_grid : (J,)
    Q_all  : (B,H,J)
    y_true : (B,H)
    horizon : int
    alphas : sequence of tail probabilities, e.g. [0.005, 0.01, 0.02, 0.05, 0.1]
    title_prefix : str
    out_path : str
    log_x : bool

    Returns
    -------
    summary : dict
        Contains alpha grid and empirical lower/upper exceedance rates.
    """

    u_grid = np.asarray(u_grid, dtype=np.float64)
    Q_all = np.asarray(Q_all, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    if alphas is None:
        alphas = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2], dtype=np.float64)
    else:
        alphas = np.asarray(alphas, dtype=np.float64)

    Q_h = Q_all[:, horizon, :]   # (B,J)
    y_h = y_true[:, horizon]     # (B,)
    B = Q_h.shape[0]

    lower_emp = []
    upper_emp = []
    lower_se = []
    upper_se = []

    for alpha in alphas:
        # skip levels outside represented grid
        if alpha < u_grid[0] or (1.0 - alpha) > u_grid[-1]:
            lower_emp.append(np.nan)
            upper_emp.append(np.nan)
            lower_se.append(np.nan)
            upper_se.append(np.nan)
            continue

        q_low = np.empty(B, dtype=np.float64)
        q_high = np.empty(B, dtype=np.float64)

        for i in range(B):
            Qi = np.maximum.accumulate(Q_h[i])
            q_low[i] = np.interp(alpha, u_grid, Qi)
            q_high[i] = np.interp(1.0 - alpha, u_grid, Qi)

        emp_low = np.mean(y_h <= q_low)
        emp_up = np.mean(y_h >= q_high)

        # simple binomial SE under empirical estimate
        se_low = np.sqrt(max(emp_low * (1.0 - emp_low), 1e-12) / B)
        se_up = np.sqrt(max(emp_up * (1.0 - emp_up), 1e-12) / B)

        lower_emp.append(emp_low)
        upper_emp.append(emp_up)
        lower_se.append(se_low)
        upper_se.append(se_up)

    lower_emp = np.asarray(lower_emp)
    upper_emp = np.asarray(upper_emp)
    lower_se = np.asarray(lower_se)
    upper_se = np.asarray(upper_se)

    # reference ± 1.96 sqrt(alpha(1-alpha)/B)
    ref_se = np.sqrt(np.maximum(alphas * (1.0 - alphas), 1e-12) / B)
    ref_lo = np.clip(alphas - 1.96 * ref_se, 0, 1)
    ref_hi = np.clip(alphas + 1.96 * ref_se, 0, 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)

    valid = np.isfinite(lower_emp)
    ax.plot(alphas[valid], alphas[valid], "k--", lw=1.5, label="ideal")
    ax.fill_between(alphas[valid], ref_lo[valid], ref_hi[valid], color="gray", alpha=0.18, label="95% ref band")

    ax.plot(alphas[valid], lower_emp[valid], marker="o", lw=2, color="royalblue", label="lower tail")
    ax.plot(alphas[valid], upper_emp[valid], marker="s", lw=2, color="crimson", label="upper tail")

    ax.set_xlabel("nominal tail probability α")
    ax.set_ylabel("empirical exceedance probability")
    ax.set_title(f"{title_prefix} Tail exceedance calibration (h={horizon})")

    if log_x:
        ax.set_xscale("log")

    ax.set_ylim(0, max(0.25, np.nanmax([lower_emp, upper_emp, alphas]) * 1.1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n": int(B),
        "alphas": alphas,
        "lower_empirical": lower_emp,
        "upper_empirical": upper_emp,
        "reference_lo": ref_lo,
        "reference_hi": ref_hi,
    }
    return summary

def plot_tail_exceedance_ratio(
    summary,
    title_prefix="",
    out_path="tail_exceedance_ratio.png",
    log_x=True,
):

    alphas = np.asarray(summary["alphas"], dtype=np.float64)
    lower_emp = np.asarray(summary["lower_empirical"], dtype=np.float64)
    upper_emp = np.asarray(summary["upper_empirical"], dtype=np.float64)

    lower_ratio = lower_emp / alphas
    upper_ratio = upper_emp / alphas

    valid = np.isfinite(lower_ratio) & np.isfinite(upper_ratio)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=200)
    ax.axhline(1.0, color="k", ls="--", lw=1.5, label="ideal")
    ax.plot(alphas[valid], lower_ratio[valid], marker="o", lw=2, color="royalblue", label="lower tail")
    ax.plot(alphas[valid], upper_ratio[valid], marker="s", lw=2, color="crimson", label="upper tail")

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel("nominal tail probability α")
    ax.set_ylabel("empirical / nominal")
    ax.set_title(f"{title_prefix} Tail exceedance ratio")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close(fig)


def plot_pit_hist(
    u_grid,
    Q_all,
    y_true,
    horizon=0,
    bins=20,
    title_prefix="",
    out_path="pit.png",
):
    """
    PIT for a fixed horizon: PIT = F(y). We approximate via inverting Q:
      PIT = interp(y, Q(u), u)

    Parameters
    ----------
    u_grid : array-like, shape (J,)
        Quantile levels.
    Q_all : array-like, shape (B,H,J)
        Quantile forecasts.
    y_true : array-like, shape (B,H)
        Realized targets.
    horizon : int
        Horizon index.
    bins : int
        Number of histogram bins.
    title_prefix : str
        Prefix for plot title.
    out_path : str
        Path to save figure.

    Returns
    -------
    pits : np.ndarray, shape (B,)
        PIT values.
    stats : dict
        Summary diagnostics for PIT uniformity.
    """
    u_grid = np.asarray(u_grid, dtype=np.float64)
    Q_all = np.asarray(Q_all, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    Q_h = Q_all[:, horizon, :]   # (B,J)
    y_h = y_true[:, horizon]     # (B,)

    pits = []
    for i in range(Q_h.shape[0]):
        Qi = np.asarray(Q_h[i], dtype=np.float64)
        yi = float(y_h[i])

        # Guard against tiny non-monotonic numerical artifacts
        Qi = np.maximum.accumulate(Qi)

        # PIT by inversion of Q(u)
        # If y falls outside the represented quantile grid support,
        # we map to 0 or 1 as in your original implementation.
        if yi <= Qi[0]:
            pit = 0.0
        elif yi >= Qi[-1]:
            pit = 1.0
        else:
            pit = float(np.interp(yi, Qi, u_grid))

        pits.append(pit)

    pits = np.asarray(pits, dtype=np.float64)
    pits = pits[np.isfinite(pits)]

    N = len(pits)

    # Summary stats
    mean_pit = float(np.mean(pits)) if N > 0 else np.nan
    var_pit = float(np.var(pits)) if N > 0 else np.nan

    stats = {
        "n": int(N),
        "pit_mean": mean_pit,
        "pit_mean_target": 0.5,
        "pit_var": var_pit,
        "pit_var_target": 1.0 / 12.0,
        "ks_stat": np.nan,
        "ks_pvalue": np.nan,
        "cvm_stat": np.nan,
        "cvm_pvalue": np.nan,
    }

    if N > 0:
        try:
            ks = kstest(pits, "uniform")
            stats["ks_stat"] = float(ks.statistic)
            stats["ks_pvalue"] = float(ks.pvalue)
        except Exception:
            pass

        try:
            cvm = cramervonmises(pits, "uniform")
            stats["cvm_stat"] = float(cvm.statistic)
            stats["cvm_pvalue"] = float(cvm.pvalue)
        except Exception:
            pass

    # Histogram with density=True, so the null reference is density = 1
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)

    counts, edges, _ = ax.hist(
        pits,
        bins=bins,
        range=(0, 1),
        density=True,
        alpha=0.72,
        facecolor="cornflowerblue",
        edgecolor="black",
        linewidth=0.8,
    )

    # Uniform reference density
    ax.axhline(1.0, color="crimson", lw=1.5, label="uniform density")

    # 95% reference band under exact uniformity
    # Count in each bin ~ Binomial(N, p=1/bins)
    # Convert count band to density band by dividing by N * bin_width
    if N > 0:
        p = 1.0 / bins
        bin_width = 1.0 / bins

        expected_count = N * p
        std_count = np.sqrt(N * p * (1.0 - p))

        lo_count = max(0.0, expected_count - 1.96 * std_count)
        hi_count = expected_count + 1.96 * std_count

        lo_density = lo_count / (N * bin_width)
        hi_density = hi_count / (N * bin_width)

        ax.axhspan(
            lo_density,
            hi_density,
            color="gray",
            alpha=0.18,
            label="95% uniform band",
        )

    # Title with quantitative diagnostics
    title = (
        f"{title_prefix} PIT histogram (h={horizon})\n"
        f"mean={stats['pit_mean']:.3f} (target 0.500), "
        f"var={stats['pit_var']:.3f} (target {stats['pit_var_target']:.3f})"
    )

    if np.isfinite(stats["ks_stat"]):
        title += f"\nKS={stats['ks_stat']:.3f}, p={stats['ks_pvalue']:.3g}"
    if np.isfinite(stats["cvm_stat"]):
        title += f", CvM={stats['cvm_stat']:.3f}, p={stats['cvm_pvalue']:.3g}"

    ax.set_xlabel("PIT = F(y)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.set_xlim(0, 1)

    ax.legend(ncols=2, bbox_to_anchor=(0.75, -0.12), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close(fig)

    return pits, stats

def plot_pit_hist_from_values(
    pits,
    bins=20,
    title_prefix="",
    out_path="pit.png",
):
    pits = np.asarray(pits, dtype=np.float64)
    pits = pits[np.isfinite(pits)]
    pits = np.clip(pits, 0.0, 1.0)

    N = len(pits)
    mean_pit = float(np.mean(pits)) if N > 0 else np.nan
    var_pit = float(np.var(pits)) if N > 0 else np.nan

    stats = {
        "n": int(N),
        "pit_mean": mean_pit,
        "pit_mean_target": 0.5,
        "pit_var": var_pit,
        "pit_var_target": 1.0 / 12.0,
        "ks_stat": np.nan,
        "ks_pvalue": np.nan,
        "cvm_stat": np.nan,
        "cvm_pvalue": np.nan,
    }

    if N > 0:
        try:
            ks = kstest(pits, "uniform")
            stats["ks_stat"] = float(ks.statistic)
            stats["ks_pvalue"] = float(ks.pvalue)
        except Exception:
            pass

        try:
            cvm = cramervonmises(pits, "uniform")
            stats["cvm_stat"] = float(cvm.statistic)
            stats["cvm_pvalue"] = float(cvm.pvalue)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)

    ax.hist(
        pits,
        bins=bins,
        range=(0, 1),
        density=True,
        alpha=0.72,
        facecolor="cornflowerblue",
        edgecolor="black",
        linewidth=0.8,
    )

    ax.axhline(1.0, color="crimson", lw=1.5, label="uniform density")

    if N > 0:
        p = 1.0 / bins
        bin_width = 1.0 / bins
        expected_count = N * p
        std_count = np.sqrt(N * p * (1.0 - p))
        lo_count = max(0.0, expected_count - 1.96 * std_count)
        hi_count = expected_count + 1.96 * std_count
        lo_density = lo_count / (N * bin_width)
        hi_density = hi_count / (N * bin_width)

        ax.axhspan(lo_density, hi_density, color="gray", alpha=0.18, label="95% uniform band")

    title = (
        f"{title_prefix} PIT histogram\n"
        f"mean={stats['pit_mean']:.3f} (target 0.500), "
        f"var={stats['pit_var']:.3f} (target {stats['pit_var_target']:.3f})"
    )
    if np.isfinite(stats["ks_stat"]):
        title += f"\nKS={stats['ks_stat']:.3f}, p={stats['ks_pvalue']:.3g}"
    if np.isfinite(stats["cvm_stat"]):
        title += f", CvM={stats['cvm_stat']:.3f}, p={stats['cvm_pvalue']:.3g}"

    ax.set_xlabel("PIT = F(y)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(ncols=2, bbox_to_anchor=(0.75, -0.12), frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close(fig)

    return pits, stats

def plot_pit_ecdf_from_values(
    pits,
    title_prefix="",
    out_path="pit_ecdf.png",
):
    pits = np.asarray(pits, dtype=np.float64)
    pits = pits[np.isfinite(pits)]
    pits = np.clip(pits, 0.0, 1.0)

    N = len(pits)
    stats = {"n": int(N), "ks_stat": np.nan, "ks_pvalue": np.nan}

    if N == 0:
        return pits, stats

    pits_sorted = np.sort(pits)
    ecdf = np.arange(1, N + 1) / N


    try:
        ks = kstest(pits, "uniform")
        stats["ks_stat"] = float(ks.statistic)
        stats["ks_pvalue"] = float(ks.pvalue)
    except Exception:
        pass

    band = 1.36 / np.sqrt(N)
    uu = np.linspace(0, 1, 500)
    lo = np.clip(uu - band, 0, 1)
    hi = np.clip(uu + band, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.plot(uu, uu, "k--", lw=1.5, label="uniform")
    ax.fill_between(uu, lo, hi, color="gray", alpha=0.18, label="95% KS band")
    ax.step(pits_sorted, ecdf, where="post", color="royalblue", lw=2, label="PIT ECDF")

    title = f"{title_prefix} PIT ECDF"
    if np.isfinite(stats["ks_stat"]):
        title += f"\nKS={stats['ks_stat']:.3f}, p={stats['ks_pvalue']:.3g}"

    ax.set_title(title)
    ax.set_xlabel("u")
    ax.set_ylabel("ECDF of PIT")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close(fig)

    return pits, stats

def compute_spliced_pit_batched(spliced_quantile, params_np, y_true_np, horizon=0, batch_size=256, device="cpu"):
    """
    Compute exact tail-aware PIT = F(y) for a SplicedGPDQuantile.

    params_np: (B,H,P)
    y_true_np: (B,H)

    returns:
      pits: (B,)
    """
    B = params_np.shape[0]
    pits = np.empty(B, dtype=np.float32)

    spliced_quantile.eval()

    with torch.inference_mode():
        for b0, b1 in _batched_range(B, batch_size):
            params_t = torch.from_numpy(params_np[b0:b1, horizon:horizon+1]).to(device=device, dtype=torch.float32)  # (b,1,P)
            y_t = torch.from_numpy(y_true_np[b0:b1, horizon:horizon+1]).to(device=device, dtype=torch.float32)       # (b,1)

            pit_t = spliced_quantile.pit(params_t, y_t)  # (b,1)
            pits[b0:b1] = pit_t[:, 0].detach().cpu().numpy().astype(np.float32)

            del params_t, y_t, pit_t

    return pits

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