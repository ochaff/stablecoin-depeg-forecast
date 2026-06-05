# ============================================================
# Comparison notebook utilities for probabilistic forecasts
# ============================================================

import os
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kstest, cramervonmises
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

# ============================================================
# Basic helpers
# ============================================================

model_paths = {
    "SAINT": "3328/46ff2d0b6aa54b2886d4ac43a5cfec81/preds_test_set.pkl",
    "TimeXer": "3328/4cdfa4f81dd941db9c271275e9c0c8c9/preds_test_set.pkl",
    "TiDE": "./benchmark_outputs/tide_hist_exog/preds_test_set.pkl",
    "GARCH": "./benchmark_outputs/garch_student_t/preds_test_set.pkl",
    "ARIMA": "./arima_benchmark_eval/arima_preds_test_set.pkl",
    "Naive": "./benchmark_outputs/naive/preds_test_set.pkl",
}


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _model_name_from_path(path):
    path = Path(path)
    if path.name == "preds_test_set.pkl":
        return path.parent.name
    return path.stem


def load_forecast_pickles(model_paths, names=None):
    """
    Parameters
    ----------
    model_paths:
        list[str | Path] or dict[name -> path]
    names:
        optional list of model names if model_paths is a list.

    Returns
    -------
    runs : dict[name -> dict]
    """
    if isinstance(model_paths, dict):
        items = list(model_paths.items())
    else:
        if names is None:
            names = [_model_name_from_path(p) for p in model_paths]
        items = list(zip(names, model_paths))

    runs = {}

    for name, path in items:
        path = Path(path)
        with open(path, "rb") as f:
            A = pkl.load(f)

        required = ["true", "Q", "u_grid"]
        missing = [k for k in required if k not in A]
        if missing:
            raise ValueError(f"{name}: missing required keys {missing}")

        A["true"] = np.asarray(A["true"], dtype=np.float64)
        A["Q"] = np.asarray(A["Q"], dtype=np.float64)
        A["u_grid"] = np.asarray(A["u_grid"], dtype=np.float64)

        if "q" in A:
            A["q"] = np.asarray(A["q"], dtype=np.float64)

        runs[str(name)] = A
        print(f"Loaded {name}: true={A['true'].shape}, Q={A['Q'].shape}, path={path}")

    return runs

def _resolve_horizons(A, horizons_ahead=None):
    """
    Convert human-readable horizons into zero-based indices.

    Parameters
    ----------
    horizons_ahead:
        None or "all" -> all horizons.
        int -> one horizon.
        iterable[int] -> selected horizons.

    Returns
    -------
    h_idx : np.ndarray
        Zero-based horizon indices.
    label : str
        Human-readable label.
    """
    H = A["true"].shape[1]

    if horizons_ahead is None or horizons_ahead == "all":
        return np.arange(H), f"1-{H}h"

    if isinstance(horizons_ahead, (int, np.integer)):
        horizons_ahead = [int(horizons_ahead)]

    h_idx = np.array([int(h) - 1 for h in horizons_ahead], dtype=int)

    if np.any(h_idx < 0) or np.any(h_idx >= H):
        raise ValueError(f"Invalid horizons {horizons_ahead}; model has H={H}")

    if len(h_idx) == 1:
        label = f"{horizons_ahead[0]}h"
    else:
        label = ",".join([f"{h}h" for h in horizons_ahead])

    return h_idx, label


def _mean_selected(loss_bh, h_idx):
    """
    Mean loss over all observations and selected horizons.
    """
    loss_bh = np.asarray(loss_bh, dtype=np.float64)
    return float(np.nanmean(loss_bh[:, h_idx]))

def check_common_grid_and_shape(runs, atol=1e-8):
    """
    Verify that all runs use same y shape and same u-grid.
    """
    names = list(runs.keys())
    ref = runs[names[0]]

    y_shape = ref["true"].shape
    q_shape = ref["Q"].shape
    u_ref = ref["u_grid"]

    for name in names[1:]:
        A = runs[name]

        if A["true"].shape != y_shape:
            raise ValueError(f"{name}: true shape {A['true'].shape} != {y_shape}")

        if A["Q"].shape[:2] != q_shape[:2]:
            raise ValueError(f"{name}: Q B,H shape {A['Q'].shape[:2]} != {q_shape[:2]}")

        if len(A["u_grid"]) != len(u_ref) or not np.allclose(A["u_grid"], u_ref, atol=atol):
            raise ValueError(f"{name}: u_grid differs from reference grid.")

    print("All models have compatible true/Q shapes and common u_grid.")


def horizon_to_index(h_ahead):
    """
    Human horizon convention:
      1  -> index 0
      12 -> index 11
      24 -> index 23
    """
    return int(h_ahead) - 1


# ============================================================
# Quantile-grid numeric utilities
# ============================================================

def monotone_Q(Q):
    """
    Enforce monotonicity along quantile axis for numerical inversion.
    """
    return np.maximum.accumulate(Q, axis=-1)


def pit_from_quantile_grid(Q_all, y_true, u_grid, horizon_idx):
    """
    Approximate PIT by inverting Q(u).

    Q_all:  (B,H,J)
    y_true: (B,H)
    u_grid: (J,)
    """
    Q_h = monotone_Q(Q_all[:, horizon_idx, :])
    y_h = y_true[:, horizon_idx]

    B, J = Q_h.shape
    pits = np.empty(B, dtype=np.float64)

    for i in range(B):
        Qi = Q_h[i]
        yi = y_h[i]

        if yi <= Qi[0]:
            pits[i] = 0.0
        elif yi >= Qi[-1]:
            pits[i] = 1.0
        else:
            pits[i] = np.interp(yi, Qi, u_grid)

    return np.clip(pits[np.isfinite(pits)], 0.0, 1.0)


def pit_stats_from_values(pits):
    pits = np.asarray(pits, dtype=np.float64)
    pits = pits[np.isfinite(pits)]
    pits = np.clip(pits, 0.0, 1.0)

    out = {
        "n": int(len(pits)),
        "pit_mean": float(np.mean(pits)) if len(pits) else np.nan,
        "pit_var": float(np.var(pits)) if len(pits) else np.nan,
        "ks_stat": np.nan,
        "ks_pvalue": np.nan,
        "cvm_stat": np.nan,
        "cvm_pvalue": np.nan,
    }

    if len(pits):
        try:
            ks = kstest(pits, "uniform")
            out["ks_stat"] = float(ks.statistic)
            out["ks_pvalue"] = float(ks.pvalue)
        except Exception:
            pass

        try:
            cvm = cramervonmises(pits, "uniform")
            out["cvm_stat"] = float(cvm.statistic)
            out["cvm_pvalue"] = float(cvm.pvalue)
        except Exception:
            pass

    return out


def get_pits(A, h_ahead, prefer_stored=True):
    """
    Return PIT values for human-readable horizon h_ahead.
    Uses stored exact PIT if present, otherwise approximates by Q-grid inversion.
    """
    if prefer_stored:
        key = f"pit_h{h_ahead}"
        if key in A:
            return np.asarray(A[key], dtype=np.float64)

        # backward compatibility: h0 means 1-step ahead
        if h_ahead == 1 and "pit_h0" in A:
            return np.asarray(A["pit_h0"], dtype=np.float64)

    h_idx = horizon_to_index(h_ahead)
    return pit_from_quantile_grid(A["Q"], A["true"], A["u_grid"], h_idx)


def quantiles_from_u_grid(Q_all, u_grid, levels):
    """
    Interpolate quantiles from Q-grid.

    Q_all: (B,H,J)
    levels: iterable of probabilities

    returns:
      (B,H,A)
    """
    Q_all = monotone_Q(np.asarray(Q_all, dtype=np.float64))
    u_grid = np.asarray(u_grid, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64).reshape(-1)

    B, H, J = Q_all.shape
    out = np.empty((B, H, len(levels)), dtype=np.float64)

    for k, tau in enumerate(levels):
        idx = np.searchsorted(u_grid, tau, side="left")

        if idx <= 0:
            out[..., k] = Q_all[..., 0]
        elif idx >= J:
            out[..., k] = Q_all[..., -1]
        else:
            ul = u_grid[idx - 1]
            ur = u_grid[idx]
            w = (tau - ul) / max(ur - ul, 1e-12)
            out[..., k] = (1.0 - w) * Q_all[..., idx - 1] + w * Q_all[..., idx]

    return out


def expected_shortfall_from_quantile_grid(Q_all, u_grid, alphas, side="lower", eps=1e-8):
    """
    Approximate ES from quantile grid.

    lower:
      ES_alpha = 1/alpha ∫_0^alpha Q(u) du

    upper:
      ES_alpha = 1/alpha ∫_{1-alpha}^1 Q(u) du

    returns:
      ES: (B,H,A)
    """
    Q_all = monotone_Q(np.asarray(Q_all, dtype=np.float64))
    u_grid = np.asarray(u_grid, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64).reshape(-1)

    B, H, J = Q_all.shape
    out = np.empty((B, H, len(alphas)), dtype=np.float64)

    for k, a in enumerate(alphas):
        a = max(float(a), eps)

        if side == "lower":
            mask = u_grid <= a
            u_sel = u_grid[mask]
            Q_sel = Q_all[..., mask]

            q_a = quantiles_from_u_grid(Q_all, u_grid, [a])[..., 0:1]

            if u_sel.size == 0:
                u_aug = np.array([a], dtype=np.float64)
                Q_aug = q_a
            elif u_sel[-1] < a:
                u_aug = np.concatenate([u_sel, [a]])
                Q_aug = np.concatenate([Q_sel, q_a], axis=-1)
            else:
                u_aug = u_sel
                Q_aug = Q_sel

            # Include approximate point at u=0 by extending flat from first available quantile.
            # This improves stability if u_grid[0] > 0.
            if u_aug[0] > 0.0:
                u_aug = np.concatenate([[0.0], u_aug])
                Q_aug = np.concatenate([Q_aug[..., 0:1], Q_aug], axis=-1)

            integ = np.trapz(Q_aug, u_aug, axis=-1)
            out[..., k] = integ / a

        elif side == "upper":
            lo = 1.0 - a
            mask = u_grid >= lo
            u_sel = u_grid[mask]
            Q_sel = Q_all[..., mask]

            q_lo = quantiles_from_u_grid(Q_all, u_grid, [lo])[..., 0:1]

            if u_sel.size == 0:
                u_aug = np.array([lo], dtype=np.float64)
                Q_aug = q_lo
            elif u_sel[0] > lo:
                u_aug = np.concatenate([[lo], u_sel])
                Q_aug = np.concatenate([q_lo, Q_sel], axis=-1)
            else:
                u_aug = u_sel
                Q_aug = Q_sel

            # Include approximate point at u=1 by extending flat from last available quantile.
            if u_aug[-1] < 1.0:
                u_aug = np.concatenate([u_aug, [1.0]])
                Q_aug = np.concatenate([Q_aug, Q_aug[..., -1:]], axis=-1)

            integ = np.trapz(Q_aug, u_aug, axis=-1)
            out[..., k] = integ / a

        else:
            raise ValueError("side must be 'lower' or 'upper'")

    return out


def tail_exceedance_summary(Q_all, y_true, u_grid, horizon_idx, alphas):
    """
    For each alpha:
      lower empirical = P(Y <= Q(alpha))
      upper empirical = P(Y >= Q(1-alpha))
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    Q_h = Q_all[:, horizon_idx:horizon_idx + 1, :]
    y_h = y_true[:, horizon_idx]

    q_low = quantiles_from_u_grid(Q_h, u_grid, alphas)[:, 0, :]
    q_high = quantiles_from_u_grid(Q_h, u_grid, 1.0 - alphas)[:, 0, :]

    lower_emp = np.mean(y_h[:, None] <= q_low, axis=0)
    upper_emp = np.mean(y_h[:, None] >= q_high, axis=0)

    return {
        "n": int(len(y_h)),
        "alphas": alphas,
        "lower_empirical": lower_emp,
        "upper_empirical": upper_emp,
    }


def get_tail_summary(A, h_ahead, alphas, prefer_stored=True):
    """
    Use stored tail exceedance if available, otherwise compute from Q-grid.
    """
    if prefer_stored:
        key = f"tail_exceedance_h{h_ahead}"
        if key in A:
            return A[key]

        if h_ahead == 1 and "tail_exceedance_h0" in A:
            return A["tail_exceedance_h0"]

    h_idx = horizon_to_index(h_ahead)
    return tail_exceedance_summary(A["Q"], A["true"], A["u_grid"], h_idx, alphas)


# ============================================================
# Optional numpy twCRPS computation
# ============================================================

def trapezoid_weights_for_u_np(u_grid):
    u = np.asarray(u_grid, dtype=np.float64)
    du = u[1:] - u[:-1]

    wu = np.zeros_like(u)
    wu[0] = du[0] / 2.0
    wu[-1] = du[-1] / 2.0
    wu[1:-1] = (du[:-1] + du[1:]) / 2.0
    return wu


def chain_threshold_np(x, threshold_low=-10.0, threshold_high=10.0,
                       side="two_sided", smooth_h=1.0):
    """
    Numpy approximation of the usual chaining function for threshold weighting.

    For hard threshold:
      below:     min(x-low, 0)
      above:     max(x-high, 0)
      two_sided: below + above

    For smooth_h > 0, uses softplus smoothing.
    """
    x = np.asarray(x, dtype=np.float64)

    if smooth_h is None or smooth_h <= 0:
        below = np.minimum(x - threshold_low, 0.0)
        above = np.maximum(x - threshold_high, 0.0)
    else:
        h = float(smooth_h)

        # stable softplus
        def softplus(z):
            return np.logaddexp(0.0, z)

        below = -h * softplus((threshold_low - x) / h)
        above = h * softplus((x - threshold_high) / h)

    if side == "below":
        return below
    elif side == "above":
        return above
    elif side == "two_sided":
        return below + above
    else:
        raise ValueError("side must be 'below', 'above', or 'two_sided'")


def compute_twcrps_per_horizon_np(
    Q_all,
    y_true,
    u_grid,
    wu=None,
    threshold_low=-10.0,
    threshold_high=10.0,
    side="two_sided",
    smooth_h=1.0,
    crps_convention=True,
):
    """
    Compute twCRPS per horizon from Q-grid.

    This is useful if stored twCRPS_per_horizon is missing.
    For exact agreement with training, prefer stored values or PyTorch loss.
    """
    Q = np.asarray(Q_all, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)

    if wu is None:
        wu = trapezoid_weights_for_u_np(u)
    wu = np.asarray(wu, dtype=np.float64)

    B, H, J = Q.shape

    u3 = u.reshape(1, 1, J)
    wu3 = wu.reshape(1, 1, J)

    cy = chain_threshold_np(
        y[:, :, None],
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        side=side,
        smooth_h=smooth_h,
    )
    cQ = chain_threshold_np(
        Q,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        side=side,
        smooth_h=smooth_h,
    )

    e = cy - cQ
    pinball = np.maximum(u3 * e, (u3 - 1.0) * e)
    loss_bh = np.sum(pinball * wu3, axis=-1)

    if crps_convention:
        loss_bh = 2.0 * loss_bh

    return np.mean(loss_bh, axis=0), float(np.mean(loss_bh))


def get_twcrps_per_horizon(A, compute_if_missing=True, twcrps_kwargs=None):
    """
    Return twCRPS per horizon, using stored value if available.
    """
    if "twcrps_per_horizon" in A:
        return np.asarray(A["twcrps_per_horizon"], dtype=np.float64)

    if "test_twcrps_per_horizon" in A:
        return np.asarray(A["test_twcrps_per_horizon"], dtype=np.float64)

    if not compute_if_missing:
        return None

    twcrps_kwargs = twcrps_kwargs or {}
    per_h, overall = compute_twcrps_per_horizon_np(
        A["Q"],
        A["true"],
        A["u_grid"],
        **twcrps_kwargs,
    )
    A["twcrps_per_horizon_computed"] = per_h
    A["twcrps_computed"] = overall
    return per_h

# ============================================================
# Extreme pinball loss comparison
# ============================================================

def pinball_loss_array(y_true, q_pred, tau):
    """
    Elementwise pinball loss.

    y_true, q_pred: same shape
    tau: quantile level
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    q_pred = np.asarray(q_pred, dtype=np.float64)
    err = y_true - q_pred
    return np.maximum(tau * err, (tau - 1.0) * err)


def compute_extreme_pinball_per_horizon(
    A,
    alphas=(0.05, 0.01, 0.005, 0.001),
):
    """
    Compute lower and upper extreme pinball losses per horizon.

    Lower:
      tau = alpha

    Upper:
      tau = 1 - alpha

    Returns dataframe with:
      alpha, side, tau, horizon, pinball
    """
    y = np.asarray(A["true"], dtype=np.float64)      # (B,H)
    Q = np.asarray(A["Q"], dtype=np.float64)         # (B,H,J)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    B, H = y.shape
    rows = []

    for alpha in alphas:
        alpha = float(alpha)

        # Lower tail quantile Q(alpha)
        q_lower = quantiles_from_u_grid(Q, u, [alpha])[..., 0]  # (B,H)
        loss_lower = pinball_loss_array(y, q_lower, alpha)      # (B,H)

        # Upper tail quantile Q(1-alpha)
        tau_upper = 1.0 - alpha
        q_upper = quantiles_from_u_grid(Q, u, [tau_upper])[..., 0]
        loss_upper = pinball_loss_array(y, q_upper, tau_upper)

        for h_idx in range(H):
            rows.append({
                "alpha": alpha,
                "side": "lower",
                "tau": alpha,
                "horizon": h_idx + 1,
                "pinball": float(np.mean(loss_lower[:, h_idx])),
            })
            rows.append({
                "alpha": alpha,
                "side": "upper",
                "tau": tau_upper,
                "horizon": h_idx + 1,
                "pinball": float(np.mean(loss_upper[:, h_idx])),
            })

    return pd.DataFrame(rows)


def plot_extreme_pinball_comparison(
    runs,
    out_dir,
    alphas=(0.05, 0.01, 0.005, 0.001),
):
    """
    Compare extreme pinball loss per horizon for all models.

    Produces:
      compare_extreme_pinball_lower.png
      compare_extreme_pinball_upper.png
      compare_extreme_pinball_summary.csv
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    all_rows = []

    for name, A in runs.items():
        df = compute_extreme_pinball_per_horizon(A, alphas=alphas)
        df["model"] = name
        all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(out_dir / "compare_extreme_pinball_summary.csv", index=False)

    for side in ["lower", "upper"]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200, sharex=True)
        axes = axes.ravel()

        for ax, alpha in zip(axes, alphas):
            dfa = out[(out["side"] == side) & (np.isclose(out["alpha"], alpha))]

            for name in runs.keys():
                dfn = dfa[dfa["model"] == name].sort_values("horizon")
                ax.plot(
                    dfn["horizon"],
                    dfn["pinball"],
                    marker="o",
                    lw=2,
                    label=name,
                )

            if side == "lower":
                title = f"Lower tail pinball, τ={alpha:g}"
            else:
                title = f"Upper tail pinball, τ={1-alpha:g}"

            ax.set_title(title)
            ax.set_xlabel("Forecast horizon")
            ax.set_ylabel("Mean pinball loss")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=8)

        fig.suptitle(f"Extreme {side}-tail pinball loss by horizon", y=1.02)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"compare_extreme_pinball_{side}.png",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close(fig)

    return out

# ============================================================
# Threshold exceedance probability comparison
# ============================================================

def cdf_at_value_from_quantile_grid_2d(Q_all, u_grid, y_val, eps=1e-12):
    """
    Invert Q(u) to approximate F(y_val).

    Q_all: (B,H,J)
    u_grid: (J,)
    y_val: scalar

    returns:
      F_y: (B,H)
    """
    Q_all = monotone_Q(np.asarray(Q_all, dtype=np.float64))
    u_grid = np.asarray(u_grid, dtype=np.float64)

    B, H, J = Q_all.shape
    Qf = Q_all.reshape(-1, J)

    out = np.empty(Qf.shape[0], dtype=np.float64)

    for i, row in enumerate(Qf):
        idx = np.searchsorted(row, y_val, side="left")

        if idx <= 0:
            out[i] = 0.0
        elif idx >= J:
            out[i] = 1.0
        else:
            ql = row[idx - 1]
            qr = row[idx]
            ul = u_grid[idx - 1]
            ur = u_grid[idx]

            w = (y_val - ql) / max(qr - ql, eps)
            out[i] = ul + w * (ur - ul)

    return np.clip(out.reshape(B, H), 0.0, 1.0)


def compute_abs_threshold_event_prob(A, abs_threshold=15.0):
    """
    Compute predicted probability of event:

      |Y| >= abs_threshold

    from quantile grid.
    """
    Q = np.asarray(A["Q"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    F_lo = cdf_at_value_from_quantile_grid_2d(Q, u, -abs_threshold)
    F_hi = cdf_at_value_from_quantile_grid_2d(Q, u, abs_threshold)

    p_event = F_lo + (1.0 - F_hi)
    return np.clip(p_event, 0.0, 1.0)


def event_probability_metrics_per_horizon(
    p_event,
    y_true,
    abs_threshold=15.0,
    eps=1e-12,
):
    """
    Compute Brier, AUC, event rate, and mean predicted probability per horizon.
    """
    p_event = np.asarray(p_event, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    event = (np.abs(y_true) >= abs_threshold).astype(int)

    B, H = y_true.shape
    rows = []

    for h_idx in range(H):
        p = np.clip(p_event[:, h_idx], eps, 1.0 - eps)
        e = event[:, h_idx]

        brier = np.mean((p - e) ** 2)
        logloss = -np.mean(e * np.log(p) + (1 - e) * np.log(1 - p))

        if roc_auc_score is not None and len(np.unique(e)) == 2:
            auc = float(roc_auc_score(e, p))
        else:
            auc = np.nan

        rows.append({
            "horizon": h_idx + 1,
            "n": int(len(e)),
            "event_rate": float(np.mean(e)),
            "mean_pred_prob": float(np.mean(p)),
            "brier": float(brier),
            "logloss": float(logloss),
            "auc": auc,
            "n_events": int(np.sum(e)),
        })

    return pd.DataFrame(rows)


def _get_time_axis_for_horizon(A, h_ahead):
    """
    Use stored datetime grid if available, otherwise sample index.
    """
    h_idx = horizon_to_index(h_ahead)

    if "ds" in A:
        ds = np.asarray(A["ds"])
        if ds.ndim == 2 and h_idx < ds.shape[1]:
            return pd.to_datetime(ds[:, h_idx])

    if "meta" in A and isinstance(A["meta"], pd.DataFrame) and "cutoff" in A["meta"]:
        return pd.to_datetime(A["meta"]["cutoff"])

    return np.arange(A["true"].shape[0])


def plot_abs_threshold_event_probability_comparison(
    runs,
    out_dir,
    abs_threshold=15.0,
    horizons_ahead=(1, 12, 24),
    max_points=3000,
):
    """
    Plot timeline of predicted P(|Y| >= threshold) and realized events.

    Produces:
      compare_abs_event_probability_h1.png
      compare_abs_event_probability_h12.png
      compare_abs_event_probability_h24.png
      compare_abs_event_probability_metrics.csv
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    metric_rows = []
    p_cache = {}

    # Compute probabilities and metrics
    for name, A in runs.items():
        p_event = compute_abs_threshold_event_prob(A, abs_threshold=abs_threshold)
        p_cache[name] = p_event

        dfm = event_probability_metrics_per_horizon(
            p_event=p_event,
            y_true=A["true"],
            abs_threshold=abs_threshold,
        )
        dfm["model"] = name
        metric_rows.append(dfm)

    metrics_df = pd.concat(metric_rows, ignore_index=True)
    metrics_df.to_csv(
        out_dir / f"compare_abs{abs_threshold:g}_event_probability_metrics.csv",
        index=False,
    )

    # Timeline plots
    for h_ahead in horizons_ahead:
        h_idx = horizon_to_index(h_ahead)

        fig, ax = plt.subplots(figsize=(13, 4.5), dpi=200)

        # Use first model for realized event markers
        first_name = next(iter(runs.keys()))
        A0 = runs[first_name]

        y_h = A0["true"][:, h_idx]
        event_h = np.abs(y_h) >= abs_threshold
        x = _get_time_axis_for_horizon(A0, h_ahead)

        # Downsample if needed
        n = len(y_h)
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(int)
        else:
            idx = np.arange(n)

        x_plot = np.asarray(x)[idx]
        event_plot = event_h[idx]

        for name, A in runs.items():
            p_h = p_cache[name][:, h_idx]
            p_plot = p_h[idx]

            row = metrics_df[
                (metrics_df["model"] == name)
                & (metrics_df["horizon"] == h_ahead)
            ].iloc[0]

            label = (
                f"{name} "
                f"Brier={row['brier']:.3g}, "
                f"AUC={row['auc']:.3g}"
            )

            ax.plot(
                x_plot,
                p_plot,
                lw=1.8,
                label=label,
            )

        # Event markers
        if np.any(event_plot):
            ax.scatter(
                x_plot[event_plot],
                np.ones(np.sum(event_plot)) * 1.02,
                color="black",
                s=18,
                marker="|",
                label="realized |y| event",
                clip_on=False,
                zorder=5,
            )

        ax.set_ylim(-0.02, 1.08)
        ax.set_ylabel(rf"$P(|Y| \geq {abs_threshold:g})$")
        ax.set_xlabel("test sample / time")
        ax.set_title(
            rf"Threshold exceedance probability timeline, "
            rf"$|Y| \geq {abs_threshold:g}$, {h_ahead}h ahead"
        )
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8, ncols=1)

        fig.tight_layout()
        fig.savefig(
            out_dir / f"compare_abs{abs_threshold:g}_event_probability_h{h_ahead}.png",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Summary plot: Brier and AUC by horizon
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

    for name in runs.keys():
        dfn = metrics_df[metrics_df["model"] == name].sort_values("horizon")

        axes[0].plot(
            dfn["horizon"],
            dfn["brier"],
            marker="o",
            lw=2,
            label=name,
        )

        axes[1].plot(
            dfn["horizon"],
            dfn["auc"],
            marker="o",
            lw=2,
            label=name,
        )

    axes[0].set_title("Brier loss by horizon")
    axes[0].set_xlabel("Forecast horizon")
    axes[0].set_ylabel("Brier loss")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("AUC by horizon")
    axes[1].set_xlabel("Forecast horizon")
    axes[1].set_ylabel("AUC")
    axes[1].grid(alpha=0.3)

    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(
        out_dir / f"compare_abs{abs_threshold:g}_event_brier_auc_by_horizon.png",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close(fig)

    return metrics_df

# ============================================================
# Plotting functions
# ============================================================

def plot_twcrps_comparison(
    runs,
    out_dir,
    title="twCRPS by forecast horizon",
    compute_if_missing=True,
    twcrps_kwargs=None,
):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

    rows = []

    for name, A in runs.items():
        per_h = get_twcrps_per_horizon(
            A,
            compute_if_missing=compute_if_missing,
            twcrps_kwargs=twcrps_kwargs,
        )

        if per_h is None:
            print(f"Skipping {name}: no twCRPS_per_horizon.")
            continue

        H = len(per_h)
        horizons = np.arange(1, H + 1)

        ax.plot(horizons, per_h, marker="o", lw=2, label=name)

        rows.append(pd.DataFrame({
            "model": name,
            "horizon": horizons,
            "twcrps": per_h,
        }))

    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("twCRPS")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_twcrps_per_horizon.png", transparent=True, bbox_inches="tight")
    plt.close(fig)

    if rows:
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(out_dir / "compare_twcrps_per_horizon.csv", index=False)
        return df

    return None


def plot_pit_hist_comparison(
    runs,
    out_dir,
    horizons_ahead=(1, 12, 24),
    bins=20,
    prefer_stored=True,
):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    for h_ahead in horizons_ahead:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)

        stats_rows = []

        for name, A in runs.items():
            pits = get_pits(A, h_ahead, prefer_stored=prefer_stored)
            stats = pit_stats_from_values(pits)
            stats_rows.append({"model": name, "horizon": h_ahead, **stats})

            ax.hist(
                pits,
                bins=bins,
                range=(0, 1),
                density=True,
                histtype="step",
                lw=2,
                label=f"{name}  KS={stats['ks_stat']:.3f}",
            )

        ax.axhline(1.0, color="black", ls="--", lw=1.2, label="Uniform")
        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT = F(y)")
        ax.set_ylabel("Density")
        ax.set_title(f"PIT histogram comparison, {h_ahead}h ahead")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_pit_hist_h{h_ahead}.png", transparent=True, bbox_inches="tight")
        plt.close(fig)

        pd.DataFrame(stats_rows).to_csv(out_dir / f"compare_pit_stats_h{h_ahead}.csv", index=False)


def plot_pit_ecdf_comparison(
    runs,
    out_dir,
    horizons_ahead=(1, 12, 24),
    prefer_stored=True,
):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    for h_ahead in horizons_ahead:
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=200)

        uu = np.linspace(0, 1, 500)
        ax.plot(uu, uu, "k--", lw=1.5, label="Uniform")

        for name, A in runs.items():
            pits = get_pits(A, h_ahead, prefer_stored=prefer_stored)
            pits = np.clip(pits[np.isfinite(pits)], 0.0, 1.0)
            pits_sorted = np.sort(pits)
            ecdf = np.arange(1, len(pits_sorted) + 1) / max(len(pits_sorted), 1)

            stats = pit_stats_from_values(pits)

            ax.step(
                pits_sorted,
                ecdf,
                where="post",
                lw=2,
                label=f"{name}  KS={stats['ks_stat']:.3f}",
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("u")
        ax.set_ylabel("ECDF of PIT")
        ax.set_title(f"PIT ECDF comparison, {h_ahead}h ahead")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_pit_ecdf_h{h_ahead}.png", transparent=True, bbox_inches="tight")
        plt.close(fig)


def plot_tail_exceedance_comparison(
    runs,
    out_dir,
    horizons_ahead=(1, 12, 24),
    alphas=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    prefer_stored=True,
    log_x=True,
):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    alphas = np.asarray(alphas, dtype=np.float64)
    rows = []

    for h_ahead in horizons_ahead:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

        for name, A in runs.items():
            s = get_tail_summary(A, h_ahead, alphas, prefer_stored=prefer_stored)

            a = np.asarray(s["alphas"], dtype=np.float64)
            lower = np.asarray(s["lower_empirical"], dtype=np.float64)
            upper = np.asarray(s["upper_empirical"], dtype=np.float64)

            axes[0].plot(a, lower, marker="o", lw=2, label=name)
            axes[1].plot(a, upper, marker="o", lw=2, label=name)

            for ak, lo, up in zip(a, lower, upper):
                rows.append({
                    "model": name,
                    "horizon": h_ahead,
                    "alpha": ak,
                    "lower_empirical": lo,
                    "upper_empirical": up,
                    "lower_ratio": lo / ak if ak > 0 else np.nan,
                    "upper_ratio": up / ak if ak > 0 else np.nan,
                })

        for ax, side_name in zip(axes, ["Lower tail", "Upper tail"]):
            ax.plot(alphas, alphas, "k--", lw=1.5, label="Ideal")
            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("Nominal tail probability α")
            ax.set_ylabel("Empirical exceedance probability")
            ax.set_title(f"{side_name} calibration, {h_ahead}h ahead")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"compare_tail_exceedance_h{h_ahead}.png", transparent=True, bbox_inches="tight")
        plt.close(fig)

        # Ratio plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

        for name, A in runs.items():
            s = get_tail_summary(A, h_ahead, alphas, prefer_stored=prefer_stored)

            a = np.asarray(s["alphas"], dtype=np.float64)
            lower = np.asarray(s["lower_empirical"], dtype=np.float64)
            upper = np.asarray(s["upper_empirical"], dtype=np.float64)

            axes[0].plot(a, lower / a, marker="o", lw=2, label=name)
            axes[1].plot(a, upper / a, marker="o", lw=2, label=name)

        for ax, side_name in zip(axes, ["Lower tail", "Upper tail"]):
            ax.axhline(1.0, color="k", ls="--", lw=1.5, label="Ideal")
            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("Nominal tail probability α")
            ax.set_ylabel("Empirical / nominal")
            ax.set_title(f"{side_name} exceedance ratio, {h_ahead}h ahead")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"compare_tail_exceedance_ratio_h{h_ahead}.png", transparent=True, bbox_inches="tight")
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "compare_tail_exceedance_summary.csv", index=False)
    return df


# ============================================================
# VaR / ES comparison
# ============================================================

def var_es_summary_from_grid(A, h_ahead, alphas, side="lower"):
    """
    Compute VaR/ES diagnostics from Q-grid for one model/horizon.
    """
    h_idx = horizon_to_index(h_ahead)
    y = A["true"][:, h_idx]
    Q = A["Q"]
    u = A["u_grid"]

    alphas = np.asarray(alphas, dtype=np.float64)

    if side == "lower":
        var = quantiles_from_u_grid(Q[:, h_idx:h_idx + 1, :], u, alphas)[:, 0, :]
        es = expected_shortfall_from_quantile_grid(Q[:, h_idx:h_idx + 1, :], u, alphas, side="lower")[:, 0, :]
        hit = y[:, None] <= var
    elif side == "upper":
        var = quantiles_from_u_grid(Q[:, h_idx:h_idx + 1, :], u, 1.0 - alphas)[:, 0, :]
        es = expected_shortfall_from_quantile_grid(Q[:, h_idx:h_idx + 1, :], u, alphas, side="upper")[:, 0, :]
        hit = y[:, None] >= var
    else:
        raise ValueError("side must be lower or upper")

    coverage = np.mean(hit, axis=0)
    n_hits = np.sum(hit, axis=0)

    realized_tail_mean = np.full(len(alphas), np.nan)
    predicted_es_on_hits = np.full(len(alphas), np.nan)
    predicted_es_all = np.mean(es, axis=0)

    for k in range(len(alphas)):
        if n_hits[k] > 0:
            realized_tail_mean[k] = np.mean(y[hit[:, k]])
            predicted_es_on_hits[k] = np.mean(es[hit[:, k], k])

    return {
        "alphas": alphas,
        "coverage": coverage,
        "n_hits": n_hits,
        "realized_tail_mean": realized_tail_mean,
        "predicted_es_on_hits": predicted_es_on_hits,
        "predicted_es_all": predicted_es_all,
        "es_bias": predicted_es_on_hits - realized_tail_mean,
    }


def var_es_summary_from_stored_h0(A, side="lower"):
    """
    Use stored exact VaR/ES at h0 if available.
    This is useful for SplicedGPD exact diagnostics.
    """
    if "es_alphas_h0" not in A:
        return None

    y = A["true"][:, 0]
    alphas = np.asarray(A["es_alphas_h0"], dtype=np.float64)

    if side == "lower":
        if "var_lower_h0" not in A or "es_lower_h0" not in A:
            return None
        var = np.asarray(A["var_lower_h0"], dtype=np.float64)
        es = np.asarray(A["es_lower_h0"], dtype=np.float64)
        hit = y[:, None] <= var
    else:
        if "var_upper_h0" not in A or "es_upper_h0" not in A:
            return None
        var = np.asarray(A["var_upper_h0"], dtype=np.float64)
        es = np.asarray(A["es_upper_h0"], dtype=np.float64)
        hit = y[:, None] >= var

    coverage = np.mean(hit, axis=0)
    n_hits = np.sum(hit, axis=0)

    realized_tail_mean = np.full(len(alphas), np.nan)
    predicted_es_on_hits = np.full(len(alphas), np.nan)
    predicted_es_all = np.mean(es, axis=0)

    for k in range(len(alphas)):
        if n_hits[k] > 0:
            realized_tail_mean[k] = np.mean(y[hit[:, k]])
            predicted_es_on_hits[k] = np.mean(es[hit[:, k], k])

    return {
        "alphas": alphas,
        "coverage": coverage,
        "n_hits": n_hits,
        "realized_tail_mean": realized_tail_mean,
        "predicted_es_on_hits": predicted_es_on_hits,
        "predicted_es_all": predicted_es_all,
        "es_bias": predicted_es_on_hits - realized_tail_mean,
    }


def plot_var_es_comparison(
    runs,
    out_dir,
    horizons_ahead=(1, 12, 24),
    alphas=(0.01, 0.02, 0.05),
    prefer_stored_spliced_h0=True,
    log_x=True,
):
    """
    Compare VaR coverage and ES bias.

    For h=1, if a model has stored exact h0 VaR/ES, use it.
    Otherwise compute from Q-grid.
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    all_rows = []

    for h_ahead in horizons_ahead:
        for side in ["lower", "upper"]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

            for name, A in runs.items():
                s = None

                if prefer_stored_spliced_h0 and h_ahead == 1:
                    s = var_es_summary_from_stored_h0(A, side=side)

                if s is None:
                    s = var_es_summary_from_grid(A, h_ahead, alphas, side=side)

                a = np.asarray(s["alphas"], dtype=np.float64)
                coverage = np.asarray(s["coverage"], dtype=np.float64)
                es_bias = np.asarray(s["es_bias"], dtype=np.float64)

                axes[0].plot(a, coverage, marker="o", lw=2, label=name)
                axes[1].plot(a, es_bias, marker="o", lw=2, label=name)

                for k in range(len(a)):
                    all_rows.append({
                        "model": name,
                        "horizon": h_ahead,
                        "side": side,
                        "alpha": a[k],
                        "coverage": coverage[k],
                        "coverage_ratio": coverage[k] / a[k],
                        "n_hits": int(s["n_hits"][k]),
                        "realized_tail_mean": s["realized_tail_mean"][k],
                        "predicted_es_on_hits": s["predicted_es_on_hits"][k],
                        "predicted_es_all": s["predicted_es_all"][k],
                        "es_bias": es_bias[k],
                    })

            # Coverage panel
            axes[0].plot(alphas, alphas, "k--", lw=1.5, label="Ideal")
            axes[0].set_xlabel("Tail probability α")
            axes[0].set_ylabel("Empirical VaR exceedance")
            axes[0].set_title(f"VaR coverage, {side}, {h_ahead}h ahead")
            axes[0].grid(alpha=0.3)

            # ES bias panel
            axes[1].axhline(0.0, color="k", ls="--", lw=1.5, label="Zero bias")
            axes[1].set_xlabel("Tail probability α")
            axes[1].set_ylabel("Predicted ES on hits - realized tail mean")
            axes[1].set_title(f"ES bias, {side}, {h_ahead}h ahead")
            axes[1].grid(alpha=0.3)

            if log_x:
                axes[0].set_xscale("log")
                axes[1].set_xscale("log")

            axes[0].legend(frameon=False, fontsize=8)
            axes[1].legend(frameon=False, fontsize=8)

            fig.tight_layout()
            fig.savefig(out_dir / f"compare_var_es_{side}_h{h_ahead}.png", transparent=True, bbox_inches="tight")
            plt.close(fig)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "compare_var_es_summary.csv", index=False)
    return df


# ============================================================
# Spliced GPD comparison plots
# ============================================================

def plot_spliced_gpd_tail_parameter_comparison(runs, out_dir):
    """
    Compare stored SplicedGPD tail parameters if A["gpd_tail_info"] is present.

    Expected keys inside gpd_tail_info may include:
      xiL, xiU, betaL, betaU, xL, xU
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    keys = ["xiL", "xiU", "betaL", "betaU", "xL", "xU"]
    available_any = False

    for key in keys:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
        plotted = False

        for name, A in runs.items():
            info = A.get("gpd_tail_info", None)
            if info is None or key not in info:
                continue

            vals = np.asarray(info[key], dtype=np.float64).reshape(-1)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            ax.hist(
                vals,
                bins=50,
                density=True,
                histtype="step",
                lw=2,
                label=f"{name} mean={np.mean(vals):.3g}",
            )
            plotted = True
            available_any = True

        if plotted:
            ax.set_title(f"SplicedGPD tail parameter comparison: {key}")
            ax.set_xlabel(key)
            ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"compare_spliced_gpd_{key}.png", transparent=True, bbox_inches="tight")
        plt.close(fig)

    if not available_any:
        print("No gpd_tail_info found in provided runs. Skipping SplicedGPD parameter plots.")




# ============================================================
# Master comparison function
# ============================================================

def compare_probabilistic_forecasts(
    model_paths,
    out_dir="./comparison_plots",
    names=None,
    horizons_ahead=(1, 12, 24),
    tail_alphas=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    es_alphas=(0.01, 0.02, 0.05),
    prefer_stored_pits=True,
    prefer_stored_tail=True,
    include_spliced_gpd=True,
    compute_twcrps_if_missing=True,
    twcrps_kwargs=None,
    include_extreme_pinball=True,
    extreme_pinball_alphas=(0.05, 0.01, 0.005, 0.001),
    include_event_probability=True,
    event_abs_threshold=15.0,
):
    """
    Main entry point.

    Parameters
    ----------
    model_paths:
        dict[name -> path] or list[path]
    out_dir:
        directory for comparison plots/tables
    horizons_ahead:
        human-readable horizons, e.g. (1, 12, 24)
    prefer_stored_pits:
        If True, use A["pit_h1"], etc. when present.
        Useful for SplicedGPD exact tail-aware PIT.
    prefer_stored_tail:
        If True, use stored tail exceedance summaries when present.
    include_spliced_gpd:
        If True, plot gpd_tail_info distributions when available.
    twcrps_kwargs:
        Used only if twCRPS is missing and must be computed from Q-grid.
        Example:
          dict(threshold_low=-10, threshold_high=10, side="two_sided", smooth_h=1.0)
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    runs = load_forecast_pickles(model_paths, names=names)
    check_common_grid_and_shape(runs)

    # Save run overview
    overview_rows = []
    for name, A in runs.items():
        overview_rows.append({
            "model": name,
            "B": A["true"].shape[0],
            "H": A["true"].shape[1],
            "J": A["Q"].shape[2],
            "u_min": float(A["u_grid"][0]),
            "u_max": float(A["u_grid"][-1]),
            "has_twcrps_per_horizon": "twcrps_per_horizon" in A or "test_twcrps_per_horizon" in A,
            "has_gpd_tail_info": "gpd_tail_info" in A,
        })
    pd.DataFrame(overview_rows).to_csv(out_dir / "model_overview.csv", index=False)

    # 1) twCRPS
    tw_df = plot_twcrps_comparison(
        runs,
        out_dir=out_dir,
        compute_if_missing=compute_twcrps_if_missing,
        twcrps_kwargs=twcrps_kwargs,
    )

    # 2) PIT histograms and ECDFs
    plot_pit_hist_comparison(
        runs,
        out_dir=out_dir,
        horizons_ahead=horizons_ahead,
        prefer_stored=prefer_stored_pits,
    )

    plot_pit_ecdf_comparison(
        runs,
        out_dir=out_dir,
        horizons_ahead=horizons_ahead,
        prefer_stored=prefer_stored_pits,
    )

    # 3) Tail exceedance calibration and ratios
    tail_df = plot_tail_exceedance_comparison(
        runs,
        out_dir=out_dir,
        horizons_ahead=horizons_ahead,
        alphas=tail_alphas,
        prefer_stored=prefer_stored_tail,
    )

    # 4) VaR / ES
    es_df = plot_var_es_comparison(
        runs,
        out_dir=out_dir,
        horizons_ahead=horizons_ahead,
        alphas=es_alphas,
        prefer_stored_spliced_h0=True,
    )
    # 5) Extreme pinball losses
    pinball_df = None
    if include_extreme_pinball:
        pinball_df = plot_extreme_pinball_comparison(
            runs,
            out_dir=out_dir,
            alphas=extreme_pinball_alphas,
        )

    # 6) Threshold event probability timeline and Brier/AUC
    event_df = None
    if include_event_probability:
        event_df = plot_abs_threshold_event_probability_comparison(
            runs,
            out_dir=out_dir,
            abs_threshold=event_abs_threshold,
            horizons_ahead=horizons_ahead,
        )

    # 7) Optional SplicedGPD parameter diagnostics
    if include_spliced_gpd:
        plot_spliced_gpd_tail_parameter_comparison(runs, out_dir=out_dir)

    print(f"\nComparison outputs saved to: {out_dir.resolve()}")

    return {
        "runs": runs,
        "twcrps_df": tw_df,
        "tail_df": tail_df,
        "es_df": es_df,
        "pinball_df": pinball_df,
        "event_df": event_df,
        "out_dir": out_dir,
    }

# ============================================================
# Comparison notebook utilities for probabilistic forecasts
# ============================================================

import os
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kstest, cramervonmises
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

# ============================================================
# Basic helpers
# ============================================================

def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _model_name_from_path(path):
    path = Path(path)
    if path.name == "preds_test_set.pkl":
        return path.parent.name
    return path.stem


def load_forecast_pickles(model_paths, names=None):
    """
    Parameters
    ----------
    model_paths:
        list[str | Path] or dict[name -> path]
    names:
        optional list of model names if model_paths is a list.

    Returns
    -------
    runs : dict[name -> dict]
    """
    if isinstance(model_paths, dict):
        items = list(model_paths.items())
    else:
        if names is None:
            names = [_model_name_from_path(p) for p in model_paths]
        items = list(zip(names, model_paths))

    runs = {}

    for name, path in items:
        path = Path(path)
        with open(path, "rb") as f:
            A = pkl.load(f)

        required = ["true", "Q", "u_grid"]
        missing = [k for k in required if k not in A]
        if missing:
            raise ValueError(f"{name}: missing required keys {missing}")

        A["true"] = np.asarray(A["true"], dtype=np.float64)
        A["Q"] = np.asarray(A["Q"], dtype=np.float64)
        A["u_grid"] = np.asarray(A["u_grid"], dtype=np.float64)

        if "q" in A:
            A["q"] = np.asarray(A["q"], dtype=np.float64)

        runs[str(name)] = A
        print(f"Loaded {name}: true={A['true'].shape}, Q={A['Q'].shape}, path={path}")

    return runs


def check_common_grid_and_shape(runs, atol=1e-8):
    """
    Verify that all runs use same y shape and same u-grid.
    """
    names = list(runs.keys())
    ref = runs[names[0]]

    y_shape = ref["true"].shape
    q_shape = ref["Q"].shape
    u_ref = ref["u_grid"]

    for name in names[1:]:
        A = runs[name]

        if A["true"].shape != y_shape:
            raise ValueError(f"{name}: true shape {A['true'].shape} != {y_shape}")

        if A["Q"].shape[:2] != q_shape[:2]:
            raise ValueError(f"{name}: Q B,H shape {A['Q'].shape[:2]} != {q_shape[:2]}")

        if len(A["u_grid"]) != len(u_ref) or not np.allclose(A["u_grid"], u_ref, atol=atol):
            raise ValueError(f"{name}: u_grid differs from reference grid.")

    print("All models have compatible true/Q shapes and common u_grid.")


def horizon_to_index(h_ahead):
    """
    Human horizon convention:
      1  -> index 0
      12 -> index 11
      24 -> index 23
    """
    return int(h_ahead) - 1


# ============================================================
# Quantile-grid numeric utilities
# ============================================================

def monotone_Q(Q):
    """
    Enforce monotonicity along quantile axis for numerical inversion.
    """
    return np.maximum.accumulate(Q, axis=-1)


def pit_from_quantile_grid(Q_all, y_true, u_grid, horizon_idx):
    """
    Approximate PIT by inverting Q(u).

    Q_all:  (B,H,J)
    y_true: (B,H)
    u_grid: (J,)
    """
    Q_h = monotone_Q(Q_all[:, horizon_idx, :])
    y_h = y_true[:, horizon_idx]

    B, J = Q_h.shape
    pits = np.empty(B, dtype=np.float64)

    for i in range(B):
        Qi = Q_h[i]
        yi = y_h[i]

        if yi <= Qi[0]:
            pits[i] = 0.0
        elif yi >= Qi[-1]:
            pits[i] = 1.0
        else:
            pits[i] = np.interp(yi, Qi, u_grid)

    return np.clip(pits[np.isfinite(pits)], 0.0, 1.0)


def pit_stats_from_values(pits):
    pits = np.asarray(pits, dtype=np.float64)
    pits = pits[np.isfinite(pits)]
    pits = np.clip(pits, 0.0, 1.0)

    out = {
        "n": int(len(pits)),
        "pit_mean": float(np.mean(pits)) if len(pits) else np.nan,
        "pit_var": float(np.var(pits)) if len(pits) else np.nan,
        "ks_stat": np.nan,
        "ks_pvalue": np.nan,
        "cvm_stat": np.nan,
        "cvm_pvalue": np.nan,
    }

    if len(pits):
        try:
            ks = kstest(pits, "uniform")
            out["ks_stat"] = float(ks.statistic)
            out["ks_pvalue"] = float(ks.pvalue)
        except Exception:
            pass

        try:
            cvm = cramervonmises(pits, "uniform")
            out["cvm_stat"] = float(cvm.statistic)
            out["cvm_pvalue"] = float(cvm.pvalue)
        except Exception:
            pass

    return out


def get_pits(A, h_ahead, prefer_stored=True):
    """
    Return PIT values for human-readable horizon h_ahead.
    Uses stored exact PIT if present, otherwise approximates by Q-grid inversion.
    """
    if prefer_stored:
        key = f"pit_h{h_ahead}"
        if key in A:
            return np.asarray(A[key], dtype=np.float64)

        # backward compatibility: h0 means 1-step ahead
        if h_ahead == 1 and "pit_h0" in A:
            return np.asarray(A["pit_h0"], dtype=np.float64)

    h_idx = horizon_to_index(h_ahead)
    return pit_from_quantile_grid(A["Q"], A["true"], A["u_grid"], h_idx)


def quantiles_from_u_grid(Q_all, u_grid, levels):
    """
    Interpolate quantiles from Q-grid.

    Q_all: (B,H,J)
    levels: iterable of probabilities

    returns:
      (B,H,A)
    """
    Q_all = monotone_Q(np.asarray(Q_all, dtype=np.float64))
    u_grid = np.asarray(u_grid, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64).reshape(-1)

    B, H, J = Q_all.shape
    out = np.empty((B, H, len(levels)), dtype=np.float64)

    for k, tau in enumerate(levels):
        idx = np.searchsorted(u_grid, tau, side="left")

        if idx <= 0:
            out[..., k] = Q_all[..., 0]
        elif idx >= J:
            out[..., k] = Q_all[..., -1]
        else:
            ul = u_grid[idx - 1]
            ur = u_grid[idx]
            w = (tau - ul) / max(ur - ul, 1e-12)
            out[..., k] = (1.0 - w) * Q_all[..., idx - 1] + w * Q_all[..., idx]

    return out


def expected_shortfall_from_quantile_grid(Q_all, u_grid, alphas, side="lower", eps=1e-8):
    """
    Approximate ES from quantile grid.

    lower:
      ES_alpha = 1/alpha ∫_0^alpha Q(u) du

    upper:
      ES_alpha = 1/alpha ∫_{1-alpha}^1 Q(u) du

    returns:
      ES: (B,H,A)
    """
    Q_all = monotone_Q(np.asarray(Q_all, dtype=np.float64))
    u_grid = np.asarray(u_grid, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64).reshape(-1)

    B, H, J = Q_all.shape
    out = np.empty((B, H, len(alphas)), dtype=np.float64)

    for k, a in enumerate(alphas):
        a = max(float(a), eps)

        if side == "lower":
            mask = u_grid <= a
            u_sel = u_grid[mask]
            Q_sel = Q_all[..., mask]

            q_a = quantiles_from_u_grid(Q_all, u_grid, [a])[..., 0:1]

            if u_sel.size == 0:
                u_aug = np.array([a], dtype=np.float64)
                Q_aug = q_a
            elif u_sel[-1] < a:
                u_aug = np.concatenate([u_sel, [a]])
                Q_aug = np.concatenate([Q_sel, q_a], axis=-1)
            else:
                u_aug = u_sel
                Q_aug = Q_sel

            # Include approximate point at u=0 by extending flat from first available quantile.
            # This improves stability if u_grid[0] > 0.
            if u_aug[0] > 0.0:
                u_aug = np.concatenate([[0.0], u_aug])
                Q_aug = np.concatenate([Q_aug[..., 0:1], Q_aug], axis=-1)

            integ = np.trapz(Q_aug, u_aug, axis=-1)
            out[..., k] = integ / a

        elif side == "upper":
            lo = 1.0 - a
            mask = u_grid >= lo
            u_sel = u_grid[mask]
            Q_sel = Q_all[..., mask]

            q_lo = quantiles_from_u_grid(Q_all, u_grid, [lo])[..., 0:1]

            if u_sel.size == 0:
                u_aug = np.array([lo], dtype=np.float64)
                Q_aug = q_lo
            elif u_sel[0] > lo:
                u_aug = np.concatenate([[lo], u_sel])
                Q_aug = np.concatenate([q_lo, Q_sel], axis=-1)
            else:
                u_aug = u_sel
                Q_aug = Q_sel

            # Include approximate point at u=1 by extending flat from last available quantile.
            if u_aug[-1] < 1.0:
                u_aug = np.concatenate([u_aug, [1.0]])
                Q_aug = np.concatenate([Q_aug, Q_aug[..., -1:]], axis=-1)

            integ = np.trapz(Q_aug, u_aug, axis=-1)
            out[..., k] = integ / a

        else:
            raise ValueError("side must be 'lower' or 'upper'")

    return out


def tail_exceedance_summary(Q_all, y_true, u_grid, horizon_idx, alphas):
    """
    For each alpha:
      lower empirical = P(Y <= Q(alpha))
      upper empirical = P(Y >= Q(1-alpha))
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    Q_h = Q_all[:, horizon_idx:horizon_idx + 1, :]
    y_h = y_true[:, horizon_idx]

    q_low = quantiles_from_u_grid(Q_h, u_grid, alphas)[:, 0, :]
    q_high = quantiles_from_u_grid(Q_h, u_grid, 1.0 - alphas)[:, 0, :]

    lower_emp = np.mean(y_h[:, None] <= q_low, axis=0)
    upper_emp = np.mean(y_h[:, None] >= q_high, axis=0)

    return {
        "n": int(len(y_h)),
        "alphas": alphas,
        "lower_empirical": lower_emp,
        "upper_empirical": upper_emp,
    }


def get_tail_summary(A, h_ahead, alphas, prefer_stored=True):
    """
    Use stored tail exceedance if available, otherwise compute from Q-grid.
    """
    if prefer_stored:
        key = f"tail_exceedance_h{h_ahead}"
        if key in A:
            return A[key]

        if h_ahead == 1 and "tail_exceedance_h0" in A:
            return A["tail_exceedance_h0"]

    h_idx = horizon_to_index(h_ahead)
    return tail_exceedance_summary(A["Q"], A["true"], A["u_grid"], h_idx, alphas)


# ============================================================
# Optional numpy twCRPS computation
# ============================================================

def trapezoid_weights_for_u_np(u_grid):
    u = np.asarray(u_grid, dtype=np.float64)
    du = u[1:] - u[:-1]

    wu = np.zeros_like(u)
    wu[0] = du[0] / 2.0
    wu[-1] = du[-1] / 2.0
    wu[1:-1] = (du[:-1] + du[1:]) / 2.0
    return wu


def chain_threshold_np(x, threshold_low=-10.0, threshold_high=10.0,
                       side="two_sided", smooth_h=1.0):
    """
    Numpy approximation of the usual chaining function for threshold weighting.

    For hard threshold:
      below:     min(x-low, 0)
      above:     max(x-high, 0)
      two_sided: below + above

    For smooth_h > 0, uses softplus smoothing.
    """
    x = np.asarray(x, dtype=np.float64)

    if smooth_h is None or smooth_h <= 0:
        below = np.minimum(x - threshold_low, 0.0)
        above = np.maximum(x - threshold_high, 0.0)
    else:
        h = float(smooth_h)

        # stable softplus
        def softplus(z):
            return np.logaddexp(0.0, z)

        below = -h * softplus((threshold_low - x) / h)
        above = h * softplus((x - threshold_high) / h)

    if side == "below":
        return below
    elif side == "above":
        return above
    elif side == "two_sided":
        return below + above
    else:
        raise ValueError("side must be 'below', 'above', or 'two_sided'")


def compute_twcrps_per_horizon_np(
    Q_all,
    y_true,
    u_grid,
    wu=None,
    threshold_low=-10.0,
    threshold_high=10.0,
    side="two_sided",
    smooth_h=1.0,
    crps_convention=True,
):
    """
    Compute twCRPS per horizon from Q-grid.

    This is useful if stored twCRPS_per_horizon is missing.
    For exact agreement with training, prefer stored values or PyTorch loss.
    """
    Q = np.asarray(Q_all, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)

    if wu is None:
        wu = trapezoid_weights_for_u_np(u)
    wu = np.asarray(wu, dtype=np.float64)

    B, H, J = Q.shape

    u3 = u.reshape(1, 1, J)
    wu3 = wu.reshape(1, 1, J)

    cy = chain_threshold_np(
        y[:, :, None],
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        side=side,
        smooth_h=smooth_h,
    )
    cQ = chain_threshold_np(
        Q,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        side=side,
        smooth_h=smooth_h,
    )

    e = cy - cQ
    pinball = np.maximum(u3 * e, (u3 - 1.0) * e)
    loss_bh = np.sum(pinball * wu3, axis=-1)

    if crps_convention:
        loss_bh = 2.0 * loss_bh

    return np.mean(loss_bh, axis=0), float(np.mean(loss_bh))


def get_twcrps_per_horizon(A, compute_if_missing=True, twcrps_kwargs=None):
    """
    Return twCRPS per horizon, using stored value if available.
    """
    if "twcrps_per_horizon" in A:
        return np.asarray(A["twcrps_per_horizon"], dtype=np.float64)

    if "test_twcrps_per_horizon" in A:
        return np.asarray(A["test_twcrps_per_horizon"], dtype=np.float64)

    if not compute_if_missing:
        return None

    twcrps_kwargs = twcrps_kwargs or {}
    per_h, overall = compute_twcrps_per_horizon_np(
        A["Q"],
        A["true"],
        A["u_grid"],
        **twcrps_kwargs,
    )
    A["twcrps_per_horizon_computed"] = per_h
    A["twcrps_computed"] = overall
    return per_h

# ============================================================
# Extreme pinball loss comparison
# ============================================================

def pinball_loss_array(y_true, q_pred, tau):
    """
    Elementwise pinball loss.

    y_true, q_pred: same shape
    tau: quantile level
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    q_pred = np.asarray(q_pred, dtype=np.float64)
    err = y_true - q_pred
    return np.maximum(tau * err, (tau - 1.0) * err)


def compute_extreme_pinball_per_horizon(
    A,
    alphas=(0.05, 0.01, 0.005, 0.001),
):
    """
    Compute lower and upper extreme pinball losses per horizon.

    Lower:
      tau = alpha

    Upper:
      tau = 1 - alpha

    Returns dataframe with:
      alpha, side, tau, horizon, pinball
    """
    y = np.asarray(A["true"], dtype=np.float64)      # (B,H)
    Q = np.asarray(A["Q"], dtype=np.float64)         # (B,H,J)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    B, H = y.shape
    rows = []

    for alpha in alphas:
        alpha = float(alpha)

        # Lower tail quantile Q(alpha)
        q_lower = quantiles_from_u_grid(Q, u, [alpha])[..., 0]  # (B,H)
        loss_lower = pinball_loss_array(y, q_lower, alpha)      # (B,H)

        # Upper tail quantile Q(1-alpha)
        tau_upper = 1.0 - alpha
        q_upper = quantiles_from_u_grid(Q, u, [tau_upper])[..., 0]
        loss_upper = pinball_loss_array(y, q_upper, tau_upper)

        for h_idx in range(H):
            rows.append({
                "alpha": alpha,
                "side": "lower",
                "tau": alpha,
                "horizon": h_idx + 1,
                "pinball": float(np.mean(loss_lower[:, h_idx])),
            })
            rows.append({
                "alpha": alpha,
                "side": "upper",
                "tau": tau_upper,
                "horizon": h_idx + 1,
                "pinball": float(np.mean(loss_upper[:, h_idx])),
            })

    return pd.DataFrame(rows)


def plot_extreme_pinball_comparison(
    runs,
    out_dir,
    alphas=(0.05, 0.01, 0.005, 0.001),
):
    """
    Compare extreme pinball loss per horizon for all models.

    Produces:
      compare_extreme_pinball_lower.png
      compare_extreme_pinball_upper.png
      compare_extreme_pinball_summary.csv
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    all_rows = []

    for name, A in runs.items():
        df = compute_extreme_pinball_per_horizon(A, alphas=alphas)
        df["model"] = name
        all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(out_dir / "compare_extreme_pinball_summary.csv", index=False)

    for side in ["lower", "upper"]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200, sharex=True)
        axes = axes.ravel()

        for ax, alpha in zip(axes, alphas):
            dfa = out[(out["side"] == side) & (np.isclose(out["alpha"], alpha))]

            for name in runs.keys():
                dfn = dfa[dfa["model"] == name].sort_values("horizon")
                ax.plot(
                    dfn["horizon"],
                    dfn["pinball"],
                    marker="o",
                    lw=2,
                    label=name,
                )

            if side == "lower":
                title = f"Lower tail pinball, τ={alpha:g}"
            else:
                title = f"Upper tail pinball, τ={1-alpha:g}"

            ax.set_title(title)
            ax.set_xlabel("Forecast horizon")
            ax.set_ylabel("Mean pinball loss")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=8)

        fig.suptitle(f"Extreme {side}-tail pinball loss by horizon", y=1.02)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"compare_extreme_pinball_{side}.png",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close(fig)

    return out

# ============================================================
# Threshold exceedance probability comparison
# ============================================================

def cdf_at_value_from_quantile_grid_2d(Q_all, u_grid, y_val, eps=1e-12):
    """
    Invert Q(u) to approximate F(y_val).

    Q_all: (B,H,J)
    u_grid: (J,)
    y_val: scalar

    returns:
      F_y: (B,H)
    """
    Q_all = monotone_Q(np.asarray(Q_all, dtype=np.float64))
    u_grid = np.asarray(u_grid, dtype=np.float64)

    B, H, J = Q_all.shape
    Qf = Q_all.reshape(-1, J)

    out = np.empty(Qf.shape[0], dtype=np.float64)

    for i, row in enumerate(Qf):
        idx = np.searchsorted(row, y_val, side="left")

        if idx <= 0:
            out[i] = 0.0
        elif idx >= J:
            out[i] = 1.0
        else:
            ql = row[idx - 1]
            qr = row[idx]
            ul = u_grid[idx - 1]
            ur = u_grid[idx]

            w = (y_val - ql) / max(qr - ql, eps)
            out[i] = ul + w * (ur - ul)

    return np.clip(out.reshape(B, H), 0.0, 1.0)


def compute_abs_threshold_event_prob(A, abs_threshold=15.0):
    """
    Compute predicted probability of event:

      |Y| >= abs_threshold

    from quantile grid.
    """
    Q = np.asarray(A["Q"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    F_lo = cdf_at_value_from_quantile_grid_2d(Q, u, -abs_threshold)
    F_hi = cdf_at_value_from_quantile_grid_2d(Q, u, abs_threshold)

    p_event = F_lo + (1.0 - F_hi)
    return np.clip(p_event, 0.0, 1.0)


def event_probability_metrics_per_horizon(
    p_event,
    y_true,
    abs_threshold=15.0,
    eps=1e-12,
):
    """
    Compute Brier, AUC, event rate, and mean predicted probability per horizon.
    """
    p_event = np.asarray(p_event, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    event = (np.abs(y_true) >= abs_threshold).astype(int)

    B, H = y_true.shape
    rows = []

    for h_idx in range(H):
        p = np.clip(p_event[:, h_idx], eps, 1.0 - eps)
        e = event[:, h_idx]

        brier = np.mean((p - e) ** 2)
        logloss = -np.mean(e * np.log(p) + (1 - e) * np.log(1 - p))

        if roc_auc_score is not None and len(np.unique(e)) == 2:
            auc = float(roc_auc_score(e, p))
        else:
            auc = np.nan

        rows.append({
            "horizon": h_idx + 1,
            "n": int(len(e)),
            "event_rate": float(np.mean(e)),
            "mean_pred_prob": float(np.mean(p)),
            "brier": float(brier),
            "logloss": float(logloss),
            "auc": auc,
            "n_events": int(np.sum(e)),
        })

    return pd.DataFrame(rows)


def _get_time_axis_for_horizon(A, h_ahead):
    """
    Use stored datetime grid if available, otherwise sample index.
    """
    h_idx = horizon_to_index(h_ahead)

    if "ds" in A:
        ds = np.asarray(A["ds"])
        if ds.ndim == 2 and h_idx < ds.shape[1]:
            return pd.to_datetime(ds[:, h_idx])

    if "meta" in A and isinstance(A["meta"], pd.DataFrame) and "cutoff" in A["meta"]:
        return pd.to_datetime(A["meta"]["cutoff"])

    return np.arange(A["true"].shape[0])


def plot_abs_threshold_event_probability_comparison(
    runs,
    out_dir,
    abs_threshold=15.0,
    horizons_ahead=(1, 12, 24),
    max_points=3000,
):
    """
    Plot timeline of predicted P(|Y| >= threshold) and realized events.

    Produces:
      compare_abs_event_probability_h1.png
      compare_abs_event_probability_h12.png
      compare_abs_event_probability_h24.png
      compare_abs_event_probability_metrics.csv
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    metric_rows = []
    p_cache = {}

    # Compute probabilities and metrics
    for name, A in runs.items():
        p_event = compute_abs_threshold_event_prob(A, abs_threshold=abs_threshold)
        p_cache[name] = p_event

        dfm = event_probability_metrics_per_horizon(
            p_event=p_event,
            y_true=A["true"],
            abs_threshold=abs_threshold,
        )
        dfm["model"] = name
        metric_rows.append(dfm)

    metrics_df = pd.concat(metric_rows, ignore_index=True)
    metrics_df.to_csv(
        out_dir / f"compare_abs{abs_threshold:g}_event_probability_metrics.csv",
        index=False,
    )

    # Timeline plots
    for h_ahead in horizons_ahead:
        h_idx = horizon_to_index(h_ahead)

        fig, ax = plt.subplots(figsize=(13, 4.5), dpi=200)

        # Use first model for realized event markers
        first_name = next(iter(runs.keys()))
        A0 = runs[first_name]

        y_h = A0["true"][:, h_idx]
        event_h = np.abs(y_h) >= abs_threshold
        x = _get_time_axis_for_horizon(A0, h_ahead)

        # Downsample if needed
        n = len(y_h)
        if n > max_points:
            idx = np.linspace(0, n - 1, max_points).astype(int)
        else:
            idx = np.arange(n)

        x_plot = np.asarray(x)[idx]
        event_plot = event_h[idx]

        for name, A in runs.items():
            p_h = p_cache[name][:, h_idx]
            p_plot = p_h[idx]

            row = metrics_df[
                (metrics_df["model"] == name)
                & (metrics_df["horizon"] == h_ahead)
            ].iloc[0]

            label = (
                f"{name} "
                f"Brier={row['brier']:.3g}, "
                f"AUC={row['auc']:.3g}"
            )

            ax.plot(
                x_plot,
                p_plot,
                lw=1.8,
                label=label,
            )

        # Event markers
        if np.any(event_plot):
            ax.scatter(
                x_plot[event_plot],
                np.ones(np.sum(event_plot)) * 1.02,
                color="black",
                s=18,
                marker="|",
                label="realized |y| event",
                clip_on=False,
                zorder=5,
            )

        ax.set_ylim(-0.02, 1.08)
        ax.set_ylabel(rf"$P(|Y| \geq {abs_threshold:g})$")
        ax.set_xlabel("test sample / time")
        ax.set_title(
            rf"Threshold exceedance probability timeline, "
            rf"$|Y| \geq {abs_threshold:g}$, {h_ahead}h ahead"
        )
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8, ncols=1)

        fig.tight_layout()
        fig.savefig(
            out_dir / f"compare_abs{abs_threshold:g}_event_probability_h{h_ahead}.png",
            transparent=True,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Summary plot: Brier and AUC by horizon
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

    for name in runs.keys():
        dfn = metrics_df[metrics_df["model"] == name].sort_values("horizon")

        axes[0].plot(
            dfn["horizon"],
            dfn["brier"],
            marker="o",
            lw=2,
            label=name,
        )

        axes[1].plot(
            dfn["horizon"],
            dfn["auc"],
            marker="o",
            lw=2,
            label=name,
        )

    axes[0].set_title("Brier loss by horizon")
    axes[0].set_xlabel("Forecast horizon")
    axes[0].set_ylabel("Brier loss")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("AUC by horizon")
    axes[1].set_xlabel("Forecast horizon")
    axes[1].set_ylabel("AUC")
    axes[1].grid(alpha=0.3)

    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(
        out_dir / f"compare_abs{abs_threshold:g}_event_brier_auc_by_horizon.png",
        transparent=True,
        bbox_inches="tight",
    )
    plt.close(fig)

    return metrics_df

# ============================================================
# Plotting functions
# ============================================================

def plot_twcrps_comparison(
    runs,
    out_dir,
    title="twCRPS by forecast horizon",
    compute_if_missing=True,
    twcrps_kwargs=None,
):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

    rows = []

    for name, A in runs.items():
        per_h = get_twcrps_per_horizon(
            A,
            compute_if_missing=compute_if_missing,
            twcrps_kwargs=twcrps_kwargs,
        )

        if per_h is None:
            print(f"Skipping {name}: no twCRPS_per_horizon.")
            continue

        H = len(per_h)
        horizons = np.arange(1, H + 1)

        ax.plot(horizons, per_h, marker="o", lw=2, label=name)

        rows.append(pd.DataFrame({
            "model": name,
            "horizon": horizons,
            "twcrps": per_h,
        }))

    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("twCRPS")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "compare_twcrps_per_horizon.png", transparent=True, bbox_inches="tight")
    plt.close(fig)

    if rows:
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(out_dir / "compare_twcrps_per_horizon.csv", index=False)
        return df

    return None


def plot_pit_hist_comparison(
    runs,
    out_dir,
    horizons_ahead=(1, 12, 24),
    bins=20,
    prefer_stored=True,
):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    for h_ahead in horizons_ahead:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)

        stats_rows = []

        for name, A in runs.items():
            pits = get_pits(A, h_ahead, prefer_stored=prefer_stored)
            stats = pit_stats_from_values(pits)
            stats_rows.append({"model": name, "horizon": h_ahead, **stats})

            ax.hist(
                pits,
                bins=bins,
                range=(0, 1),
                density=True,
                histtype="step",
                lw=2,
                label=f"{name}  KS={stats['ks_stat']:.3f}",
            )

        ax.axhline(1.0, color="black", ls="--", lw=1.2, label="Uniform")
        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT = F(y)")
        ax.set_ylabel("Density")
        ax.set_title(f"PIT histogram comparison, {h_ahead}h ahead")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_pit_hist_h{h_ahead}.png", transparent=True, bbox_inches="tight")
        plt.close(fig)

        pd.DataFrame(stats_rows).to_csv(out_dir / f"compare_pit_stats_h{h_ahead}.csv", index=False)


def plot_pit_ecdf_comparison(
    runs,
    out_dir,
    horizons_ahead=(1, 12, 24),
    prefer_stored=True,
):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    for h_ahead in horizons_ahead:
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=200)

        uu = np.linspace(0, 1, 500)
        ax.plot(uu, uu, "k--", lw=1.5, label="Uniform")

        for name, A in runs.items():
            pits = get_pits(A, h_ahead, prefer_stored=prefer_stored)
            pits = np.clip(pits[np.isfinite(pits)], 0.0, 1.0)
            pits_sorted = np.sort(pits)
            ecdf = np.arange(1, len(pits_sorted) + 1) / max(len(pits_sorted), 1)

            stats = pit_stats_from_values(pits)

            ax.step(
                pits_sorted,
                ecdf,
                where="post",
                lw=2,
                label=f"{name}  KS={stats['ks_stat']:.3f}",
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("u")
        ax.set_ylabel("ECDF of PIT")
        ax.set_title(f"PIT ECDF comparison, {h_ahead}h ahead")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_pit_ecdf_h{h_ahead}.png", transparent=True, bbox_inches="tight")
        plt.close(fig)


def plot_tail_exceedance_comparison(
    runs,
    out_dir,
    horizons_ahead=(1, 12, 24),
    alphas=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    prefer_stored=True,
    log_x=True,
):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    alphas = np.asarray(alphas, dtype=np.float64)
    rows = []

    for h_ahead in horizons_ahead:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

        for name, A in runs.items():
            s = get_tail_summary(A, h_ahead, alphas, prefer_stored=prefer_stored)

            a = np.asarray(s["alphas"], dtype=np.float64)
            lower = np.asarray(s["lower_empirical"], dtype=np.float64)
            upper = np.asarray(s["upper_empirical"], dtype=np.float64)

            axes[0].plot(a, lower, marker="o", lw=2, label=name)
            axes[1].plot(a, upper, marker="o", lw=2, label=name)

            for ak, lo, up in zip(a, lower, upper):
                rows.append({
                    "model": name,
                    "horizon": h_ahead,
                    "alpha": ak,
                    "lower_empirical": lo,
                    "upper_empirical": up,
                    "lower_ratio": lo / ak if ak > 0 else np.nan,
                    "upper_ratio": up / ak if ak > 0 else np.nan,
                })

        for ax, side_name in zip(axes, ["Lower tail", "Upper tail"]):
            ax.plot(alphas, alphas, "k--", lw=1.5, label="Ideal")
            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("Nominal tail probability α")
            ax.set_ylabel("Empirical exceedance probability")
            ax.set_title(f"{side_name} calibration, {h_ahead}h ahead")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"compare_tail_exceedance_h{h_ahead}.png", transparent=True, bbox_inches="tight")
        plt.close(fig)

        # Ratio plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

        for name, A in runs.items():
            s = get_tail_summary(A, h_ahead, alphas, prefer_stored=prefer_stored)

            a = np.asarray(s["alphas"], dtype=np.float64)
            lower = np.asarray(s["lower_empirical"], dtype=np.float64)
            upper = np.asarray(s["upper_empirical"], dtype=np.float64)

            axes[0].plot(a, lower / a, marker="o", lw=2, label=name)
            axes[1].plot(a, upper / a, marker="o", lw=2, label=name)

        for ax, side_name in zip(axes, ["Lower tail", "Upper tail"]):
            ax.axhline(1.0, color="k", ls="--", lw=1.5, label="Ideal")
            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("Nominal tail probability α")
            ax.set_ylabel("Empirical / nominal")
            ax.set_title(f"{side_name} exceedance ratio, {h_ahead}h ahead")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"compare_tail_exceedance_ratio_h{h_ahead}.png", transparent=True, bbox_inches="tight")
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "compare_tail_exceedance_summary.csv", index=False)
    return df


# ============================================================
# VaR / ES comparison
# ============================================================

def var_es_summary_from_grid(A, h_ahead, alphas, side="lower"):
    """
    Compute VaR/ES diagnostics from Q-grid for one model/horizon.
    """
    h_idx = horizon_to_index(h_ahead)
    y = A["true"][:, h_idx]
    Q = A["Q"]
    u = A["u_grid"]

    alphas = np.asarray(alphas, dtype=np.float64)

    if side == "lower":
        var = quantiles_from_u_grid(Q[:, h_idx:h_idx + 1, :], u, alphas)[:, 0, :]
        es = expected_shortfall_from_quantile_grid(Q[:, h_idx:h_idx + 1, :], u, alphas, side="lower")[:, 0, :]
        hit = y[:, None] <= var
    elif side == "upper":
        var = quantiles_from_u_grid(Q[:, h_idx:h_idx + 1, :], u, 1.0 - alphas)[:, 0, :]
        es = expected_shortfall_from_quantile_grid(Q[:, h_idx:h_idx + 1, :], u, alphas, side="upper")[:, 0, :]
        hit = y[:, None] >= var
    else:
        raise ValueError("side must be lower or upper")

    coverage = np.mean(hit, axis=0)
    n_hits = np.sum(hit, axis=0)

    realized_tail_mean = np.full(len(alphas), np.nan)
    predicted_es_on_hits = np.full(len(alphas), np.nan)
    predicted_es_all = np.mean(es, axis=0)

    for k in range(len(alphas)):
        if n_hits[k] > 0:
            realized_tail_mean[k] = np.mean(y[hit[:, k]])
            predicted_es_on_hits[k] = np.mean(es[hit[:, k], k])

    return {
        "alphas": alphas,
        "coverage": coverage,
        "n_hits": n_hits,
        "realized_tail_mean": realized_tail_mean,
        "predicted_es_on_hits": predicted_es_on_hits,
        "predicted_es_all": predicted_es_all,
        "es_bias": predicted_es_on_hits - realized_tail_mean,
    }


def var_es_summary_from_stored_h0(A, side="lower"):
    """
    Use stored exact VaR/ES at h0 if available.
    This is useful for SplicedGPD exact diagnostics.
    """
    if "es_alphas_h0" not in A:
        return None

    y = A["true"][:, 0]
    alphas = np.asarray(A["es_alphas_h0"], dtype=np.float64)

    if side == "lower":
        if "var_lower_h0" not in A or "es_lower_h0" not in A:
            return None
        var = np.asarray(A["var_lower_h0"], dtype=np.float64)
        es = np.asarray(A["es_lower_h0"], dtype=np.float64)
        hit = y[:, None] <= var
    else:
        if "var_upper_h0" not in A or "es_upper_h0" not in A:
            return None
        var = np.asarray(A["var_upper_h0"], dtype=np.float64)
        es = np.asarray(A["es_upper_h0"], dtype=np.float64)
        hit = y[:, None] >= var

    coverage = np.mean(hit, axis=0)
    n_hits = np.sum(hit, axis=0)

    realized_tail_mean = np.full(len(alphas), np.nan)
    predicted_es_on_hits = np.full(len(alphas), np.nan)
    predicted_es_all = np.mean(es, axis=0)

    for k in range(len(alphas)):
        if n_hits[k] > 0:
            realized_tail_mean[k] = np.mean(y[hit[:, k]])
            predicted_es_on_hits[k] = np.mean(es[hit[:, k], k])

    return {
        "alphas": alphas,
        "coverage": coverage,
        "n_hits": n_hits,
        "realized_tail_mean": realized_tail_mean,
        "predicted_es_on_hits": predicted_es_on_hits,
        "predicted_es_all": predicted_es_all,
        "es_bias": predicted_es_on_hits - realized_tail_mean,
    }


def plot_var_es_comparison(
    runs,
    out_dir,
    horizons_ahead=(1, 12, 24),
    alphas=(0.01, 0.02, 0.05),
    prefer_stored_spliced_h0=True,
    log_x=True,
):
    """
    Compare VaR coverage and ES bias.

    For h=1, if a model has stored exact h0 VaR/ES, use it.
    Otherwise compute from Q-grid.
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    all_rows = []

    for h_ahead in horizons_ahead:
        for side in ["lower", "upper"]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

            for name, A in runs.items():
                s = None

                if prefer_stored_spliced_h0 and h_ahead == 1:
                    s = var_es_summary_from_stored_h0(A, side=side)

                if s is None:
                    s = var_es_summary_from_grid(A, h_ahead, alphas, side=side)

                a = np.asarray(s["alphas"], dtype=np.float64)
                coverage = np.asarray(s["coverage"], dtype=np.float64)
                es_bias = np.asarray(s["es_bias"], dtype=np.float64)

                axes[0].plot(a, coverage, marker="o", lw=2, label=name)
                axes[1].plot(a, es_bias, marker="o", lw=2, label=name)

                for k in range(len(a)):
                    all_rows.append({
                        "model": name,
                        "horizon": h_ahead,
                        "side": side,
                        "alpha": a[k],
                        "coverage": coverage[k],
                        "coverage_ratio": coverage[k] / a[k],
                        "n_hits": int(s["n_hits"][k]),
                        "realized_tail_mean": s["realized_tail_mean"][k],
                        "predicted_es_on_hits": s["predicted_es_on_hits"][k],
                        "predicted_es_all": s["predicted_es_all"][k],
                        "es_bias": es_bias[k],
                    })

            # Coverage panel
            axes[0].plot(alphas, alphas, "k--", lw=1.5, label="Ideal")
            axes[0].set_xlabel("Tail probability α")
            axes[0].set_ylabel("Empirical VaR exceedance")
            axes[0].set_title(f"VaR coverage, {side}, {h_ahead}h ahead")
            axes[0].grid(alpha=0.3)

            # ES bias panel
            axes[1].axhline(0.0, color="k", ls="--", lw=1.5, label="Zero bias")
            axes[1].set_xlabel("Tail probability α")
            axes[1].set_ylabel("Predicted ES on hits - realized tail mean")
            axes[1].set_title(f"ES bias, {side}, {h_ahead}h ahead")
            axes[1].grid(alpha=0.3)

            if log_x:
                axes[0].set_xscale("log")
                axes[1].set_xscale("log")

            axes[0].legend(frameon=False, fontsize=8)
            axes[1].legend(frameon=False, fontsize=8)

            fig.tight_layout()
            fig.savefig(out_dir / f"compare_var_es_{side}_h{h_ahead}.png", transparent=True, bbox_inches="tight")
            plt.close(fig)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "compare_var_es_summary.csv", index=False)
    return df


# ============================================================
# Spliced GPD comparison plots
# ============================================================

def plot_spliced_gpd_tail_parameter_comparison(runs, out_dir):
    """
    Compare stored SplicedGPD tail parameters if A["gpd_tail_info"] is present.

    Expected keys inside gpd_tail_info may include:
      xiL, xiU, betaL, betaU, xL, xU
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    keys = ["xiL", "xiU", "betaL", "betaU", "xL", "xU"]
    available_any = False

    for key in keys:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
        plotted = False

        for name, A in runs.items():
            info = A.get("gpd_tail_info", None)
            if info is None or key not in info:
                continue

            vals = np.asarray(info[key], dtype=np.float64).reshape(-1)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            ax.hist(
                vals,
                bins=50,
                density=True,
                histtype="step",
                lw=2,
                label=f"{name} mean={np.mean(vals):.3g}",
            )
            plotted = True
            available_any = True

        if plotted:
            ax.set_title(f"SplicedGPD tail parameter comparison: {key}")
            ax.set_xlabel(key)
            ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"compare_spliced_gpd_{key}.png", transparent=True, bbox_inches="tight")
        plt.close(fig)

    if not available_any:
        print("No gpd_tail_info found in provided runs. Skipping SplicedGPD parameter plots.")




# ============================================================
# Master comparison function
# ============================================================

def compare_probabilistic_forecasts(
    model_paths,
    out_dir="./comparison_plots",
    names=None,
    horizons_ahead=(1, 12, 24),
    tail_alphas=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    es_alphas=(0.01, 0.02, 0.05),
    prefer_stored_pits=True,
    prefer_stored_tail=True,
    include_spliced_gpd=True,
    compute_twcrps_if_missing=True,
    twcrps_kwargs=None,
    include_extreme_pinball=True,
    extreme_pinball_alphas=(0.05, 0.01, 0.005, 0.001),
    include_event_probability=True,
    event_abs_threshold=15.0,
):
    """
    Main entry point.

    Parameters
    ----------
    model_paths:
        dict[name -> path] or list[path]
    out_dir:
        directory for comparison plots/tables
    horizons_ahead:
        human-readable horizons, e.g. (1, 12, 24)
    prefer_stored_pits:
        If True, use A["pit_h1"], etc. when present.
        Useful for SplicedGPD exact tail-aware PIT.
    prefer_stored_tail:
        If True, use stored tail exceedance summaries when present.
    include_spliced_gpd:
        If True, plot gpd_tail_info distributions when available.
    twcrps_kwargs:
        Used only if twCRPS is missing and must be computed from Q-grid.
        Example:
          dict(threshold_low=-10, threshold_high=10, side="two_sided", smooth_h=1.0)
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    runs = load_forecast_pickles(model_paths, names=names)
    check_common_grid_and_shape(runs)

    # Save run overview
    overview_rows = []
    for name, A in runs.items():
        overview_rows.append({
            "model": name,
            "B": A["true"].shape[0],
            "H": A["true"].shape[1],
            "J": A["Q"].shape[2],
            "u_min": float(A["u_grid"][0]),
            "u_max": float(A["u_grid"][-1]),
            "has_twcrps_per_horizon": "twcrps_per_horizon" in A or "test_twcrps_per_horizon" in A,
            "has_gpd_tail_info": "gpd_tail_info" in A,
        })
    pd.DataFrame(overview_rows).to_csv(out_dir / "model_overview.csv", index=False)

    # 1) twCRPS
    tw_df = plot_twcrps_comparison(
        runs,
        out_dir=out_dir,
        compute_if_missing=compute_twcrps_if_missing,
        twcrps_kwargs=twcrps_kwargs,
    )

    # 2) PIT histograms and ECDFs
    plot_pit_hist_comparison(
        runs,
        out_dir=out_dir,
        horizons_ahead=horizons_ahead,
        prefer_stored=prefer_stored_pits,
    )

    plot_pit_ecdf_comparison(
        runs,
        out_dir=out_dir,
        horizons_ahead=horizons_ahead,
        prefer_stored=prefer_stored_pits,
    )

    # 3) Tail exceedance calibration and ratios
    tail_df = plot_tail_exceedance_comparison(
        runs,
        out_dir=out_dir,
        horizons_ahead=horizons_ahead,
        alphas=tail_alphas,
        prefer_stored=prefer_stored_tail,
    )

    # 4) VaR / ES
    es_df = plot_var_es_comparison(
        runs,
        out_dir=out_dir,
        horizons_ahead=horizons_ahead,
        alphas=es_alphas,
        prefer_stored_spliced_h0=True,
    )
    # 5) Extreme pinball losses
    pinball_df = None
    if include_extreme_pinball:
        pinball_df = plot_extreme_pinball_comparison(
            runs,
            out_dir=out_dir,
            alphas=extreme_pinball_alphas,
        )

    # 6) Threshold event probability timeline and Brier/AUC
    event_df = None
    if include_event_probability:
        event_df = plot_abs_threshold_event_probability_comparison(
            runs,
            out_dir=out_dir,
            abs_threshold=event_abs_threshold,
            horizons_ahead=horizons_ahead,
        )

    # 7) Optional SplicedGPD parameter diagnostics
    if include_spliced_gpd:
        plot_spliced_gpd_tail_parameter_comparison(runs, out_dir=out_dir)

    print(f"\nComparison outputs saved to: {out_dir.resolve()}")

    return {
        "runs": runs,
        "twcrps_df": tw_df,
        "tail_df": tail_df,
        "es_df": es_df,
        "pinball_df": pinball_df,
        "event_df": event_df,
        "out_dir": out_dir,
    }

if __name__ == "__main__":
    pass