# ============================================================
# Section 5.2.5:
# Visual case studies around depeg episodes
# ============================================================
#
# Creates fan-chart case studies around realized depeg/stress events.
#
# Figure:
#   Panel A: realized peg deviation + median forecast + predictive intervals
#   Panel B: predicted probability of |Delta p| > threshold
#   Panel C: predicted downside/upside expected shortfall conditional on depeg
#
# Main outputs:
#   1. section_525_case_study_episode_*.png
#   2. section_525_case_study_episode_*.csv
#   3. section_525_selected_episodes.csv
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from benchmark_utils import (
    load_forecast_pickles,
    check_common_grid_and_shape,
    quantiles_from_u_grid,
    compute_abs_threshold_event_prob,
    _resolve_horizons,
    model_paths,
)


# ------------------------------------------------------------
# 1. Optional timestamp helper
# ------------------------------------------------------------

def get_time_index_for_horizon(A, horizon_idx=0):
    """
    Try to extract timestamps from the forecast dictionary.

    Supports common keys:
        timestamp, timestamps, time, times, test_times, ds, index

    If no timestamp field is found, returns integer index 0,...,B-1.

    Parameters
    ----------
    A : dict
        Forecast dictionary.
    horizon_idx : int
        Zero-based horizon index.

    Returns
    -------
    index : np.ndarray
        Length-B array suitable for plotting.
    """
    y = np.asarray(A["true"])
    B = y.shape[0]

    possible_keys = [
        "timestamp",
        "timestamps",
        "time",
        "times",
        "test_times",
        "ds",
        "index",
        "datetime",
        "datetimes",
    ]

    for key in possible_keys:
        if key not in A:
            continue

        t = np.asarray(A[key])

        # Case 1: one timestamp per forecast origin
        if t.ndim == 1 and len(t) == B:
            try:
                return pd.to_datetime(t)
            except Exception:
                return t

        # Case 2: timestamp per origin and horizon
        if t.ndim == 2 and t.shape[0] == B:
            h = min(horizon_idx, t.shape[1] - 1)
            try:
                return pd.to_datetime(t[:, h])
            except Exception:
                return t[:, h]

    return np.arange(B)


# ------------------------------------------------------------
# 2. CDF and expected shortfall from quantile grid
# ------------------------------------------------------------

def _make_strictly_increasing(x, eps=1e-10):
    """
    Enforce strict monotonicity for numerical inversion.
    """
    x = np.asarray(x, dtype=np.float64).copy()

    for j in range(1, len(x)):
        if x[j] <= x[j - 1]:
            x[j] = x[j - 1] + eps

    return x


def cdf_from_single_quantile_function(q, u, x):
    """
    Approximate F(x) by inverting a single quantile function.
    """
    q = np.asarray(q, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    mask = np.isfinite(q) & np.isfinite(u)

    if np.sum(mask) < 2 or not np.isfinite(x):
        return np.nan

    q = q[mask]
    u = u[mask]

    order = np.argsort(u)
    u = u[order]
    q = q[order]

    q = np.maximum.accumulate(q)
    q = _make_strictly_increasing(q)

    return float(np.interp(x, q, u, left=0.0, right=1.0))


def integrate_quantile_function(q, u, a, b, n_grid=200):
    """
    Numerically integrate Q(v) over v in [a,b].

    Returns:
        integral_a^b Q(v) dv
    """
    q = np.asarray(q, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return np.nan

    mask = np.isfinite(q) & np.isfinite(u)

    if np.sum(mask) < 2:
        return np.nan

    q = q[mask]
    u = u[mask]

    order = np.argsort(u)
    u = u[order]
    q = q[order]

    q = np.maximum.accumulate(q)

    grid = np.linspace(a, b, n_grid)
    q_grid = np.interp(grid, u, q, left=q[0], right=q[-1])

    return float(np.trapz(q_grid, grid))


def directional_expected_shortfall_from_quantiles(
    Q_all,
    u_grid,
    threshold=15.0,
    n_grid=200,
):
    """
    Compute directional expected shortfall conditional on depeg:

        Downside ES_t = E[Y | Y <= -threshold]
        Upside ES_t   = E[Y | Y >=  threshold]

    using the forecast quantile function.

    Parameters
    ----------
    Q_all : np.ndarray
        Forecast quantiles, shape (B,H,J).
    u_grid : np.ndarray
        Quantile levels, shape (J,).
    threshold : float
        Absolute threshold in bps.

    Returns
    -------
    es_down : np.ndarray
        Downside conditional expected shortfall, shape (B,H).
    es_up : np.ndarray
        Upside conditional expected shortfall, shape (B,H).
    p_down : np.ndarray
        Forecast probability Y <= -threshold, shape (B,H).
    p_up : np.ndarray
        Forecast probability Y >= threshold, shape (B,H).
    """
    Q = np.asarray(Q_all, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)

    B, H, J = Q.shape

    es_down = np.full((B, H), np.nan, dtype=np.float64)
    es_up = np.full((B, H), np.nan, dtype=np.float64)
    p_down = np.full((B, H), np.nan, dtype=np.float64)
    p_up = np.full((B, H), np.nan, dtype=np.float64)

    thr = float(threshold)

    for b in range(B):
        for h in range(H):
            q = Q[b, h, :]

            F_low = cdf_from_single_quantile_function(q, u, -thr)
            F_high = cdf_from_single_quantile_function(q, u, thr)

            if not np.isfinite(F_low) or not np.isfinite(F_high):
                continue

            pdn = np.clip(F_low, 0.0, 1.0)
            pup = np.clip(1.0 - F_high, 0.0, 1.0)

            p_down[b, h] = pdn
            p_up[b, h] = pup

            # E[Y | Y <= -thr] = 1/p * int_0^p Q(v) dv
            if pdn > 1e-12:
                integ = integrate_quantile_function(
                    q,
                    u,
                    0.0,
                    pdn,
                    n_grid=n_grid,
                )
                if np.isfinite(integ):
                    es_down[b, h] = integ / pdn

            # E[Y | Y >= thr] = 1/p * int_F(thr)^1 Q(v) dv
            if pup > 1e-12:
                integ = integrate_quantile_function(
                    q,
                    u,
                    F_high,
                    1.0,
                    n_grid=n_grid,
                )
                if np.isfinite(integ):
                    es_up[b, h] = integ / pup

    return es_down, es_up, p_down, p_up


# ------------------------------------------------------------
# 3. Episode selection
# ------------------------------------------------------------

def _select_non_overlapping_indices(candidate_idx, scores, n_select=2, min_gap=48):
    """
    Select high-scoring candidate indices with minimum separation.
    """
    candidate_idx = np.asarray(candidate_idx, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)

    if len(candidate_idx) == 0:
        return []

    order = np.argsort(-scores)
    selected = []

    for k in order:
        idx = int(candidate_idx[k])

        if all(abs(idx - s) >= min_gap for s in selected):
            selected.append(idx)

        if len(selected) >= n_select:
            break

    return selected


def select_depeg_case_studies(
    A,
    horizons_ahead=24,
    threshold=15.0,
    n_lower=1,
    n_upper=1,
    n_abs_fallback=1,
    min_gap=48,
):
    """
    Automatically select depeg episodes.

    Attempts to select:
        - n_lower downside episodes where y <= -threshold
        - n_upper upside episodes where y >= threshold

    If few threshold crossings exist, falls back to largest |y| episodes.

    Returns
    -------
    episodes : pd.DataFrame
        Selected episode centers.
    """
    y = np.asarray(A["true"], dtype=np.float64)

    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    if len(h_idx) != 1:
        raise ValueError(
            "For visual case studies, use a single horizon, e.g. horizons_ahead=24."
        )

    h = int(h_idx[0])
    y_h = y[:, h]

    lower_idx = np.where(y_h <= -threshold)[0]
    upper_idx = np.where(y_h >= threshold)[0]

    lower_scores = np.abs(y_h[lower_idx])
    upper_scores = np.abs(y_h[upper_idx])

    selected_lower = _select_non_overlapping_indices(
        lower_idx,
        lower_scores,
        n_select=n_lower,
        min_gap=min_gap,
    )

    selected_upper = _select_non_overlapping_indices(
        upper_idx,
        upper_scores,
        n_select=n_upper,
        min_gap=min_gap,
    )

    selected = []

    for idx in selected_lower:
        selected.append({
            "center_idx": idx,
            "direction": "downside",
            "realized": float(y_h[idx]),
            "horizon": horizon_label,
        })

    for idx in selected_upper:
        selected.append({
            "center_idx": idx,
            "direction": "upside",
            "realized": float(y_h[idx]),
            "horizon": horizon_label,
        })

    # Fallback: if no downside/upside events are found, choose largest absolute events.
    if len(selected) == 0 and n_abs_fallback > 0:
        all_idx = np.arange(len(y_h))
        valid = np.isfinite(y_h)
        all_idx = all_idx[valid]
        scores = np.abs(y_h[valid])

        fallback = _select_non_overlapping_indices(
            all_idx,
            scores,
            n_select=n_abs_fallback,
            min_gap=min_gap,
        )

        for idx in fallback:
            selected.append({
                "center_idx": idx,
                "direction": "largest_abs",
                "realized": float(y_h[idx]),
                "horizon": horizon_label,
            })

    return pd.DataFrame(selected)


# ------------------------------------------------------------
# 4. Build plotting data for one horizon
# ------------------------------------------------------------

def build_case_study_frame(
    A,
    horizons_ahead=24,
    threshold=15.0,
):
    """
    Build a long time-indexed DataFrame for plotting fan charts
    at a selected forecast horizon.
    """
    y = np.asarray(A["true"], dtype=np.float64)
    Q = np.asarray(A["Q"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    if len(h_idx) != 1:
        raise ValueError(
            "For visual case studies, use a single horizon, e.g. horizons_ahead=24."
        )

    h = int(h_idx[0])

    # Required quantiles for median and central intervals:
    # 50%: 25-75
    # 80%: 10-90
    # 95%: 2.5-97.5
    # 99%: 0.5-99.5
    taus = [
        0.005,
        0.025,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.975,
        0.995,
    ]

    qhat = quantiles_from_u_grid(Q, u, taus)

    p_depeg = compute_abs_threshold_event_prob(
        A,
        abs_threshold=threshold,
    )

    es_down, es_up, p_down, p_up = directional_expected_shortfall_from_quantiles(
        Q,
        u,
        threshold=threshold,
        n_grid=200,
    )

    time_index = get_time_index_for_horizon(A, horizon_idx=h)

    df = pd.DataFrame({
        "t": time_index,
        "idx": np.arange(y.shape[0]),
        "realized": y[:, h],
        "q005": qhat[:, h, 0],
        "q025": qhat[:, h, 1],
        "q10": qhat[:, h, 2],
        "q25": qhat[:, h, 3],
        "q50": qhat[:, h, 4],
        "q75": qhat[:, h, 5],
        "q90": qhat[:, h, 6],
        "q975": qhat[:, h, 7],
        "q995": qhat[:, h, 8],
        "p_depeg": p_depeg[:, h],
        "p_down": p_down[:, h],
        "p_up": p_up[:, h],
        "es_down": es_down[:, h],
        "es_up": es_up[:, h],
    })

    return df, horizon_label


# ------------------------------------------------------------
# 5. Plot one episode
# ------------------------------------------------------------

def plot_depeg_case_study(
    df,
    center_idx,
    out_path=None,
    threshold=15.0,
    window_pre=72,
    window_post=72,
    title=None,
):
    """
    Create the three-panel case-study figure.
    """
    center_idx = int(center_idx)

    lo = max(0, center_idx - window_pre)
    hi = min(len(df), center_idx + window_post + 1)

    d = df.iloc[lo:hi].copy()

    x = d["t"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]},
    )

    ax = axes[0]

    # Plot widest interval first.
    ax.fill_between(
        x,
        d["q005"].astype(float).values,
        d["q995"].astype(float).values,
        alpha=0.15,
        label="99% PI",
    )
    ax.fill_between(
        x,
        d["q025"].astype(float).values,
        d["q975"].astype(float).values,
        alpha=0.20,
        label="95% PI",
    )
    ax.fill_between(
        x,
        d["q10"].astype(float).values,
        d["q90"].astype(float).values,
        alpha=0.25,
        label="80% PI",
    )
    ax.fill_between(
        x,
        d["q25"].astype(float).values,
        d["q75"].astype(float).values,
        alpha=0.35,
        label="50% PI",
    )

    ax.plot(
        x,
        d["realized"].astype(float).values,
        color="black",
        linewidth=1.8,
        label="Realized",
    )

    ax.plot(
        x,
        d["q50"].astype(float).values,
        color="tab:blue",
        linewidth=1.8,
        linestyle="--",
        label="Median forecast",
    )

    ax.axhline(threshold, color="red", linestyle=":", linewidth=1.2)
    ax.axhline(-threshold, color="red", linestyle=":", linewidth=1.2)
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.6)

    ax.axvline(
        df.loc[center_idx, "t"],
        color="purple",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.set_ylabel("Peg deviation / return (bps)")
    ax.set_title("Panel A: realized deviation and predictive intervals")
    ax.legend(loc="best", ncol=3, fontsize=8)
    ax.grid(alpha=0.25)

    # Panel B: depeg probability
    ax = axes[1]

    ax.plot(
        x,
        d["p_depeg"].astype(float).values,
        color="tab:red",
        linewidth=1.8,
        label=rf"$P(|\Delta p|>{threshold:g}\mathrm{{bps}})$",
    )

    ax.plot(
        x,
        d["p_down"].astype(float).values,
        color="tab:orange",
        linewidth=1.2,
        linestyle="--",
        label="Downside probability",
    )

    ax.plot(
        x,
        d["p_up"].astype(float).values,
        color="tab:green",
        linewidth=1.2,
        linestyle="--",
        label="Upside probability",
    )

    ax.axvline(
        df.loc[center_idx, "t"],
        color="purple",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.set_ylabel("Probability")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Panel B: predicted depeg probability")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    # Panel C: directional expected shortfall
    ax = axes[2]

    ax.plot(
        x,
        d["es_down"].astype(float).values,
        color="tab:orange",
        linewidth=1.8,
        label=rf"$E[Y \mid Y<-{threshold:g}]$",
    )

    ax.plot(
        x,
        d["es_up"].astype(float).values,
        color="tab:green",
        linewidth=1.8,
        label=rf"$E[Y \mid Y>{threshold:g}]$",
    )

    ax.axhline(threshold, color="red", linestyle=":", linewidth=1.0)
    ax.axhline(-threshold, color="red", linestyle=":", linewidth=1.0)
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.6)

    ax.axvline(
        df.loc[center_idx, "t"],
        color="purple",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.set_ylabel("Conditional ES (bps)")
    ax.set_title("Panel C: predicted downside and upside expected shortfall")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    if title is not None:
        fig.suptitle(title, y=0.995, fontsize=13)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=220, bbox_inches="tight")

    return fig, axes


# ------------------------------------------------------------
# 6. Convenience wrapper
# ------------------------------------------------------------

def make_section_525_case_study_outputs(
    model_paths=None,
    runs=None,
    out_dir="./comparison_nn_vs_arima/section_525_case_studies",
    model="SAINT",
    horizons_ahead=24,
    threshold=15.0,
    n_lower=1,
    n_upper=1,
    min_gap=72,
    window_pre=72,
    window_post=72,
    manual_centers=None,
):
    """
    Create 2-3 visual case studies around stress/depeg episodes.

    Parameters
    ----------
    manual_centers : list[int] or None
        Optional manually selected center indices.
        If supplied, automatic episode selection is skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if runs is None:
        if model_paths is None:
            raise ValueError("Pass either runs or model_paths.")
        runs = load_forecast_pickles(model_paths)
        check_common_grid_and_shape(runs)

    if model not in runs:
        raise ValueError(f"model={model} not found in runs.")

    A = runs[model]

    df_plot, horizon_label = build_case_study_frame(
        A,
        horizons_ahead=horizons_ahead,
        threshold=threshold,
    )

    model_tag = model.replace(" ", "_").replace("/", "_")
    horizon_tag = horizon_label.replace(",", "_").replace("-", "_")

    if manual_centers is not None:
        episodes = pd.DataFrame([
            {
                "center_idx": int(c),
                "direction": "manual",
                "realized": float(df_plot.loc[int(c), "realized"]),
                "horizon": horizon_label,
            }
            for c in manual_centers
        ])
    else:
        episodes = select_depeg_case_studies(
            A,
            horizons_ahead=horizons_ahead,
            threshold=threshold,
            n_lower=n_lower,
            n_upper=n_upper,
            n_abs_fallback=2,
            min_gap=min_gap,
        )

    episodes.to_csv(
        out_dir / f"section_525_selected_episodes_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    print("\n==============================================")
    print(f"Section 5.2.5: Selected case studies for {model}")
    print(f"Horizon: {horizon_label}")
    print("==============================================")
    print(episodes.to_string(index=False))

    for k, row in episodes.iterrows():
        center_idx = int(row["center_idx"])
        direction = str(row["direction"])

        lo = max(0, center_idx - window_pre)
        hi = min(len(df_plot), center_idx + window_post + 1)

        episode_df = df_plot.iloc[lo:hi].copy()

        csv_path = (
            out_dir
            / f"section_525_case_study_episode_{k+1}_{direction}_{model_tag}_{horizon_tag}.csv"
        )

        png_path = (
            out_dir
            / f"section_525_case_study_episode_{k+1}_{direction}_{model_tag}_{horizon_tag}.png"
        )

        episode_df.to_csv(csv_path, index=False)

        realized_center = float(df_plot.loc[center_idx, "realized"])
        p_center = float(df_plot.loc[center_idx, "p_depeg"])

        title = (
            f"Distributional forecast fan chart around depeg episode "
            f"({model}, {horizon_label}; center y={realized_center:.2f} bps, "
            f"p={p_center:.3f})"
        )

        plot_depeg_case_study(
            df_plot,
            center_idx=center_idx,
            out_path=png_path,
            threshold=threshold,
            window_pre=window_pre,
            window_post=window_post,
            title=title,
        )

        print(f"Saved episode {k+1}: {png_path}")

    return {
        "runs": runs,
        "plot_frame": df_plot,
        "episodes": episodes,
        "out_dir": out_dir,
    }


# ============================================================
# Example usage
# ============================================================

section_525_result = make_section_525_case_study_outputs(
    model_paths=model_paths,
    out_dir="./comparison_nn_vs_arima/section_525_case_studies",
    model="SAINT",
    horizons_ahead=24,
    threshold=15.0,
    n_lower=1,
    n_upper=1,
    min_gap=72,
    window_pre=72,
    window_post=72,
    manual_centers=None,  # or e.g. [120, 740]
)
