# ============================================================
# Section 5.2.9:
# Statistical comparison with benchmarks
# ============================================================
#
# Formal pairwise forecast-comparison tests against proposed model.
#
# Tests:
#   - Diebold-Mariano / HAC t-test on loss differentials
#   - Moving-block bootstrap confidence intervals for mean loss differential
#   - Optional simple Model Confidence Set-style elimination summary
#
# Main table:
#   "Pairwise forecast comparison against proposed model"
#
# Main outputs:
#   1. section_529_pairwise_dm_bootstrap_<horizon>.csv
#   2. section_529_pairwise_dm_bootstrap_<horizon>.tex
#   3. section_529_pairwise_dm_bootstrap_<horizon>.md
#   4. section_529_loss_series_<metric>_<horizon>.csv
#   5. section_529_simple_mcs_<metric>_<horizon>.csv
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

from benchmark_utils import (
    load_forecast_pickles,
    check_common_grid_and_shape,
    _resolve_horizons,
    model_paths,
)

try:
    from scipy.stats import norm
except Exception:
    norm = None


# ------------------------------------------------------------
# 1. CRPS and twCRPS utilities
# ------------------------------------------------------------

def trapezoid_weights_for_u_np(u):
    """
    Trapezoidal quadrature weights over quantile grid u.
    """
    u = np.asarray(u, dtype=np.float64)

    if u.ndim != 1:
        raise ValueError("u must be one-dimensional.")

    if len(u) < 2:
        raise ValueError("u must contain at least two points.")

    order = np.argsort(u)
    u_sorted = u[order]

    w = np.zeros_like(u_sorted)

    w[0] = 0.5 * (u_sorted[1] - u_sorted[0])
    w[-1] = 0.5 * (u_sorted[-1] - u_sorted[-2])

    if len(u_sorted) > 2:
        w[1:-1] = 0.5 * (u_sorted[2:] - u_sorted[:-2])

    # Return in original order.
    w_orig = np.zeros_like(w)
    w_orig[order] = w

    return w_orig


def chain_threshold_np(
    x,
    threshold_low=-10.0,
    threshold_high=10.0,
    side="two_sided",
    smooth_h=1.0,
):
    """
    Threshold chaining transform for threshold-weighted CRPS.

    This maps values inside the non-tail region approximately to zero and
    accumulates distance into the selected tail region.

    side:
        "below"     focuses on x < threshold_low
        "above"     focuses on x > threshold_high
        "two_sided" focuses on both tails
    """
    x = np.asarray(x, dtype=np.float64)

    tl = float(threshold_low)
    th = float(threshold_high)

    if smooth_h is None or smooth_h <= 0:
        if side == "below":
            return np.minimum(x - tl, 0.0)

        if side == "above":
            return np.maximum(x - th, 0.0)

        if side == "two_sided":
            below = np.minimum(x - tl, 0.0)
            above = np.maximum(x - th, 0.0)
            return below + above

        raise ValueError("side must be 'below', 'above', or 'two_sided'.")

    h = float(smooth_h)

    # Smooth softplus approximations.
    # softplus(z) = h * log(1 + exp(z / h))
    def softplus(z):
        z = np.asarray(z, dtype=np.float64)
        return h * np.logaddexp(0.0, z / h)

    if side == "below":
        # Negative magnitude below tl.
        return -softplus(tl - x)

    if side == "above":
        # Positive magnitude above th.
        return softplus(x - th)

    if side == "two_sided":
        return -softplus(tl - x) + softplus(x - th)

    raise ValueError("side must be 'below', 'above', or 'two_sided'.")


def compute_crps_loss_bh_np(Q_all, y_true, u_grid, wu=None, crps_convention=True):
    """
    Standard CRPS from quantile grid.

    Returns
    -------
    loss_bh : np.ndarray
        Shape (B,H).
    """
    Q = np.asarray(Q_all, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)

    if wu is None:
        wu = trapezoid_weights_for_u_np(u)

    wu = np.asarray(wu, dtype=np.float64)

    u3 = u.reshape(1, 1, -1)
    wu3 = wu.reshape(1, 1, -1)

    e = y[:, :, None] - Q
    pinball = np.maximum(u3 * e, (u3 - 1.0) * e)

    loss_bh = np.sum(pinball * wu3, axis=-1)

    if crps_convention:
        loss_bh = 2.0 * loss_bh

    return loss_bh


def compute_twcrps_loss_bh_np(
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
    Threshold-weighted CRPS from quantile grid.

    Returns
    -------
    loss_bh : np.ndarray
        Shape (B,H).
    """
    Q = np.asarray(Q_all, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)

    if wu is None:
        wu = trapezoid_weights_for_u_np(u)

    wu = np.asarray(wu, dtype=np.float64)

    u3 = u.reshape(1, 1, -1)
    wu3 = wu.reshape(1, 1, -1)

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

    return loss_bh


# ------------------------------------------------------------
# 2. Loss-series construction
# ------------------------------------------------------------

def compute_loss_bh_for_metric(
    A,
    metric="twCRPS",
    tw_threshold_low=-10.0,
    tw_threshold_high=10.0,
    tw_smooth_h=1.0,
):
    """
    Compute observation-by-horizon loss matrix for a selected scoring rule.
    """
    y = np.asarray(A["true"], dtype=np.float64)
    Q = np.asarray(A["Q"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    metric_lower = metric.lower()

    if metric_lower == "crps":
        return compute_crps_loss_bh_np(Q, y, u)

    if metric_lower == "twcrps":
        return compute_twcrps_loss_bh_np(
            Q,
            y,
            u,
            threshold_low=tw_threshold_low,
            threshold_high=tw_threshold_high,
            side="two_sided",
            smooth_h=tw_smooth_h,
        )

    if metric_lower in ["twcrps_lower", "twcrps lower", "lower_twcrps"]:
        return compute_twcrps_loss_bh_np(
            Q,
            y,
            u,
            threshold_low=tw_threshold_low,
            threshold_high=tw_threshold_high,
            side="below",
            smooth_h=tw_smooth_h,
        )

    if metric_lower in ["twcrps_upper", "twcrps upper", "upper_twcrps"]:
        return compute_twcrps_loss_bh_np(
            Q,
            y,
            u,
            threshold_low=tw_threshold_low,
            threshold_high=tw_threshold_high,
            side="above",
            smooth_h=tw_smooth_h,
        )

    raise ValueError(f"Unknown metric: {metric}")


def aggregate_loss_series_by_origin(loss_bh, h_idx):
    """
    Convert loss matrix (B,H) into chronological series of length B.

    For one selected horizon, this is simply loss[:, h].
    For multiple horizons, average selected-horizon losses per forecast origin.
    """
    loss_bh = np.asarray(loss_bh, dtype=np.float64)
    loss_sel = loss_bh[:, h_idx]

    return np.nanmean(loss_sel, axis=1)


def build_loss_series_table(
    runs,
    metric="twCRPS",
    horizons_ahead=24,
    tw_threshold_low=-10.0,
    tw_threshold_high=10.0,
    tw_smooth_h=1.0,
):
    """
    Build DataFrame of chronological loss series.

    Columns:
        index, Model1, Model2, ...
    """
    first_A = next(iter(runs.values()))
    h_idx, horizon_label = _resolve_horizons(first_A, horizons_ahead)

    data = {"index": np.arange(first_A["true"].shape[0])}

    for model_name, A in runs.items():
        h_idx_model, _ = _resolve_horizons(A, horizons_ahead)

        if not np.array_equal(h_idx, h_idx_model):
            raise ValueError(f"Horizon mismatch for model {model_name}")

        loss_bh = compute_loss_bh_for_metric(
            A,
            metric=metric,
            tw_threshold_low=tw_threshold_low,
            tw_threshold_high=tw_threshold_high,
            tw_smooth_h=tw_smooth_h,
        )

        data[model_name] = aggregate_loss_series_by_origin(loss_bh, h_idx)

    return pd.DataFrame(data), horizon_label


# ------------------------------------------------------------
# 3. HAC / Diebold-Mariano test
# ------------------------------------------------------------

def newey_west_lrv(x, lag=None):
    """
    Newey-West long-run variance estimate for sample mean.

    Returns long-run variance of x_t, not divided by T.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    T = len(x)

    if T < 2:
        return np.nan

    x = x - np.mean(x)

    if lag is None:
        lag = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))

    lag = int(max(0, min(lag, T - 1)))

    gamma0 = np.mean(x * x)
    lrv = gamma0

    for k in range(1, lag + 1):
        gamma_k = np.mean(x[k:] * x[:-k])
        weight = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * weight * gamma_k

    return float(max(lrv, 0.0))


def diebold_mariano_hac_test(d, lag=None):
    """
    HAC Diebold-Mariano-style test on loss differential d_t.

    Here:
        d_t = loss_main,t - loss_benchmark,t

    Negative mean differential means the main model has lower loss.

    Returns:
        DM statistic, two-sided p-value, one-sided p-value for main better.
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]

    T = len(d)

    if T < 5:
        return {
            "Mean diff": np.nan,
            "DM stat": np.nan,
            "DM p-value two-sided": np.nan,
            "DM p-value main better": np.nan,
            "NW lag": lag,
            "N": T,
        }

    mean_d = float(np.mean(d))
    lrv = newey_west_lrv(d, lag=lag)

    if not np.isfinite(lrv) or lrv <= 0:
        dm_stat = np.nan
    else:
        dm_stat = mean_d / np.sqrt(lrv / T)

    if norm is None or not np.isfinite(dm_stat):
        p_two = np.nan
        p_better = np.nan
    else:
        p_two = float(2.0 * norm.sf(abs(dm_stat)))

        # H1: main has lower loss, i.e. E[d] < 0.
        p_better = float(norm.cdf(dm_stat))

    return {
        "Mean diff": mean_d,
        "DM stat": float(dm_stat) if np.isfinite(dm_stat) else np.nan,
        "DM p-value two-sided": p_two,
        "DM p-value main better": p_better,
        "NW lag": lag,
        "N": T,
    }


# ------------------------------------------------------------
# 4. Moving-block bootstrap
# ------------------------------------------------------------

def moving_block_bootstrap_mean(
    d,
    block_length=24,
    n_boot=5000,
    random_state=123,
):
    """
    Moving-block bootstrap for the mean of a dependent series.

    Parameters
    ----------
    d : array-like
        Loss differential series.
    block_length : int
        Bootstrap block length.
    n_boot : int
        Number of bootstrap samples.

    Returns
    -------
    boot_means : np.ndarray
        Bootstrap mean differentials.
    """
    rng = np.random.default_rng(random_state)

    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]

    T = len(d)

    if T == 0:
        return np.array([], dtype=np.float64)

    block_length = int(max(1, min(block_length, T)))

    starts = np.arange(0, T - block_length + 1)

    if len(starts) == 0:
        starts = np.array([0])

    n_blocks = int(np.ceil(T / block_length))

    boot_means = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sampled_starts = rng.choice(starts, size=n_blocks, replace=True)

        pieces = []
        for s in sampled_starts:
            pieces.append(d[s:s + block_length])

        sample = np.concatenate(pieces)[:T]
        boot_means[b] = np.mean(sample)

    return boot_means


def bootstrap_ci_and_pvalue(
    d,
    block_length=24,
    n_boot=5000,
    alpha=0.05,
    random_state=123,
):
    """
    Moving-block bootstrap CI for mean loss differential.

    d_t = loss_main,t - loss_benchmark,t

    Negative mean means main model is better.

    Bootstrap p-value main better is estimated as:
        P_boot(mean_d >= 0)
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]

    if len(d) == 0:
        return {
            "Bootstrap mean diff": np.nan,
            "Bootstrap CI low": np.nan,
            "Bootstrap CI high": np.nan,
            "Bootstrap p-value main better": np.nan,
            "Block length": block_length,
            "N boot": n_boot,
        }

    boot_means = moving_block_bootstrap_mean(
        d,
        block_length=block_length,
        n_boot=n_boot,
        random_state=random_state,
    )

    if len(boot_means) == 0:
        return {
            "Bootstrap mean diff": np.nan,
            "Bootstrap CI low": np.nan,
            "Bootstrap CI high": np.nan,
            "Bootstrap p-value main better": np.nan,
            "Block length": block_length,
            "N boot": n_boot,
        }

    ci_low = float(np.quantile(boot_means, alpha / 2.0))
    ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

    # One-sided bootstrap evidence for main better.
    p_better = float(np.mean(boot_means >= 0.0))

    return {
        "Bootstrap mean diff": float(np.mean(d)),
        "Bootstrap CI low": ci_low,
        "Bootstrap CI high": ci_high,
        "Bootstrap p-value main better": p_better,
        "Block length": block_length,
        "N boot": n_boot,
    }


# ------------------------------------------------------------
# 5. Pairwise comparison table
# ------------------------------------------------------------

def build_pairwise_forecast_comparison_table(
    runs,
    main_model="SAINT",
    metrics=("CRPS", "twCRPS"),
    horizons_ahead=24,
    tw_threshold_low=-10.0,
    tw_threshold_high=10.0,
    tw_smooth_h=1.0,
    nw_lag=None,
    block_length=24,
    n_boot=5000,
    random_state=123,
):
    """
    Build pairwise DM/bootstrap comparison table against main_model.

    Loss differential:
        d_t = loss_main,t - loss_benchmark,t

    Therefore:
        mean diff < 0 indicates main_model has lower loss.
        improvement (%) = 100 * (mean_benchmark - mean_main) / mean_benchmark.
    """
    if main_model not in runs:
        raise ValueError(f"main_model={main_model} not found in runs.")

    first_A = runs[main_model]
    h_idx, horizon_label = _resolve_horizons(first_A, horizons_ahead)

    # Default NW lag: useful for overlapping h-step forecasts.
    if nw_lag is None:
        if len(h_idx) == 1:
            nw_lag_eff = int(max(0, h_idx[0]))
        else:
            nw_lag_eff = int(max(0, np.max(h_idx)))
    else:
        nw_lag_eff = int(nw_lag)

    rows = []
    loss_tables = {}

    for metric in metrics:
        loss_df, _ = build_loss_series_table(
            runs,
            metric=metric,
            horizons_ahead=horizons_ahead,
            tw_threshold_low=tw_threshold_low,
            tw_threshold_high=tw_threshold_high,
            tw_smooth_h=tw_smooth_h,
        )

        loss_tables[metric] = loss_df

        main_loss = loss_df[main_model].astype(float).values

        for benchmark_model in runs.keys():
            if benchmark_model == main_model:
                continue

            bench_loss = loss_df[benchmark_model].astype(float).values

            mask = np.isfinite(main_loss) & np.isfinite(bench_loss)

            lm = main_loss[mask]
            lb = bench_loss[mask]

            d = lm - lb

            if len(d) == 0:
                continue

            mean_main = float(np.mean(lm))
            mean_bench = float(np.mean(lb))
            mean_diff = float(np.mean(d))

            if mean_bench != 0:
                improvement = 100.0 * (mean_bench - mean_main) / mean_bench
            else:
                improvement = np.nan

            dm = diebold_mariano_hac_test(
                d,
                lag=nw_lag_eff,
            )

            boot = bootstrap_ci_and_pvalue(
                d,
                block_length=block_length,
                n_boot=n_boot,
                alpha=0.05,
                random_state=random_state,
            )

            rows.append({
                "Metric": metric,
                "Proposed model": main_model,
                "Benchmark": benchmark_model,
                "Mean loss proposed": mean_main,
                "Mean loss benchmark": mean_bench,
                "Mean diff proposed-minus-benchmark": mean_diff,
                "Improvement (%)": improvement,
                "DM stat": dm["DM stat"],
                "DM p-value two-sided": dm["DM p-value two-sided"],
                "DM p-value proposed better": dm["DM p-value main better"],
                "Bootstrap CI low": boot["Bootstrap CI low"],
                "Bootstrap CI high": boot["Bootstrap CI high"],
                "Bootstrap p-value proposed better": boot["Bootstrap p-value main better"],
                "NW lag": nw_lag_eff,
                "Block length": block_length,
                "N": int(len(d)),
                "Horizons": horizon_label,
            })

    return pd.DataFrame(rows), loss_tables, horizon_label


# ------------------------------------------------------------
# 6. Optional simple MCS-style elimination
# ------------------------------------------------------------

def simple_mcs_elimination(
    loss_df,
    alpha=0.10,
    block_length=24,
    n_boot=3000,
    random_state=123,
):
    """
    Simple Model Confidence Set-style elimination.

    This is an approximate, lightweight implementation intended as a robustness
    summary rather than a full Hansen-Lunde-Nason MCS replacement.

    Procedure:
        1. Start with all models.
        2. Compute average loss for each remaining model.
        3. Compare worst remaining model against best remaining model using
           moving-block bootstrap of the loss differential.
        4. If worst is significantly worse, eliminate it.
        5. Repeat until no elimination.

    Returns
    -------
    result_df : pd.DataFrame
        Elimination sequence and final included models.
    """
    rng = np.random.default_rng(random_state)

    model_cols = [c for c in loss_df.columns if c != "index"]
    remaining = list(model_cols)

    rows = []
    step = 0

    while len(remaining) > 1:
        step += 1

        means = loss_df[remaining].mean(axis=0, skipna=True)
        best_model = means.idxmin()
        worst_model = means.idxmax()

        best_loss = loss_df[best_model].astype(float).values
        worst_loss = loss_df[worst_model].astype(float).values

        mask = np.isfinite(best_loss) & np.isfinite(worst_loss)

        d = worst_loss[mask] - best_loss[mask]

        # Positive d means worst has larger loss than best.
        boot_means = moving_block_bootstrap_mean(
            d,
            block_length=block_length,
            n_boot=n_boot,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )

        if len(d) == 0 or len(boot_means) == 0:
            p_worst_worse = np.nan
            ci_low = np.nan
            ci_high = np.nan
            eliminate = False
        else:
            mean_d = float(np.mean(d))
            ci_low = float(np.quantile(boot_means, alpha / 2.0))
            ci_high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

            # One-sided p-value for null that worst is not worse than best.
            p_worst_worse = float(np.mean(boot_means <= 0.0))
            eliminate = bool(p_worst_worse < alpha and mean_d > 0)

        rows.append({
            "Step": step,
            "Best remaining": best_model,
            "Worst remaining": worst_model,
            "Best mean loss": float(means[best_model]),
            "Worst mean loss": float(means[worst_model]),
            "Worst-minus-best diff": float(means[worst_model] - means[best_model]),
            "Bootstrap CI low": ci_low,
            "Bootstrap CI high": ci_high,
            "p-value worst worse": p_worst_worse,
            "Eliminated": worst_model if eliminate else "",
            "Remaining before step": ", ".join(remaining),
        })

        if eliminate:
            remaining.remove(worst_model)
        else:
            break

    for m in remaining:
        rows.append({
            "Step": step + 1,
            "Best remaining": "",
            "Worst remaining": "",
            "Best mean loss": np.nan,
            "Worst mean loss": np.nan,
            "Worst-minus-best diff": np.nan,
            "Bootstrap CI low": np.nan,
            "Bootstrap CI high": np.nan,
            "p-value worst worse": np.nan,
            "Eliminated": "",
            "Remaining before step": "",
            "Included in final set": m,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 7. Formatting
# ------------------------------------------------------------

def format_pairwise_table_for_latex(
    df,
    decimals=4,
    keep_cols=(
        "Metric",
        "Benchmark",
        "Mean loss proposed",
        "Mean loss benchmark",
        "Improvement (%)",
        "Mean diff proposed-minus-benchmark",
        "DM p-value proposed better",
        "Bootstrap CI low",
        "Bootstrap CI high",
    ),
):
    """
    Format pairwise comparison table for LaTeX.
    """
    df_fmt = df.copy()
    df_fmt = df_fmt[list(keep_cols)]

    for col in df_fmt.columns:
        if col in ["Metric", "Benchmark"]:
            continue

        vals = pd.to_numeric(df_fmt[col], errors="coerce").values
        formatted = []

        for v in vals:
            if not np.isfinite(v):
                formatted.append("--")
            else:
                formatted.append(f"{v:.{decimals}f}")

        df_fmt[col] = formatted

    return df_fmt.to_latex(
        index=False,
        escape=False,
        column_format="ll" + "r" * (df_fmt.shape[1] - 2),
    )


def format_pairwise_table_for_markdown(
    df,
    decimals=4,
    keep_cols=(
        "Metric",
        "Benchmark",
        "Mean loss proposed",
        "Mean loss benchmark",
        "Improvement (%)",
        "Mean diff proposed-minus-benchmark",
        "DM p-value proposed better",
        "Bootstrap CI low",
        "Bootstrap CI high",
    ),
):
    """
    Format pairwise comparison table for Markdown.
    """
    df_fmt = df.copy()
    df_fmt = df_fmt[list(keep_cols)]

    for col in df_fmt.columns:
        if col in ["Metric", "Benchmark"]:
            continue

        vals = pd.to_numeric(df_fmt[col], errors="coerce").values
        formatted = []

        for v in vals:
            if not np.isfinite(v):
                formatted.append("--")
            else:
                formatted.append(f"{v:.{decimals}f}")

        df_fmt[col] = formatted

    return df_fmt.to_markdown(index=False)


# ------------------------------------------------------------
# 8. Convenience wrapper
# ------------------------------------------------------------

def make_section_529_statistical_comparison_outputs(
    model_paths=None,
    runs=None,
    out_dir="./comparison_nn_vs_arima/section_529_statistical_comparison",
    main_model="SAINT",
    metrics=("CRPS", "twCRPS"),
    horizons_ahead=24,
    tw_threshold_low=-10.0,
    tw_threshold_high=10.0,
    tw_smooth_h=1.0,
    nw_lag=None,
    block_length=24,
    n_boot=5000,
    random_state=123,
    run_simple_mcs=True,
):
    """
    Create Section 5.2.9 pairwise statistical comparison outputs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if runs is None:
        if model_paths is None:
            raise ValueError("Pass either runs or model_paths.")
        runs = load_forecast_pickles(model_paths)
        check_common_grid_and_shape(runs)

    pairwise_df, loss_tables, horizon_label = build_pairwise_forecast_comparison_table(
        runs=runs,
        main_model=main_model,
        metrics=metrics,
        horizons_ahead=horizons_ahead,
        tw_threshold_low=tw_threshold_low,
        tw_threshold_high=tw_threshold_high,
        tw_smooth_h=tw_smooth_h,
        nw_lag=nw_lag,
        block_length=block_length,
        n_boot=n_boot,
        random_state=random_state,
    )

    horizon_tag = horizon_label.replace(",", "_").replace("-", "_")
    main_tag = main_model.replace(" ", "_").replace("/", "_")

    # Save main pairwise table.
    pairwise_csv = out_dir / f"section_529_pairwise_dm_bootstrap_{main_tag}_{horizon_tag}.csv"
    pairwise_tex = out_dir / f"section_529_pairwise_dm_bootstrap_{main_tag}_{horizon_tag}.tex"
    pairwise_md = out_dir / f"section_529_pairwise_dm_bootstrap_{main_tag}_{horizon_tag}.md"

    pairwise_df.to_csv(pairwise_csv, index=False)

    latex = format_pairwise_table_for_latex(pairwise_df, decimals=4)
    md = format_pairwise_table_for_markdown(pairwise_df, decimals=4)

    with open(pairwise_tex, "w") as f:
        f.write(latex)

    with open(pairwise_md, "w") as f:
        f.write(md)

    # Save loss series and optional MCS summaries.
    mcs_tables = {}

    for metric, loss_df in loss_tables.items():
        metric_tag = metric.replace(" ", "_").replace("/", "_")

        loss_df.to_csv(
            out_dir / f"section_529_loss_series_{metric_tag}_{horizon_tag}.csv",
            index=False,
        )

        if run_simple_mcs:
            mcs_df = simple_mcs_elimination(
                loss_df,
                alpha=0.10,
                block_length=block_length,
                n_boot=max(1000, min(n_boot, 3000)),
                random_state=random_state,
            )

            mcs_tables[metric] = mcs_df

            mcs_df.to_csv(
                out_dir / f"section_529_simple_mcs_{metric_tag}_{horizon_tag}.csv",
                index=False,
            )

    print("\n==============================================")
    print(f"Section 5.2.9: Statistical comparison against {main_model}")
    print(f"Horizons: {horizon_label}")
    print("==============================================")
    print(md)

    print("\nInterpretation:")
    print(
        "Loss differential is proposed-minus-benchmark. "
        "Negative values indicate that the proposed model has lower loss. "
        "The one-sided DM and bootstrap p-values test whether the proposed "
        "model is significantly better."
    )

    if norm is None:
        print(
            "\nWarning: scipy was not available, so DM p-values are NaN. "
            "Install scipy to enable p-values."
        )

    return {
        "runs": runs,
        "pairwise_table": pairwise_df,
        "loss_tables": loss_tables,
        "mcs_tables": mcs_tables,
        "out_dir": out_dir,
    }


# ============================================================
# Example usage
# ============================================================

section_529_result = make_section_529_statistical_comparison_outputs(
    model_paths=model_paths,
    out_dir="./comparison_nn_vs_arima/section_529_statistical_comparison",
    main_model="SAINT",
    metrics=("CRPS", "twCRPS"),
    horizons_ahead=24,
    tw_threshold_low=-10.0,
    tw_threshold_high=10.0,
    tw_smooth_h=1.0,
    nw_lag=None,          # None uses horizon-based default
    block_length=24,      # hourly dependence; try 48 or 72 as robustness
    n_boot=5000,
    random_state=123,
    run_simple_mcs=True,
)

pairwise_comparison_table = section_529_result["pairwise_table"]
loss_tables = section_529_result["loss_tables"]
mcs_tables = section_529_result["mcs_tables"]
