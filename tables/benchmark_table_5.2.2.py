# ============================================================
# Benchmark table for Section 5.2.2:
# Aggregate distributional forecasting performance
# ============================================================
#
# Assumes the utilities from your notebook are already defined:
#   - load_forecast_pickles
#   - check_common_grid_and_shape
#   - quantiles_from_u_grid
#   - pinball_loss_array
#   - compute_abs_threshold_event_prob
#   - get_twcrps_per_horizon
#   - chain_threshold_np
#   - trapezoid_weights_for_u_np
#
# Main outputs:
#   1. benchmark_distributional_all_horizons.csv
#   2. benchmark_distributional_h24.csv
#   3. benchmark_distributional_all_horizons.tex
#   4. benchmark_distributional_h24.tex
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from benchmark_utils import (
    load_forecast_pickles,
    check_common_grid_and_shape,
    compute_abs_threshold_event_prob,
    _resolve_horizons,
    quantiles_from_u_grid,
    pinball_loss_array,
    get_twcrps_per_horizon,
    chain_threshold_np,
    trapezoid_weights_for_u_np,
    model_paths,
)

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


# ------------------------------------------------------------
# 1. Standard CRPS from quantile grid
# ------------------------------------------------------------

def compute_crps_loss_bh_np(Q_all, y_true, u_grid, wu=None, crps_convention=True):
    """
    Compute standard CRPS loss for each observation and horizon.

    Parameters
    ----------
    Q_all : np.ndarray
        Forecast quantiles, shape (B, H, J).
    y_true : np.ndarray
        Realized values, shape (B, H).
    u_grid : np.ndarray
        Quantile levels, shape (J,).
    wu : np.ndarray or None
        Quadrature weights over u_grid. If None, trapezoidal weights are used.
    crps_convention : bool
        If True, uses CRPS = 2 * integral pinball loss.

    Returns
    -------
    loss_bh : np.ndarray
        CRPS per observation and horizon, shape (B, H).
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


# ------------------------------------------------------------
# 2. twCRPS from quantile grid, observation-level
# ------------------------------------------------------------

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
    Compute threshold-weighted CRPS for each observation and horizon.

    Parameters
    ----------
    side : {"two_sided", "below", "above"}
        two_sided: weights both lower and upper depeg regions.
        below: focuses on lower tail, y < threshold_low.
        above: focuses on upper tail, y > threshold_high.

    Returns
    -------
    loss_bh : np.ndarray
        twCRPS per observation and horizon, shape (B, H).
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
# 3. Horizon helpers
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 4. Pinball-loss aggregation
# ------------------------------------------------------------

def aggregate_pinball_loss(A, tau, h_idx):
    """
    Mean pinball loss at quantile tau over selected horizons.
    """
    y = np.asarray(A["true"], dtype=np.float64)
    Q = np.asarray(A["Q"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    q_tau = quantiles_from_u_grid(Q, u, [tau])[..., 0]  # (B,H)
    loss = pinball_loss_array(y, q_tau, tau)

    return float(np.nanmean(loss[:, h_idx]))


# ------------------------------------------------------------
# 5. Threshold-event probability metrics
# ------------------------------------------------------------

def aggregate_event_probability_metrics(A, abs_threshold=15.0, h_idx=None, eps=1e-12):
    """
    Aggregate Brier score, log loss, ROC-AUC, event rate, and mean predicted
    event probability for event:

        |Y| >= abs_threshold

    over selected horizons.
    """
    y = np.asarray(A["true"], dtype=np.float64)

    if h_idx is None:
        h_idx = np.arange(y.shape[1])

    p_event = compute_abs_threshold_event_prob(A, abs_threshold=abs_threshold)
    event = (np.abs(y) >= abs_threshold).astype(int)

    p = p_event[:, h_idx].reshape(-1)
    e = event[:, h_idx].reshape(-1)

    mask = np.isfinite(p) & np.isfinite(e)
    p = np.clip(p[mask], eps, 1.0 - eps)
    e = e[mask].astype(int)

    brier = float(np.mean((p - e) ** 2))
    logloss = float(-np.mean(e * np.log(p) + (1 - e) * np.log(1 - p)))

    if roc_auc_score is not None and len(np.unique(e)) == 2:
        auc = float(roc_auc_score(e, p))
    else:
        auc = np.nan

    return {
        "Event rate": float(np.mean(e)) if len(e) else np.nan,
        "Mean p(depeg)": float(np.mean(p)) if len(p) else np.nan,
        "Brier depeg": brier,
        "LogLoss depeg": logloss,
        "AUC depeg": auc,
        "N events": int(np.sum(e)),
        "N obs": int(len(e)),
    }


# ------------------------------------------------------------
# 6. Main benchmark-table builder
# ------------------------------------------------------------

def build_distributional_benchmark_table(
    runs,
    horizons_ahead="all",
    quantile_levels=(0.01, 0.05, 0.50, 0.95, 0.99),
    event_abs_threshold=15.0,
    tw_threshold_low=-10.0,
    tw_threshold_high=10.0,
    tw_smooth_h=1.0,
    use_stored_twcrps=True,
):
    """
    Build one aggregate benchmark table.

    Parameters
    ----------
    runs : dict[str -> dict]
        Loaded forecast dictionaries.
    horizons_ahead : "all", None, int, or iterable[int]
        Horizons over which to aggregate metrics.
    quantile_levels : tuple
        Quantile levels for pinball-loss columns.
    event_abs_threshold : float
        Depeg event threshold in bps.
    tw_threshold_low, tw_threshold_high : float
        Thresholds used for twCRPS in bps.
    tw_smooth_h : float
        Smoothness parameter for threshold weighting.
    use_stored_twcrps : bool
        If True, use stored two-sided twCRPS per horizon when available.

    Returns
    -------
    df : pd.DataFrame
        Benchmark table, one row per model.
    """
    rows = []

    first_A = next(iter(runs.values()))
    h_idx, horizon_label = _resolve_horizons(first_A, horizons_ahead)

    for name, A in runs.items():
        # Check horizon compatibility for each model
        h_idx_model, _ = _resolve_horizons(A, horizons_ahead)

        if not np.array_equal(h_idx, h_idx_model):
            raise ValueError(f"Horizon mismatch for model {name}")

        y = A["true"]
        Q = A["Q"]
        u = A["u_grid"]

        # -----------------------------
        # Standard CRPS
        # -----------------------------
        crps_bh = compute_crps_loss_bh_np(Q, y, u)
        crps = _mean_selected(crps_bh, h_idx)

        # -----------------------------
        # Two-sided twCRPS
        # -----------------------------
        twcrps = np.nan

        if use_stored_twcrps:
            try:
                tw_per_h = get_twcrps_per_horizon(
                    A,
                    compute_if_missing=False,
                )
                if tw_per_h is not None:
                    twcrps = float(np.nanmean(tw_per_h[h_idx]))
            except Exception:
                twcrps = np.nan

        if not np.isfinite(twcrps):
            tw_bh = compute_twcrps_loss_bh_np(
                Q,
                y,
                u,
                threshold_low=tw_threshold_low,
                threshold_high=tw_threshold_high,
                side="two_sided",
                smooth_h=tw_smooth_h,
            )
            twcrps = _mean_selected(tw_bh, h_idx)

        # -----------------------------
        # One-sided twCRPS
        # -----------------------------
        tw_lower_bh = compute_twcrps_loss_bh_np(
            Q,
            y,
            u,
            threshold_low=tw_threshold_low,
            threshold_high=tw_threshold_high,
            side="below",
            smooth_h=tw_smooth_h,
        )
        tw_upper_bh = compute_twcrps_loss_bh_np(
            Q,
            y,
            u,
            threshold_low=tw_threshold_low,
            threshold_high=tw_threshold_high,
            side="above",
            smooth_h=tw_smooth_h,
        )

        twcrps_lower = _mean_selected(tw_lower_bh, h_idx)
        twcrps_upper = _mean_selected(tw_upper_bh, h_idx)

        # -----------------------------
        # Quantile / tail pinball loss
        # -----------------------------
        qloss = {}
        for tau in quantile_levels:
            label = f"QL {100*tau:.0f}%"
            qloss[label] = aggregate_pinball_loss(A, tau=tau, h_idx=h_idx)

        # -----------------------------
        # Depeg event probability metrics
        # -----------------------------
        event_metrics = aggregate_event_probability_metrics(
            A,
            abs_threshold=event_abs_threshold,
            h_idx=h_idx,
        )

        row = {
            "Model": name,
            "Horizons": horizon_label,
            "CRPS": crps,
            "twCRPS": twcrps,
            f"twCRPS lower (<{tw_threshold_low:g})": twcrps_lower,
            f"twCRPS upper (>{tw_threshold_high:g})": twcrps_upper,
            **qloss,
            **event_metrics,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


# ------------------------------------------------------------
# 7. Relative improvement of main model over best benchmark
# ------------------------------------------------------------

def compute_main_model_improvements(
    df,
    main_model="SAINT",
    lower_better=None,
    higher_better=None,
):
    """
    Compute relative improvement of main_model against the best non-main benchmark.

    For lower-is-better metrics:
        improvement = 100 * (best_benchmark - main_model) / best_benchmark

    For higher-is-better metrics:
        improvement = 100 * (main_model - best_benchmark) / best_benchmark
    """
    if lower_better is None:
        lower_better = [
            "CRPS",
            "twCRPS",
            "twCRPS lower",
            "twCRPS upper",
            "QL",
            "Brier",
            "LogLoss",
        ]

    if higher_better is None:
        higher_better = [
            "AUC",
        ]

    main_row = df[df["Model"] == main_model]

    if len(main_row) != 1:
        raise ValueError(f"Expected exactly one row for main_model={main_model}")

    main_row = main_row.iloc[0]
    bench = df[df["Model"] != main_model].copy()

    rows = []

    for col in df.columns:
        if col in ["Model", "Horizons", "N events", "N obs"]:
            continue

        if not np.issubdtype(df[col].dtype, np.number):
            continue

        main_val = float(main_row[col])

        if not np.isfinite(main_val) or len(bench) == 0:
            continue

        is_lower = any(key in col for key in lower_better)
        is_higher = any(key in col for key in higher_better)

        if is_lower:
            ref_val = float(np.nanmin(bench[col].values))
            ref_model = bench.iloc[np.nanargmin(bench[col].values)]["Model"]

            improvement = 100.0 * (ref_val - main_val) / ref_val if ref_val != 0 else np.nan
            direction = "lower_better"

        elif is_higher:
            ref_val = float(np.nanmax(bench[col].values))
            ref_model = bench.iloc[np.nanargmax(bench[col].values)]["Model"]

            improvement = 100.0 * (main_val - ref_val) / ref_val if ref_val != 0 else np.nan
            direction = "higher_better"

        else:
            continue

        rows.append({
            "Metric": col,
            "Main model": main_model,
            "Main value": main_val,
            "Best benchmark": ref_model,
            "Benchmark value": ref_val,
            "Improvement (%)": improvement,
            "Direction": direction,
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 8. Paper-style formatting: bold best values
# ------------------------------------------------------------

def format_benchmark_table_for_latex(
    df,
    decimals=4,
    higher_better_cols=("AUC depeg",),
    exclude_cols=("Horizons", "N events", "N obs"),
):
    """
    Return a LaTeX table string with best values bolded.

    Lower is better for all numeric columns except those in higher_better_cols.
    """
    df_fmt = df.copy()

    higher_better_cols = set(higher_better_cols)
    exclude_cols = set(exclude_cols)

    for col in df.columns:
        if col in ["Model"] or col in exclude_cols:
            continue

        if not np.issubdtype(df[col].dtype, np.number):
            continue

        vals = df[col].astype(float).values

        if np.all(~np.isfinite(vals)):
            continue

        if col in higher_better_cols:
            best_val = np.nanmax(vals)
            is_best = np.isclose(vals, best_val, rtol=1e-10, atol=1e-12)
        else:
            best_val = np.nanmin(vals)
            is_best = np.isclose(vals, best_val, rtol=1e-10, atol=1e-12)

        formatted = []
        for v, b in zip(vals, is_best):
            if not np.isfinite(v):
                s = "--"
            else:
                s = f"{v:.{decimals}f}"

            if b and np.isfinite(v):
                s = r"\textbf{" + s + "}"

            formatted.append(s)

        df_fmt[col] = formatted

    # Keep useful columns, but remove overly diagnostic columns if desired
    keep_cols = [c for c in df_fmt.columns if c not in exclude_cols]
    df_fmt = df_fmt[keep_cols]

    latex = df_fmt.to_latex(
        index=False,
        escape=False,
        column_format="l" + "r" * (df_fmt.shape[1] - 1),
    )

    return latex


def format_benchmark_table_for_markdown(
    df,
    decimals=4,
    higher_better_cols=("AUC depeg",),
    exclude_cols=("Horizons", "N events", "N obs"),
):
    """
    Return a markdown table with best values bolded.
    """
    df_fmt = df.copy()

    higher_better_cols = set(higher_better_cols)
    exclude_cols = set(exclude_cols)

    for col in df.columns:
        if col in ["Model"] or col in exclude_cols:
            continue

        if not np.issubdtype(df[col].dtype, np.number):
            continue

        vals = df[col].astype(float).values

        if np.all(~np.isfinite(vals)):
            continue

        if col in higher_better_cols:
            best_val = np.nanmax(vals)
            is_best = np.isclose(vals, best_val, rtol=1e-10, atol=1e-12)
        else:
            best_val = np.nanmin(vals)
            is_best = np.isclose(vals, best_val, rtol=1e-10, atol=1e-12)

        formatted = []
        for v, b in zip(vals, is_best):
            if not np.isfinite(v):
                s = "--"
            else:
                s = f"{v:.{decimals}f}"

            if b and np.isfinite(v):
                s = f"**{s}**"

            formatted.append(s)

        df_fmt[col] = formatted

    keep_cols = [c for c in df_fmt.columns if c not in exclude_cols]
    df_fmt = df_fmt[keep_cols]

    return df_fmt.to_markdown(index=False)


# ------------------------------------------------------------
# 9. Convenience wrapper: create all tables
# ------------------------------------------------------------

def make_section_522_benchmark_outputs(
    model_paths=None,
    runs=None,
    out_dir="./comparison_nn_vs_arima",
    main_model="SAINT",
    event_abs_threshold=15.0,
    tw_threshold_low=-10.0,
    tw_threshold_high=10.0,
    tw_smooth_h=1.0,
    quantile_levels=(0.01, 0.05, 0.50, 0.95, 0.99),
):
    """
    Build benchmark tables for:
      - all horizons
      - 1h
      - 12h
      - 24h

    You can pass either:
      - model_paths: dict[name -> pickle_path]
      - runs: already loaded dict[name -> forecast dict]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if runs is None:
        if model_paths is None:
            raise ValueError("Pass either runs or model_paths.")
        runs = load_forecast_pickles(model_paths)
        check_common_grid_and_shape(runs)

    tables = {}
    improvements = {}

    horizon_specs = {
        "all_horizons": "all",
        "h1": 1,
        "h12": 12,
        "h24": 24,
    }

    for label, horizons in horizon_specs.items():
        df = build_distributional_benchmark_table(
            runs=runs,
            horizons_ahead=horizons,
            quantile_levels=quantile_levels,
            event_abs_threshold=event_abs_threshold,
            tw_threshold_low=tw_threshold_low,
            tw_threshold_high=tw_threshold_high,
            tw_smooth_h=tw_smooth_h,
            use_stored_twcrps=True,
        )

        imp = compute_main_model_improvements(
            df,
            main_model=main_model,
        )

        tables[label] = df
        improvements[label] = imp

        # Save raw numeric tables
        df.to_csv(out_dir / f"benchmark_distributional_{label}.csv", index=False)
        imp.to_csv(out_dir / f"benchmark_distributional_improvements_{label}.csv", index=False)

        # Save LaTeX
        latex = format_benchmark_table_for_latex(df, decimals=4)
        with open(out_dir / f"benchmark_distributional_{label}.tex", "w") as f:
            f.write(latex)

        # Save Markdown
        md = format_benchmark_table_for_markdown(df, decimals=4)
        with open(out_dir / f"benchmark_distributional_{label}.md", "w") as f:
            f.write(md)

        print(f"\n==============================")
        print(f"Benchmark table: {label}")
        print(f"==============================")
        print(format_benchmark_table_for_markdown(df, decimals=4))

        print(f"\nImprovement of {main_model} vs best non-{main_model} benchmark:")
        print(
            imp[["Metric", "Best benchmark", "Benchmark value", "Main value", "Improvement (%)"]]
            .to_string(index=False)
        )

    return {
        "runs": runs,
        "tables": tables,
        "improvements": improvements,
        "out_dir": out_dir,
    }


# ============================================================
# Example usage with your current models
# ============================================================



bench_result = make_section_522_benchmark_outputs(
    model_paths=model_paths,
    out_dir="./comparison_nn_vs_arima/section_522_tables",
    main_model="SAINT",
    event_abs_threshold=15.0,
    tw_threshold_low=-10.0,
    tw_threshold_high=10.0,
    tw_smooth_h=1.0,
    quantile_levels=(0.01, 0.05, 0.95, 0.99),
)

# Access the main tables
benchmark_all = bench_result["tables"]["all_horizons"]
benchmark_h24 = bench_result["tables"]["h24"]

# Access improvement summaries
improvement_all = bench_result["improvements"]["all_horizons"]
improvement_h24 = bench_result["improvements"]["h24"]