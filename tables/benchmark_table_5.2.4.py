# ============================================================
# Section 5.2.4:
# Calibration of the predictive distribution
# ============================================================
#
# Assumes utilities from previous sections are available:
#   - load_forecast_pickles
#   - check_common_grid_and_shape
#   - quantiles_from_u_grid
#   - _resolve_horizons
#
# Main outputs:
#   1. section_524_interval_coverage_<model>_<horizon>.csv
#   2. section_524_interval_coverage_<model>_<horizon>.tex
#   3. section_524_interval_coverage_<model>_<horizon>.md
#   4. section_524_quantile_calibration_<model>_<horizon>.csv
#   5. section_524_pit_histogram_<model>_<horizon>.png
#   6. section_524_quantile_calibration_<model>_<horizon>.png
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from tables.bechmark_utils import (
    load_forecast_pickles,
    check_common_grid_and_shape,
    quantiles_from_u_grid,
    _resolve_horizons,)

# ------------------------------------------------------------
# 1. PIT from quantile grid
# ------------------------------------------------------------

def _make_strictly_increasing(x, eps=1e-10):
    """
    Ensure a vector is strictly increasing by adding tiny increments
    when quantile crossings or ties are present.
    """
    x = np.asarray(x, dtype=np.float64).copy()

    for j in range(1, len(x)):
        if x[j] <= x[j - 1]:
            x[j] = x[j - 1] + eps

    return x


def pit_from_quantile_grid(
    Q_all,
    y_true,
    u_grid,
    h_idx=None,
):
    """
    Approximate PIT values from a quantile forecast grid.

    PIT is approximated by inverting the forecast quantile function:

        PIT_t = F_hat_t(y_t)

    Parameters
    ----------
    Q_all : np.ndarray
        Forecast quantiles, shape (B, H, J).
    y_true : np.ndarray
        Realized values, shape (B, H).
    u_grid : np.ndarray
        Quantile levels, shape (J,).
    h_idx : np.ndarray or None
        Selected horizon indices.

    Returns
    -------
    pit : np.ndarray
        Flattened PIT values.
    """
    Q = np.asarray(Q_all, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)

    if h_idx is None:
        h_idx = np.arange(y.shape[1])

    Q_sel = Q[:, h_idx, :]
    y_sel = y[:, h_idx]

    B, Hs, J = Q_sel.shape

    pit_vals = []

    # Sort u-grid in case it is not already sorted.
    order = np.argsort(u)
    u_sorted = u[order]

    Q_sel = Q_sel[:, :, order]

    for b in range(B):
        for h in range(Hs):
            yy = y_sel[b, h]
            qq = Q_sel[b, h, :]

            if not np.isfinite(yy) or not np.all(np.isfinite(qq)):
                continue

            # Enforce monotonicity of quantile function.
            qq_mono = np.maximum.accumulate(qq)
            qq_mono = _make_strictly_increasing(qq_mono)

            # Invert Q(u) to get F(y).
            # Outside the quantile grid, assign mass to 0 or 1.
            pit = np.interp(
                yy,
                qq_mono,
                u_sorted,
                left=0.0,
                right=1.0,
            )

            pit_vals.append(pit)

    pit_vals = np.asarray(pit_vals, dtype=np.float64)
    pit_vals = pit_vals[np.isfinite(pit_vals)]

    return pit_vals


# ------------------------------------------------------------
# 2. Quantile calibration table
# ------------------------------------------------------------

def build_quantile_calibration_table(
    A,
    horizons_ahead="all",
    quantile_levels=None,
):
    """
    Build nominal-vs-empirical quantile calibration table.

    For each q, computes:

        empirical coverage = mean{Y <= Q_hat(q)}
    """
    y = np.asarray(A["true"], dtype=np.float64)
    Q = np.asarray(A["Q"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    if quantile_levels is None:
        quantile_levels = u

    quantile_levels = np.asarray(quantile_levels, dtype=np.float64)

    qhat = quantiles_from_u_grid(Q, u, quantile_levels)  # (B,H,Jq)

    y_sel = y[:, h_idx]
    qhat_sel = qhat[:, h_idx, :]

    rows = []

    for j, q in enumerate(quantile_levels):
        qj = qhat_sel[:, :, j]

        mask = np.isfinite(y_sel) & np.isfinite(qj)

        if np.sum(mask) == 0:
            emp_cov = np.nan
            calib_error = np.nan
            n_obs = 0
        else:
            emp_cov = float(np.mean(y_sel[mask] <= qj[mask]))
            calib_error = float(emp_cov - q)
            n_obs = int(np.sum(mask))

        rows.append({
            "Nominal quantile": float(q),
            "Empirical coverage": emp_cov,
            "Calibration error": calib_error,
            "Abs calibration error": abs(calib_error) if np.isfinite(calib_error) else np.nan,
            "Horizons": horizon_label,
            "N obs": n_obs,
        })

    df = pd.DataFrame(rows)

    return df


# ------------------------------------------------------------
# 3. Central interval coverage and interval score
# ------------------------------------------------------------

def interval_score_array(y, lower, upper, alpha):
    """
    Gneiting-Raftery interval score for central (1-alpha) prediction interval:

        IS_alpha = (upper - lower)
                   + 2/alpha * (lower - y) * 1{y < lower}
                   + 2/alpha * (y - upper) * 1{y > upper}

    Lower is better.
    """
    y = np.asarray(y, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    width = upper - lower

    below = y < lower
    above = y > upper

    score = width.copy()
    score = score + (2.0 / alpha) * (lower - y) * below
    score = score + (2.0 / alpha) * (y - upper) * above

    return score


def build_interval_coverage_table(
    A,
    intervals=(0.50, 0.80, 0.90, 0.95, 0.98),
    horizons_ahead="all",
):
    """
    Build interval calibration table.

    For each nominal central interval, computes:
        - empirical coverage
        - average width
        - interval score
        - lower-tail miss rate
        - upper-tail miss rate
    """
    y = np.asarray(A["true"], dtype=np.float64)
    Q = np.asarray(A["Q"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    y_sel = y[:, h_idx]

    rows = []

    for nominal in intervals:
        nominal = float(nominal)
        alpha = 1.0 - nominal

        q_low = alpha / 2.0
        q_high = 1.0 - alpha / 2.0

        qhat = quantiles_from_u_grid(Q, u, [q_low, q_high])  # (B,H,2)

        lower = qhat[:, h_idx, 0]
        upper = qhat[:, h_idx, 1]

        mask = (
            np.isfinite(y_sel)
            & np.isfinite(lower)
            & np.isfinite(upper)
            & (upper >= lower)
        )

        if np.sum(mask) == 0:
            empirical_coverage = np.nan
            avg_width = np.nan
            avg_interval_score = np.nan
            lower_miss_rate = np.nan
            upper_miss_rate = np.nan
            n_obs = 0
        else:
            yy = y_sel[mask]
            ll = lower[mask]
            uu = upper[mask]

            inside = (yy >= ll) & (yy <= uu)

            empirical_coverage = float(np.mean(inside))
            avg_width = float(np.mean(uu - ll))

            score = interval_score_array(
                yy,
                ll,
                uu,
                alpha=alpha,
            )
            avg_interval_score = float(np.mean(score))

            lower_miss_rate = float(np.mean(yy < ll))
            upper_miss_rate = float(np.mean(yy > uu))

            n_obs = int(np.sum(mask))

        rows.append({
            "Nominal interval": f"{100 * nominal:.0f}%",
            "Nominal coverage": nominal,
            "Empirical coverage": empirical_coverage,
            "Coverage error": empirical_coverage - nominal if np.isfinite(empirical_coverage) else np.nan,
            "Avg width": avg_width,
            "Interval score": avg_interval_score,
            "Lower miss rate": lower_miss_rate,
            "Upper miss rate": upper_miss_rate,
            "Horizons": horizon_label,
            "N obs": n_obs,
        })

    df = pd.DataFrame(rows)

    return df


# ------------------------------------------------------------
# 4. Plotting functions
# ------------------------------------------------------------

def plot_pit_histogram(
    pit,
    out_path=None,
    bins=20,
    title="PIT histogram",
):
    """
    Plot PIT histogram with uniform reference line.
    """
    pit = np.asarray(pit, dtype=np.float64)
    pit = pit[np.isfinite(pit)]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(
        pit,
        bins=bins,
        range=(0.0, 1.0),
        density=True,
        alpha=0.75,
        edgecolor="black",
    )

    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Uniform reference",
    )

    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig, ax


def plot_quantile_calibration(
    calib_df,
    out_path=None,
    title="Quantile calibration",
):
    """
    Plot nominal quantile level against empirical coverage.
    """
    df = calib_df.copy()

    x = df["Nominal quantile"].astype(float).values
    y = df["Empirical coverage"].astype(float).values

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=1.5,
        label="Forecast",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    return fig, ax


# ------------------------------------------------------------
# 5. Formatting
# ------------------------------------------------------------

def format_interval_table_for_latex(
    df,
    decimals=4,
    keep_cols=(
        "Nominal interval",
        "Empirical coverage",
        "Avg width",
        "Interval score",
    ),
):
    """
    Format interval coverage table for LaTeX.
    """
    df_fmt = df.copy()
    df_fmt = df_fmt[list(keep_cols)]

    for col in df_fmt.columns:
        if col == "Nominal interval":
            continue

        vals = pd.to_numeric(df_fmt[col], errors="coerce").values
        formatted = []

        for v in vals:
            if not np.isfinite(v):
                formatted.append("--")
            else:
                formatted.append(f"{v:.{decimals}f}")

        df_fmt[col] = formatted

    latex = df_fmt.to_latex(
        index=False,
        escape=False,
        column_format="l" + "r" * (df_fmt.shape[1] - 1),
    )

    return latex


def format_interval_table_for_markdown(
    df,
    decimals=4,
    keep_cols=(
        "Nominal interval",
        "Empirical coverage",
        "Avg width",
        "Interval score",
    ),
):
    """
    Format interval coverage table for Markdown.
    """
    df_fmt = df.copy()
    df_fmt = df_fmt[list(keep_cols)]

    for col in df_fmt.columns:
        if col == "Nominal interval":
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
# 6. Convenience wrapper for Section 5.2.4
# ------------------------------------------------------------

def make_section_524_calibration_outputs(
    model_paths=None,
    runs=None,
    out_dir="./comparison_nn_vs_arima/section_524_calibration",
    model="SAINT",
    horizons_ahead="all",
    intervals=(0.50, 0.80, 0.90, 0.95, 0.98),
    quantile_levels_for_plot=None,
    pit_bins=20,
):
    """
    Create calibration tables and plots for Section 5.2.4.

    Outputs:
        - PIT histogram
        - quantile calibration plot
        - interval coverage table
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

    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    horizon_tag = horizon_label.replace(",", "_").replace("-", "_").replace("h", "h")
    model_tag = model.replace(" ", "_").replace("/", "_")

    y = np.asarray(A["true"], dtype=np.float64)
    Q = np.asarray(A["Q"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    # -----------------------------
    # PIT values and histogram
    # -----------------------------
    pit = pit_from_quantile_grid(
        Q,
        y,
        u,
        h_idx=h_idx,
    )

    pit_df = pd.DataFrame({"PIT": pit})
    pit_df.to_csv(
        out_dir / f"section_524_pit_values_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    plot_pit_histogram(
        pit,
        out_path=out_dir / f"section_524_pit_histogram_{model_tag}_{horizon_tag}.png",
        bins=pit_bins,
        title=f"PIT histogram: {model}, horizons {horizon_label}",
    )

    # -----------------------------
    # Quantile calibration
    # -----------------------------
    if quantile_levels_for_plot is None:
        # Use a clean set for the paper plot.
        quantile_levels_for_plot = np.array(
            [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99],
            dtype=np.float64,
        )

    calib_df = build_quantile_calibration_table(
        A,
        horizons_ahead=horizons_ahead,
        quantile_levels=quantile_levels_for_plot,
    )

    calib_df.to_csv(
        out_dir / f"section_524_quantile_calibration_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    plot_quantile_calibration(
        calib_df,
        out_path=out_dir / f"section_524_quantile_calibration_{model_tag}_{horizon_tag}.png",
        title=f"Quantile calibration: {model}, horizons {horizon_label}",
    )

    # -----------------------------
    # Interval coverage table
    # -----------------------------
    interval_df = build_interval_coverage_table(
        A,
        intervals=intervals,
        horizons_ahead=horizons_ahead,
    )

    interval_df.to_csv(
        out_dir / f"section_524_interval_coverage_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    latex = format_interval_table_for_latex(interval_df, decimals=4)
    md = format_interval_table_for_markdown(interval_df, decimals=4)

    with open(out_dir / f"section_524_interval_coverage_{model_tag}_{horizon_tag}.tex", "w") as f:
        f.write(latex)

    with open(out_dir / f"section_524_interval_coverage_{model_tag}_{horizon_tag}.md", "w") as f:
        f.write(md)

    print("\n==============================================")
    print(f"Section 5.2.4: Calibration outputs for {model}")
    print(f"Horizons: {horizon_label}")
    print("==============================================")

    print("\nInterval coverage table:")
    print(md)

    print("\nPIT summary:")
    if len(pit) > 0:
        print(f"N PIT values: {len(pit)}")
        print(f"Mean PIT: {np.mean(pit):.4f}")
        print(f"Std PIT: {np.std(pit):.4f}")
        print(f"Expected uniform std: {np.sqrt(1/12):.4f}")
    else:
        print("No valid PIT values.")

    print("\nQuantile calibration summary:")
    mean_abs_cal_error = float(np.nanmean(calib_df["Abs calibration error"].values))
    max_abs_cal_error = float(np.nanmax(calib_df["Abs calibration error"].values))
    print(f"Mean absolute calibration error: {mean_abs_cal_error:.4f}")
    print(f"Max absolute calibration error: {max_abs_cal_error:.4f}")

    return {
        "runs": runs,
        "pit": pit,
        "pit_df": pit_df,
        "quantile_calibration": calib_df,
        "interval_coverage": interval_df,
        "out_dir": out_dir,
    }


# ============================================================
# Example usage
# ============================================================

model_paths = {
    "SAINT": "3328/a4b021c593f044fabee4a9207a5d090f/preds_test_set.pkl",
    "TimeXer": "3328/97e7458094704027a289909bc87a8058/preds_test_set.pkl",
    "TiDE": "./benchmark_outputs/tide_hist_exog/preds_test_set.pkl",
    "GARCH": "./benchmark_outputs/garch_student_t/preds_test_set.pkl",
    "ARIMA": "./arima_benchmark_eval/arima_preds_test_set.pkl",
    "Naive": "./benchmark_outputs/naive/preds_test_set.pkl",
}

section_524_result = make_section_524_calibration_outputs(
    model_paths=model_paths,
    out_dir="./comparison_nn_vs_arima/section_524_calibration",
    model="SAINT",
    horizons_ahead="all",
    intervals=(0.50, 0.80, 0.90, 0.95, 0.98),
    pit_bins=20,
)

interval_coverage_table = section_524_result["interval_coverage"]
quantile_calibration_table = section_524_result["quantile_calibration"]
pit_values = section_524_result["pit"]
