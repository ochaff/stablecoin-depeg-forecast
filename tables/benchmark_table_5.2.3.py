# ============================================================
# Section 5.2.3:
# Tail-risk and depeg-probability evaluation
# ============================================================
#
# Assumes utilities from previous sections are available:
#   - load_forecast_pickles
#   - check_common_grid_and_shape
#   - compute_abs_threshold_event_prob
#   - _resolve_horizons
#
# Main outputs:
#   1. section_523_depeg_risk_by_threshold_<horizon>.csv
#   2. section_523_depeg_risk_by_threshold_<horizon>.tex
#   3. section_523_depeg_risk_by_threshold_<horizon>.md
#   4. section_523_depeg_risk_all_models_<horizon>.csv
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from tables.benchmark_utils import (
    load_forecast_pickles,
    check_common_grid_and_shape,
    compute_abs_threshold_event_prob,
    _resolve_horizons,
)

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None


# ------------------------------------------------------------
# 1. Helper functions
# ------------------------------------------------------------

def _safe_logit(p, eps=1e-12):
    """
    Numerically stable logit transform.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _flatten_event_forecasts_for_threshold(A, abs_threshold, h_idx, eps=1e-12):
    """
    Return flattened predicted event probabilities and binary event outcomes
    for |Y| >= abs_threshold over selected horizons.
    """
    y = np.asarray(A["true"], dtype=np.float64)

    p_event = compute_abs_threshold_event_prob(
        A,
        abs_threshold=abs_threshold,
    )

    event = (np.abs(y) >= abs_threshold).astype(int)

    p = p_event[:, h_idx].reshape(-1)
    e = event[:, h_idx].reshape(-1)

    mask = np.isfinite(p) & np.isfinite(e)

    p = np.clip(p[mask], eps, 1.0 - eps)
    e = e[mask].astype(int)

    return p, e


def _expected_calibration_error_binary(
    p,
    e,
    n_bins=10,
    strategy="quantile",
):
    """
    Expected calibration error for binary probabilities.

    Parameters
    ----------
    strategy : {"quantile", "uniform"}
        quantile:
            bins contain approximately equal numbers of observations.
            This is usually more stable for rare depeg events.
        uniform:
            bins are fixed probability intervals on [0, 1].
    """
    p = np.asarray(p, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)

    mask = np.isfinite(p) & np.isfinite(e)
    p = p[mask]
    e = e[mask]

    n = len(p)
    if n == 0:
        return np.nan

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(p, qs)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0

        # If probabilities are heavily tied, remove duplicate edges.
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) <= 2:
            return np.nan
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'.")

    ece = 0.0

    for j in range(len(bin_edges) - 1):
        lo = bin_edges[j]
        hi = bin_edges[j + 1]

        if j == len(bin_edges) - 2:
            idx = (p >= lo) & (p <= hi)
        else:
            idx = (p >= lo) & (p < hi)

        nj = int(np.sum(idx))

        if nj == 0:
            continue

        avg_p = float(np.mean(p[idx]))
        avg_e = float(np.mean(e[idx]))

        ece += (nj / n) * abs(avg_p - avg_e)

    return float(ece)


def _top_decile_lift(p, e, top_frac=0.10):
    """
    Lift in the top predicted-risk decile.

    lift = event rate among top 10% predicted probabilities / unconditional event rate.
    """
    p = np.asarray(p, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)

    mask = np.isfinite(p) & np.isfinite(e)
    p = p[mask]
    e = e[mask]

    n = len(p)

    if n == 0:
        return {
            "Top-decile event rate": np.nan,
            "Top-decile lift": np.nan,
            "Top-decile N": 0,
        }

    base_rate = float(np.mean(e))

    if base_rate <= 0:
        return {
            "Top-decile event rate": np.nan,
            "Top-decile lift": np.nan,
            "Top-decile N": 0,
        }

    k = max(1, int(np.ceil(top_frac * n)))

    # Exact top-k by predicted risk
    top_idx = np.argpartition(-p, k - 1)[:k]

    top_rate = float(np.mean(e[top_idx]))
    lift = float(top_rate / base_rate)

    return {
        "Top-decile event rate": top_rate,
        "Top-decile lift": lift,
        "Top-decile N": int(k),
    }


def _calibration_slope_intercept(p, e):
    """
    Logistic calibration regression:

        logit Pr(Y = 1) = intercept + slope * logit(p_hat)

    Returns calibration intercept and slope.

    Perfect calibration corresponds approximately to:
        intercept = 0
        slope = 1
    """
    p = np.asarray(p, dtype=np.float64)
    e = np.asarray(e, dtype=int)

    mask = np.isfinite(p) & np.isfinite(e)
    p = p[mask]
    e = e[mask]

    if len(p) == 0 or len(np.unique(e)) < 2:
        return np.nan, np.nan

    x = _safe_logit(p).reshape(-1, 1)

    if np.nanstd(x) <= 1e-12:
        return np.nan, np.nan

    if LogisticRegression is None:
        return np.nan, np.nan

    try:
        # Near-unregularized logistic calibration.
        clf = LogisticRegression(
            penalty="l2",
            C=1e6,
            solver="lbfgs",
            max_iter=2000,
        )
        clf.fit(x, e)

        intercept = float(clf.intercept_[0])
        slope = float(clf.coef_[0, 0])

        return intercept, slope

    except Exception:
        return np.nan, np.nan


def _binary_event_metrics(p, e, n_bins_ece=10):
    """
    Compute event-probability forecast metrics.
    """
    p = np.asarray(p, dtype=np.float64)
    e = np.asarray(e, dtype=int)

    mask = np.isfinite(p) & np.isfinite(e)
    p = p[mask]
    e = e[mask]

    n = len(e)

    if n == 0:
        return {
            "Event rate": np.nan,
            "Mean p": np.nan,
            "ROC-AUC": np.nan,
            "PR-AUC": np.nan,
            "Brier": np.nan,
            "LogLoss": np.nan,
            "ECE": np.nan,
            "Calibration intercept": np.nan,
            "Calibration slope": np.nan,
            "Top-decile event rate": np.nan,
            "Top-decile lift": np.nan,
            "N events": 0,
            "N obs": 0,
        }

    event_rate = float(np.mean(e))
    mean_p = float(np.mean(p))

    brier = float(np.mean((p - e) ** 2))

    eps = 1e-12
    p_clip = np.clip(p, eps, 1.0 - eps)
    logloss = float(-np.mean(e * np.log(p_clip) + (1 - e) * np.log(1 - p_clip)))

    if roc_auc_score is not None and len(np.unique(e)) == 2:
        roc_auc = float(roc_auc_score(e, p))
    else:
        roc_auc = np.nan

    if average_precision_score is not None and np.sum(e) > 0:
        pr_auc = float(average_precision_score(e, p))
    else:
        pr_auc = np.nan

    ece = _expected_calibration_error_binary(
        p,
        e,
        n_bins=n_bins_ece,
        strategy="quantile",
    )

    cal_intercept, cal_slope = _calibration_slope_intercept(p, e)

    lift_stats = _top_decile_lift(p, e, top_frac=0.10)

    return {
        "Event rate": event_rate,
        "Mean p": mean_p,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Brier": brier,
        "LogLoss": logloss,
        "ECE": ece,
        "Calibration intercept": cal_intercept,
        "Calibration slope": cal_slope,
        **lift_stats,
        "N events": int(np.sum(e)),
        "N obs": int(n),
    }


# ------------------------------------------------------------
# 2. Build threshold table for one model
# ------------------------------------------------------------

def build_depeg_risk_threshold_table(
    A,
    thresholds_bps=(5, 10, 15, 25, 50),
    horizons_ahead=24,
    n_bins_ece=10,
):
    """
    Build the Section 5.2.3 table for one model.

    Event:
        |Y| >= threshold

    Parameters
    ----------
    A : dict
        Forecast dictionary for one model.
    thresholds_bps : iterable
        Absolute depeg thresholds in bps.
    horizons_ahead : "all", None, int, or iterable[int]
        Horizons over which event metrics are evaluated.
        The text usually emphasizes h=24.
    """
    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    rows = []

    for thr in thresholds_bps:
        p, e = _flatten_event_forecasts_for_threshold(
            A,
            abs_threshold=float(thr),
            h_idx=h_idx,
        )

        metrics = _binary_event_metrics(
            p,
            e,
            n_bins_ece=n_bins_ece,
        )

        row = {
            "Threshold": f"{thr:g} bps",
            "Threshold bps": float(thr),
            "Horizons": horizon_label,
            **metrics,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


# ------------------------------------------------------------
# 3. Build long table for all models and thresholds
# ------------------------------------------------------------

def build_depeg_risk_all_models_table(
    runs,
    thresholds_bps=(5, 10, 15, 25, 50),
    horizons_ahead=24,
    n_bins_ece=10,
):
    """
    Build a long-format depeg-risk table for all models.
    Useful for appendix or robustness checks.
    """
    rows = []

    for model_name, A in runs.items():
        df_model = build_depeg_risk_threshold_table(
            A,
            thresholds_bps=thresholds_bps,
            horizons_ahead=horizons_ahead,
            n_bins_ece=n_bins_ece,
        )
        df_model.insert(0, "Model", model_name)
        rows.append(df_model)

    return pd.concat(rows, axis=0, ignore_index=True)


# ------------------------------------------------------------
# 4. Formatting
# ------------------------------------------------------------

def format_depeg_threshold_table_for_latex(
    df,
    decimals=4,
    keep_cols=(
        "Threshold",
        "Event rate",
        "ROC-AUC",
        "PR-AUC",
        "Brier",
        "Top-decile lift",
        "Calibration slope",
    ),
):
    """
    Format main Section 5.2.3 threshold table for LaTeX.
    """
    df_fmt = df.copy()
    df_fmt = df_fmt[list(keep_cols)]

    for col in df_fmt.columns:
        if col == "Threshold":
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


def format_depeg_threshold_table_for_markdown(
    df,
    decimals=4,
    keep_cols=(
        "Threshold",
        "Event rate",
        "ROC-AUC",
        "PR-AUC",
        "Brier",
        "Top-decile lift",
        "Calibration slope",
    ),
):
    """
    Format main Section 5.2.3 threshold table for Markdown.
    """
    df_fmt = df.copy()
    df_fmt = df_fmt[list(keep_cols)]

    for col in df_fmt.columns:
        if col == "Threshold":
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
# 5. Convenience wrapper for Section 5.2.3
# ------------------------------------------------------------

def make_section_523_depeg_risk_outputs(
    model_paths=None,
    runs=None,
    out_dir="./comparison_nn_vs_arima/section_523_depeg_risk",
    main_model="SAINT",
    thresholds_bps=(5, 10, 15, 25, 50),
    horizons_ahead=24,
    n_bins_ece=10,
):
    """
    Create Section 5.2.3 depeg-risk tables.

    Main table:
        Derived depeg-risk forecast performance by threshold

    By default this evaluates h=24, matching the interpretation:
        "top 10% of predicted 24-hour depeg probability".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if runs is None:
        if model_paths is None:
            raise ValueError("Pass either runs or model_paths.")
        runs = load_forecast_pickles(model_paths)
        check_common_grid_and_shape(runs)

    if main_model not in runs:
        raise ValueError(f"main_model={main_model} not found in runs.")

    h_idx, horizon_label = _resolve_horizons(runs[main_model], horizons_ahead)

    horizon_tag = horizon_label.replace(",", "_").replace("-", "_").replace("h", "h")

    # Main model threshold table
    df_main = build_depeg_risk_threshold_table(
        runs[main_model],
        thresholds_bps=thresholds_bps,
        horizons_ahead=horizons_ahead,
        n_bins_ece=n_bins_ece,
    )

    # All-model long table
    df_all_models = build_depeg_risk_all_models_table(
        runs,
        thresholds_bps=thresholds_bps,
        horizons_ahead=horizons_ahead,
        n_bins_ece=n_bins_ece,
    )

    # Save numeric CSVs
    df_main.to_csv(
        out_dir / f"section_523_depeg_risk_by_threshold_{horizon_tag}.csv",
        index=False,
    )
    df_all_models.to_csv(
        out_dir / f"section_523_depeg_risk_all_models_{horizon_tag}.csv",
        index=False,
    )

    # Save LaTeX and Markdown for main table
    latex = format_depeg_threshold_table_for_latex(df_main, decimals=4)
    md = format_depeg_threshold_table_for_markdown(df_main, decimals=4)

    with open(out_dir / f"section_523_depeg_risk_by_threshold_{horizon_tag}.tex", "w") as f:
        f.write(latex)

    with open(out_dir / f"section_523_depeg_risk_by_threshold_{horizon_tag}.md", "w") as f:
        f.write(md)

    print("\n==============================================")
    print(f"Section 5.2.3: Depeg-risk table for {main_model}")
    print(f"Horizons: {horizon_label}")
    print("==============================================")
    print(md)

    print("\nInterpretation helper:")
    for _, row in df_main.iterrows():
        thr = row["Threshold"]
        lift = row["Top-decile lift"]
        event_rate = row["Event rate"]

        if np.isfinite(lift):
            print(
                f"At {thr}, hours in the top 10% of predicted risk contain "
                f"{lift:.2f}x as many realized depegs as the unconditional base rate "
                f"of {event_rate:.4f}."
            )

    return {
        "runs": runs,
        "main_table": df_main,
        "all_models_table": df_all_models,
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

section_523_result = make_section_523_depeg_risk_outputs(
    model_paths=model_paths,
    out_dir="./comparison_nn_vs_arima/section_523_depeg_risk",
    main_model="SAINT",
    thresholds_bps=(5, 10, 15, 25, 50),
    horizons_ahead=24,
    n_bins_ece=10,
)

depeg_risk_table_h24 = section_523_result["main_table"]
depeg_risk_all_models_h24 = section_523_result["all_models_table"]
