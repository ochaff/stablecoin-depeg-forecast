# ============================================================
# Section 5.2.7:
# Tail backtesting: VaR and expected shortfall
# ============================================================
#
# Backtests lower and upper tail quantiles using:
#   - empirical violation rate
#   - Kupiec unconditional coverage test
#   - Christoffersen conditional coverage test
#   - simple ES diagnostic conditional on VaR violation
#
# Main outputs:
#   1. section_527_tail_backtesting_<model>_<horizon>.csv
#   2. section_527_tail_backtesting_<model>_<horizon>.tex
#   3. section_527_tail_backtesting_<model>_<horizon>.md
#   4. section_527_tail_backtesting_all_models_<horizon>.csv
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

from benchmark_utils import (
    load_forecast_pickles,
    check_common_grid_and_shape,
    quantiles_from_u_grid,
    _resolve_horizons,
    model_paths,
)

try:
    from scipy.stats import chi2
except Exception:
    chi2 = None


# ------------------------------------------------------------
# 1. Likelihood-ratio tests for VaR violations
# ------------------------------------------------------------

def _safe_log(x, eps=1e-12):
    return np.log(np.clip(x, eps, 1.0))


def kupiec_uc_test(violations, p0):
    """
    Kupiec unconditional coverage test.

    Null:
        violation probability equals p0.

    Parameters
    ----------
    violations : array-like of bool/int
        1 if VaR violation occurs.
    p0 : float
        Nominal violation probability.

    Returns
    -------
    dict
        LR statistic, p-value, empirical violation rate, N violations, N obs.
    """
    v = np.asarray(violations).astype(int)
    v = v[np.isfinite(v)]

    n = len(v)
    x = int(np.sum(v))

    if n == 0:
        return {
            "Kupiec LR": np.nan,
            "Kupiec p-value": np.nan,
            "N violations": 0,
            "N obs": 0,
            "Empirical violation": np.nan,
        }

    phat = x / n

    # Log-likelihood under null and unrestricted Bernoulli.
    ll_null = x * _safe_log(p0) + (n - x) * _safe_log(1.0 - p0)

    if phat <= 0.0:
        ll_alt = (n - x) * _safe_log(1.0)
    elif phat >= 1.0:
        ll_alt = x * _safe_log(1.0)
    else:
        ll_alt = x * _safe_log(phat) + (n - x) * _safe_log(1.0 - phat)

    LR_uc = -2.0 * (ll_null - ll_alt)

    if chi2 is not None:
        p_value = float(chi2.sf(LR_uc, df=1))
    else:
        p_value = np.nan

    return {
        "Kupiec LR": float(LR_uc),
        "Kupiec p-value": p_value,
        "N violations": int(x),
        "N obs": int(n),
        "Empirical violation": float(phat),
    }


def christoffersen_cc_test(violations, p0):
    """
    Christoffersen conditional coverage test.

    Combines:
        - Kupiec unconditional coverage
        - independence of violations

    Null:
        correct unconditional coverage and independent violations.

    Returns
    -------
    dict
        Independence LR, independence p-value,
        conditional coverage LR, conditional coverage p-value,
        transition counts.
    """
    v = np.asarray(violations).astype(int)
    v = v[np.isfinite(v)]

    n = len(v)

    if n < 2:
        return {
            "Christoffersen independence LR": np.nan,
            "Christoffersen independence p-value": np.nan,
            "Christoffersen LR": np.nan,
            "Christoffersen p-value": np.nan,
            "n00": 0,
            "n01": 0,
            "n10": 0,
            "n11": 0,
        }

    v0 = v[:-1]
    v1 = v[1:]

    n00 = int(np.sum((v0 == 0) & (v1 == 0)))
    n01 = int(np.sum((v0 == 0) & (v1 == 1)))
    n10 = int(np.sum((v0 == 1) & (v1 == 0)))
    n11 = int(np.sum((v0 == 1) & (v1 == 1)))

    n0 = n00 + n01
    n1 = n10 + n11

    pi = (n01 + n11) / max(n0 + n1, 1)

    pi01 = n01 / n0 if n0 > 0 else np.nan
    pi11 = n11 / n1 if n1 > 0 else np.nan

    # Restricted likelihood: common transition probability pi
    ll_restricted = (
        (n00 + n10) * _safe_log(1.0 - pi)
        + (n01 + n11) * _safe_log(pi)
    )

    # Unrestricted first-order Markov likelihood
    ll_unrestricted = 0.0

    if n0 > 0:
        ll_unrestricted += n00 * _safe_log(1.0 - pi01) + n01 * _safe_log(pi01)

    if n1 > 0:
        ll_unrestricted += n10 * _safe_log(1.0 - pi11) + n11 * _safe_log(pi11)

    LR_ind = -2.0 * (ll_restricted - ll_unrestricted)

    uc = kupiec_uc_test(v, p0)
    LR_uc = uc["Kupiec LR"]

    LR_cc = LR_uc + LR_ind

    if chi2 is not None:
        p_ind = float(chi2.sf(LR_ind, df=1))
        p_cc = float(chi2.sf(LR_cc, df=2))
    else:
        p_ind = np.nan
        p_cc = np.nan

    return {
        "Christoffersen independence LR": float(LR_ind),
        "Christoffersen independence p-value": p_ind,
        "Christoffersen LR": float(LR_cc),
        "Christoffersen p-value": p_cc,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


# ------------------------------------------------------------
# 2. Expected shortfall from quantile grid
# ------------------------------------------------------------

def expected_shortfall_from_quantile_grid(
    Q_all,
    u_grid,
    tau,
    tail,
    n_grid=200,
):
    """
    Approximate forecast expected shortfall from quantile function.

    Lower tail:
        ES_tau = E[Y | Y <= VaR_tau]
               = (1/tau) int_0^tau Q(u) du

    Upper tail:
        ES_tau = E[Y | Y >= VaR_tau]
               = (1/(1-tau)) int_tau^1 Q(u) du

    Parameters
    ----------
    Q_all : np.ndarray
        Forecast quantiles, shape (B,H,J).
    u_grid : np.ndarray
        Quantile grid, shape (J,).
    tau : float
        Quantile level.
    tail : {"lower", "upper"}

    Returns
    -------
    es : np.ndarray
        Forecast ES, shape (B,H).
    """
    Q = np.asarray(Q_all, dtype=np.float64)
    u = np.asarray(u_grid, dtype=np.float64)

    B, H, J = Q.shape

    order = np.argsort(u)
    u_sorted = u[order]
    Q_sorted = Q[..., order]

    es = np.full((B, H), np.nan, dtype=np.float64)

    if tail == "lower":
        a, b = 0.0, float(tau)
        denom = tau
    elif tail == "upper":
        a, b = float(tau), 1.0
        denom = 1.0 - tau
    else:
        raise ValueError("tail must be 'lower' or 'upper'.")

    if denom <= 0:
        return es

    integ_grid = np.linspace(a, b, n_grid)

    for i in range(B):
        for h in range(H):
            q = Q_sorted[i, h, :]

            mask = np.isfinite(q) & np.isfinite(u_sorted)

            if np.sum(mask) < 2:
                continue

            uu = u_sorted[mask]
            qq = q[mask]

            qq = np.maximum.accumulate(qq)

            q_interp = np.interp(
                integ_grid,
                uu,
                qq,
                left=qq[0],
                right=qq[-1],
            )

            es[i, h] = np.trapz(q_interp, integ_grid) / denom

    return es


# ------------------------------------------------------------
# 3. Backtest one model
# ------------------------------------------------------------

def _flatten_for_backtest(y, q, h_idx):
    """
    Flatten selected horizons for unconditional tests.

    If h_idx contains one horizon, order is chronological in forecast origin.
    If h_idx contains multiple horizons, data are flattened by origin/horizon.
    """
    y_sel = y[:, h_idx]
    q_sel = q[:, h_idx]

    mask = np.isfinite(y_sel) & np.isfinite(q_sel)

    return y_sel[mask], q_sel[mask]


def _flatten_violations_chronological(y, q, h_idx, tail):
    """
    Return violations in an order suitable for Christoffersen test.

    Best practice:
        use a single horizon, e.g. horizons_ahead=24.

    For multiple horizons, violations are concatenated horizon-by-horizon.
    """
    viols = []

    for h in h_idx:
        yy = y[:, h]
        qq = q[:, h]

        mask = np.isfinite(yy) & np.isfinite(qq)

        if tail == "lower":
            v = yy[mask] < qq[mask]
        else:
            v = yy[mask] > qq[mask]

        viols.append(v.astype(int))

    if len(viols) == 0:
        return np.array([], dtype=int)

    return np.concatenate(viols)


def build_tail_backtesting_table(
    A,
    horizons_ahead=24,
    quantile_levels=(0.01, 0.025, 0.05, 0.95, 0.975, 0.99),
    n_grid_es=200,
):
    """
    Build VaR and ES backtesting table for one model.

    For lower quantiles tau < 0.5:
        violation = Y < Q_tau
        nominal violation = tau

    For upper quantiles tau > 0.5:
        violation = Y > Q_tau
        nominal violation = 1 - tau
    """
    y = np.asarray(A["true"], dtype=np.float64)
    Q = np.asarray(A["Q"], dtype=np.float64)
    u = np.asarray(A["u_grid"], dtype=np.float64)

    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    taus = np.asarray(quantile_levels, dtype=np.float64)

    qhat_all = quantiles_from_u_grid(Q, u, taus)

    rows = []

    for j, tau in enumerate(taus):
        tau = float(tau)

        if tau < 0.5:
            tail = "Lower"
            nominal_violation = tau
            violation_rule = "Y < VaR"
            qhat = qhat_all[:, :, j]
            violations_ordered = _flatten_violations_chronological(
                y,
                qhat,
                h_idx,
                tail="lower",
            )

            es_forecast = expected_shortfall_from_quantile_grid(
                Q,
                u,
                tau=tau,
                tail="lower",
                n_grid=n_grid_es,
            )

        elif tau > 0.5:
            tail = "Upper"
            nominal_violation = 1.0 - tau
            violation_rule = "Y > VaR"
            qhat = qhat_all[:, :, j]
            violations_ordered = _flatten_violations_chronological(
                y,
                qhat,
                h_idx,
                tail="upper",
            )

            es_forecast = expected_shortfall_from_quantile_grid(
                Q,
                u,
                tau=tau,
                tail="upper",
                n_grid=n_grid_es,
            )

        else:
            continue

        # Flatten y, q and ES for diagnostic summaries.
        y_sel = y[:, h_idx]
        q_sel = qhat[:, h_idx]
        es_sel = es_forecast[:, h_idx]

        mask = np.isfinite(y_sel) & np.isfinite(q_sel)

        if tail == "Lower":
            viol_mask = mask & (y_sel < q_sel)
        else:
            viol_mask = mask & (y_sel > q_sel)

        uc = kupiec_uc_test(
            violations_ordered,
            p0=nominal_violation,
        )

        cc = christoffersen_cc_test(
            violations_ordered,
            p0=nominal_violation,
        )

        # Simple ES diagnostic among VaR violations.
        if np.sum(viol_mask) > 0:
            realized_tail_mean = float(np.mean(y_sel[viol_mask]))
            forecast_es_mean = float(np.nanmean(es_sel[viol_mask]))
            es_error = realized_tail_mean - forecast_es_mean
        else:
            realized_tail_mean = np.nan
            forecast_es_mean = np.nan
            es_error = np.nan

        row = {
            "Tail": tail,
            "Quantile": tau,
            "Nominal violation": nominal_violation,
            "Empirical violation": uc["Empirical violation"],
            "Kupiec p-value": uc["Kupiec p-value"],
            "Christoffersen p-value": cc["Christoffersen p-value"],
            "Kupiec LR": uc["Kupiec LR"],
            "Christoffersen LR": cc["Christoffersen LR"],
            "Christoffersen independence p-value": cc["Christoffersen independence p-value"],
            "N violations": uc["N violations"],
            "N obs": uc["N obs"],
            "Violation rule": violation_rule,
            "Mean realized tail outcome": realized_tail_mean,
            "Mean forecast ES on violations": forecast_es_mean,
            "ES error realized-minus-forecast": es_error,
            "Horizons": horizon_label,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


# ------------------------------------------------------------
# 4. All-model table
# ------------------------------------------------------------

def build_tail_backtesting_all_models_table(
    runs,
    horizons_ahead=24,
    quantile_levels=(0.01, 0.025, 0.05, 0.95, 0.975, 0.99),
    n_grid_es=200,
):
    """
    Long-format tail backtesting table for all models.
    """
    out = []

    for model_name, A in runs.items():
        df_model = build_tail_backtesting_table(
            A,
            horizons_ahead=horizons_ahead,
            quantile_levels=quantile_levels,
            n_grid_es=n_grid_es,
        )
        df_model.insert(0, "Model", model_name)
        out.append(df_model)

    return pd.concat(out, axis=0, ignore_index=True)


# ------------------------------------------------------------
# 5. Formatting
# ------------------------------------------------------------

def format_tail_backtesting_table_for_latex(
    df,
    decimals=4,
    keep_cols=(
        "Tail",
        "Quantile",
        "Nominal violation",
        "Empirical violation",
        "Kupiec p-value",
        "Christoffersen p-value",
    ),
):
    """
    Format main tail-backtesting table for LaTeX.
    """
    df_fmt = df.copy()
    df_fmt = df_fmt[list(keep_cols)]

    for col in df_fmt.columns:
        if col == "Tail":
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


def format_tail_backtesting_table_for_markdown(
    df,
    decimals=4,
    keep_cols=(
        "Tail",
        "Quantile",
        "Nominal violation",
        "Empirical violation",
        "Kupiec p-value",
        "Christoffersen p-value",
    ),
):
    """
    Format main tail-backtesting table for Markdown.
    """
    df_fmt = df.copy()
    df_fmt = df_fmt[list(keep_cols)]

    for col in df_fmt.columns:
        if col == "Tail":
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
# 6. Convenience wrapper
# ------------------------------------------------------------

def make_section_527_tail_backtesting_outputs(
    model_paths=None,
    runs=None,
    out_dir="./comparison_nn_vs_arima/section_527_tail_backtesting",
    model="SAINT",
    horizons_ahead=24,
    quantile_levels=(0.01, 0.025, 0.05, 0.95, 0.975, 0.99),
    n_grid_es=200,
):
    """
    Create tail-risk backtesting outputs.

    Recommended:
        Use a single horizon, especially h=24, for Christoffersen tests.
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

    _, horizon_label = _resolve_horizons(A, horizons_ahead)
    horizon_tag = horizon_label.replace(",", "_").replace("-", "_")
    model_tag = model.replace(" ", "_").replace("/", "_")

    df_main = build_tail_backtesting_table(
        A,
        horizons_ahead=horizons_ahead,
        quantile_levels=quantile_levels,
        n_grid_es=n_grid_es,
    )

    df_all = build_tail_backtesting_all_models_table(
        runs,
        horizons_ahead=horizons_ahead,
        quantile_levels=quantile_levels,
        n_grid_es=n_grid_es,
    )

    # Save numeric outputs.
    df_main.to_csv(
        out_dir / f"section_527_tail_backtesting_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    df_all.to_csv(
        out_dir / f"section_527_tail_backtesting_all_models_{horizon_tag}.csv",
        index=False,
    )

    # Save paper-style LaTeX and Markdown.
    latex = format_tail_backtesting_table_for_latex(df_main, decimals=4)
    md = format_tail_backtesting_table_for_markdown(df_main, decimals=4)

    with open(out_dir / f"section_527_tail_backtesting_{model_tag}_{horizon_tag}.tex", "w") as f:
        f.write(latex)

    with open(out_dir / f"section_527_tail_backtesting_{model_tag}_{horizon_tag}.md", "w") as f:
        f.write(md)

    print("\n==============================================")
    print(f"Section 5.2.7: Tail backtesting for {model}")
    print(f"Horizon: {horizon_label}")
    print("==============================================")
    print(md)

    print("\nES diagnostic, realized-minus-forecast among VaR violations:")
    es_cols = [
        "Tail",
        "Quantile",
        "Mean realized tail outcome",
        "Mean forecast ES on violations",
        "ES error realized-minus-forecast",
    ]
    print(df_main[es_cols].to_string(index=False))

    if chi2 is None:
        print(
            "\nWarning: scipy was not available, so Kupiec and Christoffersen "
            "p-values are NaN. Install scipy to enable p-values."
        )

    return {
        "runs": runs,
        "main_table": df_main,
        "all_models_table": df_all,
        "out_dir": out_dir,
    }


# ============================================================
# Example usage
# ============================================================


section_527_result = make_section_527_tail_backtesting_outputs(
    model_paths=model_paths,
    out_dir="./comparison_nn_vs_arima/section_527_tail_backtesting",
    model="SAINT",
    horizons_ahead=24,
    quantile_levels=(0.01, 0.025, 0.05, 0.95, 0.975, 0.99),
    n_grid_es=200,
)

tail_backtesting_table = section_527_result["main_table"]
tail_backtesting_all_models = section_527_result["all_models_table"]
