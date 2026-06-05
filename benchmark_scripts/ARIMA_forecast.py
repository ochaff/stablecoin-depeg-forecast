# ============================================================
# Rolling SARIMA benchmark with fixed orders and Student-t predictive innovations
# Saves:
#   ./benchmark_outputs/sarima_student_t_fixed/preds_test_set.pkl
#
# Fixed orders:
#   order          = (5, 0, 1)
#   seasonal_order = (0, 0, 2, 24)
#
# Requires:
#   pip install statsmodels scipy tqdm
#
# Important:
#   statsmodels SARIMAX is fitted by Gaussian quasi-MLE.
#   This script then forms Student-t predictive quantiles using:
#       forecast mean + forecast standard error * standardized t-quantile.
#
#   Therefore this is a SARIMA model with Student-t predictive innovations,
#   not a full Student-t likelihood SARIMA.
# ============================================================

import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm, t as student_t
from tqdm.auto import tqdm

from statsmodels.tsa.statespace.sarimax import SARIMAX


# ============================================================
# 0. Data setup
# ============================================================

df = pd.read_parquet("./preprocessed_datasets/dataset_alpha_0.1_full.parquet").copy()

df["ds"] = df.index.tz_localize(None)
df["y"] = df["depeg_bps"]
df["unique_id"] = "stablecoin_depeg"

Y_df = df[["unique_id", "ds", "y"]].reset_index(drop=True)

h = 24
test_size = Y_df.shape[0] - int(0.7 * Y_df.shape[0])
n_windows = test_size - h + 1
train_size = len(Y_df) - test_size

y_full = Y_df["y"].to_numpy(dtype=np.float64)
ds_full = pd.to_datetime(Y_df["ds"]).to_numpy(dtype="datetime64[ns]")

print("Y_df:", Y_df.shape)
print("train_size:", train_size)
print("test_size:", test_size)
print("h:", h)
print("n_windows:", n_windows)


# ============================================================
# 1. Fixed SARIMA specification
# ============================================================

ORDER = (5, 0, 1)
SEASONAL_ORDER = (0, 0, 2, 24)

print("Using fixed SARIMA order:", ORDER)
print("Using fixed seasonal order:", SEASONAL_ORDER)


# ============================================================
# 2. Quantile grid
# ============================================================

def load_reference_u_grid(reference_pickle=None):
    """
    Use the same quantile grid as your main SAINT model if available.
    Otherwise use a tail-focused default grid.
    """
    if reference_pickle is not None and Path(reference_pickle).exists():
        with open(reference_pickle, "rb") as f:
            A = pkl.load(f)

        u_grid = np.asarray(A["u_grid"], dtype=np.float64)
        print(f"Loaded reference u_grid from {reference_pickle}: J={len(u_grid)}")
        return u_grid

    u_grid = np.array(
        [
            0.001, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05,
            0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
            0.95, 0.975, 0.98, 0.99, 0.995, 0.9975, 0.999,
        ],
        dtype=np.float64,
    )

    print(f"Using default u_grid: J={len(u_grid)}")
    return u_grid


REFERENCE_PICKLE = "3328/a4b021c593f044fabee4a9207a5d090f/preds_test_set.pkl"
u_grid = load_reference_u_grid(REFERENCE_PICKLE)


# ============================================================
# 3. Helpers
# ============================================================

def save_preds_pickle(A, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pkl.dump(A, f)

    print(f"Saved: {path.resolve()}")


def make_sarimax_model(
    y_train,
    order=ORDER,
    seasonal_order=SEASONAL_ORDER,
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
):
    """
    Construct fixed-order SARIMAX model.

    trend:
      "c"  : constant / intercept
      "n"  : no trend
      "t"  : linear trend
      "ct" : constant + trend
    """
    return SARIMAX(
        endog=y_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
        simple_differencing=False,
    )


def estimate_student_df_from_residuals(
    resid,
    fallback_nu=8.0,
    min_nu=2.25,
    max_nu=200.0,
    max_fit_resid=5000,
):
    """
    Estimate Student-t degrees of freedom from standardized SARIMA residuals.

    Returns
    -------
    nu : float
    """
    resid = np.asarray(resid, dtype=np.float64)
    resid = resid[np.isfinite(resid)]

    if len(resid) < 100:
        return float(fallback_nu)

    # Remove extreme numerical artifacts from diffuse initialization / failed periods
    lo, hi = np.nanpercentile(resid, [0.1, 99.9])
    resid = resid[(resid >= lo) & (resid <= hi)]

    if len(resid) < 100:
        return float(fallback_nu)

    # Use recent residuals if long: faster and more relevant under regime shifts
    if len(resid) > max_fit_resid:
        resid = resid[-max_fit_resid:]

    resid = resid - np.mean(resid)
    sd = np.std(resid)

    if not np.isfinite(sd) or sd <= 1e-12:
        return float(fallback_nu)

    z = resid / sd

    try:
        nu_hat, loc_hat, scale_hat = student_t.fit(z, floc=0.0)

        if not np.isfinite(nu_hat):
            return float(fallback_nu)

        nu_hat = float(np.clip(nu_hat, min_nu, max_nu))
        return nu_hat

    except Exception:
        return float(fallback_nu)


def innovation_quantiles(
    u_grid,
    dist="t",
    nu=8.0,
    standardize_t_to_unit_variance=True,
):
    """
    Return standardized innovation quantiles.

    For Student-t:
      If standardize_t_to_unit_variance=True, use
          sqrt((nu-2)/nu) * t_nu^{-1}(u)
      so the innovation has unit variance.

    This is appropriate because statsmodels' forecast standard error already
    gives the forecast-error standard deviation.
    """
    dist_lower = dist.lower()

    if dist_lower in {"normal", "gaussian"}:
        return norm.ppf(u_grid)

    if dist_lower in {"t", "student", "studentt", "student-t", "students-t"}:
        nu = max(float(nu), 2.25)

        z = student_t.ppf(u_grid, df=nu)

        if standardize_t_to_unit_variance:
            z = np.sqrt((nu - 2.0) / nu) * z

        return z

    raise ValueError(f"Unknown dist={dist}")


def get_forecast_mean_and_se(fitted_res, h):
    """
    Extract forecast mean and forecast standard error from statsmodels result.
    """
    fc = fitted_res.get_forecast(steps=h)

    mean = np.asarray(fc.predicted_mean, dtype=np.float64)

    # statsmodels forecast-error variance
    var = np.asarray(fc.var_pred_mean, dtype=np.float64)
    se = np.sqrt(np.maximum(var, 1e-12))

    return mean, se


# ============================================================
# 4. Rolling fixed-order SARIMA benchmark
# ============================================================

def fit_save_fixed_sarima_student_t_benchmark(
    y_full,
    ds_full,
    train_size,
    h,
    n_windows,
    u_grid,
    out_path="./benchmark_outputs/sarima_student_t_fixed/preds_test_set.pkl",

    # Fixed SARIMA specification
    order=(5, 0, 1),
    seasonal_order=(0, 0, 2, 24),
    trend="c",

    # Predictive distribution
    dist="t",
    t_df_mode="fit_residuals",
    fixed_nu=8.0,
    standardize_t_to_unit_variance=True,

    # Rolling refit
    refit_every=24,

    # Numerical options
    scale_y=1.0,
    enforce_stationarity=False,
    enforce_invertibility=False,
    maxiter=200,
    use_previous_params_as_start=True,
):
    """
    Rolling expanding-window fixed-order SARIMA benchmark.

    Parameters
    ----------
    order:
        Non-seasonal ARIMA order. Here default is (5,0,1).

    seasonal_order:
        Seasonal order. Here default is (0,0,2,24).

    dist:
        "t" for Student-t predictive innovations.
        "normal" for Gaussian predictive innovations.

    t_df_mode:
        "fit_residuals":
            Estimate Student-t df from residuals at each refit.
        "fixed":
            Use fixed_nu.
        "df_resid":
            Use approximate residual degrees of freedom nobs-k_params.

    refit_every:
        Refit SARIMA parameters every this many rolling windows.
        Larger values are much faster.
        Suggested:
          1    exact rolling refit, very slow
          24   daily refit for hourly data
          168  weekly refit, much faster

    use_previous_params_as_start:
        If True, use the previous chunk's fitted parameters as starting values
        for the next optimization. This can substantially speed up rolling fits.
    """
    y_scaled = np.asarray(y_full, dtype=np.float64) * scale_y
    ds_full = np.asarray(ds_full)

    B = int(n_windows)
    H = int(h)
    J = len(u_grid)

    true = np.empty((B, H), dtype=np.float64)
    Q = np.empty((B, H, J), dtype=np.float64)
    ds_mat = np.empty((B, H), dtype="datetime64[ns]")
    cutoffs = np.empty(B, dtype="datetime64[ns]")

    nu_used = np.full(B, np.nan, dtype=np.float64)
    did_refit = np.zeros(B, dtype=bool)

    current_params = None
    current_nu = fixed_nu

    n_chunks = int(np.ceil(B / refit_every))

    print("Fixed SARIMA benchmark configuration:")
    print("  order:", order)
    print("  seasonal_order:", seasonal_order)
    print("  trend:", trend)
    print("  dist:", dist)
    print("  t_df_mode:", t_df_mode)
    print("  refit_every:", refit_every)

    outer = tqdm(
        range(0, B, refit_every),
        total=n_chunks,
        desc=f"SARIMA{order}x{seasonal_order} rolling chunks",
    )

    for chunk_start in outer:
        chunk_end = min(chunk_start + refit_every, B)

        # ----------------------------------------------------
        # Refit parameters at start of chunk
        # ----------------------------------------------------
        fit_end = train_size + chunk_start
        y_train_chunk = y_scaled[:fit_end]

        model = make_sarimax_model(
            y_train=y_train_chunk,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )

        fit_kwargs = dict(
            disp=False,
            maxiter=maxiter,
        )

        if (
            use_previous_params_as_start
            and current_params is not None
            and len(current_params) == len(model.start_params)
        ):
            fit_kwargs["start_params"] = current_params

        try:
            res = model.fit(**fit_kwargs)
        except Exception as e:
            print(f"\nFit failed at chunk_start={chunk_start}, retrying without start_params.")
            print("Error:", repr(e))

            fit_kwargs.pop("start_params", None)
            res = model.fit(**fit_kwargs)

        current_params = res.params.copy()
        did_refit[chunk_start] = True

        # ----------------------------------------------------
        # Student-t df for this chunk
        # ----------------------------------------------------
        dist_lower = dist.lower()

        if dist_lower in {"t", "student", "studentt", "student-t", "students-t"}:
            if t_df_mode == "fixed":
                current_nu = float(fixed_nu)

            elif t_df_mode == "df_resid":
                k_params = len(current_params)
                current_nu = max(float(res.nobs - k_params), 2.25)

            elif t_df_mode == "fit_residuals":
                current_nu = estimate_student_df_from_residuals(
                    res.resid,
                    fallback_nu=fixed_nu,
                )

            else:
                raise ValueError(
                    "t_df_mode must be one of: "
                    "'fit_residuals', 'fixed', 'df_resid'"
                )

            z = innovation_quantiles(
                u_grid=u_grid,
                dist="t",
                nu=current_nu,
                standardize_t_to_unit_variance=standardize_t_to_unit_variance,
            )

            outer.set_postfix({"nu": f"{current_nu:.2f}"})

        else:
            z = innovation_quantiles(
                u_grid=u_grid,
                dist="normal",
                nu=np.nan,
            )

            current_nu = np.nan
            outer.set_postfix({"nu": "NA"})

        # ----------------------------------------------------
        # Forecast all origins in the current chunk
        # using fixed parameters and updated filtered state
        # ----------------------------------------------------
        for b in tqdm(
            range(chunk_start, chunk_end),
            desc=f"chunk {chunk_start}-{chunk_end - 1}",
            leave=False,
        ):
            fit_end_b = train_size + b
            forecast_start = fit_end_b
            forecast_end = fit_end_b + h

            true[b, :] = y_scaled[forecast_start:forecast_end] / scale_y
            ds_mat[b, :] = ds_full[forecast_start:forecast_end]
            cutoffs[b] = ds_full[fit_end_b - 1]

            y_train_b = y_scaled[:fit_end_b]

            model_b = make_sarimax_model(
                y_train=y_train_b,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            )

            try:
                # Re-filter expanded sample with fixed parameters.
                # This updates the latent state without re-optimizing.
                fixed_res = model_b.filter(current_params)

            except Exception as e:
                # Fallback: fully refit this origin if filtering fails.
                print(f"\nFiltering failed at window b={b}, fully refitting.")
                print("Error:", repr(e))

                fixed_res = model_b.fit(
                    disp=False,
                    maxiter=maxiter,
                )

            mu, se = get_forecast_mean_and_se(fixed_res, h=h)

            Q[b, :, :] = (mu[:, None] + se[:, None] * z[None, :]) / scale_y
            nu_used[b] = current_nu if np.isfinite(current_nu) else np.nan

    # Ensure quantile monotonicity
    Q = np.maximum.accumulate(Q, axis=-1)

    A = {
        "true": true,
        "Q": Q,
        "u_grid": np.asarray(u_grid, dtype=np.float64),
        "ds": ds_mat,
        "meta": pd.DataFrame(
            {
                "cutoff": pd.to_datetime(cutoffs),
                "did_refit": did_refit,
                "nu_used": nu_used,
            }
        ),
        "model_col": f"SARIMA{order}x{seasonal_order}_{dist}",
        "sarima_config": {
            "order": order,
            "seasonal_order": seasonal_order,
            "trend": trend,
            "dist": dist,
            "t_df_mode": t_df_mode,
            "fixed_nu": fixed_nu,
            "standardize_t_to_unit_variance": standardize_t_to_unit_variance,
            "refit_every": refit_every,
            "scale_y": scale_y,
            "enforce_stationarity": enforce_stationarity,
            "enforce_invertibility": enforce_invertibility,
            "maxiter": maxiter,
            "use_previous_params_as_start": use_previous_params_as_start,
        },
    }

    save_preds_pickle(A, out_path)

    return A


# ============================================================
# 5. Run fixed SARIMA Student-t benchmark
# ============================================================

A_sarima_t = fit_save_fixed_sarima_student_t_benchmark(
    y_full=y_full,
    ds_full=ds_full,
    train_size=train_size,
    h=h,
    n_windows=n_windows,
    u_grid=u_grid,

    # Output
    out_path="./benchmark_outputs/sarima_student_t_fixed/preds_test_set.pkl",

    # Your selected orders
    order=(5, 0, 1),
    seasonal_order=(0, 0, 2, 24),
    trend="c",

    # Predictive distribution
    dist="t",
    t_df_mode="fit_residuals",   # alternatives: "fixed", "df_resid"
    fixed_nu=8.0,
    standardize_t_to_unit_variance=True,

    # Speed/accuracy trade-off
    refit_every=24,              # daily refit; use 168 for faster weekly refit

    # Numerical settings
    scale_y=1.0,
    enforce_stationarity=False,
    enforce_invertibility=False,
    maxiter=200,
    use_previous_params_as_start=True,
)

print("Saved fixed SARIMA Student-t benchmark:")
print("./benchmark_outputs/sarima_student_t_fixed/preds_test_set.pkl")