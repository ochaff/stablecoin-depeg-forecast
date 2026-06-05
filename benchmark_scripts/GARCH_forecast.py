# ============================================================
# Rolling GARCH benchmark with Student-t innovations
# Saves:
#   ./benchmark_outputs/garch_student_t/preds_test_set.pkl
#
# Requires:
#   pip install arch tqdm
#
# Key speed features:
#   - Refit GARCH parameters only every `refit_every` windows.
#   - Between refits, reuse fixed parameters and update the conditional
#     volatility filter on the expanded sample.
#   - tqdm progress bars.
#
# Default:
#   dist="t"  -> Student-t innovations.
# ============================================================

import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm, t as student_t
from tqdm.auto import tqdm

from arch import arch_model


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
# 1. Quantile grid
# ============================================================

def load_reference_u_grid(reference_pickle=None):
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
# 2. Helpers
# ============================================================

def save_preds_pickle(A, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pkl.dump(A, f)

    print(f"Saved: {path.resolve()}")


def make_arch_model(y_train, p=1, q=1, dist="t", mean="Constant", rescale=False):
    """
    Create an arch_model instance.

    dist:
      "t" / "student" / "student-t" -> Student-t innovations
      "normal"                      -> Gaussian innovations
    """
    dist_lower = dist.lower()

    if dist_lower in {"t", "student", "studentt", "student-t", "students-t"}:
        arch_dist = "StudentsT"
    elif dist_lower in {"normal", "gaussian"}:
        arch_dist = "normal"
    else:
        raise ValueError(f"Unknown dist={dist}")

    return arch_model(
        y_train,
        mean=mean,
        vol="GARCH",
        p=p,
        q=q,
        dist=arch_dist,
        rescale=rescale,
    )


def standardized_innovation_quantiles(u_grid, fitted_params, dist="t"):
    """
    Innovation quantiles compatible with arch forecast variance.

    For Student-t, arch uses a standardized Student-t distribution with unit variance.
    If T_nu has variance nu/(nu-2), the standardized quantile is:

        sqrt((nu-2)/nu) * t_nu^{-1}(u)

    Returns
    -------
    z : np.ndarray, shape (J,)
    nu : float or np.nan
    """
    dist_lower = dist.lower()

    if dist_lower in {"normal", "gaussian"}:
        return norm.ppf(u_grid), np.nan

    # Student-t case
    params = fitted_params

    nu = None
    for key in params.index:
        if key.lower() in {"nu", "eta"}:
            nu = float(params[key])
            break

    if nu is None or not np.isfinite(nu):
        nu = 8.0

    # Need nu > 2 for finite variance.
    nu = max(nu, 2.05)

    z_raw = student_t.ppf(u_grid, df=nu)

    # Standardize to variance one.
    z = np.sqrt((nu - 2.0) / nu) * z_raw

    return z, nu


# ============================================================
# 3. Main rolling GARCH function
# ============================================================

def fit_save_garch_student_t_benchmark(
    y_full,
    ds_full,
    train_size,
    h,
    n_windows,
    u_grid,
    out_path="./benchmark_outputs/garch_student_t/preds_test_set.pkl",
    p=1,
    q=1,
    dist="t",
    mean="Constant",
    refit_every=24,
    scale_y=1.0,
    rescale=False,
    optimizer_update_freq=0,
):
    """
    Rolling expanding-window GARCH benchmark.

    Parameters
    ----------
    y_full:
        Full target array.

    ds_full:
        Full datetime array.

    train_size:
        Number of observations before the test period.

    h:
        Forecast horizon.

    n_windows:
        Number of rolling forecast windows.

    u_grid:
        Quantile grid.

    dist:
        "t" uses Student-t innovations.
        "normal" uses Gaussian innovations.

    refit_every:
        Refit GARCH parameters every this many rolling windows.
        Larger values are much faster.
        Suggested:
          1    -> exact rolling refit, very slow
          24   -> daily parameter refit for hourly data
          168  -> weekly parameter refit, much faster

    scale_y:
        Optional scaling for numerical stability.
        If y is in bps, scale_y=1.0 is usually fine.

    rescale:
        Passed to arch_model. If arch complains about scaling, try rescale=True.

    Notes
    -----
    Between refits, this function fixes the latest estimated GARCH parameters
    and re-filters volatility on the expanding sample. This is substantially faster
    than fully re-optimizing at every forecast origin while still updating the
    conditional variance with newly observed data.
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
    current_res = None

    n_chunks = int(np.ceil(B / refit_every))

    outer = tqdm(
        range(0, B, refit_every),
        total=n_chunks,
        desc=f"GARCH({p},{q}) rolling chunks, dist={dist}, refit_every={refit_every}",
    )

    for chunk_start in outer:
        chunk_end = min(chunk_start + refit_every, B)

        # ----------------------------------------------------
        # Refit parameters at the first origin of the chunk
        # ----------------------------------------------------
        fit_end = train_size + chunk_start
        y_train_chunk = y_scaled[:fit_end]

        am = make_arch_model(
            y_train_chunk,
            p=p,
            q=q,
            dist=dist,
            mean=mean,
            rescale=rescale,
        )

        try:
            current_res = am.fit(
                disp="off",
                update_freq=optimizer_update_freq,
                show_warning=False,
            )
        except TypeError:
            current_res = am.fit(disp="off", show_warning=False)

        current_params = current_res.params.copy()
        did_refit[chunk_start] = True

        z, nu = standardized_innovation_quantiles(
            u_grid=u_grid,
            fitted_params=current_params,
            dist=dist,
        )

        outer.set_postfix({"nu": f"{nu:.2f}" if np.isfinite(nu) else "NA"})

        # ----------------------------------------------------
        # Forecast all origins in this chunk using fixed params
        # ----------------------------------------------------
        inner = range(chunk_start, chunk_end)

        for b in tqdm(inner, desc=f"chunk {chunk_start}-{chunk_end-1}", leave=False):
            fit_end_b = train_size + b
            forecast_start = fit_end_b
            forecast_end = fit_end_b + h

            # Store realized future path
            true[b, :] = y_scaled[forecast_start:forecast_end] / scale_y
            ds_mat[b, :] = ds_full[forecast_start:forecast_end]
            cutoffs[b] = ds_full[fit_end_b - 1]

            # Re-filter the GARCH state on the expanded sample, using fixed params
            y_train_b = y_scaled[:fit_end_b]

            am_b = make_arch_model(
                y_train_b,
                p=p,
                q=q,
                dist=dist,
                mean=mean,
                rescale=rescale,
            )

            fixed_res = am_b.fix(current_params)

            fc = fixed_res.forecast(horizon=h, reindex=False)

            mu = np.asarray(fc.mean.values[-1, :], dtype=np.float64)
            var = np.asarray(fc.variance.values[-1, :], dtype=np.float64)
            sigma = np.sqrt(np.maximum(var, 1e-12))

            Q[b, :, :] = (mu[:, None] + sigma[:, None] * z[None, :]) / scale_y
            nu_used[b] = nu

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
        "model_col": f"GARCH({p},{q})_{dist}",
        "garch_config": {
            "p": p,
            "q": q,
            "dist": dist,
            "mean": mean,
            "refit_every": refit_every,
            "scale_y": scale_y,
            "rescale": rescale,
        },
    }

    save_preds_pickle(A, out_path)

    return A


# ============================================================
# 4. Run GARCH Student-t benchmark
# ============================================================

A_garch = fit_save_garch_student_t_benchmark(
    y_full=y_full,
    ds_full=ds_full,
    train_size=train_size,
    h=h,
    n_windows=n_windows,
    u_grid=u_grid,

    # Output
    out_path="./benchmark_outputs/garch_student_t/preds_test_set.pkl",

    # GARCH specification
    p=1,
    q=1,
    dist="t",              # Student-t innovations
    mean="Constant",

    # Speed/accuracy trade-off
    refit_every=20000,        # daily refit for hourly data; use 168 for faster weekly refit

    # Numerical settings
    scale_y=1.0,
    rescale=False,
)

print("Saved GARCH Student-t benchmark:")
print("./benchmark_outputs/garch_student_t/preds_test_set.pkl")