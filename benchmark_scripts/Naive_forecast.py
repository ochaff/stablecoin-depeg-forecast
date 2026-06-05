# ============================================================
# Fast Naive benchmark only
# Saves:
#   ./benchmark_outputs/naive/preds_test_set.pkl
# ============================================================

import re
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

from statsforecast import StatsForecast
from statsforecast.models import Naive


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

print("Y_df:", Y_df.shape)
print("h:", h)
print("test_size:", test_size)
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


def central_levels_from_u_grid(u_grid, decimals=6):
    levels = []
    for tau in np.asarray(u_grid, dtype=np.float64):
        if np.isclose(tau, 0.5):
            continue
        level = 100.0 * abs(2.0 * tau - 1.0)
        if 0.0 < level < 100.0:
            levels.append(round(float(level), decimals))
    return sorted(set(levels))


REFERENCE_PICKLE = "3328/a4b021c593f044fabee4a9207a5d090f/preds_test_set.pkl"
u_grid = load_reference_u_grid(REFERENCE_PICKLE)
levels = central_levels_from_u_grid(u_grid)

print("u_grid:", u_grid)
print("central interval levels:", levels)


# ============================================================
# 2. Conversion helpers
# ============================================================

def _safe_level_str(x):
    if np.isclose(x, round(x)):
        return str(int(round(x)))
    return f"{float(x):g}"


def find_interval_col(columns, interval_prefix, side, level, atol=1e-4):
    columns = list(columns)

    direct_candidates = [
        f"{interval_prefix}-{side}-{_safe_level_str(level)}",
        f"{interval_prefix}-{side}-{level}",
        f"{interval_prefix}-{side}-{float(level):.1f}",
        f"{interval_prefix}-{side}-{float(level):.2f}",
        f"{interval_prefix}-{side}-{float(level):.3f}",
        f"{interval_prefix}-{side}-{float(level):.4f}",
        f"{interval_prefix}-{side}-{float(level):.6f}",
    ]

    for c in direct_candidates:
        if c in columns:
            return c

    pattern = re.compile(
        rf"^{re.escape(interval_prefix)}-{re.escape(side)}-([0-9]+(?:\.[0-9]+)?)$"
    )

    matches = []
    for c in columns:
        m = pattern.match(c)
        if m is not None:
            lvl = float(m.group(1))
            matches.append((abs(lvl - float(level)), lvl, c))

    if matches:
        matches = sorted(matches, key=lambda z: z[0])
        if matches[0][0] <= atol:
            return matches[0][2]

    available = [c for c in columns if c.startswith(f"{interval_prefix}-{side}-")]
    raise KeyError(
        f"Could not find {interval_prefix}-{side}-{level}. "
        f"Available columns: {available[:20]}"
    )


def cv_df_to_quantile_pickle_dict(
    cv_df,
    u_grid,
    point_col="Naive",
    interval_prefix="Naive",
    enforce_monotone=True,
):
    cv_df = cv_df.copy()
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])

    cutoffs = np.array(sorted(cv_df["cutoff"].unique()))
    B = len(cutoffs)

    first = cv_df[cv_df["cutoff"] == cutoffs[0]].sort_values("ds")
    H = first.shape[0]
    J = len(u_grid)

    true = np.empty((B, H), dtype=np.float64)
    Q = np.empty((B, H, J), dtype=np.float64)
    ds_mat = np.empty((B, H), dtype="datetime64[ns]")

    for b, cutoff in enumerate(cutoffs):
        g = cv_df[cv_df["cutoff"] == cutoff].sort_values("ds")

        if g.shape[0] != H:
            raise ValueError(f"Cutoff {cutoff} has {g.shape[0]} rows, expected H={H}.")

        true[b, :] = g["y"].to_numpy(dtype=np.float64)
        ds_mat[b, :] = g["ds"].to_numpy(dtype="datetime64[ns]")

        point = g[point_col].to_numpy(dtype=np.float64)

        for j, tau in enumerate(u_grid):
            tau = float(tau)

            if np.isclose(tau, 0.5):
                Q[b, :, j] = point

            elif tau < 0.5:
                level = 100.0 * (1.0 - 2.0 * tau)
                col = find_interval_col(g.columns, interval_prefix, "lo", level)
                Q[b, :, j] = g[col].to_numpy(dtype=np.float64)

            else:
                level = 100.0 * (2.0 * tau - 1.0)
                col = find_interval_col(g.columns, interval_prefix, "hi", level)
                Q[b, :, j] = g[col].to_numpy(dtype=np.float64)

    if enforce_monotone:
        Q = np.maximum.accumulate(Q, axis=-1)

    return {
        "true": true,
        "Q": Q,
        "u_grid": np.asarray(u_grid, dtype=np.float64),
        "ds": ds_mat,
        "meta": pd.DataFrame({"cutoff": pd.to_datetime(cutoffs)}),
        "point_col": point_col,
        "interval_prefix": interval_prefix,
    }


def save_preds_pickle(A, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pkl.dump(A, f)
    print(f"Saved: {path.resolve()}")


# ============================================================
# 3. Fit Naive and save preds_test_set.pkl
# ============================================================

sf = StatsForecast(
    models=[Naive()],
    freq="h",
    n_jobs=-1,
)

print("Running StatsForecast Naive cross_validation...")

cv_df = sf.cross_validation(
    df=Y_df,
    h=h,
    n_windows=n_windows,
    step_size=1,
    level=levels,
)

out_dir = Path("./benchmark_outputs/naive")
out_dir.mkdir(parents=True, exist_ok=True)

cv_df.to_parquet(out_dir / "naive_cv.parquet", index=False)

print(cv_df.head())
print(cv_df.columns)

A_naive = cv_df_to_quantile_pickle_dict(
    cv_df=cv_df,
    u_grid=u_grid,
    point_col="Naive",
    interval_prefix="Naive",
    enforce_monotone=True,
)

save_preds_pickle(A_naive, out_dir / "preds_test_set.pkl")