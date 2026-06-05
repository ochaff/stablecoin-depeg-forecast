# ============================================================
# NeuralForecast TiDE benchmark with full historical exogenous features
# ============================================================
#
# Output:
#   ./benchmark_outputs/tide_hist_exog/preds_test_set.pkl
#
# This file has:
#   A["true"]   : (B,H)
#   A["Q"]      : (B,H,J)
#   A["u_grid"] : (J,)
#   A["ds"]     : (B,H)
#   A["meta"]   : cutoff dataframe
#
# ============================================================

import re
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. Load data and construct NeuralForecast dataframe
# ============================================================

df = pd.read_parquet("./preprocessed_datasets/dataset_alpha_0.1_full.parquet").copy()

# Ensure timestamp column
df["ds"] = df.index.tz_localize(None)

# Target
df["y"] = df["depeg_bps"]

# Single series id
df["unique_id"] = "stablecoin_depeg"

h = 24
test_size = df.shape[0] - int(0.7 * df.shape[0])
n_windows = test_size - h + 1

print("Full dataframe:", df.shape)
print("h:", h)
print("test_size:", test_size)
print("n_windows:", n_windows)


# ============================================================
# 1. Select historical exogenous features
# ============================================================
#
# Important:
#   - We exclude y, ds, unique_id.
#   - We exclude raw depeg_bps because it is identical to y.
#     The target history is already available to TiDE through y.
#   - Lagged depeg variables, liquidity variables, market variables, etc.
#     are retained if numeric.
#
# If you have columns that are labels or future-looking by construction,
# add them to EXCLUDE_COLS manually.
# ============================================================

EXCLUDE_COLS = {
    "unique_id",
    "ds",
    "y",
    "depeg_bps",          # duplicate of target
    "target",
    "label",
    "event",
    "depeg_event",
    "depeg_within_24h",
}

# Keep numeric columns only
numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

hist_exog_list = [
    c for c in numeric_cols
    if c not in EXCLUDE_COLS
]

print(f"Number of historical exogenous features: {len(hist_exog_list)}")
print("First 20 hist exog features:")
print(hist_exog_list[:20])

# Optional: save feature list for reproducibility
out_dir = Path("./benchmark_outputs/tide_hist_exog")
out_dir.mkdir(parents=True, exist_ok=True)

pd.Series(hist_exog_list, name="hist_exog").to_csv(
    out_dir / "tide_hist_exog_features.csv",
    index=False,
)

# Build NeuralForecast dataframe
Y_df_exog = df[["unique_id", "ds", "y"] + hist_exog_list].reset_index(drop=True)

# Clean numerical issues
Y_df_exog[hist_exog_list] = (
    Y_df_exog[hist_exog_list]
    .replace([np.inf, -np.inf], np.nan)
    .ffill()
    .bfill()
)

Y_df_exog["y"] = (
    Y_df_exog["y"]
    .replace([np.inf, -np.inf], np.nan)
    .ffill()
    .bfill()
)

print("Y_df_exog:", Y_df_exog.shape)
print(Y_df_exog.head())


# ============================================================
# 2. Quantile grid and central interval levels
# ============================================================

def load_reference_u_grid(reference_pickle=None):
    """
    Use the same quantile grid as the main model if available.
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


def central_levels_from_u_grid(u_grid, decimals=6):
    """
    Convert quantile levels into central interval levels.

    tau < 0.5:
        lower quantile tau corresponds to central interval level:
        100 * (1 - 2*tau)

    tau > 0.5:
        upper quantile tau corresponds to central interval level:
        100 * (2*tau - 1)
    """
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
# 3. Conversion helpers: NeuralForecast CV dataframe -> pickle
# ============================================================

def _safe_level_str(x):
    if np.isclose(x, round(x)):
        return str(int(round(x)))
    return f"{float(x):g}"


def find_interval_col(columns, interval_prefix, side, level, atol=1e-4):
    """
    Robustly find columns like:
      TiDE-lo-90
      TiDE-hi-90
      TiDE-lo-99.8
      TiDE-hi-99.8
    """
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
            try:
                lvl = float(m.group(1))
                matches.append((abs(lvl - float(level)), lvl, c))
            except Exception:
                pass

    if len(matches) > 0:
        matches = sorted(matches, key=lambda z: z[0])

        if matches[0][0] <= atol:
            return matches[0][2]

    available = [c for c in columns if c.startswith(f"{interval_prefix}-{side}-")]

    raise KeyError(
        f"Could not find {interval_prefix}-{side}-{level}. "
        f"Available columns: {available[:20]}"
    )


def detect_tide_point_col(cv_df):
    """
    Detect TiDE median / point forecast column.
    NeuralForecast usually outputs "TiDE".
    """
    reserved = {"unique_id", "ds", "cutoff", "y"}

    if "TiDE" in cv_df.columns:
        return "TiDE"

    if "TiDE-median" in cv_df.columns:
        return "TiDE-median"

    candidates = [
        c for c in cv_df.columns
        if c not in reserved
        and "-lo-" not in c
        and "-hi-" not in c
        and "tide" in c.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"Could not detect TiDE point column. "
        f"Candidates: {candidates}. "
        f"Columns: {list(cv_df.columns)}"
    )


def cv_df_to_quantile_pickle_dict(
    cv_df,
    u_grid,
    point_col,
    interval_prefix="TiDE",
    enforce_monotone=True,
):
    """
    Convert NeuralForecast cross-validation output to comparison format.

    Output:
      true:   (B,H)
      Q:      (B,H,J)
      u_grid: (J,)
      ds:     (B,H)
      meta:   cutoff dataframe
    """
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
            raise ValueError(
                f"Cutoff {cutoff} has {g.shape[0]} rows, expected H={H}."
            )

        true[b, :] = g["y"].to_numpy(dtype=np.float64)
        ds_mat[b, :] = g["ds"].to_numpy(dtype="datetime64[ns]")

        point = g[point_col].to_numpy(dtype=np.float64)

        for j, tau in enumerate(u_grid):
            tau = float(tau)

            if np.isclose(tau, 0.5):
                Q[b, :, j] = point

            elif tau < 0.5:
                level = 100.0 * (1.0 - 2.0 * tau)
                col = find_interval_col(
                    g.columns,
                    interval_prefix=interval_prefix,
                    side="lo",
                    level=level,
                )
                Q[b, :, j] = g[col].to_numpy(dtype=np.float64)

            else:
                level = 100.0 * (2.0 * tau - 1.0)
                col = find_interval_col(
                    g.columns,
                    interval_prefix=interval_prefix,
                    side="hi",
                    level=level,
                )
                Q[b, :, j] = g[col].to_numpy(dtype=np.float64)

    if enforce_monotone:
        Q = np.maximum.accumulate(Q, axis=-1)

    A = {
        "true": true,
        "Q": Q,
        "u_grid": np.asarray(u_grid, dtype=np.float64),
        "ds": ds_mat,
        "meta": pd.DataFrame({"cutoff": pd.to_datetime(cutoffs)}),
        "point_col": point_col,
        "interval_prefix": interval_prefix,
        "hist_exog_list": hist_exog_list,
    }

    print(
        f"Converted TiDE hist-exog forecasts: "
        f"true={A['true'].shape}, Q={A['Q'].shape}, J={len(u_grid)}"
    )

    return A


def save_preds_pickle(A, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pkl.dump(A, f)

    print(f"Saved: {path.resolve()}")


# ============================================================
# 4. Fit TiDE with historical exogenous variables
# ============================================================

def fit_save_tide_hist_exog(
    Y_df_exog,
    hist_exog_list,
    h,
    n_windows,
    levels,
    u_grid,
    out_dir="./benchmark_outputs/tide_hist_exog",
    freq="h",
    input_size=168,
    max_steps=1000,
    val_size=24 * 30,
    random_seed=1,
):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TiDE
    from neuralforecast.losses.pytorch import MQLoss

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss = MQLoss(level=levels)

    model = TiDE(
        h=h,
        input_size=input_size,

        # Key change: use full historical exogenous feature set
        hist_exog_list=hist_exog_list,

        loss=loss,
        valid_loss=loss,
        max_steps=max_steps,
        scaler_type="robust",
        random_seed=random_seed,

        # Training settings
        learning_rate=1e-3,
        batch_size=32,
        windows_batch_size=1024,
        val_check_steps=50,
        early_stop_patience_steps=10,
    )

    nf = NeuralForecast(
        models=[model],
        freq=freq,
    )

    print("Running NeuralForecast TiDE with historical exogenous variables...")
    print(f"Number of hist_exog variables: {len(hist_exog_list)}")

    cv_df = nf.cross_validation(
        df=Y_df_exog,
        n_windows=n_windows,
        step_size=1,
        val_size=val_size,
        verbose=True,
        refit=False,
    )

    raw_path = out_dir / "tide_hist_exog_cv.parquet"
    cv_df.to_parquet(raw_path, index=False)

    print(f"Saved raw TiDE CV dataframe: {raw_path.resolve()}")
    print(cv_df.head())
    print(cv_df.columns)

    point_col = detect_tide_point_col(cv_df)

    A_tide = cv_df_to_quantile_pickle_dict(
        cv_df=cv_df,
        u_grid=u_grid,
        point_col=point_col,
        interval_prefix="TiDE",
        enforce_monotone=True,
    )

    preds_path = out_dir / "preds_test_set.pkl"
    save_preds_pickle(A_tide, preds_path)

    return cv_df, str(preds_path), A_tide


tide_cv_df, tide_path, A_tide = fit_save_tide_hist_exog(
    Y_df_exog=Y_df_exog,
    hist_exog_list=hist_exog_list,
    h=h,
    n_windows=n_windows,
    levels=levels,
    u_grid=u_grid,
    out_dir="./benchmark_outputs/tide_hist_exog",
    freq="h",
    input_size=168,
    max_steps=1000,
    val_size=24 * 30,
    random_seed=1,
)

print("\nSaved TiDE hist-exog benchmark path:")
print(tide_path)