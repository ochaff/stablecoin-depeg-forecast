# ============================================================
# Section 5.2.10:
# Interpretability of the probabilistic model
# ============================================================
#
# This version is tailored to your LightningModule output.
#
# It expects diagnostics to be saved inside:
#
#   preds_test_set.pkl["diagnostics"]
#
# Expected useful diagnostic keys:
#   - selection_weights
#   - hard_gates
#   - expected_open
#   - effective_selection
#   - effective_selection_norm
#   - expected_effective_selection
#   - cross_attn_mean_layers
#   - selected_cross_attn
#   - selected_cross_attn_norm
#
# Main outputs:
#   1. section_5210_feature_selection_table.csv
#   2. section_5210_group_gate_table.csv
#   3. section_5210_high_risk_episode_base.csv
#   4. section_5210_high_risk_episode_gates_long.csv
#   5. section_5210_variable_selection_frequency.png
#   6. section_5210_group_gate_probabilities.png
#   7. section_5210_selected_covariates_high_risk_episode.png
# ============================================================

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from benchmark_utils import (
    load_forecast_pickles,
    check_common_grid_and_shape,
    compute_abs_threshold_event_prob,
    _resolve_horizons,
    model_paths,
)

# ------------------------------------------------------------
# 1. Feature-name and paper-specific group helpers
# ------------------------------------------------------------

from collections import OrderedDict


def get_feature_names_from_A(A, n_features=None, feature_names=None):
    """
    Extract feature names from saved prediction dictionary.

    Priority:
      1. user-supplied feature_names
      2. A["covariate_names"]
      3. A["feature_names"]
      4. A["var_names"]
      5. generic feature_000, feature_001, ...

    IMPORTANT:
    Your LightningModule currently comments out A["covariate_names"].
    For readable figures, uncomment this before saving preds_test_set.pkl:

        if var_names is not None:
            A["covariate_names"] = list(var_names)
    """
    if feature_names is not None:
        names = list(feature_names)
    elif "covariate_names" in A:
        names = list(A["covariate_names"])
    elif "feature_names" in A:
        names = list(A["feature_names"])
    elif "var_names" in A:
        names = list(A["var_names"])
    else:
        if n_features is None:
            raise ValueError("n_features is required when feature names are unavailable.")
        names = [f"feature_{j:03d}" for j in range(n_features)]

    names = [
        x.decode("utf-8") if isinstance(x, bytes) else str(x)
        for x in names
    ]

    if n_features is not None and len(names) != n_features:
        print(
            f"Warning: number of feature names ({len(names)}) does not match "
            f"diagnostic feature dimension ({n_features}). Using generic names."
        )
        names = [f"feature_{j:03d}" for j in range(n_features)]

    return names


def build_paper_feature_groups(features):
    """
    Build paper-specific feature groups for Section 5.2.10.

    Parameters
    ----------
    features : list[str]
        Ordered list of feature names used by the SAINT covariate branch.

    Returns
    -------
    feature_groups : OrderedDict[str, list[str]]
        Group name -> list of feature names.
    """
    features = [str(f) for f in features]
    feature_set = set(features)

    feature_groups = OrderedDict()

    # --------------------------------------------------------
    # Main Uniswap V3 USDC/USDT liquidity-curve shape.
    # Includes Gegenbauer coefficients, energy features derived
    # from the decomposition, ratios, tangents of swap-size impact
    # curve, and swap-size imbalance.
    # --------------------------------------------------------
    feature_groups["Uniswap V3 liquidity-curve shape"] = [
        f for f in features
        if (
            f.startswith("Gegenbauer_")
            or f.startswith("E_")
            or ("ratio" in f)
            or f in [
                "tangent_up",
                "tangent_down",
                "swap_size_imbalance",
                "tvlUSD_100",
                "tvlUSD_500",
            ]
        )
    ]

    # --------------------------------------------------------
    # Curve 3pool broad stablecoin liquidity conditions.
    # --------------------------------------------------------
    feature_groups["Curve 3pool liquidity conditions"] = [
        f for f in [
            "w_USDC",
            "w_USDT",
            "curve_entropy",
            "gauge_share_3crv",
            "totalValueLockedUSD",
        ]
        if f in feature_set
    ]

    # --------------------------------------------------------
    # Broader market conditions: ETH/BTC technical indicators,
    # dollar index, FX volatility, sentiment.
    # --------------------------------------------------------
    feature_groups["Broader market conditions"] = [
        f for f in features
        if (
            f.startswith("eth_")
            or f.startswith("btc_")
            or f in [
                "usd_index",
                "fx_volatility",
                "fear_greed_index",
            ]
        )
    ]

    # --------------------------------------------------------
    # Historical stablecoin peg deviation.
    # --------------------------------------------------------
    feature_groups["Historical peg deviation"] = [
        f for f in features
        if f == "depeg_bps" or f.startswith("depeg_bps_lag")
    ]

    # --------------------------------------------------------
    # Liquidity ownership, concentration, and position structure.
    # --------------------------------------------------------
    feature_groups["Liquidity ownership and position structure"] = [
        f for f in [
            "hhi_24h_rolling_mean",
            "tick_width_24h_rolling_median",
            "n_in_range_log_return",
            "weighted_mean_age_hours",
        ]
        if f in feature_set
    ]

    # --------------------------------------------------------
    # Market velocity and flows on Uniswap / aggregate DEX activity.
    # --------------------------------------------------------
    feature_groups["Trading velocity and flows"] = [
        f for f in [
            "swap_count_100",
            "swap_count_500",
            "net_amountUSD_100",
            "net_amountUSD_500",
            "net_amount0",
            "hourlyVolumeUSD",
        ]
        if f in feature_set
    ]

    # --------------------------------------------------------
    # AAVE lending market variables.
    # --------------------------------------------------------
    feature_groups["AAVE lending market conditions"] = [
        f for f in [
            "supplied_USD_usdt",
            "utilisation_rate_usdt",
            "supplied_USD_usdc",
            "utilisation_rate_usdc",
            "liquidation_USD",
        ]
        if f in feature_set
    ]

    return feature_groups


def build_feature_group_mapping(
    feature_names,
    custom_group_map=None,
    assign_unmatched_to="Other",
    verbose=True,
):
    """
    Build feature -> group mapping using the paper-specific groups.

    Parameters
    ----------
    feature_names : list[str]
        Feature names in model order.
    custom_group_map : dict or None
        Optional explicit overrides:
            {"feature_name": "Group name", ...}
    assign_unmatched_to : str
        Group assigned to features not captured by the paper-specific rules.
    verbose : bool
        If True, print group coverage diagnostics.

    Returns
    -------
    group_map_df : pd.DataFrame
        Columns:
            Feature
            Feature group
    """
    feature_names = [str(f) for f in feature_names]
    custom_group_map = custom_group_map or {}

    paper_groups = build_paper_feature_groups(feature_names)

    # Start with unmatched.
    feature_to_group = {
        f: assign_unmatched_to
        for f in feature_names
    }

    # Assign groups in OrderedDict order.
    # If a feature appears in multiple groups, the first group wins.
    duplicate_assignments = []

    for group_name, group_features in paper_groups.items():
        for f in group_features:
            if f not in feature_to_group:
                continue

            if feature_to_group[f] != assign_unmatched_to:
                duplicate_assignments.append(
                    {
                        "Feature": f,
                        "Existing group": feature_to_group[f],
                        "New group ignored": group_name,
                    }
                )
                continue

            feature_to_group[f] = group_name

    # User overrides take final priority.
    for f, g in custom_group_map.items():
        f = str(f)
        if f in feature_to_group:
            feature_to_group[f] = str(g)

    rows = [
        {
            "Feature": f,
            "Feature group": feature_to_group[f],
        }
        for f in feature_names
    ]

    group_map_df = pd.DataFrame(rows)

    if verbose:
        print("\nSection 5.2.10 feature-group coverage:")
        print("======================================")
        coverage = (
            group_map_df
            .groupby("Feature group", as_index=False)
            .agg(N_features=("Feature", "count"))
            .sort_values("N_features", ascending=False)
        )
        print(coverage.to_string(index=False))

        n_other = int(np.sum(group_map_df["Feature group"] == assign_unmatched_to))
        if n_other > 0:
            print(f"\nWarning: {n_other} features assigned to '{assign_unmatched_to}'.")
            print(
                group_map_df.loc[
                    group_map_df["Feature group"] == assign_unmatched_to,
                    "Feature",
                ]
                .head(50)
                .to_string(index=False)
            )

        if len(duplicate_assignments) > 0:
            print("\nWarning: duplicate feature-group assignments detected.")
            print("The first group in OrderedDict order was kept.")
            print(pd.DataFrame(duplicate_assignments).to_string(index=False))

    return group_map_df

# ------------------------------------------------------------
# 2. Diagnostics extraction from preds_test_set.pkl
# ------------------------------------------------------------

def print_diagnostic_keys(A):
    """
    Print available diagnostic keys and shapes.
    """
    if "diagnostics" not in A:
        print("No A['diagnostics'] found.")
        return

    print("\nAvailable A['diagnostics'] keys:")
    print("================================")
    for k, v in A["diagnostics"].items():
        shape = getattr(v, "shape", None)
        print(f"{k:35s} shape={shape}")


def get_diag_array(A, key, required=False):
    """
    Safely extract a diagnostic array from A["diagnostics"].
    """
    diag = A.get("diagnostics", None)

    if diag is None:
        if required:
            raise KeyError(
                "A['diagnostics'] not found. "
                "Rerun testing with --save_test_diagnostics 1."
            )
        return None

    if key not in diag:
        if required:
            raise KeyError(
                f"A['diagnostics']['{key}'] not found. "
                f"Available keys: {list(diag.keys())}"
            )
        return None

    return np.asarray(diag[key], dtype=np.float64)


def choose_gate_probability_array(A):
    """
    Choose the best diagnostic to interpret as soft gate probability.

    Priority:
      1. expected_open
      2. expected_effective_selection
      3. effective_selection_norm
      4. selection_weights

    In your model:
      - expected_open is the hard-concrete expected open probability.
      - effective_selection_norm is closer to the normalized weight actually
        applied to covariate tokens.
    """
    priority = [
        "expected_open",
        "expected_effective_selection",
        "effective_selection_norm",
        "selection_weights",
    ]

    for key in priority:
        arr = get_diag_array(A, key, required=False)
        if arr is not None:
            return key, arr

    raise KeyError(
        "No suitable gate probability diagnostic found. Expected one of: "
        f"{priority}"
    )


def choose_hard_gate_array(A):
    """
    Return hard_gates if available.
    """
    return get_diag_array(A, "hard_gates", required=False)


def normalize_feature_diag_to_BF(arr, A=None, horizons_ahead=24):
    """
    Normalize diagnostic array to shape (B,F).

    Supported:
      - (B,F)
      - (B,H,F)
      - (B,T,F)
      - (B,H,T,F)

    For arrays with horizon dimension matching A["true"].shape[1],
    select / average the requested horizons.
    Otherwise average over non-feature middle dimensions.
    """
    if arr is None:
        return None

    X = np.asarray(arr, dtype=np.float64)

    if X.ndim == 2:
        return X

    if A is not None:
        H = np.asarray(A["true"]).shape[1]
        h_idx, _ = _resolve_horizons(A, horizons_ahead)
    else:
        H = None
        h_idx = None

    if X.ndim == 3:
        B, D, F = X.shape

        if H is not None and D == H:
            return np.nanmean(X[:, h_idx, :], axis=1)

        # Otherwise treat D as time/lookback and average.
        return np.nanmean(X, axis=1)

    if X.ndim == 4:
        B, D1, D2, F = X.shape

        if H is not None and D1 == H:
            return np.nanmean(X[:, h_idx, :, :], axis=(1, 2))

        return np.nanmean(X, axis=(1, 2))

    raise ValueError(f"Unsupported diagnostic shape {X.shape}")


def align_B_length(X, A):
    """
    Align diagnostic length to prediction length.
    """
    if X is None:
        return None

    B_pred = np.asarray(A["true"]).shape[0]
    B_x = X.shape[0]
    B = min(B_pred, B_x)

    if B_pred != B_x:
        print(
            f"Warning: diagnostic length {B_x} differs from prediction length {B_pred}. "
            f"Using first {B} rows."
        )

    return X[:B]


# ------------------------------------------------------------
# 3. Feature and group summary tables
# ------------------------------------------------------------

def build_feature_selection_table(
    A,
    horizons_ahead=24,
    active_threshold=1e-6,
    feature_names=None,
    custom_group_map=None,
):
    """
    Build feature-level interpretability table.

    Uses:
      - expected_open or fallback as soft gate probability
      - hard_gates as hard selection indicator if available
      - effective_selection_norm as actual normalized applied selection if available
      - selected_cross_attn_norm if available for selected cross-attention mass
    """
    gate_key, gate_raw = choose_gate_probability_array(A)
    hard_raw = choose_hard_gate_array(A)

    gate_BF = normalize_feature_diag_to_BF(
        gate_raw,
        A=A,
        horizons_ahead=horizons_ahead,
    )
    hard_BF = normalize_feature_diag_to_BF(
        hard_raw,
        A=A,
        horizons_ahead=horizons_ahead,
    )

    gate_BF = align_B_length(gate_BF, A)
    hard_BF = align_B_length(hard_BF, A)

    n_features = gate_BF.shape[1]

    eff_norm_raw = get_diag_array(A, "effective_selection_norm", required=False)
    eff_norm_BF = normalize_feature_diag_to_BF(
        eff_norm_raw,
        A=A,
        horizons_ahead=horizons_ahead,
    )
    eff_norm_BF = align_B_length(eff_norm_BF, A)

    selected_cross_raw = get_diag_array(A, "selected_cross_attn_norm", required=False)
    selected_cross_BF = normalize_feature_diag_to_BF(
        selected_cross_raw,
        A=A,
        horizons_ahead=horizons_ahead,
    )
    selected_cross_BF = align_B_length(selected_cross_BF, A)

    names = get_feature_names_from_A(
        A,
        n_features=n_features,
        feature_names=feature_names,
    )

    group_map = build_feature_group_mapping(
        names,
        custom_group_map=custom_group_map,
    )

    groups = group_map["Feature group"].tolist()

    rows = []

    for j in range(n_features):
        gp = gate_BF[:, j]

        if hard_BF is not None:
            hg = hard_BF[:, j]
            selection_frequency = float(np.nanmean(hg > active_threshold))
            mean_hard_gate = float(np.nanmean(hg))
        else:
            selection_frequency = float(np.nanmean(gp > active_threshold))
            mean_hard_gate = np.nan

        if eff_norm_BF is not None:
            mean_effective_selection_norm = float(np.nanmean(eff_norm_BF[:, j]))
        else:
            mean_effective_selection_norm = np.nan

        if selected_cross_BF is not None:
            mean_selected_cross_attention = float(np.nanmean(selected_cross_BF[:, j]))
        else:
            mean_selected_cross_attention = np.nan

        rows.append({
            "Feature": names[j],
            "Feature group": groups[j],
            "Gate diagnostic used": gate_key,
            "Selection frequency": selection_frequency,
            "Mean hard gate": mean_hard_gate,
            "Average gate probability": float(np.nanmean(gp)),
            "Median gate probability": float(np.nanmedian(gp)),
            "P90 gate probability": float(np.nanquantile(gp, 0.90)),
            "Mean effective selection norm": mean_effective_selection_norm,
            "Mean selected cross-attention norm": mean_selected_cross_attention,
        })

    feature_table = pd.DataFrame(rows)

    feature_table = feature_table.sort_values(
        [
            "Selection frequency",
            "Average gate probability",
            "Mean effective selection norm",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return feature_table, group_map, gate_BF, hard_BF, eff_norm_BF, selected_cross_BF


def build_group_gate_table(feature_table):
    """
    Aggregate feature-level diagnostics by group.
    """
    agg = (
        feature_table
        .groupby("Feature group", as_index=False)
        .agg(
            **{
                "Average gate probability": ("Average gate probability", "mean"),
                "Selection frequency": ("Selection frequency", "mean"),
                "Mean hard gate": ("Mean hard gate", "mean"),
                "Mean effective selection norm": ("Mean effective selection norm", "mean"),
                "Mean selected cross-attention norm": ("Mean selected cross-attention norm", "mean"),
                "N features": ("Feature", "count"),
            }
        )
    )

    agg = agg.sort_values(
        "Average gate probability",
        ascending=False,
    ).reset_index(drop=True)

    return agg


# ------------------------------------------------------------
# 4. High-risk episode selection
# ------------------------------------------------------------

def get_time_index_for_horizon(A, horizon_idx=0):
    """
    Try to extract timestamps; otherwise return integer index.
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

        if t.ndim == 1 and len(t) == B:
            try:
                return pd.to_datetime(t)
            except Exception:
                return t

        if t.ndim == 2 and t.shape[0] == B:
            h = min(horizon_idx, t.shape[1] - 1)
            try:
                return pd.to_datetime(t[:, h])
            except Exception:
                return t[:, h]

    return np.arange(B)


def select_high_risk_episode_center(
    A,
    horizons_ahead=24,
    threshold=15.0,
    manual_center=None,
    prefer_realized_event=True,
):
    """
    Select an episode center.

    If manual_center is provided, use it.
    Otherwise choose the highest predicted depeg probability among realized
    depeg events if available; otherwise choose highest predicted risk overall.
    """
    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    if len(h_idx) != 1:
        raise ValueError("Use a single horizon for case-study plots, e.g. horizons_ahead=24.")

    h = int(h_idx[0])

    if manual_center is not None:
        return int(manual_center), horizon_label

    y = np.asarray(A["true"], dtype=np.float64)
    p_depeg = compute_abs_threshold_event_prob(A, abs_threshold=threshold)

    y_h = y[:, h]
    p_h = p_depeg[:, h]

    valid = np.isfinite(y_h) & np.isfinite(p_h)

    if prefer_realized_event:
        event = valid & (np.abs(y_h) >= threshold)

        if np.any(event):
            idx = np.where(event)[0]
            center = idx[np.nanargmax(p_h[idx])]
            return int(center), horizon_label

    idx = np.where(valid)[0]

    if len(idx) == 0:
        raise ValueError("No valid observations for high-risk episode selection.")

    center = idx[np.nanargmax(p_h[idx])]
    return int(center), horizon_label


def build_high_risk_episode_frames(
    A,
    feature_table,
    gate_BF,
    eff_norm_BF=None,
    selected_cross_BF=None,
    horizons_ahead=24,
    threshold=15.0,
    manual_center=None,
    window_pre=72,
    window_post=72,
    top_n_features=25,
):
    """
    Build data frames for high-risk interpretability case study.
    """
    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    if len(h_idx) != 1:
        raise ValueError("Use a single horizon for case-study plots, e.g. horizons_ahead=24.")

    h = int(h_idx[0])

    center_idx, _ = select_high_risk_episode_center(
        A,
        horizons_ahead=horizons_ahead,
        threshold=threshold,
        manual_center=manual_center,
        prefer_realized_event=True,
    )

    y = np.asarray(A["true"], dtype=np.float64)
    p_depeg = compute_abs_threshold_event_prob(A, abs_threshold=threshold)

    B = min(y.shape[0], gate_BF.shape[0], p_depeg.shape[0])

    y = y[:B]
    p_depeg = p_depeg[:B]
    gate_BF = gate_BF[:B]

    if eff_norm_BF is not None:
        eff_norm_BF = eff_norm_BF[:B]

    if selected_cross_BF is not None:
        selected_cross_BF = selected_cross_BF[:B]

    center_idx = int(np.clip(center_idx, 0, B - 1))

    lo = max(0, center_idx - window_pre)
    hi = min(B, center_idx + window_post + 1)

    time_index = get_time_index_for_horizon(A, horizon_idx=h)
    time_index = time_index[:B]

    base = pd.DataFrame({
        "idx": np.arange(lo, hi),
        "t": time_index[lo:hi],
        "realized": y[lo:hi, h],
        "p_depeg": p_depeg[lo:hi, h],
    })

    # Choose top features by global importance.
    ft = feature_table.copy()

    sort_cols = [
        "Average gate probability",
        "Mean effective selection norm",
        "Selection frequency",
    ]

    ft = ft.sort_values(sort_cols, ascending=False)
    top_features = ft.head(top_n_features)["Feature"].tolist()

    all_features = feature_table["Feature"].tolist()
    feature_to_col = {f: j for j, f in enumerate(all_features)}
    cols = [feature_to_col[f] for f in top_features]

    gate_mat = gate_BF[lo:hi, :][:, cols].T

    if eff_norm_BF is not None:
        eff_mat = eff_norm_BF[lo:hi, :][:, cols].T
    else:
        eff_mat = None

    if selected_cross_BF is not None:
        attn_mat = selected_cross_BF[lo:hi, :][:, cols].T
    else:
        attn_mat = None

    records = []

    for local_t, global_idx in enumerate(range(lo, hi)):
        for row_j, f in enumerate(top_features):
            rec = {
                "idx": global_idx,
                "t": time_index[global_idx],
                "Feature": f,
                "Feature group": feature_table.loc[
                    feature_table["Feature"] == f,
                    "Feature group",
                ].iloc[0],
                "Gate probability": gate_mat[row_j, local_t],
            }

            if eff_mat is not None:
                rec["Effective selection norm"] = eff_mat[row_j, local_t]

            if attn_mat is not None:
                rec["Selected cross-attention norm"] = attn_mat[row_j, local_t]

            records.append(rec)

    gates_long = pd.DataFrame(records)

    return {
        "base": base,
        "gates_long": gates_long,
        "top_features": top_features,
        "gate_matrix": gate_mat,
        "effective_selection_matrix": eff_mat,
        "selected_cross_attention_matrix": attn_mat,
        "center_idx": center_idx,
        "window_lo": lo,
        "window_hi": hi,
        "horizon_label": horizon_label,
    }


# ------------------------------------------------------------
# 5. Plotting
# ------------------------------------------------------------

def plot_variable_selection_frequency(
    feature_table,
    out_path=None,
    top_n=30,
    title="Variable selection frequency",
):
    df = feature_table.head(top_n).copy()
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.30 * len(df))))

    ax.barh(
        df["Feature"],
        df["Selection frequency"],
        color="tab:blue",
        alpha=0.85,
    )

    ax.set_xlabel("Fraction of test windows active")
    ax.set_title(title)
    ax.set_xlim(0, max(1.0, float(np.nanmax(df["Selection frequency"])) * 1.05))
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=220, bbox_inches="tight")

    return fig, ax


def plot_group_gate_probabilities(
    group_table,
    out_path=None,
    title="Average gate probability by feature group",
):
    df = group_table.copy()
    df = df.sort_values("Average gate probability", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(df))))

    ax.barh(
        df["Feature group"],
        df["Average gate probability"],
        color="tab:green",
        alpha=0.85,
    )

    ax.set_xlabel("Average expected open probability")
    ax.set_title(title)
    ax.set_xlim(0, max(1.0, float(np.nanmax(df["Average gate probability"])) * 1.05))
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=220, bbox_inches="tight")

    return fig, ax


def plot_selected_covariates_high_risk_episode(
    episode,
    feature_table,
    group_table,
    out_path=None,
    threshold=15.0,
    title="Selected covariates during high-risk forecasts",
    heatmap_kind="gate",
):
    """
    Three-panel interpretability plot.

    Panel A:
        average gate probability by feature group

    Panel B:
        gate probabilities or effective selection during selected episode

    Panel C:
        predicted depeg probability and realized deviation
    """
    base = episode["base"]
    top_features = episode["top_features"]
    center_idx = episode["center_idx"]

    if heatmap_kind == "effective" and episode["effective_selection_matrix"] is not None:
        M = episode["effective_selection_matrix"]
        cbar_label = "Effective selection norm"
        panel_b_title = "Panel B: normalized effective selection during episode"
    elif heatmap_kind == "attention" and episode["selected_cross_attention_matrix"] is not None:
        M = episode["selected_cross_attention_matrix"]
        cbar_label = "Selected cross-attention norm"
        panel_b_title = "Panel B: selected cross-attention during episode"
    else:
        M = episode["gate_matrix"]
        cbar_label = "Expected open probability"
        panel_b_title = "Panel B: gate probabilities during episode"

    fig = plt.figure(figsize=(13, 10))

    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.1, 2.1, 1.2],
        hspace=0.38,
    )

    # -----------------------------
    # Panel A
    # -----------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    g = group_table.copy()
    g = g.sort_values("Average gate probability", ascending=True)

    ax1.barh(
        g["Feature group"],
        g["Average gate probability"],
        color="tab:green",
        alpha=0.85,
    )

    ax1.set_xlabel("Average expected open probability")
    ax1.set_title("Panel A: average gate probability by feature group")
    ax1.grid(axis="x", alpha=0.25)

    # -----------------------------
    # Panel B
    # -----------------------------
    ax2 = fig.add_subplot(gs[1, 0])

    vmax = np.nanmax(M)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    im = ax2.imshow(
        M,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )

    ax2.set_yticks(np.arange(len(top_features)))
    ax2.set_yticklabels(top_features, fontsize=8)

    n_time = len(base)
    xticks = np.linspace(0, n_time - 1, min(8, n_time), dtype=int)

    xlabels = []
    for i in xticks:
        val = base.iloc[i]["t"]
        if isinstance(val, pd.Timestamp):
            xlabels.append(val.strftime("%Y-%m-%d\n%H:%M"))
        else:
            xlabels.append(str(val))

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels, fontsize=8)

    local_center = int(center_idx - int(base["idx"].iloc[0]))

    if 0 <= local_center < n_time:
        ax2.axvline(
            local_center,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="Episode center",
        )

    ax2.set_title(panel_b_title)

    cbar = fig.colorbar(
        im,
        ax=ax2,
        orientation="vertical",
        fraction=0.018,
        pad=0.01,
    )
    cbar.set_label(cbar_label)

    # -----------------------------
    # Panel C
    # -----------------------------
    ax3 = fig.add_subplot(gs[2, 0])

    x = np.arange(n_time)

    ax3.plot(
        x,
        base["p_depeg"].astype(float).values,
        color="tab:red",
        linewidth=1.9,
        label=rf"$P(|\Delta p|>{threshold:g}\mathrm{{bps}})$",
    )

    ax3.set_ylabel("Predicted tail probability", color="tab:red")
    ax3.tick_params(axis="y", labelcolor="tab:red")
    ax3.set_ylim(bottom=0.0)
    ax3.grid(alpha=0.25)

    ax3b = ax3.twinx()

    ax3b.plot(
        x,
        base["realized"].astype(float).values,
        color="black",
        linewidth=1.5,
        label="Realized deviation",
    )

    ax3b.axhline(threshold, color="gray", linestyle=":", linewidth=1.0)
    ax3b.axhline(-threshold, color="gray", linestyle=":", linewidth=1.0)
    ax3b.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    ax3b.set_ylabel("Realized deviation, bps", color="black")
    ax3b.tick_params(axis="y", labelcolor="black")

    if 0 <= local_center < n_time:
        ax3.axvline(
            local_center,
            color="red",
            linestyle="--",
            linewidth=1.2,
        )

    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xlabels, fontsize=8)
    ax3.set_title("Panel C: predicted tail probability and realized deviation")

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if out_path is not None:
        fig.savefig(out_path, dpi=240, bbox_inches="tight")

    return fig


# ------------------------------------------------------------
# 6. Main wrapper
# ------------------------------------------------------------

def make_section_5210_interpretability_outputs(
    model_paths=None,
    runs=None,
    out_dir="./comparison_nn_vs_arima/section_5210_interpretability",
    model="SAINT",
    horizons_ahead=24,
    threshold=15.0,
    active_threshold=1e-6,
    feature_names=None,
    custom_group_map=None,
    manual_center=None,
    window_pre=72,
    window_post=72,
    top_n_features=25,
    heatmap_kind="gate",
    print_keys=True,
):
    """
    Create Section 5.2.10 interpretability tables and plots from
    diagnostics saved in preds_test_set.pkl.

    Parameters
    ----------
    heatmap_kind : {"gate", "effective", "attention"}
        gate:
            Panel B uses expected_open or fallback gate diagnostic.
        effective:
            Panel B uses effective_selection_norm if available.
        attention:
            Panel B uses selected_cross_attn_norm if available.
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

    if "diagnostics" not in A:
        raise KeyError(
            f"{model} preds_test_set.pkl does not contain A['diagnostics']. "
            "Rerun test with --save_test_diagnostics 1."
        )

    if print_keys:
        print_diagnostic_keys(A)

    h_idx, horizon_label = _resolve_horizons(A, horizons_ahead)

    feature_table, group_map, gate_BF, hard_BF, eff_norm_BF, selected_cross_BF = (
        build_feature_selection_table(
            A,
            horizons_ahead=horizons_ahead,
            active_threshold=active_threshold,
            feature_names=feature_names,
            custom_group_map=custom_group_map,
        )
    )

    group_table = build_group_gate_table(feature_table)

    episode = build_high_risk_episode_frames(
        A,
        feature_table=feature_table,
        gate_BF=gate_BF,
        eff_norm_BF=eff_norm_BF,
        selected_cross_BF=selected_cross_BF,
        horizons_ahead=horizons_ahead,
        threshold=threshold,
        manual_center=manual_center,
        window_pre=window_pre,
        window_post=window_post,
        top_n_features=top_n_features,
    )

    model_tag = model.replace(" ", "_").replace("/", "_")
    horizon_tag = horizon_label.replace(",", "_").replace("-", "_")

    # -----------------------------
    # Save tables
    # -----------------------------
    feature_table.to_csv(
        out_dir / f"section_5210_feature_selection_table_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    group_table.to_csv(
        out_dir / f"section_5210_group_gate_table_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    group_map.to_csv(
        out_dir / f"section_5210_feature_group_mapping_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    episode["base"].to_csv(
        out_dir / f"section_5210_high_risk_episode_base_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    episode["gates_long"].to_csv(
        out_dir / f"section_5210_high_risk_episode_gates_long_{model_tag}_{horizon_tag}.csv",
        index=False,
    )

    # -----------------------------
    # Save plots
    # -----------------------------
    plot_variable_selection_frequency(
        feature_table,
        out_path=out_dir / f"section_5210_variable_selection_frequency_{model_tag}_{horizon_tag}.png",
        top_n=top_n_features,
        title=f"Variable selection frequency: {model}, horizons {horizon_label}",
    )

    plot_group_gate_probabilities(
        group_table,
        out_path=out_dir / f"section_5210_group_gate_probabilities_{model_tag}_{horizon_tag}.png",
        title=f"Average gate probability by feature group: {model}",
    )

    plot_selected_covariates_high_risk_episode(
        episode,
        feature_table=feature_table,
        group_table=group_table,
        out_path=out_dir / f"section_5210_selected_covariates_high_risk_episode_{model_tag}_{horizon_tag}.png",
        threshold=threshold,
        title=f"Selected covariates during high-risk forecasts: {model}, horizons {horizon_label}",
        heatmap_kind=heatmap_kind,
    )

    print("\n==============================================")
    print(f"Section 5.2.10 interpretability outputs for {model}")
    print(f"Horizons: {horizon_label}")
    print("==============================================")

    print("\nTop selected features:")
    print(
        feature_table[
            [
                "Feature",
                "Feature group",
                "Selection frequency",
                "Average gate probability",
                "Mean effective selection norm",
                "Mean selected cross-attention norm",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )

    print("\nAverage gate probability by feature group:")
    print(group_table.to_string(index=False))

    print(f"\nSelected high-risk episode center index: {episode['center_idx']}")

    return {
        "runs": runs,
        "feature_table": feature_table,
        "group_table": group_table,
        "feature_group_mapping": group_map,
        "episode": episode,
        "gate_BF": gate_BF,
        "hard_BF": hard_BF,
        "effective_selection_norm_BF": eff_norm_BF,
        "selected_cross_attention_norm_BF": selected_cross_BF,
        "out_dir": out_dir,
    }


# ============================================================
# Example usage
# ============================================================

section_5210_result = make_section_5210_interpretability_outputs(
    model_paths=model_paths,
    out_dir="./comparison_nn_vs_arima/section_5210_interpretability",
    model="SAINT",
    horizons_ahead=24,
    threshold=15.0,
    active_threshold=1e-6,
    feature_names=None,       # optionally pass list of covariate names here
    custom_group_map=None,    # optionally pass explicit feature -> group map
    manual_center=None,       # or manually set, e.g. manual_center=740
    window_pre=72,
    window_post=72,
    top_n_features=25,

    # Options:
    #   "gate"      -> Panel B uses expected_open
    #   "effective" -> Panel B uses effective_selection_norm
    #   "attention" -> Panel B uses selected_cross_attn_norm if available
    heatmap_kind="gate",

    print_keys=True,
)

feature_selection_table = section_5210_result["feature_table"]
group_gate_table = section_5210_result["group_table"]
high_risk_episode = section_5210_result["episode"]