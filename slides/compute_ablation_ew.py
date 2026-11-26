"""Holdout early-warning tallies of the SCQF (formerly ANGEL) ablations, on the probability grid of
compute_calibrated.py, so the ablations get the same value curves as the main models.

    .venv/bin/python slides/compute_ablation_ew.py      # -> slides/data/ablation/ablation_policy_grid.csv

Scores: the raw exceedance probability S_t = max_{h=2..24} P(|d_{t+h}| >= 15 bp) of every ablation seed
(lightning_logs/holdout_exceedance_eval/scores.parquet, columns ANGEL_<variant>_seed_<seed>; six seeds per variant,
MLflow experiment 3603). Same events, 2-24 h window and 12 h alert reset as the main comparison. The rule is
"act when S_t >= p" on the raw probability (a forecaster needs no fitted cutoff); tallies include the lead-time-aware
loss terms. Hourly ROC AUC / AUPRC of S_t against a breach in the 2-24 h window are written alongside.
"""
from __future__ import annotations

import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "slides")]
import compute_calibrated as CC  # noqa: E402
import compute_results as CR  # noqa: E402
from early_warning.events import build_target  # noqa: E402

OUT = REPO / "slides/data/ablation"
VARIANT = {"full": "Full SCQF", "crps_only": "CRPS loss only", "no_gpd": "No GPD tails", "gaussian": "Gaussian output",
           "no_sparsity": "No sparsity", "no_liq_shape": "No liquidity shape", "price_only": "Price only"}


def job(col):
    ew, context, _, events = CR.setup()
    h, _ = CR.holdout_scores(context)
    absd = context.set_index("timestamp").depeg_bps.abs()
    frame = pd.DataFrame({"timestamp": h.index,
                          "depeg_bps": context.set_index("timestamp").depeg_bps.reindex(h.index).to_numpy()})
    sc = h[col].to_numpy(float)
    t = CC.policy_tallies(frame, sc, events, context, absd)
    variant, seed = re.match(r"ANGEL_(.+)_seed_(\d+)$", col).groups()
    t = t.assign(variant=variant, model=VARIANT[variant], seed=int(seed), period="holdout", calibration="raw")
    y, obs, _ = build_target(frame, context, ew.target_threshold, ew.min_lead_hours, ew.max_lead_hours,
                             ew.depeg_event_reset_hours)
    m = obs & np.isfinite(sc)
    rank = dict(variant=variant, model=VARIANT[variant], seed=int(seed), auc=roc_auc_score(y[m], sc[m]),
                auprc=average_precision_score(y[m], sc[m]))
    return t, rank


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ew, context, _, _ = CR.setup()
    h, _ = CR.holdout_scores(context)
    cols = [c for c in h if re.match(r"ANGEL_(.+)_seed_\d+$", c) and not c.startswith("ANGEL_trial100")]
    with Pool(8) as p:
        res = p.map(job, cols)
    pd.concat([t for t, _ in res]).to_csv(OUT / "ablation_policy_grid.csv", index=False)
    pd.DataFrame([r for _, r in res]).to_csv(OUT / "ablation_ranking.csv", index=False)
    print(pd.DataFrame([r for _, r in res]).groupby("model")[["auc", "auprc"]].mean().round(3))


if __name__ == "__main__":
    main()
