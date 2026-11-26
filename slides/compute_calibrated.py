"""Calibrated decision-rule evaluation for the slide deck (15 bp onsets, 2-24 h lead): "act when the calibrated
probability of an onset is >= p", tallied for p on a grid, on the 2024 CV and on the 2025-26 holdout.

    .venv/bin/python slides/compute_calibrated.py        # -> slides/data/calibrated_*.csv

Calibration (Platt / logistic recalibration on logit(score); log1p(score) for |d|, which is not a probability)
  CV       per fold, fitted on the fold's validation block against the onset label, applied to its test block.
  holdout  fitted on the same model specification's pooled 2024 CV out-of-sample (test-fold) scores, frozen,
           applied to the holdout scores. Only pre-holdout data are used for the map.

For every (period, model, run, p) the policy's tallies are stored: D caught events, N events, A alert episodes,
F false alerts, H follow-up hours, plus the lead-time-aware loss terms (avoided bp of caught events, and the bp a
perfect warning at 24 h lead would avoid). From these the notebook derives
  * V(r) of a fixed policy (p = 0.1) for every cost-loss ratio r,
  * the decision-rule curve V(r) at p = r (each user acts on its own r: the deployable analogue of the envelope),
  * the (p, r) value heatmap, and the lead-time-aware value with a fixed cost per alert episode.
Brier score / skill score of the raw and calibrated scores against the hourly onset label are stored as well.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "slides"))
import threshold_free_eval as T  # noqa: E402
from compute_results import CV_MODELS, Q8, holdout_scores, setup, MIN_LEAD, MAX_LEAD, COOLDOWN, PEAK_HOURS  # noqa: E402
from early_warning.events import build_target, score_events  # noqa: E402

DATA = REPO / "slides/data"
P_GRID = np.round(np.arange(0.01, 0.501, 0.01), 4)
POOL = REPO / "lightning_logs/earlywarning_pool_15bps_2026-11-19"
TREES_HO = REPO / "lightning_logs/earlywarning_trees_holdout_2026-11-23"

CV_KEEP = ["ANGEL exceedance", "ARIMA", "GARCH-t", "TiDE forecast", "TimeXer forecast", "Kumo", "TabICL",
           "Random forest", "CatBoost", "XGBoost", "LightGBM", "ANGEL classifier"]
CV_RUNS = {m: CV_MODELS[m] for m in CV_KEEP}
CV_RUNS["|d| now"] = ("naive", [POOL / "baseline_naive_abs_depeg_alpha_0.1"])
PROB = lambda m: m != "|d| now"


# ------------------------------------------------------------------------------------------------ helpers
def platt_fit(score, y, prob):
    x = T._feature(score, prob)
    ok = np.isfinite(x) & np.isfinite(y)
    return T.platt(x[ok], y[ok].astype(float))


def onset_labels(frame, ctx, ew):
    y, obs, _ = build_target(frame, ctx, ew.target_threshold, ew.min_lead_hours, ew.max_lead_hours,
                             ew.depeg_event_reset_hours)
    return np.where(obs, y, np.nan)


def brier_row(y, p_raw, p_cal, base_rate):
    m = np.isfinite(y) & np.isfinite(p_raw)
    y, p_raw, p_cal = y[m], np.clip(p_raw[m], 0, 1), p_cal[m]
    b_ref = np.mean((base_rate - y) ** 2)
    b_raw, b_cal = np.mean((p_raw - y) ** 2), np.mean((p_cal - y) ** 2)
    return dict(n=len(y), prevalence=y.mean(), base_rate=base_rate, brier_ref=b_ref, brier_raw=b_raw,
                brier_cal=b_cal, bss_raw=1 - b_raw / b_ref, bss_cal=1 - b_cal / b_ref)


def policy_tallies(frame, pc, events, ctx, absd):
    """Tallies of 'act when pc >= p' for p in P_GRID, with lead-time-aware loss terms."""
    rows = []
    for p in P_GRID:
        m = score_events(frame, pc, p, events, MIN_LEAD, MAX_LEAD, COOLDOWN, outcome_context=ctx)
        avoided = perfect = 0.0
        for o in m["_event_outcomes"]:
            e = o["event_time"]
            peak = absd.loc[e:e + pd.Timedelta(hours=PEAK_HOURS)].max()
            perfect += max(peak - absd.get(e - pd.Timedelta(hours=MAX_LEAD), np.nan), 0.0)
            if o["detected"]:
                avoided += max(peak - absd.get(e - pd.Timedelta(hours=o["lead_hours"]), np.nan), 0.0)
        rows.append(dict(p=p, D=m["detected"], N=m["events"], A=m["timely_alert_episodes"] + m["false_alerts"],
                         F=m["false_alerts"], H=m["false_alert_exposure_hours"], avoided=avoided, perfect=perfect))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------ CV
def cv_job(job):
    name, run = job
    ew, context, ctx_cv, events = setup()
    absd = context.set_index("timestamp").depeg_bps.abs()
    s = pd.read_parquet(run / "artifacts/cv/operational_scores.parquet")
    s["timestamp"] = pd.to_datetime(s.timestamp, utc=True)
    tallies, briers = [], []
    for k, g in s.groupby("fold"):
        val = g[g.split.eq("validation")].sort_values("timestamp").reset_index(drop=True)
        test = g[g.split.eq("test")].sort_values("timestamp").reset_index(drop=True)
        vctx = ctx_cv[ctx_cv.timestamp <= val.timestamp.iloc[-1]]
        yv = onset_labels(val, vctx, ew)
        f = platt_fit(val.probability.to_numpy(float), yv, PROB(name))
        sc = test.probability.to_numpy(float)
        pc = f(T._feature(sc, PROB(name)))
        tallies.append(policy_tallies(test, pc, events, ctx_cv, absd).assign(fold=k, split="test", calibration="calibrated"))
        if PROB(name):
            tallies.append(policy_tallies(test, sc, events, ctx_cv, absd).assign(fold=k, split="test", calibration="raw"))
        # validation tallies (calibrated in-sample on validation) for choosing p per r without seeing the test block
        pv = f(T._feature(val.probability.to_numpy(float), PROB(name)))
        tallies.append(policy_tallies(val, pv, events, vctx, absd).assign(fold=k, split="validation",
                                                                         calibration="calibrated"))
        yt = onset_labels(test, ctx_cv, ew)
        if PROB(name):
            briers.append(dict(fold=k, **brier_row(yt, sc, pc, np.nanmean(yv)), _y=yt, _raw=sc, _cal=pc))
    t = pd.concat(tallies).assign(period="CV", model=name, run=run.name)        # per fold and split
    b = None
    if briers:                                     # pooled over folds (each fold's own validation base rate)
        y = np.concatenate([x["_y"] for x in briers]); raw = np.concatenate([x["_raw"] for x in briers])
        cal = np.concatenate([x["_cal"] for x in briers])
        br = np.concatenate([np.full(len(x["_y"]), x["base_rate"]) for x in briers])
        m = np.isfinite(y)
        b_ref = np.mean((br[m] - y[m]) ** 2)
        b_raw = np.mean((np.clip(raw[m], 0, 1) - y[m]) ** 2); b_cal = np.mean((cal[m] - y[m]) ** 2)
        b = dict(period="CV", model=name, run=run.name, n=int(m.sum()), prevalence=y[m].mean(), brier_ref=b_ref,
                 brier_raw=b_raw, brier_cal=b_cal, bss_raw=1 - b_raw / b_ref, bss_cal=1 - b_cal / b_ref)
    return t, b


# ------------------------------------------------------------------------------------------------- holdout
def holdout_series():
    """model -> (family, list of raw score Series on the holdout index)."""
    ew, context, _, _ = setup()
    h, series = holdout_scores(context)
    out = {m: (fam, [h[c] for c in cs]) for m, (fam, cs) in series.items()}
    for m, tag in (("Random forest", "random_forest"), ("CatBoost", "catboost"), ("XGBoost", "xgboost"),
                   ("LightGBM", "lightgbm"), ("TabICL", "tabicl_v2")):
        cols = []
        for d in sorted(TREES_HO.glob(f"holdout*_seed*_{tag}_alpha_0.1")):
            s = pd.read_parquet(d / "artifacts/cv/operational_scores.parquet")
            s = s[s.split.eq("test")]
            cols.append(pd.Series(s.probability.to_numpy(float), pd.to_datetime(s.timestamp, utc=True)).reindex(h.index))
        out[m] = ("classifier", cols)
    return out, h.index


def cv_oos(name):
    """Pooled 2024 CV test-fold scores of a model specification (all its CV runs) -> (scores, onset labels)."""
    ew, _, ctx_cv, _ = setup()
    xs, ys = [], []
    for run in CV_RUNS[name][1]:
        s = pd.read_parquet(run / "artifacts/cv/operational_scores.parquet")
        s["timestamp"] = pd.to_datetime(s.timestamp, utc=True)
        for _, g in s[s.split.eq("test")].groupby("fold"):
            g = g.sort_values("timestamp").reset_index(drop=True)
            xs.append(g.probability.to_numpy(float)); ys.append(onset_labels(g, ctx_cv, ew))
    return np.concatenate(xs), np.concatenate(ys)


def holdout_job(job):
    name, j = job
    ew, context, _, events = setup()
    absd = context.set_index("timestamp").depeg_bps.abs()
    series, times = holdout_series()
    sc = series[name][1][j].to_numpy(float)
    x_cv, y_cv = cv_oos(name)
    f = platt_fit(x_cv, y_cv, PROB(name))
    pc = f(T._feature(sc, PROB(name)))
    frame = pd.DataFrame({"timestamp": times, "depeg_bps": context.set_index("timestamp").depeg_bps.reindex(times).to_numpy()})
    t = policy_tallies(frame, pc, events, context, absd).assign(calibration="calibrated")
    if PROB(name):
        t = pd.concat([t, policy_tallies(frame, sc, events, context, absd).assign(calibration="raw")])
    t = t.assign(period="holdout", model=name, run=f"col{j}", fold=0, split="test")
    b = None
    if PROB(name):
        y = onset_labels(frame, context, ew)
        b = dict(period="holdout", model=name, run=f"col{j}", **brier_row(y, sc, pc, np.nanmean(y_cv)))
    return t, b


def main():
    series, _ = holdout_series()
    cv_jobs = [(m, d) for m, (_, dirs) in CV_RUNS.items() for d in dirs]
    ho_jobs = [(m, j) for m, (_, cols) in series.items() for j in range(len(cols))]
    with Pool(8) as p:
        cv = p.map(cv_job, cv_jobs)
        ho = p.map(holdout_job, ho_jobs)
    fam = {**{m: f for m, (f, _) in CV_RUNS.items()}, **{m: f for m, (f, _) in series.items()}}
    tall = pd.concat([t for t, _ in cv + ho]).assign(family=lambda x: x.model.map(fam))
    brier = pd.DataFrame([b for _, b in cv + ho if b is not None]).assign(family=lambda x: x.model.map(fam))
    tall.to_csv(DATA / "calibrated_policy_grid.csv", index=False)
    brier.to_csv(DATA / "calibrated_brier.csv", index=False)
    pd.set_option("display.width", 200)
    print(brier.groupby(["period", "model"])[["brier_raw", "brier_cal", "bss_raw", "bss_cal"]].mean().round(4).to_string())
    v = tall[tall.p.eq(0.1)].assign(V=lambda x: (x.D - 0.1 * x.A) / (x.N * 0.9))
    print(v.groupby(["period", "calibration", "model"])[["D", "N", "A", "V"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
