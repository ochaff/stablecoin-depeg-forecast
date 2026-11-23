"""Early-warning results for the slide deck: cost-loss value curves (CV 2024, holdout 2025-26) and a lead-time-aware
cost-loss on the holdout. Heavy event tallies are cached in slides/data/; the notebook only reads the CSVs.

    .venv/bin/python slides/compute_results.py cv        # resumable (one cache file per model x seed run)
    .venv/bin/python slides/compute_results.py holdout

Protocol (identical to threshold_free_eval.py): 15 bp declustered onsets (24 h reset), timely = alert 2-24 h before
onset, 12 h alert cooldown, false alerts need their full 24 h follow-up observed.

Cost-loss value  V(r) = (D - r A) / (N (1 - r)): D caught events, A alert episodes, N events, r = C/L. 1 = perfect
warning, 0 = never alerting. The curves are value envelopes V*(r) = max over cutoffs (floored at 0); in CV the cutoff
is one validation quantile q applied per fold ("q-sweep"), on the holdout a cutoff on the score itself.

Lead-time-aware value (holdout): the loss avoided by a warning at time t_a for event e is the part of the move still
ahead, L_e(t_a) = max_{onset <= t <= onset + 24 h} |d_t| - |d_{t_a}| (bp, floored at 0). A perfect warning comes at the
earliest admissible time (24 h before onset). With C = r * mean_e L_e(perfect):
    V_lead(r) = (sum_caught L_e(t_a) - C A) / (sum_e L_e(perfect) - C N),
which reduces to V(r) when every warning avoids the same loss.
"""
from __future__ import annotations

import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import cv_earlywarning as C  # noqa: E402
import threshold_free_eval as T  # noqa: E402
from early_warning.events import event_onsets, score_events  # noqa: E402

DATA = REPO / "slides/data"
POOL = REPO / "lightning_logs/earlywarning_pool_15bps_2026-11-19"
Q8 = REPO.parent / "onchain-insights/8. Early-Warning Classifier Cross Validation"
THRESHOLD, MIN_LEAD, MAX_LEAD, COOLDOWN, PEAK_HOURS = 15, 2, 24, 12, 24

# model -> (family, run directories). Same pool experiment and feature set (full) for every model.
CV_MODELS = {
    "ANGEL exceedance": ("forecast", sorted(POOL.glob("fc_full_seed*_ANGEL_fc_exceedance_alpha_0.1"))),
    "ARIMA": ("forecast", [POOL / "bench_arima_seed1_ARIMA_fc_exceedance_alpha_0.1"]),
    "GARCH-t": ("forecast", [POOL / "bench_garch_t_seed1_GARCH_t_fc_exceedance_alpha_0.1"]),
    "TiDE forecast": ("forecast", [POOL / "bench_tide_nf_seed1233_TiDE_NF_fc_exceedance_alpha_0.1"]),
    "TimeXer forecast": ("forecast", [POOL / "bench_timexer_nf_seed1_TimeXer_NF_fc_exceedance_alpha_0.1"]),
    "ANGEL classifier": ("classifier", [POOL / "full_seed1233_ANGEL_alpha_0.1"]),
    "TiDE classifier": ("classifier", [POOL / "full_seed1233_TiDE_alpha_0.1"]),
    "TimeXer classifier": ("classifier", [POOL / "full_seed1233_TimeXer_alpha_0.1"]),
    "iTransformer classifier": ("classifier", [POOL / "full_seed1233_iTransformer_alpha_0.1"]),
    # Kumo: the Quantlet 8 specification (all fitting rows as context), the one with holdout predictions
    "Kumo": ("classifier", [Q8 / "lightning_logs/kumo_context_cv_2026-11-18/all_seed1233_kumo_tabular_alpha_1.0"]
             + [Q8 / f"lightning_logs/cv_reproduction_2026-11-18_seeds/lead6_24_threshold_15_alpha_1.0_seed{s}"
                     "_kumo_tabular_alpha_1.0" for s in (7, 42)]),
    **{name: ("classifier", sorted(POOL.glob(f"full_seed*_{tag}_alpha_0.1")))
       for name, tag in [("TabICL", "tabicl_v2"), ("CatBoost", "catboost"),
                         ("LightGBM", "lightgbm"), ("XGBoost", "xgboost"), ("Random forest", "random_forest")]},
}


def setup():
    ew = C.parse_args(["--alpha", "0.1", "--dataset_file", str(REPO / "preprocessed_datasets/dataset_alpha_0.1_full.parquet")])
    ew.target_threshold = THRESHOLD
    data, context, _, _ = C.load_dataset(ew)
    n = len(data)
    holdout_start = data["timestamp"].iloc[n - int(ew.holdout_frac * n)]
    ctx_cv = context[context.timestamp < holdout_start].reset_index(drop=True)
    return ew, context, ctx_cv, event_onsets(context, THRESHOLD, ew.depeg_event_reset_hours)


# ------------------------------------------------------------------------------------------------------------ CV
def _cv_job(job):
    name, run = job
    out = DATA / "cv_seed" / f"{name.replace(' ', '_')}__{run.name}.pkl"
    if out.exists():
        return name, run.name
    ew, _, ctx_cv, events = setup()
    T.CV_RUNS = {name: [run]}
    tab, curves = T.cv_eval(ew, events, ctx_cv)
    pickle.dump((tab, curves), open(out, "wb"))
    return name, run.name


def run_cv():
    (DATA / "cv_seed").mkdir(parents=True, exist_ok=True)
    T.OUT = DATA / "cv_seed"                        # cv_eval writes a per-seed CSV there; keep it out of figs/
    jobs = [(m, d) for m, (_, dirs) in CV_MODELS.items() for d in dirs]
    with Pool(6, maxtasksperchild=1) as p:
        for m, r in p.imap_unordered(_cv_job, jobs):
            print("done", m, r, flush=True)
    rows, curves = [], []
    for m, (fam, dirs) in CV_MODELS.items():
        tabs, envs = [], []
        for d in dirs:
            tab, cur = pickle.load(open(DATA / "cv_seed" / f"{m.replace(' ', '_')}__{d.name}.pkl", "rb"))
            tabs.append(tab.iloc[0])
            envs.append(pd.DataFrame(cur).query("kind == 'envelope'").set_index("x").y)
        t = pd.DataFrame(tabs)
        rows.append({"model": m, "family": fam, "n_seeds": len(dirs),
                     "AUVC": t["q-sweep AUVC_0.5"].mean(), "AUVC_sd": t["q-sweep AUVC_0.5"].std(),
                     "AUVC_0.2": t["q-sweep AUVC_0.2"].mean(), "pAUC_6": t["q-sweep pAUC_6"].mean(),
                     "LOFO_min": t["LOFO AUVC min"].min(), "LOFO_max": t["LOFO AUVC max"].max()})
        env = pd.concat(envs, axis=1)
        curves += [dict(model=m, family=fam, r=r, V=v, V_min=lo, V_max=hi)
                   for r, v, lo, hi in zip(env.index, env.mean(1), env.min(1), env.max(1))]
    pd.DataFrame(rows).to_csv(DATA / "cv_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(DATA / "cv_curves.csv", index=False)
    print(pd.DataFrame(rows).sort_values("AUVC", ascending=False).round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------------- holdout
def holdout_scores(context):
    h = pd.read_parquet(REPO / "lightning_logs/holdout_exceedance_eval/scores.parquet")
    k = pd.read_parquet(Q8 / "artifacts/kumo_context_design_holdout/holdout_predictions.parquet")
    k["timestamp"] = pd.to_datetime(k.timestamp, utc=True)
    k = k.set_index("timestamp")
    times = h.index.intersection(k.index)
    h = h.loc[times].copy()
    h["Kumo"] = k["Native full context"].reindex(times)             # the CV specification (all fitting rows)
    absd = np.abs(context.set_index("timestamp").depeg_bps.reindex(times).to_numpy())
    h["|d| now"] = absd / (1 + absd)
    series = {"ANGEL exceedance": ("forecast", [c for c in h if c.startswith("ANGEL_full_seed_")]),
              "ARIMA": ("forecast", ["ARIMA"]), "GARCH-t": ("forecast", ["GARCH-t"]),
              "TiDE forecast": ("forecast", ["TiDE (NF)"]), "TimeXer forecast": ("forecast", ["TimeXer (NF)"]),
              "Kumo": ("classifier", ["Kumo"]), "|d| now": ("naive", ["|d| now"])}
    return h, series


def event_losses(events, context, times):
    """Per eligible holdout event: peak |d| in [onset, onset + 24 h] and |d| at the earliest admissible warning."""
    d = context.set_index("timestamp").depeg_bps.abs()
    rows = []
    for e in events[(events - pd.Timedelta(hours=MAX_LEAD) >= times[0]) & (events <= times[-1])]:
        rows.append(dict(event_time=e, peak=d.loc[e:e + pd.Timedelta(hours=PEAK_HOURS)].max(),
                         d_perfect=d.get(e - pd.Timedelta(hours=MAX_LEAD), np.nan)))
    return pd.DataFrame(rows).set_index("event_time")


def _holdout_job(col):
    ew, context, _, events = setup()
    h, _ = holdout_scores(context)
    frame = pd.DataFrame({"timestamp": h.index,
                          "depeg_bps": context.set_index("timestamp").depeg_bps.reindex(h.index).to_numpy()})
    d = context.set_index("timestamp").depeg_bps.abs()
    ev = event_losses(events, context, h.index)
    sc = h[col].to_numpy(float)
    grid = np.unique(np.quantile(sc[np.isfinite(sc)], np.linspace(0.3, 0.9995, 200)))
    rows = []
    for c in grid:
        m = score_events(frame, sc, c, events, MIN_LEAD, MAX_LEAD, COOLDOWN, outcome_context=context)
        avoided = 0.0
        for o in m["_event_outcomes"]:
            if o["detected"]:
                t_a = o["event_time"] - pd.Timedelta(hours=o["lead_hours"])
                avoided += max(ev.peak.get(o["event_time"], np.nan) - d.get(t_a, np.nan), 0.0)
        rows.append(dict(column=col, cutoff=c, D=m["detected"], N=m["events"],
                         A=m["timely_alert_episodes"] + m["false_alerts"], F=m["false_alerts"],
                         H=m["false_alert_exposure_hours"], avoided=avoided,
                         leads=[o["lead_hours"] for o in m["_event_outcomes"] if o["detected"]]))
    return rows


def envelope(tallies, losses, lead_aware):
    """V*(r) over the cutoff grid; lead_aware uses avoided bp instead of caught-event counts."""
    R = T.R
    L_perf = (losses.peak - losses.d_perfect).clip(lower=0)
    best = np.full(len(R), -np.inf)
    for _, t in tallies.iterrows():
        if lead_aware:
            cost = R * L_perf.mean()
            v = (t.avoided - cost * t.A) / (L_perf.sum() - cost * t.N)
        else:
            v = (t.D - R * t.A) / (t.N * (1 - R))
        best = np.maximum(best, v)
    return np.maximum(best, 0)


def run_holdout():
    ew, context, _, events = setup()
    h, series = holdout_scores(context)
    cache = DATA / "holdout_tallies.pkl"
    cols = [c for _, cs in series.values() for c in cs]
    if cache.exists():
        tallies = pd.read_pickle(cache)
    else:
        with Pool(8) as p:
            tallies = pd.DataFrame([r for rows in p.map(_holdout_job, cols) for r in rows])
        tallies.to_pickle(cache)
    losses = event_losses(events, context, h.index)
    losses.to_csv(DATA / "holdout_events.csv")
    rows, curves = [], []
    for m, (fam, cs) in series.items():
        env = {k: np.mean([envelope(tallies[tallies.column.eq(c)], losses, k == "lead") for c in cs], axis=0)
               for k in ("standard", "lead")}
        for k, e in env.items():
            curves += [dict(model=m, family=fam, kind=k, r=r, V=v) for r, v in zip(T.R, e)]
        rows.append({"model": m, "family": fam, "n_seeds": len(cs), "AUVC": env["standard"].mean(),
                     "AUVC_lead": env["lead"].mean()})
    pd.DataFrame(rows).to_csv(DATA / "holdout_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(DATA / "holdout_curves.csv", index=False)
    print(losses.round(1).to_string())
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # cost-loss explainer: V(r) of single cutoffs of one ANGEL seed, plus its envelope
    t = tallies[tallies.column.eq("ANGEL_full_seed_1000")].reset_index(drop=True)
    t[["cutoff", "D", "N", "A", "F", "H"]].to_csv(DATA / "explainer_angel_seed1000_cutoffs.csv", index=False)

    # event paths and the first timely alert of each model, at the cutoff maximising V(0.1) on the holdout
    # (ex post, identical treatment for every score; ANGEL = mean exceedance probability over its 6 seeds)
    d = context.set_index("timestamp").depeg_bps
    paths = []
    for e in losses.index:
        seg = d.loc[e - pd.Timedelta(hours=36):e + pd.Timedelta(hours=PEAK_HOURS)]
        paths += [dict(event_time=e, hours=(ts - e).total_seconds() / 3600, depeg_bps=v) for ts, v in seg.items()]
    pd.DataFrame(paths).to_csv(DATA / "holdout_event_paths.csv", index=False)
    frame = pd.DataFrame({"timestamp": h.index, "depeg_bps": d.reindex(h.index).to_numpy()})
    alerts = []
    for m, (fam, cs) in series.items():
        sc = h[cs].mean(axis=1).to_numpy(float)
        grid = np.unique(np.quantile(sc[np.isfinite(sc)], np.linspace(0.3, 0.9995, 200)))
        best, best_v = None, -np.inf
        for c in grid:
            r = score_events(frame, sc, c, events, MIN_LEAD, MAX_LEAD, COOLDOWN, outcome_context=context)
            v = (r["detected"] - 0.1 * (r["timely_alert_episodes"] + r["false_alerts"])) / (r["events"] * 0.9)
            if v > best_v:
                best, best_v = r, v
        for o in best["_event_outcomes"]:
            t_a = o["event_time"] - pd.Timedelta(hours=o["lead_hours"]) if o["detected"] else pd.NaT
            alerts.append(dict(model=m, family=fam, event_time=o["event_time"], detected=o["detected"],
                               lead_hours=o["lead_hours"], d_at_alert=abs(d.get(t_a, np.nan)) if o["detected"] else np.nan,
                               false_alerts_per_month=best["false_alerts_per_month"], V_0p1=best_v))
    alerts = pd.DataFrame(alerts)
    alerts["avoided_bp"] = (alerts.event_time.map(losses.peak) - alerts.d_at_alert).clip(lower=0)
    alerts.to_csv(DATA / "holdout_event_alerts.csv", index=False)
    print(alerts.groupby("model")[["detected", "lead_hours", "d_at_alert", "avoided_bp", "false_alerts_per_month",
                                   "V_0p1"]].mean().round(2).to_string())


if __name__ == "__main__":
    {"cv": run_cv, "holdout": run_holdout}[sys.argv[1]]()
