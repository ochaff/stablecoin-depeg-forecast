"""Value curves and WA-NBC table for the paper (CSV for plotting, LaTeX table), models renamed ANGEL -> SCQF.

    .venv/bin/python slides/compute_value_curves.py
    -> slides/data/paper/value_curves.csv          one row per (period, model, run, calibration, rule, r)
       slides/data/paper/value_curves_summary.csv  mean / min / max over runs (headline probability only)
       slides/data/paper/wanbc.csv, slides/tables/wanbc.tex

Sources: slides/data/calibrated_policy_grid.csv (main models; compute_calibrated.py) and
slides/data/ablation/ablation_policy_grid.csv (SCQF ablations, holdout, raw probability; compute_ablation_ew.py).
Tallies per probability cutoff p: D caught events, N events, A alert episodes, avoided / perfect lead-time losses (bp).

Rules
  decision   act when P >= r, valued at r (each user acts on its own cost-loss ratio; the decision curve)
  p=0.1      the fixed policy P >= 0.1, valued at every r
Value
  V(r)       = (D - r A) / (N (1 - r))                                         standard cost-loss value
  V_lead(r)  = (avoided - C A) / (perfect - C N),  C = r * perfect / N (bp)    lead-time-aware value: a warning
               avoids only the part of the move still ahead at the alert; C is the cost per alert episode in bp
Probability: raw for the forecasters (exceedance probability), Platt-recalibrated for the direct classifiers (map
fitted before the evaluation data); both are in value_curves.csv, `headline` marks the reported one.
WA-NBC = sum_r f(r) V(r) of the decision rule on r in [0.01, 0.5], f normalised on the grid
(slides/data/simple_cost_loss_weights.csv): simple expected-loss prior (headline), its excl.-USDC-2023-03 variant,
the LP prior; WA-NBC lead uses V_lead with the simple prior.
"""
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).resolve().parent
DATA, OUT, TAB = S / "data", S / "data/paper", S / "tables"
R = np.round(np.arange(0.01, 0.501, 0.01), 4)
RENAME = {"ANGEL exceedance": "SCQF", "ANGEL classifier": "SCQF classifier", "TiDE forecast": "TiDE",
          "TimeXer forecast": "TimeXer"}
PRIORS = {"simple": "simple: expected loss", "simple excl. USDC": "simple: excl. USDC 2023-03", "LP": "LP prior"}
ORDER = ["SCQF", "TiDE", "TimeXer", "GARCH-t", "ARIMA", "Kumo", "TabICL", "CatBoost", "XGBoost", "LightGBM",
         "Random forest", "SCQF classifier", "|d| now"]


def load():
    g = pd.read_csv(DATA / "calibrated_policy_grid.csv")
    g = g[g.split.eq("test")].groupby(["period", "model", "family", "calibration", "run", "p"],
                                      as_index=False)[["D", "N", "A", "F", "H", "avoided", "perfect"]].sum()
    g["model"] = g.model.replace(RENAME)
    a = pd.read_csv(DATA / "ablation/ablation_policy_grid.csv")
    a = a.assign(family="SCQF ablation", run="seed" + a.seed.astype(str), model="Ablation: " + a.model)
    g = pd.concat([g, a[g.columns]], ignore_index=True)
    g["p"] = g.p.round(4)
    g["headline"] = np.where(g.family.eq("classifier"), g.calibration.eq("calibrated"), g.calibration.eq("raw"))
    return g


def curves(g):
    rows = []
    for key, t in g.groupby(["period", "model", "family", "calibration", "run", "headline"]):
        t = t.set_index("p")
        if not set(R) <= set(t.index):
            continue
        tr, p1 = t.loc[R], t.loc[0.1]
        for rule, src in (("decision", tr), ("p=0.1", pd.DataFrame([p1] * len(R), index=R))):
            C = R * src.perfect.to_numpy() / src.N.to_numpy()
            rows.append(pd.DataFrame(dict(zip(["period", "model", "family", "calibration", "run", "headline"], key)),
                                     index=range(len(R))).assign(
                rule=rule, r=R, p=src.index.to_numpy() if rule == "decision" else 0.1, D=src.D.to_numpy(),
                N=src.N.to_numpy(), A=src.A.to_numpy(), C_bp=C,
                V=(src.D.to_numpy() - R * src.A.to_numpy()) / (src.N.to_numpy() * (1 - R)),
                V_lead=(src.avoided.to_numpy() - C * src.A.to_numpy()) / (src.perfect.to_numpy() - C * src.N.to_numpy())))
    return pd.concat(rows, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True); TAB.mkdir(exist_ok=True)
    c = curves(load())
    c.to_csv(OUT / "value_curves.csv", index=False)
    h = c[c.headline]
    summ = h.groupby(["period", "model", "family", "calibration", "rule", "r"]).agg(
        runs=("run", "nunique"), V=("V", "mean"), V_min=("V", "min"), V_max=("V", "max"),
        V_lead=("V_lead", "mean"), V_lead_min=("V_lead", "min"), V_lead_max=("V_lead", "max"),
        C_bp=("C_bp", "mean")).reset_index()
    summ.to_csv(OUT / "value_curves_summary.csv", index=False)

    W = pd.read_csv(DATA / "simple_cost_loss_weights.csv", index_col=0)
    assert np.allclose(W.index.to_numpy(), R)
    W = {k: W[col].to_numpy() / W[col].sum() for k, col in PRIORS.items()}
    d = h[h.rule.eq("decision")]
    rows = []
    for (period, model, cal), t in d.groupby(["period", "model", "calibration"]):
        per = t.pivot(index="run", columns="r", values="V")[R].to_numpy()
        lead = t.pivot(index="run", columns="r", values="V_lead")[R].to_numpy()
        r = dict(period=period, model=model, probability="Platt" if cal == "calibrated" else "raw", runs=len(per))
        for k, w in W.items():
            v = per @ w
            r[f"WA-NBC {k}"], r[f"WA-NBC {k} min"], r[f"WA-NBC {k} max"] = v.mean(), v.min(), v.max()
        v = lead @ W["simple"]
        r["WA-NBC lead"], r["WA-NBC lead min"], r["WA-NBC lead max"] = v.mean(), v.min(), v.max()
        r["V(0.1)"] = per[:, list(R).index(0.1)].mean()
        rows.append(r)
    wt = pd.DataFrame(rows)
    wt.to_csv(OUT / "wanbc.csv", index=False)

    # LaTeX: main models, CV and holdout side by side
    def cell(r, k):
        if r is None:
            return "--"
        s = f"{r[f'WA-NBC {k}']:.3f}".replace("-", "$-$")
        if r.runs > 1:
            s += rf" {{\scriptsize [{r[f'WA-NBC {k} min']:.2f}, {r[f'WA-NBC {k} max']:.2f}]}}".replace("-", "$-$")
        return s
    main_ = wt[~wt.model.str.startswith("Ablation") & wt.model.isin(ORDER)]
    L = [r"% generated by slides/compute_value_curves.py -- do not edit by hand",
         r"\begin{table}[pos=!htbp,align=\centering]",
         r"\caption{Weighted area under the net-benefit curve (WA-NBC) of the decision rule `alert when $\hat P\ge r$' "
         r"for $r\in[0.01,0.5]$. Weights: the expected-loss cost--loss prior (simple), the same prior without the "
         r"March 2023 USDC event (excl.\ USDC), and the lead-time-aware value under the simple prior (lead). "
         r"Forecasters use their raw exceedance probability; direct classifiers use a Platt recalibration fitted before "
         r"the evaluation data. CV: 2024 test folds; holdout: 2025-01-15 to 2026-05-06, five eligible events. "
         r"Brackets: range over seeds. 0 = never alerting, 1 = perfect warnings.}",
         r"\label{tab:wanbc}", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l l rrr rrr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{CV 2024} & \multicolumn{3}{c}{Holdout 2025--26} \\",
         r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
         r"Model & Prob. & simple & excl.\ USDC & lead & simple & excl.\ USDC & lead \\", r"\midrule"]
    fam = c.drop_duplicates("model").set_index("model").family
    last = None
    for m in ORDER:
        rs = {p: (main_[(main_.model == m) & (main_.period == p)].iloc[0]
                  if ((main_.model == m) & (main_.period == p)).any() else None) for p in ("CV", "holdout")}
        if all(v is None for v in rs.values()):
            continue
        f = fam.get(m)
        if last is not None and f != last:
            L.append(r"\addlinespace[2pt]")
        last = f
        any_r = next(v for v in rs.values() if v is not None)
        L.append(rf"{m} & {any_r.probability} & " + " & ".join(cell(rs[p], k) for p in ("CV", "holdout")
                                                               for k in ("simple", "simple excl. USDC", "lead"))
                 + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    (TAB / "wanbc.tex").write_text("\n".join(L) + "\n")
    pd.set_option("display.width", 250)
    show = ["period", "model", "probability", "runs", "WA-NBC simple", "WA-NBC simple excl. USDC", "WA-NBC lead",
            "WA-NBC LP", "V(0.1)"]
    print(wt[show].sort_values(["period", "WA-NBC simple"], ascending=[True, False]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
