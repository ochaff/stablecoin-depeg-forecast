"""Simple cost-loss prior f(r) for the decision-curve summary (WA-NBC), and a comparison with the LP prior.

    .venv/bin/python slides/compute_simple_cost_loss_prior.py   # -> slides/data/simple_cost_loss_{weights,wanbc}.csv

User: holds a stablecoin position and, on a warning, swaps it out of the pool's pair and back after the episode.
  C  = round-trip transaction cost c, in bp of the position (2 x 1 bp pool fee plus slippage/gas);
       users differ in c: log-uniform on [C_LO, C_HI] bp.
  L  = loss avoided by a timely warning = peak |deviation| (bp) in the 24 h after onset, for the 35 pool 15 bp
       onsets before the holdout (same events as compute_cost_loss_prior.py, read from its cached sims).
  Because C and L are both in bp of the position, r = C/L does not depend on position size.
Priors on r:
  expected loss       r = c / E[L], E[L] = mean peak, bootstrap over events (risk-neutral user, one draw per user)
  excl. USDC 2023-03  same without the March 2023 event
  median event        r = c / median(L)  (the typical event; robust to the USDC tail)
r < 0.01 is put at 0.01 (acting pays for all evaluated cutoffs); r > 0.5 lies outside the evaluated grid and is
dropped (share reported). Weights: Gaussian kernel (bandwidth 0.02) on the r grid, as for the LP prior.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
R = np.round(np.arange(0.01, 0.501, 0.01), 4)
BW, N_DRAW, SEED = 0.02, 20000, 7
C_LO, C_HI = 2.0, 10.0
EXCL = pd.Timestamp("2023-03-10 21:00", tz="UTC")


def kde(r):
    w = np.exp(-0.5 * ((R[:, None] - r[None, :]) / BW) ** 2).sum(1)
    return w / w.sum()


def draws(L, rng, stat):
    c = np.exp(rng.uniform(np.log(C_LO), np.log(C_HI), N_DRAW))
    if stat == "mean":
        Lb = rng.choice(L, size=(N_DRAW, len(L))).mean(1)          # bootstrap E[L]
    else:
        Lb = np.median(rng.choice(L, size=(N_DRAW, len(L))), axis=1)
    return c / Lb


def main():
    sims = pd.read_csv(DATA / "cost_loss_sims.csv", parse_dates=["t"])
    ev = sims[sims.kind.eq("event")].drop_duplicates("t").dropna(subset=["peak"])
    L_all = ev.peak.to_numpy()
    L_ex = ev.loc[ev.t.ne(EXCL), "peak"].to_numpy()
    rng = np.random.default_rng(SEED)
    raw = {"simple: expected loss": draws(L_all, rng, "mean"),
           "simple: excl. USDC 2023-03": draws(L_ex, rng, "mean"),
           "simple: median event": draws(L_all, rng, "median")}
    W, summ = {}, []
    for k, r in raw.items():
        summ.append(dict(prior=k, median_r=np.median(r), q05=np.quantile(r, .05), q95=np.quantile(r, .95),
                         share_gt_0p5=(r > 0.5).mean(), share_lt_0p01=(r < 0.01).mean()))
        W[k] = kde(np.clip(r[r <= 0.5], 0.01, None))
    lp = pd.read_csv(DATA / "cost_loss_weights.csv", index_col=0)
    for k, col in (("LP prior", "sample mean"), ("LP: excl. USDC 2023-03", "excl. USDC 2023-03")):
        W[k] = lp[col].to_numpy() / lp[col].sum()
    pd.DataFrame(W, index=pd.Index(R, name="r")).to_csv(DATA / "simple_cost_loss_weights.csv")
    print(f"L (bp): n={len(L_all)}, mean {L_all.mean():.1f}, excl. USDC {L_ex.mean():.1f}, median {np.median(L_all):.1f}")
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print("\nmean r under each weight:", {k: round(float(R @ w), 3) for k, w in W.items()})

    # WA-NBC of the headline models (same rule and data as the slides notebook, section 5)
    grid = pd.read_csv(DATA / "calibrated_policy_grid.csv")
    grid["p"] = grid.p.round(4)
    fam = grid.drop_duplicates("model").set_index("model").family
    rows = []
    for (period, model), g in grid[grid.split.eq("test")].groupby(["period", "model"]):
        cal = "calibrated" if fam[model] == "classifier" else "raw"
        g = g[g.calibration.eq(cal)].groupby(["run", "p"])[["D", "N", "A"]].sum().reset_index()
        curves = []
        for _, gr in g.groupby("run"):
            gr = gr.set_index("p").reindex(R)
            if gr.D.isna().any():
                continue
            curves.append(((gr.D - R * gr.A) / (gr.N * (1 - R))).to_numpy())
        if curves:
            c = np.array(curves)
            rows.append(dict(period=period, model=model, runs=len(c), **{k: float((c @ w).mean()) for k, w in W.items()}))
    t = pd.DataFrame(rows)
    t.to_csv(DATA / "simple_cost_loss_wanbc.csv", index=False)
    for period in ("CV", "holdout"):
        s = t[t.period.eq(period)].set_index("model").drop(columns=["period", "runs"]).sort_values("LP prior", ascending=False)
        print(f"\n{period}\n", s.round(3).to_string())
        print("Spearman rank corr with LP prior:", s.corr(method="spearman")["LP prior"].round(3).to_dict())


if __name__ == "__main__":
    main()
