"""Cost-loss prior f(r) for the decision-curve summary (WA-NBC), estimated from LP economics before the holdout.

    .venv/bin/python slides/compute_cost_loss_prior.py      # -> slides/data/cost_loss_*.csv

A user is an LP running the Gamma-style vault of the LP backtest (onchain-insights/14, `lpsim`), on the Uniswap V3
USDC/USDT 0.01% pool, who reacts to a warning by exiting to a 50/50 inventory for the alert episode.
PnL is the backtest's hedged PnL (fees + LVR - gas - swap costs, i.e. against the rebalancing portfolio), bp of V0.

  C  cost of acting on a false alert: hedged PnL lost by exiting during an episode with no event (fees forgone +
     burn/mint gas and swap cost - LVR avoided). Placebo episodes at random quiet hours with fee data
     (2024-03-20 .. holdout start), with the durations of the event alert windows.
  L_e  loss a perfect warning avoids on event e: hedged PnL gained by exiting from 24 h before the onset to the end of
     the breach episode (12 h quiet reset), counted to 48 h after the onset, plus C (the cost-loss scheme books
     C for every action: gain = L - C). All pool 15 bp onsets before the holdout (2022-05 .. 2025-01).
     Before 2024-03-19 the pool data have no fees; the fees the base vault would have earned in the window are
     imputed with the mean fee rate (bp of V0 per hour) of the base vault during the fee-observed event windows.

Users: vault size V0 in {100k, 1M, 5M} x cost scenario {low: $2 gas, no slippage; base: $10 gas, 0.5 bp; high: $50
gas, 1 bp} (the backtest's sensitivity grid), equally weighted. r = C / E[L]:
  sample mean         E[L] = mean of L_e, bootstrap over events; r draws of all users pooled form f(r)
  GPD tail            E[L] from a body + GPD-tail fit (tail above the 75th percentile); infinite when the fitted
                      shape is >= 1, then r = 0 (acting always pays)
  excl. USDC 2023-03  sample mean without the USDC/SVB event of March 2023
  per event           r_e = C / L_e, every event equally weighted (typical event rather than expectation)
r <= 0 (C <= 0 or E[L] infinite): acting always pays, put at the grid's lowest r; E[L] <= 0: acting never pays,
dropped (share reported). The weights on the r grid [0.01, 0.5] are a Gaussian kernel density (bandwidth 0.02) of the
draws (cost_loss_weights.csv); a Beta(a, b) fitted by moments to the draws in (0, 1) is reported as a summary.
Events with missing prices in the window (2022-05-11) are dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto

REPO = Path(__file__).resolve().parents[1]
S14 = REPO.parent / "onchain-insights/14. LP vault backtest"
sys.path.insert(0, str(S14))
import config as C14  # noqa: E402
from data import market, events as pool_events, panel  # noqa: E402
from lpsim import Sim  # noqa: E402
from strategies import Recenter, AlertReaction, episodes_from_flags  # noqa: E402

DATA = REPO / "slides/data"
HOLDOUT_START = pd.Timestamp("2025-01-15 00:00", tz="UTC")
FEE_START = pd.Timestamp(C14.FEE_DATA_START, tz="UTC")
PRE, POST, WARM = 24, 48, 72            # exit from 24 h before onset; count to 48 h after; 72 h warm-up
USERS = [(v, c, g, s) for v in (1e5, 1e6, 5e6)
         for c, g, s in (("low", 2.0, 0.0), ("base", 10.0, 0.5), ("high", 50.0, 1.0))]
N_PLACEBO, N_BOOT, SEED = 150, 2000, 7
EXCL = pd.Timestamp("2023-03-10 21:00", tz="UTC")
PRIORS = ("sample mean", "GPD tail", "excl. USDC 2023-03", "per event")
R = np.round(np.arange(0.01, 0.501, 0.01), 4)     # the r grid of the early-warning evaluation
BW = 0.02                                         # kernel bandwidth for the weights on the grid


def exit_gain(t0, t_exit, t_end, alert_len, V0, gas, slip):
    """Hedged PnL (bp of V0) of exiting for `alert_len` hours from t_exit, minus that of the base vault, plus the
    base vault's fees over the exit hours (for the imputation)."""
    m = market(t0, t_end)
    a = np.zeros(m.n, bool)
    k = m.idx.get_loc(t_exit)
    a[k:k + alert_len] = True
    sim = Sim(m, V0=V0, gas=gas, slip_bps=slip, fee_bps=C14.POOL_FEE_BPS)
    base = sim.run(Recenter(C14.GAMMA_HALF_WIDTH))
    ex = sim.run(AlertReaction(Recenter(C14.GAMMA_HALF_WIDTH), a, "exit", np.zeros(m.n), np.zeros(m.n)))
    bp = 1e4 / V0
    return ((ex.hedged.sum() - base.hedged.sum()) * bp, base.fees.values[a].sum() * bp, int(a.sum()))


def event_windows():
    p = panel()
    d = p.depeg_bps.abs().values
    breach = pd.Series(episodes_from_flags(d >= C14.EVENT_BP, C14.KUMO_QUIET), index=p.index)
    m = market(p.index[0], HOLDOUT_START)
    rows = []
    for i in pool_events(m):
        t = m.idx[i]
        if t - pd.Timedelta(hours=PRE + WARM) < p.index[0]:
            continue
        j = p.index.get_loc(t)
        e = j
        while e + 1 < len(p) and breach.iloc[e + 1]:
            e += 1
        L = PRE + (e - j + 1)
        rows.append(dict(onset=t, alert_len=L, peak=float(d[j:j + 25].max()),
                         end=min(t + pd.Timedelta(hours=max(POST, e - j + 1)), HOLDOUT_START - pd.Timedelta(hours=1))))
    return pd.DataFrame(rows)


def placebo_starts(ev, rng):
    p = panel().loc[FEE_START:HOLDOUT_START - pd.Timedelta(hours=1)]
    far = (p.depeg_bps.abs().rolling(24 * 7, center=True, min_periods=1).max() < C14.EVENT_BP)   # 3.5 days quiet
    cand = p.index[far.values]
    cand = cand[(cand - pd.Timedelta(hours=WARM) >= FEE_START) & (cand + pd.Timedelta(hours=96) < HOLDOUT_START)]
    starts = rng.choice(cand, N_PLACEBO, replace=False)
    lens = rng.choice(ev.alert_len.clip(upper=72).values, N_PLACEBO)
    return [(pd.Timestamp(s), int(n)) for s, n in zip(starts, lens)]


def tail_mean(x, q=0.75):
    """E[X] with a GPD for the excesses over the q-quantile (body: empirical); inf if the fitted shape >= 1."""
    x = np.asarray(x, float)
    u = np.quantile(x, q)
    exc = x[x > u] - u
    body = x[x <= u]
    if len(exc) < 4:
        return x.mean(), np.nan
    xi, _, sc = genpareto.fit(exc, floc=0)
    tail = u + (sc / (1 - xi) if xi < 1 else np.inf)
    w = len(exc) / len(x)
    return (1 - w) * body.mean() + w * tail, xi


def ratio(c, l):
    """r = C / L; inf when L <= 0 (acting never pays), 0 when L = inf."""
    if not np.isfinite(l):
        return 0.0
    return c / l if l > 0 else np.inf


def beta_mom(r):
    r = np.asarray(r, float)
    r = r[(r > 0) & (r < 1)]
    m, v = r.mean(), r.var()
    k = m * (1 - m) / v - 1
    return m * k, (1 - m) * k


def simulate():
    rng = np.random.default_rng(SEED)
    ev = event_windows()
    placebo = placebo_starts(ev, rng)
    rows = []
    for V0, cname, gas, slip in USERS:
        tag = dict(V0=V0, cost=cname)
        for s, n in placebo:
            g, f, h = exit_gain(s - pd.Timedelta(hours=WARM), s, s + pd.Timedelta(hours=n + POST), n, V0, gas, slip)
            rows.append(dict(**tag, kind="placebo", t=s, hours=h, gain=g, fees_forgone=f))
        for e in ev.itertuples():
            t_exit = e.onset - pd.Timedelta(hours=PRE)
            g, f, h = exit_gain(t_exit - pd.Timedelta(hours=WARM), t_exit, e.end, e.alert_len, V0, gas, slip)
            rows.append(dict(**tag, kind="event", t=e.onset, hours=h, gain=g, fees_forgone=f, peak=e.peak))
        print(f"V0={V0:,.0f} {cname}: done", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(DATA / "cost_loss_sims.csv", index=False)
    return ev, T


def summarise(ev, T):
    rng = np.random.default_rng(SEED + 1)
    T = T[T.t.isin(ev.onset) | T.kind.eq("placebo")]

    out, draws = [], {k: [] for k in PRIORS}
    for (V0, cname), g in T.groupby(["V0", "cost"], sort=False):
        pl, evr = g[g.kind.eq("placebo")], g[g.kind.eq("event")].copy()
        Cu = -pl.gain.mean()                                           # cost per false-alert episode (bp of V0)
        obs = evr.t >= FEE_START
        fee_rate = (evr.fees_forgone[obs] / evr.hours[obs]).mean()
        evr["L"] = evr.gain + Cu - np.where(obs, 0.0, fee_rate * evr.hours)   # impute forgone fees pre-2024-03
        L, usdc = evr.L.values, (evr.t == EXCL).values
        EL_gpd, xi = tail_mean(L)
        out.append(dict(V0=V0, cost=cname, C_bp=Cu, C_sd=pl.gain.std(), fee_rate_bp_h=fee_rate,
                        L_median=np.median(L), L_mean=L.mean(), L_mean_excl=L[~usdc].mean(), gpd_xi=xi,
                        share_L_pos=(L > 0).mean(), r_mean=ratio(Cu, L.mean()), r_mean_excl=ratio(Cu, L[~usdc].mean()),
                        r_gpd=ratio(Cu, EL_gpd)))
        for _ in range(N_BOOT // len(USERS)):
            b = rng.choice(len(L), len(L))
            draws["sample mean"].append(ratio(Cu, L[b].mean()))
            draws["GPD tail"].append(ratio(Cu, tail_mean(L[b])[0]))
            bx = rng.choice(np.flatnonzero(~usdc), (~usdc).sum())
            draws["excl. USDC 2023-03"].append(ratio(Cu, L[bx].mean()))
        draws["per event"] += [ratio(Cu, x) for x in L]
    U = pd.DataFrame(out)
    U.to_csv(DATA / "cost_loss_users.csv", index=False)
    pri, W = [], {}
    for k, r in draws.items():
        r = np.asarray(r, float)
        never = ~np.isfinite(r)                                        # E[L] <= 0: acting never pays
        rr = np.clip(r[~never], 0.0, None)                             # C <= 0 or E[L] = inf: acting always pays (r = 0)
        a, b = beta_mom(np.clip(rr[rr < 1], 1e-3, None))
        w = np.exp(-0.5 * ((R[:, None] - np.clip(rr[rr <= 0.5], R[0], None)[None, :]) / BW) ** 2).sum(1)
        W[k] = w / w.sum()
        pri.append(dict(prior=k, n=len(r), share_never=never.mean(), share_r_lt_0p01=np.mean(rr < 0.01) * (~never).mean(),
                        share_r_gt_0p5=np.mean(rr > 0.5) * (~never).mean(), median_r=np.median(rr), mean_r_in_grid=
                        np.average(R, weights=W[k]), beta_a=a, beta_b=b))
    P = pd.DataFrame(pri)
    P.to_csv(DATA / "cost_loss_prior.csv", index=False)
    pd.DataFrame(W, index=pd.Index(R, name="r")).to_csv(DATA / "cost_loss_weights.csv")
    pd.set_option("display.width", 220)
    print(U.round(3).to_string(index=False))
    print(P.round(3).to_string(index=False))


if __name__ == "__main__":
    cache = DATA / "cost_loss_sims.csv"
    if "--from-cache" in sys.argv and cache.exists():
        ev = event_windows()
        T = pd.read_csv(cache, parse_dates=["t"])
    else:
        ev, T = simulate()
    summarise(ev[ev.peak.notna()], T)
