"""SCQF ablations in calm and stressed forecast origins (six seeds per variant, all ablations).

    .venv/bin/python slides/compute_ablation_stress.py
    -> slides/data/ablation/scqf_ablation_stress.csv     % change vs full SCQF per regime, metric, with 95% joint CIs
       slides/tables/scqf_ablation_stress.tex           LaTeX table
       slides/figures/scqf_ablation_stress.{png,pdf}    figure

Input: comparison_nn_vs_arima/angel_ablation_study/test_loss_paths.parquet (per variant, seed and forecast origin:
the origin's losses averaged over the 24 horizons; written by ANGEL_ablation_analysis.ipynb from the MLflow 3603
runs listed in slides/data/ablation/run_manifest.csv; its pooled means equal forecast_summary_long.csv).
Regimes: an origin is stressed when at least one realised target of its 24-hour window lies outside +-10 bp (the
twCRPS thresholds), calm otherwise. % change = 100 * mean over seeds and regime origins of (variant - full) / mean of
full, seeds matched. 95% interval: joint seed-and-time bootstrap (2000 replicates; whole seeds resampled with
replacement, circular blocks of 168 consecutive forecast origins resampled over the whole test period, so each
replicate's regime sample keeps the time clustering of stress episodes), as for panel A of the ablation table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).resolve().parent
REPO = S.parent
PATHS = REPO / "comparison_nn_vs_arima/angel_ablation_study/test_loss_paths.parquet"
OUT, TAB, FIG = S / "data/ablation", S / "tables", S / "figures"
ORDER = [("crps_only", "CRPS loss only"), ("no_gpd", "No GPD tails"), ("gaussian", "Gaussian output"),
         ("no_sparsity", "No sparsity"), ("no_liq_shape", "No liquidity shape"), ("price_only", "Price only")]
GROUP = {"crps_only": "Loss and distributional head", "no_sparsity": "Input features"}
METRICS = ["CRPS", "twCRPS", "twCRPS low", "twCRPS high", "QL 1%", "QL 99%"]
TEX = {"CRPS": "CRPS", "twCRPS": "twCRPS", "twCRPS low": r"twCRPS$_{\mathrm{low}}$",
       "twCRPS high": r"twCRPS$_{\mathrm{high}}$", "QL 1%": r"QL$_{1\%}$", "QL 99%": r"QL$_{99\%}$"}
SHOW = ["CRPS", "twCRPS", "QL 1%", "QL 99%"]          # table columns per regime (all six metrics are in the csv)
BLOCK, N_BOOT, RNG_SEED = 168, 2000, 2026


def block_sums(x, length):
    """Sums of every circular block of `length` consecutive rows (axis 0)."""
    n = len(x)
    ext = np.concatenate([x, x[:length - 1]], axis=0)
    c = np.concatenate([np.zeros_like(x[:1]), np.cumsum(ext, axis=0)], axis=0)
    return c[length:length + n] - c[:n]


def main():
    for d in (OUT, TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    lp = pd.read_parquet(PATHS)
    seeds = sorted(lp.seed.unique())
    arr, stress = {}, None
    for (v, s), g in lp.groupby(["variant", "seed"]):
        g = g.sort_values("cutoff_idx")
        arr[(v, s)] = g[METRICS].to_numpy(float)
        st = g.stressed.to_numpy(bool)
        assert stress is None or np.array_equal(stress, st), (v, s)
        stress = st
    n, nS = len(stress), len(seeds)
    full = np.stack([arr[("full", s)] for s in seeds], 1)                         # (n, S, M)
    rng = np.random.default_rng(RNG_SEED)
    nb, rem = divmod(n, BLOCK)
    starts = rng.integers(0, n, size=(N_BOOT, nb + bool(rem)))
    draw = rng.integers(0, nS, size=(N_BOOT, nS))

    def boot_sum(x):
        """(N_BOOT, S, M) sums over each replicate's resampled origins."""
        out = block_sums(x, BLOCK)[starts[:, :nb]].sum(1)
        if rem:
            out += block_sums(x, rem)[starts[:, -1]]
        return out

    def joint(x):                                                                  # resample seeds
        return np.take_along_axis(x, draw[:, :, None], 1).mean(1)

    rows = []
    for regime, m in (("all", np.ones(n, bool)), ("calm", ~stress), ("stressed", stress)):
        w = m[:, None, None].astype(float)
        base_pt = (full * w).sum(0).mean(0)
        base_bt = joint(boot_sum(full * w))
        rows += [dict(variant="full", model="Full SCQF", regime=regime, metric=k, n_origins=int(m.sum()),
                      n_seeds=nS, full_mean=base_pt[j] / m.sum()) for j, k in enumerate(METRICS)]
        for v, name in ORDER:
            diff = np.stack([arr[(v, s)] for s in seeds], 1) - full
            pt = 100 * (diff * w).sum(0).mean(0) / base_pt
            bt = 100 * joint(boot_sum(diff * w)) / base_bt
            lo, hi = np.quantile(bt, [0.025, 0.975], axis=0)
            per_seed = 100 * (diff * w).sum(0) / (full * w).sum(0)                  # (S, M)
            for j, k in enumerate(METRICS):
                rows.append(dict(variant=v, model=name, regime=regime, metric=k, n_origins=int(m.sum()),
                                 n_seeds=nS, pct=pt[j], ci95_low=lo[j], ci95_high=hi[j],
                                 seeds_worse=int((per_seed[:, j] > 0).sum())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "scqf_ablation_stress.csv", index=False)

    # ------------------------------------------------------------------ LaTeX
    idx = t.set_index(["variant", "regime", "metric"])
    sgn = lambda x: f"{x:+.1f}".replace("-", "$-$").replace("+", "$+$")
    rng_ = lambda a, b: rf"{{\scriptsize [{a:.1f}, {b:.1f}]}}".replace("-", "$-$")
    nc, ns = int((~stress).sum()), int(stress.sum())
    k = len(SHOW)
    L = [r"% generated by slides/compute_ablation_stress.py -- do not edit by hand",
         r"\begin{table}[pos=!htbp,align=\centering]",
         r"\caption{Ablations of the SCQF model in calm and stressed forecast origins, six seeds per variant. An origin "
         r"is stressed when at least one realised peg deviation in its 24-hour forecast window lies outside "
         rf"$\pm10$\,bps ({ns:,} of {n:,} test origins), calm otherwise ({nc:,}). The full-model row reports the mean "
         r"loss in each regime; ablation rows report the percentage change relative to the full model (positive is "
         r"worse), with 95\% joint seed-and-time block-bootstrap intervals (circular blocks of 168 forecast origins) "
         r"below. Bold marks changes whose interval excludes zero.}",
         r"\label{tab:scqf_ablation_stress}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}l {'r' * k} {'r' * k}@{{}}}}", r"\toprule",
         rf" & \multicolumn{{{k}}}{{c}}{{Calm}} & \multicolumn{{{k}}}{{c}}{{Stressed}} \\",
         rf"\cmidrule(lr){{2-{k + 1}}}\cmidrule(l){{{k + 2}-{2 * k + 1}}}",
         "Ablation & " + " & ".join(TEX[x] for x in SHOW * 2) + r" \\", r"\midrule",
         "Full SCQF & " + " & ".join(f"{idx.loc[('full', r, x), 'full_mean']:.4f}"
                                      for r in ("calm", "stressed") for x in SHOW) + r" \\"]
    for v, name in ORDER:
        if v in GROUP:
            L += [r"\addlinespace[2pt]", rf"\multicolumn{{{2 * k + 1}}}{{@{{}}l}}{{\textit{{{GROUP[v]}}}}} \\"]
        cells, cis = [], []
        for r in ("calm", "stressed"):
            for x in SHOW:
                q = idx.loc[(v, r, x)]
                s = sgn(q.pct)
                cells.append(rf"$\mathbf{{{s.replace('$', '')}}}$" if (q.ci95_low > 0 or q.ci95_high < 0) else s)
                cis.append(rng_(q.ci95_low, q.ci95_high))
        L += [rf"\quad {name} & " + " & ".join(cells) + r" \\", " & " + " & ".join(cis) + r" \\"]
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    (TAB / "scqf_ablation_stress.tex").write_text("\n".join(L) + "\n")

    # ------------------------------------------------------------------ figure: % change with CI, calm vs stressed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"calm": "#2a78b5", "stressed": "#c8553d"}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 4.2), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, x in zip(axes, METRICS):
        for off, r in ((0.17, "calm"), (-0.17, "stressed")):
            q = idx.loc[[(v, r, x) for v, _ in ORDER]]
            ax.errorbar(q.pct, y + off, xerr=[q.pct - q.ci95_low, q.ci95_high - q.pct], fmt="o", ms=5, lw=1.5,
                        color=col[r], capsize=2, label=r.capitalize())
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_title(x, fontsize=10)
        ax.set_xlabel("% change vs full SCQF")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, [nm for _, nm in ORDER])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"Calm ({nc:,} origins)", f"Stressed ({ns:,} origins)"], frameon=False, fontsize=9,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("SCQF ablations, 6 seeds: % change vs full SCQF (positive = worse), "
                 "95% joint seed-and-time bootstrap intervals", fontsize=10, y=1.06)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"scqf_ablation_stress.{ext}", dpi=300, bbox_inches="tight")

    pd.set_option("display.width", 220)
    p = t[t.variant.ne("full") & t.regime.ne("all")]
    p = p.assign(cell=p.pct.map("{:+.1f}".format) + " [" + p.ci95_low.map("{:.1f}".format) + ", "
                 + p.ci95_high.map("{:.1f}".format) + "]")
    print(p.pivot_table(index=["model", "regime"], columns="metric", values="cell", aggfunc="first",
                        sort=False)[METRICS].to_string())


if __name__ == "__main__":
    main()
