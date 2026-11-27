# Paper tables and figures: where they are

Paths are relative to the parent folder that holds both repositories:
- `onchain-insights/` (`S12` = `onchain-insights/12. Evaluation of Probabilistic Forecasts`)
- `stablecoin-depeg-forecast/` (this repository; `SL` = `stablecoin-depeg-forecast/slides`, the folder of this file)

The model is called **SCQF** (formerly ANGEL) in every table listed here.

## Seeds

| Part | Seeds |
|---|---|
| Forecast evaluation (Section 5.2: benchmark, DM, VaR/ES, calibration figure) | 6 per trained model: SCQF = the six full-model ablation runs (MLflow 3603, seeds 1000, 1001, 1010, 1011, 1101, 1111); TiDE and TimeXer (spline + GPD head) = seeds 1, 7, 42, 100, 1233, 2026. GARCH, ARIMA, Naive: deterministic, 1 run |
| Ablation | 6 matched seeds per variant (same seeds as above) |
| Early-warning comparison | 3 per trained model (SCQF holdout: 1000, 1001, 1010, the three lowest of the six; the 6-seed holdout WA-NBC is 0.292 and belongs in a footnote) |

## Forecast evaluation (Section 5.2)

| Paper table / figure | File (LaTeX unless stated) | Data | Generator |
|---|---|---|---|
| Benchmark, all horizons (`tab:benchmark`) | `S12/benchmark_tables/seed_benchmark/benchmark_distributional_all_horizons.tex` | `.csv`, SD over seeds in `_sd.csv`, improvements in `benchmark_distributional_improvements_all_horizons.csv` | `S12/tables/table_seed_benchmark.py` |
| Benchmark at h = 1, 12, 24 | `.../seed_benchmark/benchmark_distributional_{h1,h12,h24}.tex` | same pattern | same |
| Pairwise DM tests (24 h and pooled 1-24 h) | `.../seed_benchmark/section_529_pairwise_dm_seed_bootstrap_SCQF_{24h,1_24h}.tex` | `.csv` | same |
| VaR / ES backtests per seed (`tab:var_es_backtests_seeds`) | `S12/benchmark_tables/var_es_backtests/var_es_backtests_seed_summary.tex` | `var_es_backtests_seed_summary.csv`, every seed in `var_es_backtests_per_seed.csv` | `S12/tables/table_var_es_seed_backtests.py` |
| PIT / VaR / ES calibration figure | `S12/PIT_VAR_ES_calibration_seeds.{png,pdf}` (replaces `PIT_VAR_ES_calibration.png`) | `S12/benchmark_tables/var_es_backtests/pit_var_es_seed_bands.csv` | `S12/tables/figure_pit_var_es_seeds.py` |

Three-seed versions (SCQF 1000/1001/1010; TiDE/TimeXer 1/7/42), kept for reference: `S12/benchmark_tables/seed_benchmark_3seeds/`,
`var_es_backtests/*_3seeds.*`, `S12/PIT_VAR_ES_calibration_seeds_3seeds.{png,pdf}`.
Old single-run tables (superseded, do not cite): `S12/benchmark_tables/full_benchmark/`, `dm_tests/`,
`var_es_backtests/var_es_backtests_24h_comparison.tex`, `angel_var_es_backtests.tex`.

Key six-seed numbers (all horizons, seed means): SCQF CRPS 0.8253, twCRPS 0.0687; best benchmark TimeXer 1.0279 / 0.0858.
SCQF vs TimeXer: CRPS -19.7%, twCRPS -19.9%, QL 1% -29.6%, QL 99% -35.7%. SCQF vs GARCH (1-24 h): CRPS -62.7%,
twCRPS -35.2%. Every DM comparison is significant (24 h twCRPS vs GARCH is the weakest: p = 0.021, CI [-0.037, -0.002];
24 h twCRPS vs TiDE: 29/36 seed pairs significant). Exceptions to "SCQF best everywhere": TiDE has the lowest QL 1% and
QL 99% at h = 1; TimeXer the lowest lower-tail twCRPS at h = 24.

VaR / ES (six seeds): SCQF is not rejected at 0.5% for nearly all seeds; the 12 h lower tail at 1% is under-covered for
every seed (17-21 breaches vs 9.5 expected); at 24 h / 1% two of six seeds are rejected per tail. TiDE and GARCH are
rejected in every cell, TimeXer in most (strong seed dependence: 5-90 breaches at 12 h / 0.5% lower).

## Ablation

| Paper table | File | Data | Generator |
|---|---|---|---|
| SCQF ablation (`tab:scqf_ablation`): A. forecast losses, B. holdout early warning | `SL/tables/scqf_ablation.tex` | `SL/data/ablation/scqf_ablation_table.csv`, per seed `scqf_ablation_early_warning_seeds.csv` | `SL/compute_ablation_table.py` (early-warning inputs from `SL/compute_ablation_ew.py`) |
| Ablations in calm vs stressed origins (`tab:scqf_ablation_stress`) and figure | `SL/tables/scqf_ablation_stress.tex`; figure `SL/figures/scqf_ablation_stress.{png,pdf}` | `SL/data/ablation/scqf_ablation_stress.csv` (all six metrics; calm / stressed / all origins; CIs; seeds worse than full) | `SL/compute_ablation_stress.py` |

Early warning in the ablation is evaluated on the holdout only (5 eligible events); the CV folds were not
rerun for the ablation variants.

The manuscript currently contains an older hand-edited version (`tab:ablation`); a standalone copy is in
`stablecoin-depeg-forecast/comparison_nn_vs_arima/angel_ablation_study/ablation_table_manuscript.tex`. Use
`SL/tables/scqf_ablation.tex` instead. The older calm/stressed table without intervals
(`angel_ablation_study/stress_vs_calm.tex`) is superseded by `SL/tables/scqf_ablation_stress.tex`.

## Early warning: value and WA-NBC

| Paper table / figure | File | Data | Generator |
|---|---|---|---|
| WA-NBC of every model, CV and holdout (`tab:wanbc`) | `SL/tables/wanbc.tex` | `SL/data/paper/wanbc.csv` (also LP prior, V(0.1), seed ranges) | `SL/compute_value_curves.py` |
| Value curves (standard and lead-time-aware), for plotting | n/a (figure to be drawn) | `SL/data/paper/value_curves.csv` (every run), `value_curves_summary.csv` (mean / min / max over seeds) | same |
| Cost-loss prior (weights of WA-NBC) | n/a | `SL/data/simple_cost_loss_weights.csv` | `SL/compute_simple_cost_loss_prior.py` |

Upstream of these: `SL/compute_results.py` (`cv`, `holdout`, `policy` stages) and `SL/compute_calibrated.py`
(`SL/data/calibrated_policy_grid.csv`). Rebuild order: `compute_results.py {cv,holdout,policy}` ->
`compute_calibrated.py` -> `compute_value_curves.py`.

Key numbers (WA-NBC, simple prior, holdout / CV): SCQF 0.314 / 0.322, Kumo 0.084 / 0.096, GARCH-t 0.011 / 0.228,
TimeXer 0.009 / 0.180, XGBoost 0.000 / 0.078, CatBoost -0.025 / 0.084, TabICL -0.048 / 0.035, TiDE -0.079 / 0.031,
LightGBM -0.083 / 0.023, random forest -0.087 / 0.072.

## Not covered here

LP vault backtest tables (`14. LP vault backtest/paper/generated/`, from `paper_tables.py`) and the early-warning
classifier CV tables of Section 8 (e.g. `tab:cv_model_alpha_results`, `tab:ablation_15_kumo_tabular`) were not changed
in this update.
