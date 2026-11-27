#!/usr/bin/env bash
# Extra seeds / missing holdout runs for the early-warning comparison (slides + paper), 2026-11-26.
#   CPU  : (GPU conda env, which has xgboost/lightgbm) XGBoost and LightGBM holdout runs, seeds 1233/7/42 (same protocol as the RF/CatBoost holdout runs)
#   GPU  : TabICL holdout (1233/7/42), Kumo holdout seeds 7/42, TiDE/TimeXer holdout forecasts seeds 7/42,
#          TiDE/TimeXer per-fold CV exceedance runs seeds 7/42 -- run one after another on the single GPU.
# Usage: bash scripts/run_extra_seeds.sh cpu|gpu      (logs in logs/extra_seeds/)
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=.venv/bin/python
GPU=/home/jovyan/.conda/envs/GPU/bin/python
Q8="../onchain-insights/8. Early-Warning Classifier Cross Validation"
LOG=logs/extra_seeds; mkdir -p "$LOG"
DS=preprocessed_datasets/dataset_alpha_0.1_full.parquet
HOLDOUT_ARGS=(--alpha 0.1 --dataset_path "$DS" --dataset_file "$DS" --experiment_name earlywarning_trees_holdout_2026-11-23
              --cv_start 2025-01-15T15:00 --cv_end 2026-05-07 --n_splits 1 --holdout_start 2030-01-01T00:00:00+00:00
              --n_bootstrap 10)

run() { echo "[$(date -u +%FT%TZ)] START $1"; shift; "$@"; local rc=$?; echo "[$(date -u +%FT%TZ)] END rc=$rc"; }

case "${1:-}" in
cpu)
  for s in 1233 7 42; do
    run "xgb/lgbm holdout seed $s" $GPU cv_earlywarning.py "${HOLDOUT_ARGS[@]}" --run_name holdout_xl_seed$s \
        --random_state $s --model_names xgboost lightgbm
  done ;;
gpu)
  for s in 1233 7 42; do
    run "tabicl holdout seed $s" $GPU cv_earlywarning.py "${HOLDOUT_ARGS[@]}" --run_name holdout_tabicl_seed$s \
        --random_state $s --model_names tabicl_v2
  done
  (cd "$Q8" && run "kumo holdout seeds 7 42" $GPU kumo_native_holdout_seeds.py --seeds 7 42)
  for s in 7 42; do
    BENCH_SEED=$s BENCH_OUT=./benchmark_outputs/tide_hist_exog_seed$s run "TiDE holdout seed $s" $GPU benchmark_scripts/TiDE_forecast.py
    BENCH_SEED=$s BENCH_OUT=./benchmark_outputs/timeXer_hist_exog_seed$s run "TimeXer holdout seed $s" $GPU benchmark_scripts/TimeXer_forecast.py
  done
  for s in 7 42; do
    run "TiDE/TimeXer CV seed $s" $GPU cv_benchmark_exceedance.py --models tide_nf timexer_nf --random_state $s
  done ;;
*) echo "usage: $0 cpu|gpu"; exit 1 ;;
esac
