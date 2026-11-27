#!/usr/bin/env bash
# CV refits (every fold) of the paper's spline + GPD TiDE / TimeXer benchmarks for the early-warning comparison,
# seeds 1 (the release seed), 7, 42. Waits for the section 12 holdout seed runs to free the GPU.
cd "$(dirname "$0")/.."
GPU=/home/jovyan/.conda/envs/GPU/bin/python
while pgrep -f run_spline_seeds.sh >/dev/null; do sleep 30; done
for s in 1 7 42; do
  echo "[$(date -u +%FT%TZ)] START spline-GPD TiDE/TimeXer CV seed $s"
  $GPU cv_benchmark_exceedance.py --models tide_sg timexer_sg --random_state $s; rc=$?
  echo "[$(date -u +%FT%TZ)] END seed $s rc=$rc"
done
