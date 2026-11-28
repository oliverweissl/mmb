#!/usr/bin/env bash
# Run dirs -> per-run CSV -> tables -> figure (continuous genotype; budget-bound metrics use in-budget data only).
#   scripts/analysis/run_analysis.sh <runs_root> <exp_id> [ref_arm=a0_directed]
# Writes results/<exp_id>.csv, results/tables/<exp_id>/* (summary, curve, paired vs the reference incl. the
# smallest-crossing magnitude), results/figures/<exp_id>_crossing_curve.pdf.
# Calibration of (w_img, w_txt): python -m mmb.analysis.calibration analyze <step runs or CSVs ...> --out results/calib
set -euo pipefail
PY="${PYTHON:-python}"
ROOT="${1:?runs root}"; EXP="${2:?exp id}"; REF="${3:-a0_directed}"
cd "$(dirname "$0")/../.."
"$PY" -m mmb.analysis.collect "$ROOT" --out "results/$EXP.csv" --exp-id "$EXP" --require-stats
"$PY" -m mmb.analysis.stats "results/$EXP.csv" --ref "$REF" --out "results/tables/$EXP"
"$PY" -m mmb.analysis.plots "results/$EXP.csv" --ref "$REF" --out "results/figures/${EXP}_crossing_curve.pdf" --vline 50
