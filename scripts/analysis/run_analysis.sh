#!/usr/bin/env bash
# Run dirs -> per-run CSV -> tables -> figure.
#   scripts/analysis/run_analysis.sh <runs_root> <exp_id> [ref_arm]
# Writes results/<exp_id>.csv, results/tables/<exp_id>/*, results/figures/<exp_id>_crossing_curve.pdf
set -euo pipefail
PY="${PYTHON:-/home/oliver/gardens/mmb/.venv/bin/python}"
ROOT="${1:?runs root}"; EXP="${2:?exp id}"; REF="${3:-a0_directed}"
cd "$(dirname "$0")/../.."
"$PY" -m mmb.analysis.collect "$ROOT" --out "results/$EXP.csv" --exp-id "$EXP"
"$PY" -m mmb.analysis.stats "results/$EXP.csv" --ref "$REF" --out "results/tables/$EXP"
"$PY" -m mmb.analysis.plots "results/$EXP.csv" --ref "$REF" --out "results/figures/${EXP}_crossing_curve.pdf" --vline 206
