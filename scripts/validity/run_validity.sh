#!/usr/bin/env bash
# Crossing runs -> validity proxies (CSV). Models load from the HF cache (first run downloads ~1 GB).
#   scripts/validity/run_validity.sh <runs_root> <exp_id>
# Optional calibration, needs HS-01 stimuli scored as one directory per item_id:
#   python -m mmb.validity.calibrate --proxies results/validity_hs01.csv --human <hs01_pair_item_validity_seed.csv> --out results/validity_calibration.csv
set -euo pipefail
PY="${PYTHON:-python}"
ROOT="${1:?runs root}"; EXP="${2:?exp id}"
cd "$(dirname "$0")/../.."
"$PY" -m mmb.validity.score "$ROOT" --out "results/validity_$EXP.csv" --default exp_id="$EXP"
