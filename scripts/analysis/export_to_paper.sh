#!/usr/bin/env bash
# Results -> paper: collect (MAIN) -> stats/predictors/plots/gallery/numbers -> <paper>/generated/.
#   scripts/analysis/export_to_paper.sh <paper_repo> [runs_root=runs/MAIN] [exp_id=MAIN]
# Env: PYTHON (default python), SKIP_VALIDITY=1 to skip the CPU validity proxies (they load CLIP/DINOv2/MiniLM
#      read-only from the HF cache), N_BOOT (default 10000), GRID_SUTS (default qwen,internvl,ministral).
# Writes results/<exp>.csv, results/validity_<exp>.csv and <paper_repo>/generated/{tables,figures,numbers.tex,numbers_map.csv}.
# CPU only. The paper repo is not committed to; review the diff and commit it there.
set -euo pipefail
PY="${PYTHON:-python}"
PAPER="${1:?paper repo}"; ROOT="${2:-runs/MAIN}"; EXP="${3:-MAIN}"
cd "$(dirname "$0")/../.."
[ -f "$PAPER/main.tex" ] || { echo "not a paper repo: $PAPER"; exit 2; }
"$PY" -m mmb.analysis.collect "$ROOT" --out "results/$EXP.csv" --exp-id "$EXP" --require-stats
VAL=()
if [ "${SKIP_VALIDITY:-0}" = "0" ] || [ -z "${SKIP_VALIDITY:-}" ]; then
  if "$PY" -m mmb.validity.score "$ROOT" --out "results/validity_$EXP.csv" --default exp_id="$EXP" --offline; then
    VAL=(--validity "results/validity_$EXP.csv")
  else
    echo "validity proxies failed; continuing without them" >&2
  fi
fi
AGR=()
for f in results/agree_${EXP}_*.csv; do [ -f "$f" ] && AGR+=("$f"); done
[ ${#AGR[@]} -gt 0 ] && AGR=(--agree "${AGR[@]}")
TRF=()
for f in results/transfer_${EXP}*.csv; do [ -f "$f" ] && TRF+=("$f"); done
[ ${#TRF[@]} -gt 0 ] && TRF=(--transfer "${TRF[@]}")
MIN=()
for f in results/minimal_${EXP}_*.csv; do [ -f "$f" ] && MIN+=("$f"); done
[ ${#MIN[@]} -gt 0 ] && MIN=(--minimal "${MIN[@]}")
PLN=()
[ -d "configs/generated/$EXP" ] && PLN=(--planned "configs/generated/$EXP")
"$PY" -m mmb.analysis.export "results/$EXP.csv" --paper "$PAPER" --runs-root "$ROOT" "${PLN[@]}" "${MIN[@]}" \
  --grid-suts "${GRID_SUTS:-qwen,internvl,ministral}" --n-boot "${N_BOOT:-10000}" "${VAL[@]}" "${AGR[@]}" "${TRF[@]}"
echo "generated files: $PAPER/generated"
