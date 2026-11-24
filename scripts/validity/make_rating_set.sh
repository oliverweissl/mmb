#!/usr/bin/env bash
# Item sampler + static rating page for the human validity study on MAIN crossings.
#   scripts/validity/make_rating_set.sh <results_csv> <out_dir> [n=120] [controls=24]
# Env: PYTHON, AGREE (space-separated agreement_replay CSVs: keep only single-item verified crossings),
#      CONSENT_FILE (verbatim HS-01 consent text; REQUIRED before real use).
# Then: give <out_dir>/rate.html (self-contained) to raters, collect their ratings_*.json and run
#   python scripts/validity/analyze_ratings.py <out_dir>/manifest.csv ratings_*.json --out results/validity_human.csv
# manifest.csv holds method/SUT per item and must stay with the analyst.
set -euo pipefail
PY="${PYTHON:-python}"
RES="${1:?results csv}"; OUT="${2:?out dir}"; N="${3:-120}"; C="${4:-24}"
cd "$(dirname "$0")/../.."
ARGS=()
[ -n "${AGREE:-}" ] && ARGS+=(--agree $AGREE)
[ -n "${CONSENT_FILE:-}" ] && ARGS+=(--consent-file "$CONSENT_FILE")
"$PY" -m mmb.validity.rating_set "$RES" --out "$OUT" --n "$N" --controls "$C" "${ARGS[@]}"
[ -n "${CONSENT_FILE:-}" ] || echo "WARNING: placeholder consent text in use; set CONSENT_FILE before any rater sees the page" >&2
