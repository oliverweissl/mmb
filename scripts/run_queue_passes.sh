#!/usr/bin/env bash
# Wait for a running queue unit to finish, then run up to N more passes of the queue
# so configs that failed (e.g. transient crashes) are retried; finished configs are skipped.
#   scripts/run_queue_passes.sh <config_dir> <logdir> [wait_unit] [passes]
set -u
dir=$1; logdir=$2; wait_unit=${3:-}; n=${4:-3}
if [ -n "$wait_unit" ]; then
  while systemctl --user is-active --quiet "$wait_unit"; do sleep 60; done
fi
for p in $(seq 1 "$n"); do
  left=$(for c in "$dir"/*.yaml; do b=$(basename "$c" .yaml); [ -f "$logdir/$b.done" ] || echo "$b"; done | wc -l)
  echo "[$(date +%F' '%T)] pass $p: $left config(s) left"
  [ "$left" -eq 0 ] && break
  bash "$(dirname "$0")/run_queue.sh" "$dir" "$logdir"
done
