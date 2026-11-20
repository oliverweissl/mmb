#!/usr/bin/env bash
# Run every generated config of one experiment, strictly serially (one GPU), in file-name order.
# Resumable and fault-tolerant: each config is retried up to 3 times with --resume, so finished
# cells are skipped and an intermittent crash only repeats the unfinished cells.
#   scripts/run_queue.sh configs/generated/MAIN [logdir]
set -u
dir=$1; logdir=${2:-runs/logs}; mkdir -p "$logdir"
export MMB_IMAGENET_DIR=${MMB_IMAGENET_DIR:-$HOME/data/imagenet-val-hf}
PY=${PY:-python}
for cfg in $(ls "$dir"/*.yaml | sort); do
  name=$(basename "$cfg" .yaml)
  [ -f "$logdir/$name.done" ] && continue
  for attempt in 1 2 3; do
    echo "[$(date +%F' '%T)] start $name (attempt $attempt)"
    if $PY -m mmb.runners.run_boundary_test "$cfg" --device cuda --resume >> "$logdir/$name.log" 2>&1; then
      touch "$logdir/$name.done"; echo "[$(date +%F' '%T)] done  $name"; break
    fi
    echo "[$(date +%F' '%T)] FAIL  $name attempt $attempt (see $logdir/$name.log)"
    sleep 5
  done
done
echo "[$(date +%F' '%T)] queue finished"
