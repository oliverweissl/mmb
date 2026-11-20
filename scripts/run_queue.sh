#!/usr/bin/env bash
# Run every generated config of one experiment, strictly serially (one GPU).
# Resumable: each config is run with --resume, so finished cells are skipped.
#   scripts/run_queue.sh configs/generated/E1 [logdir]
set -u
dir=$1; logdir=${2:-runs/logs}; mkdir -p "$logdir"
export MMB_IMAGENET_DIR=${MMB_IMAGENET_DIR:-$HOME/data/imagenet-val-hf}
PY=${PY:-python}
for cfg in $(ls "$dir"/*.yaml | sort); do
  name=$(basename "$cfg" .yaml)
  if [ -f "$logdir/$name.done" ]; then continue; fi
  echo "[$(date +%F' '%T)] start $name"
  if $PY -m mmb.runners.run_boundary_test "$cfg" --device cuda --resume > "$logdir/$name.log" 2>&1; then
    touch "$logdir/$name.done"; echo "[$(date +%F' '%T)] done  $name"
  else
    echo "[$(date +%F' '%T)] FAIL  $name (see $logdir/$name.log)"
  fi
done
