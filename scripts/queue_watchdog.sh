#!/usr/bin/env bash
# Watchdog for the GPU queue: if a run_boundary_test process exists but no log in
# <logdir> has been written for STALL_MIN minutes (a hang, e.g. stuck JIT or driver),
# kill it; run_queue.sh then retries the config with --resume.
#   scripts/queue_watchdog.sh runs/logs/MAIN [stall_min]
logdir=$1; stall=${2:-20}
while true; do
  pids=$(pgrep -f "mmb.runners.run_boundary_test" || true)
  if [ -n "$pids" ]; then
    newest=$(ls -t "$logdir"/../*/*.log "$logdir"/*.log 2>/dev/null | head -1)
    age=$(( ( $(date +%s) - $(stat -c %Y "$newest") ) / 60 ))
    if [ "$age" -ge "$stall" ]; then
      echo "[$(date +%F' '%T)] watchdog: no log output for ${age} min, killing $pids"
      kill $pids; sleep 30; kill -9 $pids 2>/dev/null
    fi
  fi
  sleep 60
done
