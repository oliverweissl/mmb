#!/usr/bin/env bash
# Pin every run_boundary_test process (all threads) to the E-cores (CPUs 16-31) of the
# i9-13900KF: the high-boost P-cores show random SIGSEGV/SIGILL under load.
while true; do
  for p in $(pgrep -f "mmb.runners.run_boundary_test"); do
    cur=$(taskset -pc "$p" 2>/dev/null | awk -F': ' '{print $2}')
    [ "$cur" = "16-31" ] || taskset -a -pc 16-31 "$p" >/dev/null 2>&1
  done
  sleep 1
done
