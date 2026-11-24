#!/usr/bin/env bash
# Pause the MAIN queue, re-score partners + agreement + transfer replays, restart the queue.
# Run as its own transient unit so it survives the caller:
#   systemd-run --user --unit mmb-replay -p WorkingDirectory=$PWD bash scripts/replay_maintenance.sh
set -u
PYBIN=/home/oliver/gardens/mmb/.venv/bin/python
PY="taskset -c ${PIN_CPUS:-0-3,8-31} $PYBIN"
export MMB_IMAGENET_DIR=$HOME/data/imagenet-val-hf
log=runs/logs/replay.out
echo "[$(date +%T)] stopping queue" >> $log
systemctl --user stop mmb-queue14; pkill -f "[m]mb.runners.run_boundary_test"; sleep 10
for s in ${CONFIRM_SUTS-qwen internvl ministral}; do $PY scripts/confirm_crossings.py runs/MAIN --sut $s >> $log 2>&1; echo "confirm $s exit $?" >> $log; done
for s in ${AGREE_SUTS-qwen internvl ministral}; do $PY scripts/agreement_replay.py runs/MAIN --sut $s --out results/agree_MAIN_$s.csv >> $log 2>&1; echo "agree $s exit $?" >> $log; done
[ -n "${SKIP_TRANSFER:-}" ] || $PY scripts/transfer_replay.py runs/MAIN --out results/transfer_MAIN.csv >> $log 2>&1
echo "[$(date +%T)] replays done, restarting queue" >> $log
systemd-run --user --unit mmb-queue14 -p WorkingDirectory=$PWD \
  -p StandardOutput=append:$PWD/runs/logs/MAIN/queue.out -p StandardError=append:$PWD/runs/logs/MAIN/queue.out \
  --setenv=PY=$PYBIN --setenv=MMB_IMAGENET_DIR=$MMB_IMAGENET_DIR \
  /usr/bin/bash $PWD/scripts/run_queue_passes.sh configs/generated/MAIN runs/logs/MAIN "" 3 >> $log 2>&1
