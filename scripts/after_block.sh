#!/usr/bin/env bash
# Wait until every config with prefix $1 is done, then run the replay maintenance (confirm crossings).
#   systemd-run --user --unit mmb-after-A0x -p WorkingDirectory=$PWD bash scripts/after_block.sh A0x
set -u
pre=$1
n=$(ls configs/generated/MAIN/${pre}_*.yaml configs/generated/MAIN/${pre}__*.yaml 2>/dev/null | wc -l)
while [ "$(ls runs/logs/MAIN/${pre}_*.done runs/logs/MAIN/${pre}__*.done 2>/dev/null | wc -l)" -lt "$n" ]; do sleep 120; done
echo "[$(date +%T)] block $pre done ($n configs)" >> runs/logs/replay.out
AGREE_SUTS="${AGREE_SUTS-}" SKIP_TRANSFER="${SKIP_TRANSFER-1}" bash scripts/replay_maintenance.sh
