#!/usr/bin/env bash
# Pilot of the MMBA search on held-out cells, then resume the MAIN queue.
cd /home/oliver/gardens/mmb/mmb
bash scripts/run_queue.sh configs/generated/PILOT runs/logs/PILOT
bash scripts/run_queue_passes.sh configs/generated/MAIN runs/logs/MAIN "" 3
