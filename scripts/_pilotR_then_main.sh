#!/usr/bin/env bash
cd /home/oliver/gardens/mmb/mmb
bash scripts/run_queue.sh configs/generated/PILOTR runs/logs/PILOTR
bash scripts/run_queue_passes.sh configs/generated/MAIN runs/logs/MAIN "" 3
