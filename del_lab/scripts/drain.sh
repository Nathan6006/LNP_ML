#!/bin/bash
# drain.sh - autonomous engine for the delivery A/B loop.
# Runs run_next.py (one variant per call, lock-protected) in a loop until the queue is empty
# (exit code 2). Each variant logs to results/DEL_EXPERIMENTS.md + registry.json as it finishes.
# Survives the launching turn; dies on session reset -> the heartbeat cron revives it.
cd "$(dirname "$0")" || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE
echo "$(date '+%F %T') drain.sh START (pid $$)" >> ../results/run.log
while true; do
  python3 run_next.py >> ../results/run.log 2>&1
  code=$?
  if [ "$code" -eq 2 ]; then
    echo "$(date '+%F %T') QUEUE EMPTY — drain complete" >> ../results/run.log
    break
  fi
  sleep 1
done
