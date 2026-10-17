#!/bin/bash
# Run queued GPU jobs one after another, so the GPU never waits for a person.
#
#   Q=/workspace/Capstone_Project/queue
#   Each job is a bash script in $Q/pending. Names sort the order: 100_x.sh runs
#   before 500_y.sh. A job may queue further jobs.
#   start:   nohup bash gpu/run_queue.sh > /workspace/Capstone_Project/queue/runner.log 2>&1 &
#   stop:    touch /workspace/Capstone_Project/queue/STOP   (after the current job)
#   status:  ls /workspace/Capstone_Project/queue/*; tail /workspace/Capstone_Project/queue/runner.log
#
# Jobs run from the repo with the project environment sourced; each job's
# output goes to <job>.log beside it, and the job moves to done/ or failed/.
set -u
Q=${QUEUE_DIR:-/workspace/Capstone_Project/queue}
REPO=/workspace/Capstone_Project/DeepErase
mkdir -p "$Q"/pending "$Q"/running "$Q"/done "$Q"/failed
if [ -f "$Q/runner.pid" ] && kill -0 "$(cat "$Q/runner.pid")" 2>/dev/null; then
  echo "a runner is already active (pid $(cat "$Q/runner.pid"))"; exit 1
fi
echo $$ > "$Q/runner.pid"
[ -f /workspace/Capstone_Project/.sshd/container_env.sh ] && source /workspace/Capstone_Project/.sshd/container_env.sh
source "$REPO/gpu/env.sh" >/dev/null 2>&1
cd "$REPO" || exit 1
echo "$(date '+%F %T') runner started (pid $$)"
while [ ! -e "$Q/STOP" ]; do
  job=$(ls "$Q/pending" 2>/dev/null | grep '\.sh$' | sort | head -1)
  if [ -z "$job" ]; then sleep 20; continue; fi
  mv "$Q/pending/$job" "$Q/running/$job"
  echo "$(date '+%F %T') START $job"
  ( cd "$REPO" && bash "$Q/running/$job" ) > "$Q/running/${job%.sh}.log" 2>&1
  rc=$?
  dest=done; [ "$rc" -ne 0 ] && dest=failed
  mv "$Q/running/$job" "$Q/$dest/$job"
  mv "$Q/running/${job%.sh}.log" "$Q/$dest/${job%.sh}.log"
  echo "$(date '+%F %T') END   $job rc=$rc -> $dest"
done
rm -f "$Q/STOP" "$Q/runner.pid"
echo "$(date '+%F %T') runner stopped"
