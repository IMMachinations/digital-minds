#!/bin/zsh
cd /Users/isaiah/repos/digital-minds/p1
LOG=results/surf/probe_versions/qwen25-7b/logs
until grep -q "DONE  4bc turn 6" $LOG/chain_status.log; do sleep 5; done
pkill -f gpu_chain4_resumable.sh; sleep 1; pkill -f "surf_4bc_resumable.py turn"; sleep 3
echo "[$(date '+%H:%M:%S')] STOPPED after turn 6 (deadline); launching finish chain" >> $LOG/chain_status.log
nohup $LOG/gpu_chain5_finish.sh > $LOG/gpu_chain5.out 2>&1 &
