#!/bin/zsh
# One-turn-per-process 4B/4C (memory resets every turn), then resumable probe pass, 4D, Sonnet judge, analyze.
cd /Users/isaiah/repos/digital-minds/p1
LOG=results/surf/probe_versions/qwen25-7b/logs
X=results/surf/probe_versions/qwen25-7b/extra_dirs.pt
DS=(utility_v3 utility_v10 utility_v15L18)
COMMON=(qwen25-7b --tag _versions --dirsets $DS --extra $X --n-cell 5)
step() { echo "[$(date '+%H:%M:%S')] START $1" >> $LOG/chain_status.log; }
done_() { echo "[$(date '+%H:%M:%S')] DONE  $1 (rc=$2)" >> $LOG/chain_status.log; }
for t in 0 1 2 3 4 5 6 7 8 9; do
  step "4bc turn $t"
  uv run python scripts/surf_4bc_resumable.py turn $COMMON --gen-batch 6 >> $LOG/4bc_resumable.out 2>&1; rc=$?
  done_ "4bc turn $t" $rc
  [ $rc -ne 0 ] && { echo "[$(date '+%H:%M:%S')] ABORT at turn $t" >> $LOG/chain_status.log; exit 1; }
done
step probes;  uv run python scripts/surf_4bc_resumable.py probes $COMMON >> $LOG/4bc_resumable.out 2>&1; done_ probes $?
step 4d;      uv run python stage4.py 4d qwen25-7b --dirsets $DS --tag _versions --extra $X > $LOG/4d_versions.log 2>&1; done_ 4d $?
step judge;   uv run python stage4.py judge qwen25-7b --judge sonnet --tag _versions > $LOG/judge_versions.log 2>&1; done_ judge $?
step analyze; uv run python stage4.py analyze qwen25-7b --tag _versions > $LOG/analyze_versions.log 2>&1; done_ analyze $?
echo "[$(date '+%H:%M:%S')] RESUMABLE CHAIN COMPLETE" >> $LOG/chain_status.log
