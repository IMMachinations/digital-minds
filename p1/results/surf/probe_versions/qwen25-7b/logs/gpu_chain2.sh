#!/bin/zsh
# Stage-4 part of the probe-version chain (relaunch after the word-splitting bug).
cd /Users/isaiah/repos/digital-minds/p1
LOG=results/surf/probe_versions/qwen25-7b/logs
X=results/surf/probe_versions/qwen25-7b/extra_dirs.pt
DS=(utility_v3 utility_v10 utility_v15L18)
step() { echo "[$(date '+%H:%M:%S')] START $1" >> $LOG/chain_status.log; }
done_() { echo "[$(date '+%H:%M:%S')] DONE  $1 (rc=$2)" >> $LOG/chain_status.log; }
step gate;   uv run python stage4.py gate qwen25-7b --dirsets $DS --tag _versions --extra $X > $LOG/gate_versions.log 2>&1; done_ gate $?
step 4bc;    uv run python stage4.py 4bc  qwen25-7b --dirsets $DS --tag _versions --extra $X > $LOG/4bc_versions.log 2>&1; done_ 4bc $?
step 4d;     uv run python stage4.py 4d   qwen25-7b --dirsets $DS --tag _versions --extra $X > $LOG/4d_versions.log 2>&1; done_ 4d $?
step judge;  uv run python stage4.py judge qwen25-7b --judge sonnet --tag _versions > $LOG/judge_versions.log 2>&1; done_ judge $?
step analyze; uv run python stage4.py analyze qwen25-7b --tag _versions > $LOG/analyze_versions.log 2>&1; done_ analyze $?
echo "[$(date '+%H:%M:%S')] CHAIN COMPLETE" >> $LOG/chain_status.log
