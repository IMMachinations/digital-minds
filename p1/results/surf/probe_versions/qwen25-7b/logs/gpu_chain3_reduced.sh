#!/bin/zsh
# Fallback: reduced 4B/4C (10 rollouts/cell, v10 + v15_L18 only) with per-turn progress; tag _versions_r.
cd /Users/isaiah/repos/digital-minds/p1
LOG=results/surf/probe_versions/qwen25-7b/logs
X=results/surf/probe_versions/qwen25-7b/extra_dirs.pt
DS=(utility_v10 utility_v15L18)
step() { echo "[$(date '+%H:%M:%S')] START $1" >> $LOG/chain_status.log; }
done_() { echo "[$(date '+%H:%M:%S')] DONE  $1 (rc=$2)" >> $LOG/chain_status.log; }
cp results/stage4/qwen25-7b/gate_versions.json results/stage4/qwen25-7b/gate_versions_r.json
step 4bc_r;    uv run python stage4.py 4bc  qwen25-7b --dirsets $DS --tag _versions_r --extra $X --n-cell 10 > $LOG/4bc_versions_r.log 2>&1; done_ 4bc_r $?
step 4d_r;     uv run python stage4.py 4d   qwen25-7b --dirsets $DS --tag _versions_r --extra $X > $LOG/4d_versions_r.log 2>&1; done_ 4d_r $?
step judge_r;  uv run python stage4.py judge qwen25-7b --judge sonnet --tag _versions_r > $LOG/judge_versions_r.log 2>&1; done_ judge_r $?
step analyze_r; uv run python stage4.py analyze qwen25-7b --tag _versions_r > $LOG/analyze_versions_r.log 2>&1; done_ analyze_r $?
echo "[$(date '+%H:%M:%S')] REDUCED CHAIN COMPLETE" >> $LOG/chain_status.log
