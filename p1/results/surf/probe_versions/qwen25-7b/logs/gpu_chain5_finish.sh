#!/bin/zsh
# Finish the resumable 4B/4C after 7 turns: finalize transcripts, probe pass, Sonnet judge, analyze (4D skipped).
cd /Users/isaiah/repos/digital-minds/p1
LOG=results/surf/probe_versions/qwen25-7b/logs
X=results/surf/probe_versions/qwen25-7b/extra_dirs.pt
DS=(utility_v3 utility_v10 utility_v15L18)
COMMON=(qwen25-7b --tag _versions --dirsets $DS --extra $X --n-cell 5)
step() { echo "[$(date '+%H:%M:%S')] START $1" >> $LOG/chain_status.log; }
done_() { echo "[$(date '+%H:%M:%S')] DONE  $1 (rc=$2)" >> $LOG/chain_status.log; }
step finalize; uv run python scripts/surf_4bc_resumable.py finalize $COMMON >> $LOG/4bc_resumable.out 2>&1; done_ finalize $?
step probes;   uv run python scripts/surf_4bc_resumable.py probes $COMMON >> $LOG/4bc_resumable.out 2>&1; done_ probes $?
step judge;    uv run python stage4.py judge qwen25-7b --judge sonnet --tag _versions > $LOG/judge_versions.log 2>&1; done_ judge $?
step analyze;  uv run python stage4.py analyze qwen25-7b --tag _versions > $LOG/analyze_versions.log 2>&1; done_ analyze $?
echo "[$(date '+%H:%M:%S')] FINISH CHAIN COMPLETE (7-turn sessions, 4D skipped)" >> $LOG/chain_status.log
