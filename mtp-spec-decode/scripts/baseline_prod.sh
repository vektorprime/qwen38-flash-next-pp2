#!/bin/bash
# Baseline of the live prod container (no restart): generation suite + KLD corpus at K=20 (prod's max-logprobs).
cd /home/user/qwen3nextflash/mtp
echo "[$(date -u +%T)] gen suite PROD_base"
python3 gen_suite.py http://127.0.0.1:8001 PROD_base 2>&1 | grep -v "^  t1\|^  pad" | tail -60
echo "[$(date -u +%T)] KLD corpus K=20"
python3 ../kld/score.py collect --url http://127.0.0.1:8001/v1 --model qwen38-flash-next-awq --k 20 --out results/PROD_base_k20.npz --windows ../kld/corpus/windows.jsonl 2>&1 | grep -E "77/77|wrote|retry|Error|Traceback" | tail -5
echo "[$(date -u +%T)] done"
