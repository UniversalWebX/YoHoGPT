#!/usr/bin/env bash
# End-to-end CPU smoke test of the whole YoHoGPT pipeline (no network, ~2 min):
# data -> tokenizer -> prepare -> pretrain -> SFT -> serve -> chat + agent API.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
rm -rf data/raw/pretrain/sample data/processed/sample checkpoints/smoke checkpoints/smoke-sft artifacts/smoke-tok

python scripts/make_sample_data.py --out data/raw/pretrain/sample --sft data/sft/sample.jsonl
python -m yohogpt.cli train-tokenizer --input data/raw/pretrain/sample --out artifacts/smoke-tok --vocab-size 4096
python -m yohogpt.cli prepare --raw data/raw/pretrain/sample --out data/processed/sample \
    --tokenizer artifacts/smoke-tok/tokenizer.json --workers 2
python -m yohogpt.cli train configs/train/smoke.yaml model=configs/model/yoho-tiny.yaml \
    out_dir=checkpoints/smoke
# SFT on top of the pretrained smoke checkpoint
python -m yohogpt.cli train configs/train/smoke.yaml stage=sft init_from=checkpoints/smoke \
    out_dir=checkpoints/smoke-sft "sft_files=[data/sft/sample.jsonl]" tokenizer=artifacts/smoke-tok max_steps=20 \
    eval_interval=0 ckpt_interval=20

PORT=18080
YOHO_SANDBOX=local python -m yohogpt.cli serve --backend local --checkpoint checkpoints/smoke-sft \
    --tokenizer artifacts/smoke-tok --port $PORT --sandbox local > /tmp/yoho-serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 1; done

echo "--- /v1/chat/completions"
curl -sf localhost:$PORT/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python function that returns 3 + 8."}],"max_tokens":40,"temperature":0}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
echo "--- /v1/completions (FIM)"
curl -sf localhost:$PORT/v1/completions -H 'content-type: application/json' \
  -d '{"prompt":"def add(a, b):\n    ","suffix":"\n","max_tokens":16,"temperature":0}' >/dev/null && echo ok
echo "--- /v1/agent/runs"
curl -sf localhost:$PORT/v1/agent/runs -H 'content-type: application/json' -d '{"task":"say hi","max_steps":2}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])'
echo "SMOKE TEST PASSED"
