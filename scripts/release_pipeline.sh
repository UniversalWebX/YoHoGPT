#!/usr/bin/env bash
# The full YoHoGPT-1 release pipeline, stage by stage. Each step is resumable.
# Expect weeks of wall-clock on a GPU cluster; see docs/RELEASE_PLAN.md.
set -euo pipefail
TOK=artifacts/tokenizer

step() { echo -e "\n=== $* ==="; }

step "1. download raw data (pretrain, long-context, SFT, DPO sources from data/sources.yaml)"
yoho download --stage all --out data/raw

step "2. download eval sets for decontamination"
python - <<'PY'
import json, pathlib
from datasets import load_dataset
out = pathlib.Path("data/eval"); out.mkdir(parents=True, exist_ok=True)
for name, hf, split in [("humaneval", "openai/openai_humaneval", "test"),
                        ("mbpp", "google-research-datasets/mbpp", "test"),
                        ("mbpp_train", "google-research-datasets/mbpp", "train")]:
    with open(out / f"{name}.jsonl", "w") as f:
        for r in load_dataset(hf, split=split):
            f.write(json.dumps(dict(r)) + "\n")
PY

step "3. train tokenizer on a 5B-character sample"
yoho train-tokenizer --input data/raw/pretrain --out $TOK --vocab-size 49152 --max-chars 5000000000

step "4. filter, dedup, decontaminate and tokenize every pretraining source"
for src in data/raw/pretrain/* data/raw/long_context/*; do
  name=$(basename "$src")
  [ -f "data/processed/$name/meta.json" ] && continue
  yoho prepare --raw "$src" --out "data/processed/$name" --tokenizer $TOK/tokenizer.json \
    --eval-files data/eval/humaneval.jsonl data/eval/mbpp.jsonl --fim-rate 0.5
done

step "5. pretrain (multi-node)";       echo "sbatch --nodes=32 scripts/slurm_train.sbatch configs/train/pretrain-7b.yaml"
step "6. long-context extension";      echo "sbatch --nodes=16 scripts/slurm_train.sbatch configs/train/long-context-7b.yaml"
step "7. SFT (chat + agent)";          echo "sbatch --nodes=4  scripts/slurm_train.sbatch configs/train/sft-7b.yaml"
step "8. build execution-verified DPO pairs + self-play SFT data (needs a served SFT model)"
echo "yoho eval mbpp --problems data/eval/mbpp_train.jsonl -n 8 --emit-pairs data/sft/dpo_pairs.jsonl --emit-sft data/sft/self_play.jsonl"
echo "yoho download --stage dpo --only yoho-exec-pairs && yoho download --stage sft --only yoho-self-play"
step "9. DPO";                         echo "sbatch --nodes=2  scripts/slurm_train.sbatch configs/train/dpo-7b.yaml"
step "10. export + evaluate + ship"
echo "yoho export-hf --checkpoint checkpoints/yoho-7b-dpo --tokenizer $TOK --out artifacts/hf/yoho-7b"
echo "yoho eval humaneval -n 20 -k 1 10 && yoho eval-swebench --limit 500"
echo "docker compose -f deploy/docker-compose.yml up -d"
