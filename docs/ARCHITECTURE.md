# YoHoGPT architecture

```
                      data/sources.yaml
                             │
   yoho download ──► data/raw/<stage>/<source>/*.jsonl.gz      (licensed, redacted)
                             │
   yoho train-tokenizer ──► artifacts/tokenizer/tokenizer.json (BPE, 49,152 vocab)
                             │
   yoho prepare ──► quality filter → exact dedup → MinHash dedup → decontaminate → FIM → tokenize
                             │
                   data/processed/<source>/tokens-*.bin        (uint16 memmap)
                             │
   yoho train  ──► pretrain → long-context → SFT → DPO         (checkpoints/<run>/step-*/)
                             │
   yoho export-hf ──► artifacts/hf/<model>  ──► vLLM
                             │
   yoho serve  ──► gateway: OpenAI API + agent runs ──► clients / IDEs / `yoho agent`
                             │
   yoho eval   ──► pass@k, SWE-bench ──► self-play SFT data + DPO pairs ──┐
                             ▲                                            │
                             └─────────────── next release ◄──────────────┘
```

## Model (`yohogpt/model.py`)

- A decoder-only transformer with pre-norm RMSNorm, rotary embeddings (θ = 5e5 to 1e6, so long contexts work well), grouped-query attention (8 KV heads, which keeps the KV cache small for long agent sessions) and a SwiGLU MLP.
- Parameter names match HF `LlamaForCausalLM`, so export is a pure rename. `tests/test_export.py` checks that the logits are identical.
- Residual projections use a scaled init (std / √(2L)). Gradient checkpointing is optional, and the attention path works with `torch.compile`.

## Tokenizer (`yohogpt/tokenizer/`)

- Byte-level BPE with a GPT-4-style split regex. Digits are split individually and runs of spaces/tabs are kept together, which makes indentation cheap.
- Special tokens: `<|endoftext|>`, `<|pad|>`, ChatML `<|im_start|>`/`<|im_end|>`, FIM `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>`, repo packing `<|repo_name|>`/`<|file_sep|>`, and agent `<tool_call>`/`</tool_call>` and `<think>`/`</think>`.
- Text that isn't from the assistant (user messages, file contents, tool output) **can never produce control tokens**. A literal `<|im_end|>` inside a file is encoded as plain characters, which blocks prompt injection through file contents.

## Agent protocol (`yohogpt/agent/`)

```
system: SYSTEM_PROMPT_AGENT + JSON schemas of tools
user:   task
assistant: short reasoning + <tool_call>{"name": "bash", "arguments": {"command": "pytest -x"}}</tool_call>
tool:   exit_code=1 ... output (truncated head+tail to 12k chars)
assistant: ... <tool_call>{"name": "str_replace", ...}</tool_call>
...
assistant: <tool_call>{"name": "finish", "arguments": {"summary": "..."}}</tool_call>
```

- Generation stops at `</tool_call>`, so the model never invents tool output.
- Malformed calls, unknown tools and tool errors go back to the model as `tool` messages, so it can recover.
- When the transcript grows too long, old tool outputs are replaced with short stubs and the most recent ones are kept.
- A user can interject during a run (`send_user_message`) or cancel it.
- When the run ends, the agent reports its `git diff`. That diff is used for SWE-bench and shown to the user.
- The same format appears in SFT data, at inference time and in OpenAI `tools` compatibility mode, so the model sees one contract everywhere.

## Training loop (`yohogpt/train/trainer.py`)

- One loop for every stage. `stage` picks the data source (packed tokens, SFT conversations with assistant-only loss, or preference pairs) and the loss (cross-entropy or DPO against a frozen reference).
- Distributed: DDP or FSDP (FULL_SHARD, auto-wrapped per transformer block, bf16 params with fp32 reductions), via torchrun or SLURM.
- Schedules: cosine, linear or warmup-stable-decay. The global batch is set in tokens, and gradient accumulation is derived from it.
- Checkpoints are full state dicts, loadable on any world size and written atomically. Old checkpoints are rotated out. Runs resume automatically.
- Logging: JSONL plus optional W&B, including tokens/s and MFU.
