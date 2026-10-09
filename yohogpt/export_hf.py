"""Export a YoHoGPT checkpoint to HuggingFace ``LlamaForCausalLM`` format so it
can be served by vLLM / TGI / SGLang or converted to GGUF for llama.cpp.

    yoho export-hf --checkpoint checkpoints/yoho-7b-sft --tokenizer artifacts/tokenizer --out artifacts/hf/yoho-7b
    vllm serve artifacts/hf/yoho-7b --served-model-name yohogpt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from .tokenizer.tokenizer import EOS, IM_END, IM_START, PAD, SPECIAL_TOKENS
from .train.checkpoint import load_model

# Jinja chat template equivalent to YohoTokenizer.encode_chat
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + (message['content'] or '') + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def export(checkpoint: str, tokenizer: str, out: str, dtype: str = "bfloat16") -> Path:
    model = load_model(checkpoint, "cpu", getattr(torch, dtype))
    cfg = model.cfg
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("lm_head."):
            if cfg.tie_embeddings:
                continue
            sd[k] = v.contiguous()
        else:
            sd["model." + k] = v.contiguous()

    try:
        from safetensors.torch import save_file

        save_file(sd, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    except ImportError:
        torch.save(sd, out_dir / "pytorch_model.bin")

    tok_path = Path(tokenizer)
    tok_file = tok_path / "tokenizer.json" if tok_path.is_dir() else tok_path
    shutil.copy(tok_file, out_dir / "tokenizer.json")
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(tok_file))
    eos_id, im_end_id = tk.token_to_id(EOS), tk.token_to_id(IM_END)

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.ffn_hidden,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.max_seq_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.norm_eps,
        "hidden_act": "silu",
        "tie_word_embeddings": cfg.tie_embeddings,
        "attention_bias": False,
        "mlp_bias": False,
        "bos_token_id": eos_id,
        "eos_token_id": [eos_id, im_end_id],
        "pad_token_id": tk.token_to_id(PAD),
        "torch_dtype": dtype,
    }
    (out_dir / "config.json").write_text(json.dumps(hf_config, indent=2))
    (out_dir / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": [eos_id, im_end_id], "pad_token_id": tk.token_to_id(PAD), "temperature": 0.2,
         "top_p": 0.95, "do_sample": True}, indent=2))
    (out_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": cfg.max_seq_len,
        "bos_token": None, "eos_token": IM_END, "pad_token": PAD, "unk_token": None,
        "additional_special_tokens": [t for t in SPECIAL_TOKENS if t not in (EOS, PAD, IM_END)],
        "chat_template": CHAT_TEMPLATE,
        "add_bos_token": False, "add_eos_token": False, "clean_up_tokenization_spaces": False,
    }, indent=2))
    (out_dir / "special_tokens_map.json").write_text(json.dumps(
        {"eos_token": IM_END, "pad_token": PAD, "additional_special_tokens": [IM_START, EOS]}, indent=2))
    return out_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args(argv)
    print(f"exported to {export(args.checkpoint, args.tokenizer, args.out, args.dtype)}")


if __name__ == "__main__":
    main()
