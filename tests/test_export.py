import pytest
import torch

transformers = pytest.importorskip("transformers")


def test_export_matches_hf_llama(tmp_path, tokenizer, tokenizer_dir):
    from yohogpt.config import ModelConfig
    from yohogpt.export_hf import export
    from yohogpt.model import YohoGPT
    from yohogpt.train.checkpoint import save_checkpoint

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=128, rope_theta=10000.0)
    model = YohoGPT(cfg).eval()
    save_checkpoint(tmp_path / "ckpt", 1, model, None, cfg, {}, True)
    out = export(str(tmp_path / "ckpt"), str(tokenizer_dir), str(tmp_path / "hf"), dtype="float32")

    hf = transformers.AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()
    x = torch.tensor([tokenizer.encode("def add(a, b):\n    return a + b\n")])
    with torch.no_grad():
        ours, _ = model(x)
        theirs = hf(x).logits
    assert torch.allclose(ours, theirs, atol=1e-4)

    tk = transformers.AutoTokenizer.from_pretrained(out)
    msgs = [{"role": "user", "content": "hi"}]
    assert tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) == \
        tokenizer.decode(tokenizer.encode_chat(msgs, add_generation_prompt=True).ids)
