import json

import pytest

from yohogpt.config import TrainConfig


@pytest.fixture
def model_cfg(tmp_path, tokenizer):
    p = tmp_path / "model.yaml"
    p.write_text(f"vocab_size: {tokenizer.vocab_size}\nd_model: 64\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\n"
                 "max_seq_len: 128\n")
    return p


def _cfg(tmp_path, model_cfg, **kw):
    base = dict(model=str(model_cfg), out_dir=str(tmp_path / "ckpt"), seq_len=32, micro_batch=4,
                global_batch_tokens=256, lr=3e-3, min_lr=3e-4, warmup_steps=5, max_steps=40, precision="fp32",
                log_interval=5, eval_interval=20, eval_batches=2, ckpt_interval=20)
    base.update(kw)
    p = tmp_path / "train.yaml"
    p.write_text(json.dumps(base))
    return TrainConfig.load(p)


def test_pretrain_learns_and_resumes(tmp_path, raw_corpus, tokenizer_dir, model_cfg):
    from yohogpt.data.prepare import prepare
    from yohogpt.train.checkpoint import latest_checkpoint, load_model
    from yohogpt.train.trainer import train

    prepare(raw_corpus, tmp_path / "proc", str(tokenizer_dir / "tokenizer.json"), workers=1)
    cfg = _cfg(tmp_path, model_cfg, mixture=[{"name": "s", "path": str(tmp_path / "proc"), "weight": 1}])
    out = train(cfg)
    logs = [json.loads(l) for l in (tmp_path / "ckpt" / "log.jsonl").read_text().splitlines() if '"loss"' in l]
    assert logs[-1]["loss"] < logs[0]["loss"] - 1.0
    assert latest_checkpoint(tmp_path / "ckpt").name == "step-00000040"

    # resume: extend the run; must start at step 40, not 0
    cfg.max_steps = 50
    out2 = train(cfg)
    assert out2["step"] == 49
    m = load_model(tmp_path / "ckpt")
    assert m.cfg.d_model == 64


def _sft_file(tmp_path):
    f = tmp_path / "sft.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"add {i} and {i}"},
                          {"role": "assistant", "content": f"def f(): return {2 * i}"}]} for i in range(40)]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    return f


def test_sft_then_dpo(tmp_path, tokenizer_dir, model_cfg):
    from yohogpt.train.trainer import train

    sft = _sft_file(tmp_path)
    cfg = _cfg(tmp_path, model_cfg, stage="sft", sft_files=[str(sft)], tokenizer=str(tokenizer_dir),
               out_dir=str(tmp_path / "sft"), max_steps=30, eval_interval=0, ckpt_interval=30)
    res = train(cfg)
    assert res["loss"] < 3.0

    pref = tmp_path / "pref.jsonl"
    rows = [{"prompt": [{"role": "user", "content": f"q{i}"}], "chosen": "def good(): return 1",
             "rejected": "def bad(): raise"} for i in range(20)]
    pref.write_text("\n".join(json.dumps(r) for r in rows))
    cfg = _cfg(tmp_path, model_cfg, stage="dpo", sft_files=[str(pref)], tokenizer=str(tokenizer_dir),
               init_from=str(tmp_path / "sft"), out_dir=str(tmp_path / "dpo"), max_steps=15, lr=1e-3,
               warmup_steps=1, eval_interval=0, ckpt_interval=15, micro_batch=2, global_batch_tokens=64)
    res = train(cfg)
    assert res["reward_acc"] == 1.0


def test_lr_schedules():
    from yohogpt.train.trainer import lr_at

    for decay in ("cosine", "linear", "wsd"):
        c = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100, decay=decay)
        assert lr_at(0, c) == pytest.approx(0.1)
        assert lr_at(10, c) == pytest.approx(1.0)
        assert lr_at(100, c) == pytest.approx(0.1)
