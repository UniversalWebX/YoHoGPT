import json

import numpy as np
import pytest
import torch

from yohogpt import hardware
from yohogpt.config import ModelConfig, TrainConfig
from yohogpt.model import YohoGPT
from yohogpt.train.optim import OPTIMIZERS, make_optimizer


def tiny(**kw):
    cfg = dict(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    cfg.update(kw)
    return YohoGPT(ModelConfig(**cfg))


def test_chunked_loss_matches_full_loss_and_grads():
    torch.manual_seed(0)
    a = tiny()
    b = tiny()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 97, (3, 20))
    y = x.clone()
    y[0, :5] = -100
    _, full = a(x, y)
    _, chunked = b(x, y, return_logits=False, loss_chunk=7)  # chunk does not divide 60 tokens
    assert torch.allclose(full, chunked, atol=1e-5)
    full.backward()
    chunked.backward()
    for (n, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5), n


@pytest.mark.parametrize("name", [o for o in OPTIMIZERS if o != "adamw8bit"])
def test_optimizers_reduce_loss(name):
    if name == "adafactor" and not hasattr(torch.optim, "Adafactor"):
        pytest.skip("torch too old for Adafactor")
    torch.manual_seed(0)
    m = tiny()
    lr = {"adamw": 1e-2, "lion": 1e-3, "adafactor": 3e-2}[name]
    opt = make_optimizer(name, m, lr, 0.0, (0.9, 0.95), "cpu")
    x = torch.randint(0, 97, (1, 33))
    first = None
    for _ in range(40):
        _, loss = m(x[:, :-1], x[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_precision_choice_by_gpu_generation():
    assert hardware.precision_for_capability((6, 1)) == "fp32"   # GTX 1050 Ti
    assert hardware.precision_for_capability((7, 5)) == "fp16"   # GTX 1650 Super / RTX 20xx
    assert hardware.precision_for_capability((8, 6)) == "bf16"   # RTX 30xx
    assert hardware.best_precision(torch.device("cpu")) == "fp32"
    assert hardware.supports_compile(torch.device("cpu"))


def test_gpu_lookup():
    assert hardware.lookup("NVIDIA GeForce GTX 1050 Ti").name == "gtx1050ti"
    assert hardware.lookup("GTX 1050").name == "gtx1050"
    assert hardware.lookup("1650 super").name == "gtx1650super"
    assert hardware.lookup("NVIDIA H100 80GB HBM3").name == "h100"
    with pytest.raises(KeyError):
        hardware.lookup("voodoo 3")


def test_resolve_runtime_on_cpu():
    from yohogpt.train.distributed import DistInfo
    from yohogpt.train.trainer import resolve_runtime

    cfg = TrainConfig(precision="auto", parallelism="fsdp")
    assert resolve_runtime(cfg, DistInfo(), log=False) == torch.float32
    assert cfg.precision == "fp32" and cfg.parallelism == "ddp"


def test_planner_recommendations():
    from yohogpt.plan import plan, recommend

    small = recommend(plan(hardware.lookup("gtx1050ti"), 1, 30 * 24, "fp32", "adamw", 32768, 1024, 0.4))
    assert 50e6 < small.params < 200e6 and small.fits and small.memory_gb < 4
    big = recommend(plan(hardware.lookup("h100"), 256, 10 * 24, "bf16", "adamw", 49152, 4096, 0.4))
    assert big.params > 5e9
    # a lower-memory optimiser lets a bigger model fit on the same card
    lion = plan(hardware.lookup("gtx1050ti"), 1, 1e6, "fp32", "adafactor", 32768, 1024, 0.4)
    adam = plan(hardware.lookup("gtx1050ti"), 1, 1e6, "fp32", "adamw", 32768, 1024, 0.4)
    assert sum(e.fits for e in lion) > sum(e.fits for e in adam)


def test_planner_writes_loadable_configs(tmp_path):
    from yohogpt.plan import plan, recommend, write_configs

    rec = recommend(plan(hardware.lookup("gtx1650super"), 1, 24 * 7, "fp16", "adamw", 32768, 1024, 0.4))
    m, t = write_configs(rec, tmp_path, 1024, "fp16", "adamw", 24 * 7, 1)
    assert ModelConfig.load(m).num_params() == rec.params
    cfg = TrainConfig.load(t)
    assert cfg.micro_batch == rec.micro_batch and cfg.mixture and cfg.max_steps >= 100


def test_benchmark_runs(tmp_path):
    from yohogpt.train.benchmark import run_benchmark

    mp = tmp_path / "m.yaml"
    mp.write_text("vocab_size: 97\nd_model: 64\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nmax_seq_len: 64\n")
    for opt in ("adamw", "lion"):
        cfg = TrainConfig(model=str(mp), seq_len=32, micro_batch=2, optimizer=opt, grad_checkpointing=True,
                          max_steps=100_000, global_batch_tokens=4096)
        res = run_benchmark(cfg, steps=2, warmup=1)
        assert res["tokens_per_s"] > 0 and res["eta_hours"] > 0 and res["precision"] == "fp32"


def test_prefetcher_matches_direct_batches():
    from yohogpt.train.trainer import Prefetcher

    def get_batch(step, micro, val=False):
        rng = np.random.default_rng([step, micro])
        t = torch.from_numpy(rng.integers(0, 100, (2, 8)))
        return t, t + 1

    pf = Prefetcher(get_batch, 3, 7, 2, torch.device("cpu"), depth=3)
    for step in range(3, 7):
        for micro in range(2):
            x, y = pf(step, micro)
            ex, ey = get_batch(step, micro)
            assert torch.equal(x, ex) and torch.equal(y, ey)


def test_prefetcher_surfaces_errors():
    from yohogpt.train.trainer import Prefetcher

    def bad(step, micro, val=False):
        raise FileNotFoundError("shard missing")

    pf = Prefetcher(bad, 0, 2, 1, torch.device("cpu"), depth=1)
    with pytest.raises(FileNotFoundError):
        pf(0, 0)
