import torch

from yohogpt.config import ModelConfig
from yohogpt.model import YohoGPT


def tiny(**kw):
    cfg = dict(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    cfg.update(kw)
    return YohoGPT(ModelConfig(**cfg))


def test_forward_backward():
    m = tiny()
    x = torch.randint(0, 97, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, 97)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_param_count_matches_config():
    m = tiny()
    assert m.num_params() == m.cfg.num_params()
    t = tiny(tie_embeddings=True)
    assert t.num_params() == t.cfg.num_params()


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    m = tiny().eval()
    x = torch.randint(0, 97, (1, 20))
    with torch.no_grad():
        full, _ = m(x)
        m.setup_cache(1, 32)
        a, _ = m(x[:, :12], use_cache=True)                 # prefill
        b, _ = m(x[:, 12:15], start_pos=12, use_cache=True)  # chunked prefill
        steps = [m(x[:, i : i + 1], start_pos=i, use_cache=True)[0] for i in range(15, 20)]
    inc = torch.cat([a, b, *steps], dim=1)
    assert torch.allclose(full, inc, atol=1e-5)


def test_ignore_index_and_grad_checkpointing():
    m = tiny()
    m.grad_checkpointing = True
    x = torch.randint(0, 97, (2, 8))
    y = x.clone()
    y[:, :4] = -100
    _, loss = m(x, y)
    loss.backward()
    assert torch.isfinite(loss)


def test_overfits_tiny_sequence():
    torch.manual_seed(0)
    m = tiny()
    opt = m.configure_optimizer(1e-2, 0.0, (0.9, 0.95), "cpu")
    x = torch.randint(0, 97, (1, 33))
    for _ in range(60):
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5
