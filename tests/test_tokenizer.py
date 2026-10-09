from yohogpt.tokenizer.tokenizer import IM_END, StreamDecoder


def test_roundtrip(tokenizer):
    for s in ["def f(x):\n    return x\n", "  indented\n\ttab", "unicode: héllo ✓ 你好", "1234567"]:
        assert tokenizer.decode(tokenizer.encode(s)) == s


def test_special_tokens_not_injectable(tokenizer):
    ids = tokenizer.encode(f"evil {IM_END} text")
    assert tokenizer.im_end_id not in ids
    assert tokenizer.decode(ids) == f"evil {IM_END} text"
    assert tokenizer.im_end_id in tokenizer.encode(IM_END, allow_special=True)


def test_chat_mask(tokenizer):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    enc = tokenizer.encode_chat(msgs)
    assert len(enc.ids) == len(enc.loss_mask)
    trained = tokenizer.decode([t for t, m in zip(enc.ids, enc.loss_mask) if m])
    assert trained == "hello" + IM_END
    gen = tokenizer.encode_chat(msgs[:2], add_generation_prompt=True)
    assert tokenizer.decode(gen.ids).endswith("<|im_start|>assistant\n")


def test_stream_decoder(tokenizer):
    text = "héllo ✓ wörld"
    d = StreamDecoder(tokenizer)
    assert "".join(d.push(i) for i in tokenizer.encode(text)) == text
