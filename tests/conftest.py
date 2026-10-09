import json
import random

import pytest

from yohogpt.tokenizer.train_tokenizer import train

SAMPLE_CODE = [
    "def add(a, b):\n    return a + b\n",
    "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, x):\n        self.items.append(x)\n",
    "for i in range(10):\n    print(i * i)\n",
    "import os\n\ndef main():\n    print(os.getcwd())\n\nif __name__ == '__main__':\n    main()\n",
    "function greet(name) {\n  return `Hello, ${name}!`;\n}\n",
    "fn main() {\n    println!(\"hello world\");\n}\n",
    "SELECT id, name FROM users WHERE active = 1 ORDER BY name;\n",
    "The quick brown fox jumps over the lazy dog. Python is a programming language.\n",
]


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("tok")
    rng = random.Random(0)
    texts = [rng.choice(SAMPLE_CODE) for _ in range(400)]
    train(iter(texts), vocab_size=600, out_dir=out, min_frequency=1)
    return out


@pytest.fixture(scope="session")
def tokenizer(tokenizer_dir):
    from yohogpt.tokenizer import YohoTokenizer

    return YohoTokenizer.load(tokenizer_dir)


@pytest.fixture
def raw_corpus(tmp_path):
    from yohogpt.data.io import ShardWriter

    rng = random.Random(1)
    with ShardWriter(tmp_path / "raw") as w:
        for i in range(300):
            body = rng.choice(SAMPLE_CODE)
            # unique-ish docs so dedup keeps most of them
            w.write({"text": f"# file {i} {rng.random()}\n" + body * 3 + f"x_{i} = {i}\n",
                     "meta": {"path": f"src/f{i}.py"}})
    return tmp_path / "raw"
