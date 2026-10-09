import json
import time

import pytest
from fastapi.testclient import TestClient

from yohogpt.inference.backends import ScriptedBackend
from yohogpt.server.app import ServerSettings, create_app


def make(tmp_path, responses=None, keys=None, rate=1000):
    s = ServerSettings(api_keys=keys or [], rate_limit_per_minute=rate, sandbox="local",
                       workspaces=str(tmp_path / "ws"), max_concurrent_runs=2)
    return TestClient(create_app(ScriptedBackend(responses), s))


def test_health_models_metrics(tmp_path):
    c = make(tmp_path)
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/v1/models").json()["data"][0]["id"] == "yohogpt-scripted"
    assert "http_requests_total" in c.get("/metrics").text


def test_chat_non_stream_and_stream(tmp_path):
    c = make(tmp_path, ["Hello there friend", "Streaming works fine"])
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hello there friend"
    assert body["object"] == "chat.completion"

    with c.stream("POST", "/v1/chat/completions",
                  json={"messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [l for l in r.iter_lines() if l.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    text = "".join(json.loads(l[5:])["choices"][0]["delta"].get("content", "") for l in lines[:-1])
    assert text == "Streaming works fine"


def test_openai_tool_calls(tmp_path):
    c = make(tmp_path, ['Let me check.<tool_call>{"name": "get_weather", "arguments": {"city": "Oslo"}}</tool_call>'])
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "weather?"}], "tools": tools})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    fn = choice["message"]["tool_calls"][0]["function"]
    assert fn["name"] == "get_weather" and json.loads(fn["arguments"]) == {"city": "Oslo"}


def test_completions_fim(tmp_path):
    c = make(tmp_path, ["return a + b"])
    r = c.post("/v1/completions", json={"prompt": "def add(a, b):\n    ", "suffix": "\n"})
    assert r.json()["choices"][0]["text"] == "return a + b"


def test_auth_and_rate_limit(tmp_path):
    c = make(tmp_path, keys=["secret-key"], rate=2)
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    h = {"Authorization": "Bearer secret-key"}
    assert c.get("/v1/models", headers=h).status_code == 200
    assert c.get("/v1/models", headers=h).status_code == 200
    assert c.get("/v1/models", headers=h).status_code == 429
    assert c.get("/health").status_code == 200  # health is unauthenticated


def _call(name, **args):
    return f"<tool_call>{json.dumps({'name': name, 'arguments': args})}</tool_call>"


def test_agent_run_rest_sse_and_ws(tmp_path):
    script = [_call("write_file", path="hello.py", content="print('hi')\n"),
              _call("bash", command="python3 hello.py"),
              _call("finish", summary="created hello.py")]
    c = make(tmp_path, script)
    r = c.post("/v1/agent/runs", json={"task": "make hello.py", "files": {"README.md": "# demo\n"}})
    assert r.status_code == 201
    run_id = r.json()["id"]

    with c.stream("GET", f"/v1/agent/runs/{run_id}/events") as resp:
        events = [l for l in resp.iter_lines() if l.startswith("event:")]
    assert events[-1] == "event: end"
    assert "event: tool_call" in events

    st = c.get(f"/v1/agent/runs/{run_id}").json()
    assert st["status"] == "finished" and st["summary"] == "created hello.py"
    assert (tmp_path / "ws" / run_id / "hello.py").exists()
    assert (tmp_path / "ws" / run_id / "README.md").exists()

    with c.websocket_connect(f"/v1/agent/runs/{run_id}/ws") as ws:
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] == "end":
                break
    assert msgs[-1]["data"]["status"] == "finished"
    assert len(msgs) == st["num_events"] + 1


def test_agent_run_validation_and_isolation(tmp_path):
    c = make(tmp_path, keys=["k1", "k2"])
    h1, h2 = {"Authorization": "Bearer k1"}, {"Authorization": "Bearer k2"}
    assert c.post("/v1/agent/runs", json={"task": "x", "files": {"../evil": "x"}}, headers=h1).status_code == 400
    assert c.post("/v1/agent/runs", json={"task": "x", "repo_url": "file:///etc"}, headers=h1).status_code == 400
    run_id = c.post("/v1/agent/runs", json={"task": "x", "max_steps": 1}, headers=h1).json()["id"]
    assert c.get(f"/v1/agent/runs/{run_id}", headers=h2).status_code == 404  # other tenant can't see it
    with pytest.raises(Exception):
        with c.websocket_connect(f"/v1/agent/runs/{run_id}/ws") as ws:
            ws.receive_json()
