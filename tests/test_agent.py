import json

import pytest

from yohogpt.agent.loop import Agent
from yohogpt.agent.sandbox import LocalSandbox
from yohogpt.agent.tools import ToolError, build_tools, parse_tool_call
from yohogpt.inference.backends import ScriptedBackend


def call(name, **args):
    return f"Next step.\n<tool_call>{json.dumps({'name': name, 'arguments': args})}</tool_call>"


def test_parse_tool_call_variants():
    assert parse_tool_call('x <tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>') == (
        "bash", {"command": "ls"})
    assert parse_tool_call('<tool_call>{"name": "finish", "arguments": {"summary": "ok"}} junk') == (
        "finish", {"summary": "ok"})
    assert parse_tool_call("no call here") is None
    with pytest.raises(ToolError):
        parse_tool_call("<tool_call>{not json</tool_call>")


def test_tools_and_workspace_escape(tmp_path):
    sb = LocalSandbox(tmp_path)
    tools = build_tools(sb)
    tools["write_file"]({"path": "pkg/a.py", "content": "x = 1\ny = 2\n"})
    assert "x = 1" in tools["read_file"]({"path": "pkg/a.py"})
    tools["str_replace"]({"path": "pkg/a.py", "old": "y = 2", "new": "y = 3"})
    assert (tmp_path / "pkg/a.py").read_text() == "x = 1\ny = 3\n"
    assert "pkg/" in tools["list_dir"]({})
    assert "a.py:2" in tools["search"]({"pattern": "y = 3"})
    assert "exit_code=0" in tools["bash"]({"command": "python3 -c 'print(40+2)'"})
    with pytest.raises(PermissionError):
        tools["read_file"]({"path": "../../etc/passwd"})
    with pytest.raises(ToolError):
        tools["str_replace"]({"path": "pkg/a.py", "old": "missing", "new": ""})
    with pytest.raises(ToolError):
        tools["bash"]({})


def test_bash_timeout(tmp_path):
    r = LocalSandbox(tmp_path).run("sleep 5", timeout=1)
    assert r.timed_out


def test_agent_fixes_bug_end_to_end(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_calc.py").write_text("from calc import add\n\nassert add(2, 3) == 5\nprint('ok')\n")
    script = [
        call("read_file", path="calc.py"),
        call("bash", command="python3 test_calc.py"),
        call("str_replace", path="calc.py", old="return a - b", new="return a + b"),
        call("bash", command="python3 test_calc.py"),
        call("finish", summary="Fixed add() to use +; test_calc.py passes."),
    ]
    events = []
    agent = Agent(ScriptedBackend(script), LocalSandbox(tmp_path), on_event=events.append)
    res = agent.run("Fix the failing test")
    assert res.status == "finished" and res.steps == 5
    assert (tmp_path / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    results = [e.data["content"] for e in events if e.type == "tool_result"]
    assert "exit_code=1" in results[1] and "exit_code=0" in results[3]
    assert events[-1].type == "done"


def test_agent_recovers_from_bad_calls(tmp_path):
    script = ["<tool_call>{bad json</tool_call>", call("nope"), call("finish", summary="done")]
    res = Agent(ScriptedBackend(script), LocalSandbox(tmp_path)).run("t")
    assert res.status == "finished"
    tool_msgs = [m["content"] for m in res.messages if m["role"] == "tool"]
    assert tool_msgs[0].startswith("error") and "unknown tool" in tool_msgs[1]


def test_agent_plain_answer_and_step_limit(tmp_path):
    res = Agent(ScriptedBackend(["just text", "still text"]), LocalSandbox(tmp_path)).run("hi")
    assert res.status == "finished" and res.summary == "still text"
    res = Agent(ScriptedBackend([call("list_dir")] * 5), LocalSandbox(tmp_path), max_steps=3).run("loop")
    assert res.status == "max_steps"


def test_agent_context_compaction(tmp_path):
    (tmp_path / "big.txt").write_text("y" * 5000)
    script = [call("read_file", path="big.txt")] * 6 + [call("finish", summary="ok")]
    agent = Agent(ScriptedBackend(script), LocalSandbox(tmp_path), context_chars=20_000, keep_recent_results=2)
    res = agent.run("read")
    assert any(m["content"].startswith("[elided") for m in res.messages if m["role"] == "tool")
