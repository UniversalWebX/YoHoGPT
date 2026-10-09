"""YoHoGPT API server.

Endpoints
---------
GET  /health                              liveness / readiness
GET  /metrics                             Prometheus metrics
GET  /v1/models                           OpenAI-compatible model list
POST /v1/chat/completions                 OpenAI-compatible chat (SSE streaming, tool calls)
POST /v1/completions                      completions + fill-in-the-middle (`suffix`) for IDEs
POST /v1/agent/runs                       start an autonomous agent run
GET  /v1/agent/runs                       list your runs
GET  /v1/agent/runs/{id}                  run status, summary, diff, events
GET  /v1/agent/runs/{id}/events           Server-Sent Events stream of agent events
WS   /v1/agent/runs/{id}/ws               bidirectional: events out, {"type":"message"|"cancel"} in
POST /v1/agent/runs/{id}/messages         inject guidance into a running agent
POST /v1/agent/runs/{id}/cancel           cancel a run

Run:  yoho serve --backend local --checkpoint checkpoints/yoho-sft --tokenizer artifacts/tokenizer
      yoho serve --backend openai --upstream http://vllm:8000   (gateway in front of vLLM)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from ..agent.tools import ToolError, parse_tool_call
from ..inference.backends import Backend, collect
from ..inference.generate import SamplingParams
from ..tokenizer.tokenizer import TOOL_CALL, TOOL_CALL_END
from .agent_runs import AgentRunManager
from .schemas import AgentMessage, AgentRunRequest, ChatCompletionRequest, CompletionRequest, SamplingFields
from .security import APIKeyAuth, Metrics, RateLimiter

log = logging.getLogger("yohogpt.server")


@dataclass
class ServerSettings:
    api_keys: list[str] = field(default_factory=lambda: [k for k in os.environ.get("YOHO_API_KEYS", "").split(",") if k])
    rate_limit_per_minute: int = int(os.environ.get("YOHO_RATE_LIMIT", "120"))
    cors_origins: list[str] = field(default_factory=lambda: [o for o in os.environ.get("YOHO_CORS", "").split(",") if o])
    enable_agent: bool = os.environ.get("YOHO_ENABLE_AGENT", "1") == "1"
    sandbox: str = os.environ.get("YOHO_SANDBOX", "docker")
    sandbox_image: str = os.environ.get("YOHO_SANDBOX_IMAGE", "yohogpt/sandbox:latest")
    workspaces: str = os.environ.get("YOHO_WORKSPACES", "workspaces")
    max_concurrent_runs: int = int(os.environ.get("YOHO_MAX_RUNS", "4"))
    max_concurrent_generations: int = int(os.environ.get("YOHO_MAX_GENERATIONS", "64"))


def _sampling(req: SamplingFields) -> SamplingParams:
    stop = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
    return SamplingParams(max_new_tokens=req.max_tokens or 1024, temperature=req.temperature, top_p=req.top_p,
                          top_k=req.top_k, repetition_penalty=req.repetition_penalty, stop=stop, seed=req.seed)


def _tools_system_prompt(tools: list[dict]) -> str:
    specs = "\n".join(json.dumps(t.get("function", t)) for t in tools)
    return ("\n\n# Tools\nYou may call one of these functions by replying with "
            f'{TOOL_CALL}{{"name": ..., "arguments": {{...}}}}{TOOL_CALL_END}\n{specs}')


def to_internal_messages(req: ChatCompletionRequest) -> list[dict]:
    """Convert OpenAI-style messages (incl. tool_calls) to YoHoGPT ChatML messages."""
    out: list[dict] = []
    for m in req.messages:
        content = m.text()
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                args = json.loads(args) if isinstance(args, str) else args
                content += f'{TOOL_CALL}{json.dumps({"name": fn.get("name"), "arguments": args})}{TOOL_CALL_END}'
        out.append({"role": m.role, "content": content})
    if req.tools:
        extra = _tools_system_prompt(req.tools)
        if out and out[0]["role"] == "system":
            out[0] = {"role": "system", "content": out[0]["content"] + extra}
        else:
            out.insert(0, {"role": "system", "content": extra.strip()})
    return out


def create_app(backend: Backend, settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    auth = APIKeyAuth(settings.api_keys)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    metrics = Metrics()
    gen_slots = asyncio.Semaphore(settings.max_concurrent_generations)
    runs = AgentRunManager(backend, settings.workspaces, settings.sandbox, settings.max_concurrent_runs,
                           settings.sandbox_image) if settings.enable_agent else None
    if not auth.enabled:
        log.warning("YOHO_API_KEYS is not set: the API is UNAUTHENTICATED. Do not expose it publicly.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if runs:
            runs.shutdown()

    app = FastAPI(title="YoHoGPT API", version="0.1.0", lifespan=lifespan)
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                           allow_headers=["*"], allow_credentials=False)

    @app.middleware("http")
    async def request_metrics(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        metrics.inc("http_requests_total", path=request.url.path if request.url.path.count("/") < 4 else "agent_run",
                    status=str(response.status_code))
        metrics.inc("http_request_seconds_sum", time.perf_counter() - start)
        response.headers["X-Request-Id"] = request.headers.get("x-request-id", uuid.uuid4().hex)
        return response

    def client(request: Request) -> str:
        cid = auth.http(request)
        limiter.check(cid)
        return cid

    def require_agent():
        if runs is None:
            raise HTTPException(404, "agent API disabled")
        return runs

    # ------------------------------------------------------------- basics
    @app.get("/health")
    async def health():
        return {"status": "ok", "model": backend.model_name}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus():
        return metrics.render()

    @app.get("/v1/models")
    async def models(_: str = Depends(client)):
        return {"object": "list", "data": [{"id": backend.model_name, "object": "model", "owned_by": "yohogpt"}]}

    # --------------------------------------------------------------- chat
    @app.post("/v1/chat/completions")
    async def chat(req: ChatCompletionRequest, _: str = Depends(client)):
        messages = to_internal_messages(req)
        params = _sampling(req)
        if req.tools:
            params.stop = list({*params.stop, TOOL_CALL_END})
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        metrics.inc("chat_requests_total", stream=str(req.stream).lower())

        def finalize(text: str, finish: str) -> tuple[dict, str]:
            msg: dict = {"role": "assistant", "content": text}
            if req.tools and TOOL_CALL in text:
                try:
                    call = parse_tool_call(text + (TOOL_CALL_END if TOOL_CALL_END not in text else ""))
                except ToolError:
                    call = None
                if call:
                    name, args = call
                    msg["content"] = text.split(TOOL_CALL)[0].strip() or None
                    msg["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                                          "function": {"name": name, "arguments": json.dumps(args)}}]
                    finish = "tool_calls"
            return msg, finish

        if not req.stream:
            async with gen_slots:
                res = await run_in_threadpool(lambda: collect(backend.chat_stream(messages, params)))
            msg, finish = finalize(res.text, res.finish_reason or "stop")
            metrics.inc("completion_tokens_total", res.completion_tokens)
            return {
                "id": cid, "object": "chat.completion", "created": created, "model": backend.model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                          "total_tokens": res.prompt_tokens + res.completion_tokens},
            }

        async def sse():
            def chunk(delta: dict, finish=None, usage=None):
                body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": backend.model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage:
                    body["usage"] = usage
                return f"data: {json.dumps(body)}\n\n"

            async with gen_slots:
                yield chunk({"role": "assistant", "content": ""})
                buffered, finish, usage = [], "stop", None
                async for ch in iterate_in_threadpool(backend.chat_stream(messages, params)):
                    if ch.text:
                        if req.tools:  # tool calls must be parsed whole; buffer
                            buffered.append(ch.text)
                        else:
                            yield chunk({"content": ch.text})
                    if ch.finish_reason:
                        finish = ch.finish_reason
                        usage = {"prompt_tokens": ch.prompt_tokens, "completion_tokens": ch.completion_tokens,
                                 "total_tokens": ch.prompt_tokens + ch.completion_tokens}
                        metrics.inc("completion_tokens_total", ch.completion_tokens)
                if req.tools:
                    msg, finish = finalize("".join(buffered), finish)
                    delta = {k: v for k, v in msg.items() if k != "role" and v is not None}
                    if "tool_calls" in delta:
                        delta["tool_calls"] = [{"index": 0, **tc} for tc in delta["tool_calls"]]
                    yield chunk(delta)
                yield chunk({}, finish, usage)
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # --------------------------------------------------------- completions
    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, _: str = Depends(client)):
        params = _sampling(req)
        cid, created = "cmpl-" + uuid.uuid4().hex[:24], int(time.time())
        metrics.inc("completion_requests_total", fim=str(req.suffix is not None).lower())
        if not req.stream:
            async with gen_slots:
                res = await run_in_threadpool(lambda: collect(backend.completion_stream(req.prompt, params, req.suffix)))
            return {"id": cid, "object": "text_completion", "created": created, "model": backend.model_name,
                    "choices": [{"index": 0, "text": res.text, "finish_reason": res.finish_reason}],
                    "usage": {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                              "total_tokens": res.prompt_tokens + res.completion_tokens}}

        async def sse():
            async with gen_slots:
                async for ch in iterate_in_threadpool(backend.completion_stream(req.prompt, params, req.suffix)):
                    body = {"id": cid, "object": "text_completion", "created": created, "model": backend.model_name,
                            "choices": [{"index": 0, "text": ch.text, "finish_reason": ch.finish_reason}]}
                    yield f"data: {json.dumps(body)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # --------------------------------------------------------------- agent
    @app.post("/v1/agent/runs", status_code=201)
    async def create_run(req: AgentRunRequest, owner: str = Depends(client)):
        mgr = require_agent()
        try:
            run = await run_in_threadpool(mgr.create, req, owner)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        metrics.inc("agent_runs_total")
        return run.public()

    @app.get("/v1/agent/runs")
    async def list_runs(owner: str = Depends(client)):
        return {"data": [r.public() for r in require_agent().list(owner)]}

    def get_run(run_id: str, owner: str):
        run = require_agent().get(run_id, owner)
        if run is None:
            raise HTTPException(404, "run not found")
        return run

    @app.get("/v1/agent/runs/{run_id}")
    async def run_status(run_id: str, owner: str = Depends(client)):
        return get_run(run_id, owner).public(include_events=True)

    @app.post("/v1/agent/runs/{run_id}/messages")
    async def run_message(run_id: str, body: AgentMessage, owner: str = Depends(client)):
        run = get_run(run_id, owner)
        if run.done or run.agent is None:
            raise HTTPException(409, f"run is {run.status}")
        run.agent.send_user_message(body.content)
        return {"ok": True}

    @app.post("/v1/agent/runs/{run_id}/cancel")
    async def run_cancel(run_id: str, owner: str = Depends(client)):
        run = get_run(run_id, owner)
        if run.agent:
            run.agent.cancel()
        return {"ok": True, "status": run.status}

    @app.get("/v1/agent/runs/{run_id}/events")
    async def run_events(run_id: str, owner: str = Depends(client)):
        run = get_run(run_id, owner)

        async def sse():
            sent = 0
            while True:
                new = await run_in_threadpool(run.wait_events, sent, 15.0)
                for ev in new:
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                sent += len(new)
                if not new:
                    yield ": keep-alive\n\n"
                if run.done and sent >= len(run.events):
                    yield f"event: end\ndata: {json.dumps(run.public())}\n\n"
                    return

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.websocket("/v1/agent/runs/{run_id}/ws")
    async def run_ws(websocket: WebSocket, run_id: str):
        try:
            owner = auth.ws(websocket)
        except HTTPException:
            await websocket.close(code=4401)
            return
        run = runs.get(run_id, owner) if runs else None
        if run is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()

        async def pump_in():
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "message" and run.agent and msg.get("content"):
                    run.agent.send_user_message(str(msg["content"]))
                elif msg.get("type") == "cancel" and run.agent:
                    run.agent.cancel()

        reader = asyncio.create_task(pump_in())
        try:
            sent = 0
            while True:
                new = await run_in_threadpool(run.wait_events, sent, 1.0)
                for ev in new:
                    await websocket.send_json(ev)
                sent += len(new)
                if run.done and sent >= len(run.events):
                    await websocket.send_json({"type": "end", "data": run.public()})
                    break
        except WebSocketDisconnect:
            pass
        finally:
            reader.cancel()
            try:
                await websocket.close()
            except RuntimeError:
                pass

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"error": {"message": "internal server error", "type": "server_error"}}, status_code=500)

    return app
