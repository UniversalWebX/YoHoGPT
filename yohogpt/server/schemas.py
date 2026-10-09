"""Request/response schemas (OpenAI-compatible where applicable)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def text(self) -> str:
        if isinstance(self.content, list):
            return "".join(p.get("text", "") for p in self.content if p.get("type", "text") == "text")
        return self.content or ""


class SamplingFields(BaseModel):
    model: str = "yohogpt"
    max_tokens: int | None = Field(default=1024, ge=1, le=32768)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)
    user: str | None = None


class ChatCompletionRequest(SamplingFields):
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class CompletionRequest(SamplingFields):
    prompt: str
    suffix: str | None = None  # fill-in-the-middle for IDE completion


class AgentRunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=50_000)
    repo_url: str | None = Field(default=None, description="git URL cloned into the workspace")
    branch: str | None = None
    files: dict[str, str] | None = Field(default=None, description="files to seed the workspace with")
    max_steps: int = Field(default=50, ge=1, le=500)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class AgentMessage(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
