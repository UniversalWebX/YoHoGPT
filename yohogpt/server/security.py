"""API-key authentication, per-key rate limiting and Prometheus-style metrics."""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request, WebSocket


class APIKeyAuth:
    """Bearer-token auth. Keys are compared in constant time; only hashes are kept."""

    def __init__(self, keys: list[str]) -> None:
        self.hashes = {hashlib.sha256(k.encode()).digest() for k in keys if k}

    @property
    def enabled(self) -> bool:
        return bool(self.hashes)

    def check(self, token: str | None) -> str:
        """Return a stable client id for the key, or raise 401."""
        if not self.enabled:
            return "anonymous"
        if not token:
            raise HTTPException(401, "missing API key", headers={"WWW-Authenticate": "Bearer"})
        digest = hashlib.sha256(token.encode()).digest()
        if not any(hmac.compare_digest(digest, h) for h in self.hashes):
            raise HTTPException(401, "invalid API key", headers={"WWW-Authenticate": "Bearer"})
        return digest.hex()[:16]

    @staticmethod
    def token_from(headers, query=None) -> str | None:
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return headers.get("x-api-key") or (query.get("api_key") if query is not None else None)

    def http(self, request: Request) -> str:
        return self.check(self.token_from(request.headers))

    def ws(self, websocket: WebSocket) -> str:
        return self.check(self.token_from(websocket.headers, websocket.query_params))


class RateLimiter:
    """Token bucket per client: ``rate`` requests per minute with ``burst`` capacity."""

    def __init__(self, per_minute: int, burst: int | None = None) -> None:
        self.rate = per_minute / 60.0
        self.capacity = float(burst or max(1, per_minute))
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, client: str) -> bool:
        if self.rate <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(client, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                self.buckets[client] = (tokens, now)
                return False
            self.buckets[client] = (tokens - 1, now)
            return True

    def check(self, client: str) -> None:
        if not self.allow(client):
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "1"})


class Metrics:
    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = name + ("{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}" if labels else "")
        with self.lock:
            self.counters[key] += value

    def render(self) -> str:
        with self.lock:
            return "".join(f"yoho_{k} {v}\n" for k, v in sorted(self.counters.items()))
