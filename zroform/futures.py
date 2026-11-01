"""In-memory futures with long-poll retrieval (Tinker wire protocol).

Results (which carry token-level content) live only in this process's memory and are dropped
`ttl` seconds after completion. Nothing here is ever written to disk.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Result:
    kind: str                 # "json" | "proto" | "error"
    payload: object           # dict | bytes | {"error": str, "category": "user"|"server"}


@dataclass
class _Future:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: Result | None = None
    done_at: float = 0.0


class Futures:
    def __init__(self, ttl_sec: float = 600):
        self._ttl = ttl_sec
        self._d: dict[str, _Future] = {}

    def create(self) -> str:
        self._sweep()
        rid = uuid.uuid4().hex
        self._d[rid] = _Future()
        return rid

    def resolve(self, rid: str, result: Result) -> None:
        f = self._d.get(rid)
        if f is None or f.result is not None:
            return
        f.result, f.done_at = result, time.monotonic()
        f.event.set()

    def fail(self, rid: str, msg: str, category: str = "server") -> None:
        self.resolve(rid, Result("error", {"error": msg, "category": category}))

    def known(self, rid: str) -> bool:
        return rid in self._d

    async def wait(self, rid: str, timeout: float) -> Result | None:
        """Long-poll: the result, or None on timeout. Raises KeyError if unknown/expired."""
        f = self._d[rid]
        if f.result is None:
            try:
                await asyncio.wait_for(f.event.wait(), timeout)
            except TimeoutError:
                return None
        return f.result

    def _sweep(self) -> None:
        now = time.monotonic()
        for rid in [r for r, f in self._d.items() if f.result is not None and now - f.done_at > self._ttl]:
            del self._d[rid]
