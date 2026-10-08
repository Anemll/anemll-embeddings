"""Request-size cap for the demo server (pure ASGI, no extra dependencies).

Rejects a request with **413** when its ``Content-Length`` is over the cap,
and also while streaming, so a chunked upload without a length cannot slip
past. Multipart (file uploads) and other bodies (JSON) have separate caps;
see ``demo/settings.py`` for the defaults and environment overrides.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class RequestTooLarge(Exception):
    def __init__(self, limit: int) -> None:
        super().__init__(f"request body is larger than {limit} bytes")
        self.limit = limit


def _detail(limit: int) -> bytes:
    mb = limit / (1024 * 1024)
    text = f"request body too large (limit {mb:.1f} MB)" if mb >= 1 else (
        f"request body too large (limit {limit // 1024} KB)"
    )
    return json.dumps({"detail": text}).encode("utf-8")


class BodySizeLimit:
    def __init__(self, app, *, max_upload: int, max_other: int) -> None:
        self.app = app
        self.max_upload = int(max_upload)
        self.max_other = int(max_other)

    def limit_for(self, scope: Scope) -> int:
        headers = dict(scope.get("headers") or [])
        ctype = headers.get(b"content-type", b"").decode("latin-1").lower()
        return self.max_upload if ctype.startswith("multipart/") else self.max_other

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_for(scope)
        headers = dict(scope.get("headers") or [])
        raw_len = headers.get(b"content-length")
        if raw_len is not None:
            try:
                declared = int(raw_len)
            except ValueError:
                declared = -1
            if declared > limit:
                await _reject(send, limit)
                return

        seen = 0
        started = False

        async def counted_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise RequestTooLarge(limit)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counted_receive, tracking_send)
        except RequestTooLarge:
            if not started:
                await _reject(send, limit)


async def _reject(send: Send, limit: int) -> None:
    body = _detail(limit)
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
