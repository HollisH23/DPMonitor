"""Real-time WebSocket consumer (Phase 2.1).

The connection is authenticated by `TokenAuthMiddleware` before this
consumer runs. We reject anonymous connections in `connect()`.

Protocol (newline-delimited JSON over a single WebSocket):

    client → server:
        {"type": "hello",   "session_seed": 1337, "exercise_type": "squat"}
        {"type": "frame",   "frame_index": 12, "timestamp_ms": 400.0,
         "points": {...}, "angles": {"left_knee": 92.3}}
        {"type": "bye"}

    server → client:
        {"type": "ready",   "analyzer": {...}, "user": "<username>"}
        {"type": "result",  "frame_index": 12, "count": 3,
         "quality_score": 0.91, "is_compensatory": false,
         "feedback": ["Rep counted"], "diagnostics": {...}}
        {"type": "summary", ...}     (sent on `bye` or socket close)
        {"type": "error",   "detail": "..."}

The active analyzer is resolved via `analyzer.get_analyzer(...)` so the
consumer is independent of which concrete model is wired in (see
``settings.REHAB_ANALYZER``). The trajectory is NOT persisted from
here — the client batches the full buffer to
`POST /api/sessions/ingest/` at COMPLETED state.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

from analyzer import AnalyzerFrame, BaseAnalyzer, get_analyzer

logger = logging.getLogger("rehab")


class MonitorConsumer(AsyncWebsocketConsumer):
    """One consumer instance per live monitoring session, owned by one user."""

    analyzer: Optional[BaseAnalyzer]

    async def connect(self) -> None:
        user = self.scope.get("user")
        # Reject unauthenticated connections with code 4401 ("custom 401")
        # rather than silently accepting them — this surfaces auth bugs
        # in the client during development.
        if user is None or not getattr(user, "is_authenticated", False):
            await self.close(code=4401)
            return
        self.analyzer = None
        self.user = user
        await self.accept()

    async def disconnect(self, code: int) -> None:
        if getattr(self, "analyzer", None) is not None:
            await self._send_summary()

    async def receive(self, text_data: Optional[str] = None, bytes_data: Optional[bytes] = None) -> None:
        if text_data is None:
            return
        try:
            payload: Dict[str, Any] = json.loads(text_data)
        except json.JSONDecodeError:
            await self._send({"type": "error", "detail": "Invalid JSON"})
            return

        msg_type = payload.get("type")
        if msg_type == "hello":
            seed = int(payload.get("session_seed", settings.REHAB_RANDOM_SEED))
            exercise = payload.get("exercise_type", "custom")
            try:
                self.analyzer = self._build_analyzer(seed=seed, exercise=exercise)
            except Exception as exc:
                # If CTR-GCN fails to construct (e.g. torch missing in a CI
                # smoke env), fall back to the placeholder so the live view
                # keeps working rather than dropping the socket.
                logger.exception("analyzer build failed (%s); falling back to placeholder", exc)
                self.analyzer = get_analyzer(
                    "placeholder", seed=seed, exercise_type=exercise
                )
            await self._send({
                "type": "ready",
                "analyzer": self.analyzer.describe(),
                "user": self.user.username,
            })
            return

        if msg_type == "frame":
            if self.analyzer is None:
                await self._send({"type": "error", "detail": "Send 'hello' first"})
                return
            try:
                frame = AnalyzerFrame(
                    frame_index=int(payload["frame_index"]),
                    timestamp_ms=float(payload.get("timestamp_ms", 0.0)),
                    points=payload.get("points", {}),
                    angles=payload.get("angles", {}),
                )
            except (KeyError, TypeError, ValueError) as exc:
                await self._send({"type": "error", "detail": f"Bad frame: {exc}"})
                return
            result = self.analyzer.analyze_frame(frame)
            await self._send({"type": "result", **result.to_json()})
            return

        if msg_type == "bye":
            await self._send_summary()
            return

        await self._send({"type": "error", "detail": f"Unknown type: {msg_type!r}"})

    # ------------------------------------------------------------------

    def _build_analyzer(self, *, seed: int, exercise: str) -> BaseAnalyzer:
        """Construct the active analyzer per Django settings."""
        name = getattr(settings, "REHAB_ANALYZER", "placeholder")
        kwargs: Dict[str, Any] = {}
        if name.lower().startswith("ctrgcn"):
            weights = getattr(settings, "REHAB_CTRGCN_WEIGHTS", "") or None
            window = int(getattr(settings, "REHAB_CTRGCN_WINDOW", 64))
            kwargs["weights_path"] = weights
            kwargs["window_size"] = window
        return get_analyzer(name, seed=seed, exercise_type=exercise, **kwargs)

    async def _send_summary(self) -> None:
        if self.analyzer is None:
            return
        summary = self.analyzer.generate_summary()
        await self._send({"type": "summary", **summary.to_json()})

    async def _send(self, payload: Dict[str, Any]) -> None:
        await self.send(text_data=json.dumps(payload))
