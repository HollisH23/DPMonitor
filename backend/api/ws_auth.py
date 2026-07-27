"""Channels token-auth middleware.

Channels' built-in `AuthMiddleware` reads from the Django session cookie,
which doesn't help us — our REST clients hold a DRF Token in localStorage
and pass it via `Authorization: Token <key>` for HTTP. WebSocket browsers
cannot set custom headers on `new WebSocket(...)`, so we accept the token
as a query-string parameter instead:

    ws://host/ws/monitor/?token=<key>

The middleware swaps `scope["user"]` for the resolved user (or leaves
`AnonymousUser` in place). The consumer is then free to reject anonymous
connections at `connect()` time.
"""
from __future__ import annotations

from urllib.parse import parse_qs

from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser


@database_sync_to_async
def _user_for_token(key: str):
    # Import lazily so Django apps are guaranteed to be loaded by the time
    # this runs (the ASGI app does that before importing routing).
    from rest_framework.authtoken.models import Token

    try:
        return Token.objects.select_related("user").get(key=key).user
    except Token.DoesNotExist:
        return AnonymousUser()


class TokenAuthMiddleware:
    """ASGI middleware that resolves `?token=<key>` to `scope['user']`."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        # Default to anonymous if anything goes wrong.
        scope["user"] = AnonymousUser()
        qs = scope.get("query_string", b"")
        if qs:
            params = parse_qs(qs.decode("utf-8") if isinstance(qs, bytes) else qs)
            tokens = params.get("token") or []
            if tokens:
                scope["user"] = await _user_for_token(tokens[0])
        return await self.inner(scope, receive, send)
