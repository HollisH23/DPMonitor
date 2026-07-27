"""ASGI entrypoint with Channels routing for the WebSocket transport.

Phase 2.1: WebSocket connections are authenticated by the
`TokenAuthMiddleware` before any consumer code runs. The consumer
inspects `scope["user"]` and rejects anonymous connections immediately.

Run with:
    daphne -b 0.0.0.0 -p 8000 rehab_backend.asgi:application
or simply `python manage.py runserver` (Channels auto-installs the ASGI handler).
"""
import os

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rehab_backend.settings")

# Initialise Django ASGI app first so models are loaded before consumers import them.
django_asgi_app = get_asgi_application()

from api.routing import websocket_urlpatterns  # noqa: E402  (must follow django setup)
from api.ws_auth import TokenAuthMiddleware    # noqa: E402

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AllowedHostsOriginValidator(
            TokenAuthMiddleware(URLRouter(websocket_urlpatterns)),
        ),
    }
)
