"""Channels URL routing for the WebSocket transport."""
from django.urls import re_path

from .consumers import MonitorConsumer

websocket_urlpatterns = [
    re_path(r"^ws/monitor/$", MonitorConsumer.as_asgi()),
]
