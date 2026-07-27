#!/usr/bin/env python
"""Django command-line utility for administrative tasks.

Use `python manage.py runserver` for HTTP-only smoke testing and
`daphne rehab_backend.asgi:application` (or `python manage.py runserver`
with channels' ASGI auto-wiring) for full WebSocket support.
"""
import os
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rehab_backend.settings")
    # Ensure the backend directory is on sys.path so relative imports work
    # regardless of where the script is invoked from.
    BASE_DIR = Path(__file__).resolve().parent
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "Couldn't import Django. Make sure it's installed and available "
            "on your PYTHONPATH environment variable. Did you forget to "
            "activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
