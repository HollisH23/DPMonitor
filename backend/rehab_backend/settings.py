"""Django settings for the rehab_backend project.

Privacy-first / edge-computing oriented:
  * SQLite at a stable local path (cross-platform via pathlib).
  * No third-party telemetry.
  * CORS opened only to the local dev frontend by default.

For production deployment on Windows or macOS, override SECRET_KEY,
DEBUG and ALLOWED_HOSTS via environment variables.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Repo root holds the ``ctrgcn`` package (sibling of ``backend/``). Putting
# it on sys.path lets ``analyzer.mediapipe_graph`` and the CTR-GCN
# analyzer import ``ctrgcn.*`` without copying or symlinking the package.
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SECRET_KEY = os.environ.get(
    "REHAB_SECRET_KEY",
    "dev-insecure-secret-key-change-me-in-production",
)
DEBUG = os.environ.get("REHAB_DEBUG", "1") == "1"
ALLOWED_HOSTS: list[str] = os.environ.get(
    "REHAB_ALLOWED_HOSTS", "localhost,127.0.0.1"
).split(",")

# -----------------------------------------------------------------------------
# Applications
# -----------------------------------------------------------------------------
INSTALLED_APPS = [
    "daphne",  # must come before django.contrib.staticfiles for ASGI dev server
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # third party
    "rest_framework",
    "rest_framework.authtoken",  # Phase 2.1: token-based REST auth
    "corsheaders",
    "channels",
    # local apps (Phase 2.1: patients app deprecated, sessions own the user FK)
    "clinical_sessions.apps.ClinicalSessionsConfig",
    "api.apps.ApiConfig",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "rehab_backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "rehab_backend.wsgi.application"
ASGI_APPLICATION = "rehab_backend.asgi.application"

# -----------------------------------------------------------------------------
# Database — SQLite for zero-config local clinical record keeping.
# `pathlib` everywhere keeps macOS dev and Windows deployment paths sane.
# -----------------------------------------------------------------------------
DB_DIR = BASE_DIR / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(DB_DIR / "rehab_local.sqlite3"),
    }
}

# -----------------------------------------------------------------------------
# Channels — in-memory layer is sufficient for single-machine MVP.
# Swap to redis layer for multi-process / multi-node deployments.
# -----------------------------------------------------------------------------
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}

# -----------------------------------------------------------------------------
# Auth / i18n / static
# -----------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# -----------------------------------------------------------------------------
# DRF / CORS — keep CORS narrow to the local dev frontend.
# -----------------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
    ],
    # Phase 2.1: every endpoint is gated by Token auth. Endpoints that
    # need to be reachable pre-login (health, login, register) opt out
    # via @permission_classes([AllowAny]) at the view level.
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
}

CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
CORS_ALLOW_HEADERS = list(
    # Default CORS headers + Authorization for our Token auth scheme.
    ("accept", "accept-encoding", "authorization", "content-type", "dnt",
     "origin", "user-agent", "x-csrftoken", "x-requested-with")
)

# -----------------------------------------------------------------------------
# Logging — file-based audit log written via pathlib for portability.
# -----------------------------------------------------------------------------
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "rehab_backend.log"),
        },
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "rehab": {
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

# -----------------------------------------------------------------------------
# Determinism — single source of truth for the global random seed used by the
# analyzer layer (see analyzer/seed.py).
# -----------------------------------------------------------------------------
REHAB_RANDOM_SEED = int(os.environ.get("REHAB_RANDOM_SEED", "1337"))

# -----------------------------------------------------------------------------
# Live analyzer selection. Flipping this routes the WebSocket consumer to a
# different concrete analyzer without any consumer-side code changes. Valid
# values: "placeholder" (geometric counter, no ML deps), "ctrgcn" (PyTorch
# CTR-GCN-backed quality scorer + heuristic counter).
# -----------------------------------------------------------------------------
REHAB_ANALYZER = os.environ.get("REHAB_ANALYZER", "ctrgcn")

# Optional path to a fine-tuned CTR-GCN checkpoint. Empty string ≡ "no
# checkpoint — use the freshly-initialised weights (seeded deterministically)".
REHAB_CTRGCN_WEIGHTS = os.environ.get("REHAB_CTRGCN_WEIGHTS", "")

# Sliding-window length (frames) for CTR-GCN inference. 64 ≈ 2.1 s at 30 FPS.
REHAB_CTRGCN_WINDOW = int(os.environ.get("REHAB_CTRGCN_WINDOW", "64"))
