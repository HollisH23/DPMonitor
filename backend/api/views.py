"""REST endpoints (Phase 2.1).

All session endpoints are gated by `IsAuthenticated` and strictly scoped
to `request.user` at the queryset level — there is no way for the
frontend to address another user's data, even by guessing IDs.

Endpoints (all under `/api/`):

    GET    /sessions/                       list MY sessions
    GET    /sessions/<id>/                  retrieve one of MY sessions
    DELETE /sessions/<id>/                  delete one of MY sessions
    POST   /sessions/ingest/                batch-upload a completed session
                                            (user taken from request.user)

    GET    /health/                         public liveness probe
    GET    /trend/                          last-7 metrics for current user
    POST   /auth/register/                  create account
    POST   /auth/login/                     get token
    POST   /auth/logout/                    revoke token
    GET    /auth/me/                        whoami
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from django.db import transaction
from django.db.models import Avg, Count, Sum
from django.utils import timezone
from rest_framework import generics, status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.exceptions import NotFound
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response

from analyzer import PlaceholderAnalyzer
from clinical_sessions.models import Session, TrajectoryData

from .serializers import (
    SessionDetailSerializer,
    SessionIngestSerializer,
    SessionListSerializer,
)

logger = logging.getLogger("rehab")


# ---------------------------------------------------------------------------
# Sessions — owner-scoped by default.
# ---------------------------------------------------------------------------


class SessionList(generics.ListAPIView):
    """List the *current user's* sessions only."""

    serializer_class = SessionListSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Session.objects.filter(user=self.request.user)


class SessionDetail(generics.RetrieveDestroyAPIView):
    """GET / DELETE one of the current user's sessions.

    By filtering the queryset on `user`, requests for someone else's
    session id deterministically return 404 — never 403 — which means
    we don't even leak the existence of foreign rows.
    """

    serializer_class = SessionDetailSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return (Session.objects
                .filter(user=self.request.user)
                .prefetch_related("trajectory"))


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def session_ingest(request: Request) -> Response:
    """Batch-upload a completed session.

    Phase 2.1: the user is taken from `request.user`. The serializer
    rejects any incoming `patient` / `user` field implicitly because
    they are not declared.
    """
    serializer = SessionIngestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    data = serializer.validated_data

    with transaction.atomic():
        session = Session.objects.create(
            user=request.user,
            exercise_type=data["exercise_type"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            rep_count=data["rep_count"],
            overall_stability_score=data["overall_stability_score"],
            quality_score=data["quality_score"],
            progress_trend=data.get("progress_trend") or {},
            random_seed=data.get("random_seed"),
            notes=data.get("notes", ""),
        )
        TrajectoryData.objects.create(
            session=session,
            sample_rate_hz=data["sample_rate_hz"],
            frame_count=len(data["frames"]),
            frames=data["frames"],
        )

    logger.info(
        "ingest_session id=%s user=%s frames=%s reps=%s",
        session.id,
        request.user.id,
        len(data["frames"]),
        data["rep_count"],
    )
    return Response(
        SessionDetailSerializer(session).data,
        status=status.HTTP_201_CREATED,
    )


# ---------------------------------------------------------------------------
# Dashboard support — last-7 trends (per-user, retained from Phase 1).
# ---------------------------------------------------------------------------


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def trend_last_seven(request: Request) -> Response:
    """Aggregate metrics over the last 7 days, scoped to the caller."""
    cutoff = timezone.now() - timedelta(days=7)
    qs = Session.objects.filter(user=request.user, started_at__gte=cutoff)
    agg = qs.aggregate(
        sessions=Count("id"),
        total_reps=Sum("rep_count"),
        avg_stability=Avg("overall_stability_score"),
        avg_quality=Avg("quality_score"),
    )
    # Per-exercise completion counts power the dashboard's "tasks" view.
    by_exercise = list(
        qs.values("exercise_type")
          .annotate(count=Count("id"))
          .order_by("-count")
    )
    return Response({
        "window_days": 7,
        "sessions": agg["sessions"] or 0,
        "total_reps": agg["total_reps"] or 0,
        "avg_stability": float(agg["avg_stability"] or 0.0),
        "avg_quality": float(agg["avg_quality"] or 0.0),
        "by_exercise": by_exercise,
    })


# ---------------------------------------------------------------------------
# Health / metadata — public.
# ---------------------------------------------------------------------------


@api_view(["GET"])
@authentication_classes([])
@permission_classes([AllowAny])
def health(_request: Request) -> Response:
    """Liveness + analyzer metadata. Public (no auth)."""
    from django.conf import settings

    seed = settings.REHAB_RANDOM_SEED
    payload: dict[str, Any] = {
        "status": "ok",
        "edge_computing": True,
        "random_seed": seed,
        "analyzer": {
            "analyzer": PlaceholderAnalyzer.name,
            "supported_exercises": list(PlaceholderAnalyzer.supported_exercises),
        },
    }
    return Response(payload)
