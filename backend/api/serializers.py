"""DRF serializers (Phase 2.1).

`patient` foreign-key fields have been removed; sessions are bound to
`request.user` server-side. Frontend payloads therefore no longer carry
any identifier — the token in the request header is the sole source of
identity.
"""
from __future__ import annotations

from rest_framework import serializers

from clinical_sessions.models import Session, TrajectoryData


class TrajectoryDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrajectoryData
        fields = ["id", "sample_rate_hz", "frame_count", "frames", "created_at"]
        read_only_fields = ["id", "created_at"]


class SessionListSerializer(serializers.ModelSerializer):
    """Light payload for dashboard lists.

    Note: `user` is intentionally omitted — the caller is always reading
    their own sessions, so echoing the FK adds no information and might
    accidentally leak someone else's ID into a UI cache.
    """

    class Meta:
        model = Session
        fields = [
            "id",
            "exercise_type",
            "started_at",
            "ended_at",
            "rep_count",
            "overall_stability_score",
            "quality_score",
        ]
        read_only_fields = fields


class SessionDetailSerializer(serializers.ModelSerializer):
    trajectory = TrajectoryDataSerializer(read_only=True)

    class Meta:
        model = Session
        fields = [
            "id",
            "exercise_type",
            "started_at",
            "ended_at",
            "rep_count",
            "overall_stability_score",
            "quality_score",
            "progress_trend",
            "random_seed",
            "notes",
            "trajectory",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class SessionIngestSerializer(serializers.Serializer):
    """Batch upload at session completion.

    Phase 2.1: no `patient` field. The user is taken from `request.user`
    by the view, never from the client payload.
    """

    exercise_type = serializers.ChoiceField(
        choices=Session.EXERCISE_CHOICES, default="custom"
    )
    started_at = serializers.DateTimeField()
    ended_at = serializers.DateTimeField(required=False, allow_null=True)
    rep_count = serializers.IntegerField(min_value=0, default=0)
    overall_stability_score = serializers.FloatField(min_value=0.0, max_value=1.0, default=0.0)
    quality_score = serializers.FloatField(min_value=0.0, max_value=1.0, default=0.0)
    progress_trend = serializers.JSONField(required=False, default=dict)
    random_seed = serializers.IntegerField(required=False, allow_null=True)
    notes = serializers.CharField(required=False, allow_blank=True, default="")
    frames = serializers.ListField(child=serializers.DictField(), required=True)
    sample_rate_hz = serializers.IntegerField(min_value=1, max_value=120, default=15)
