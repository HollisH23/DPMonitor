"""Exercise-session persistence layer (Phase 2.1).

Sessions are now owned directly by an authenticated user — the Patient
table has been deprecated and removed. The user FK ties trajectory data
to the logged-in account so strict per-user data isolation can be
enforced at the queryset level.

Two models:

* `Session` — per-attempt summary record with denormalised metrics for
  fast historical comparisons.
* `TrajectoryData` — the raw 15-FPS skeletal stream held in a single
  JSONField per session. Storing it as one row keeps writes cheap (one
  INSERT at session-completion time) and simplifies the future XAI hook,
  which needs whole-session windows rather than individual frames.
"""
from __future__ import annotations

from django.conf import settings
from django.db import models


class Session(models.Model):
    """A single rehabilitation-exercise attempt owned by one user."""

    EXERCISE_CHOICES = [
        ("squat", "Squat"),
        ("lunge", "Lunge"),
        ("shoulder_raise", "Shoulder Raise"),
        ("knee_extension", "Knee Extension"),
        ("chest_expansion", "Chest Expansion"),
        ("custom", "Custom"),
    ]

    # Direct link to the authenticated user. `AUTH_USER_MODEL` is read
    # lazily so a future swap to a CustomUser model is a one-line change.
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="sessions",
    )
    exercise_type = models.CharField(
        max_length=32,
        choices=EXERCISE_CHOICES,
        default="custom",
    )
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField(null=True, blank=True)

    # Denormalised summary metrics produced by the analyzer at session-end.
    rep_count = models.PositiveIntegerField(default=0)
    overall_stability_score = models.FloatField(
        default=0.0,
        help_text="0.0–1.0; higher = smoother trajectory.",
    )
    quality_score = models.FloatField(
        default=0.0,
        help_text="0.0–1.0; clinician-facing aggregate of form quality.",
    )
    progress_trend = models.JSONField(
        default=dict,
        blank=True,
        help_text=(
            "Free-form summary metrics for trend charts on the dashboard, "
            "e.g. {\"rom_avg\": 0.82, \"compensation_events\": 3}."
        ),
    )

    # Reproducibility: which deterministic seed produced these scores.
    random_seed = models.IntegerField(null=True, blank=True)

    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-started_at"]
        indexes = [
            # Fast per-user history queries.
            models.Index(fields=["user", "started_at"]),
            models.Index(fields=["started_at"]),
        ]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"Session<{self.user_id}@{self.started_at:%Y-%m-%d %H:%M}>"


class TrajectoryData(models.Model):
    """Raw keypoint stream + per-frame metrics for a session.

    The `frames` field stores a list of dicts shaped:
        {"frame": <int>, "t": <float ms>, "points": {<landmark>: [x,y,z,v]},
         "metrics": {"angle_left_knee": 92.1, ...}}
    """

    session = models.OneToOneField(
        Session,
        on_delete=models.CASCADE,
        related_name="trajectory",
    )
    # 15 FPS persistence (the 30 FPS UI stream is downsampled client-side).
    sample_rate_hz = models.PositiveSmallIntegerField(default=15)
    frame_count = models.PositiveIntegerField(default=0)
    frames = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["session"])]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"TrajectoryData<session={self.session_id} frames={self.frame_count}>"
