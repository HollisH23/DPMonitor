"""Seed a demo USER + N synthetic sessions (Phase 2.1).

Useful for the plan's manual-verification path when a reviewer wants to
click through Login → Dashboard → Report without a working webcam.

Usage:
    python manage.py seed_demo                  # one user, one session
    python manage.py seed_demo --sessions 5     # one user, five sessions
    python manage.py seed_demo --clear          # remove demo data first

After running, log in as:
    username: demo
    password: demopass
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction
from rest_framework.authtoken.models import Token

from analyzer import AnalyzerFrame, PlaceholderAnalyzer
from clinical_sessions.models import Session, TrajectoryData


DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demopass"


def _synthetic_frames(n: int = 240, fps: int = 30, cycles: int = 4) -> list[dict]:
    """Build a deterministic, MediaPipe-shaped 30-FPS stream of squats."""
    frames: list[dict] = []
    for i in range(n):
        phase = (i / n) * cycles * 2 * math.pi
        hip_y = 0.60 - 0.10 * math.cos(phase)
        knee = 170 - 90 * (0.5 - 0.5 * math.cos(phase))
        back = 170 - 10 * (0.5 - 0.5 * math.cos(phase))
        frames.append({
            "frame_index": i,
            "timestamp_ms": i * (1000.0 / fps),
            "points": {
                "left_shoulder":  [0.45, 0.30, 0.0, 1.0],
                "right_shoulder": [0.55, 0.30, 0.0, 1.0],
                "left_hip":  [0.45, hip_y, 0.0, 1.0],
                "right_hip": [0.55, hip_y, 0.0, 1.0],
                "left_knee":  [0.46, hip_y + 0.13, 0.0, 1.0],
                "right_knee": [0.54, hip_y + 0.13, 0.0, 1.0],
                "left_ankle":  [0.47, hip_y + 0.30, 0.0, 1.0],
                "right_ankle": [0.53, hip_y + 0.30, 0.0, 1.0],
            },
            "angles": {"left_knee": knee, "right_knee": knee, "back": back},
        })
    return frames


def _persist_buffer(frames: list[dict], persist_fps: int = 15) -> list[dict]:
    if not frames:
        return []
    step = max(1, int(round((1000.0 / persist_fps) /
                            (frames[1]["timestamp_ms"] - frames[0]["timestamp_ms"]))))
    out = []
    for i in range(0, len(frames), step):
        f = frames[i]
        out.append({
            "frame": f["frame_index"],
            "t": round(f["timestamp_ms"]),
            "points": f["points"],
            "angles": f["angles"],
        })
    return out


class Command(BaseCommand):
    help = "Seed a demo user + N synthetic sessions for manual verification."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--sessions", type=int, default=1)
        parser.add_argument("--clear", action="store_true",
                            help="Delete the demo user + cascade their sessions first.")
        parser.add_argument("--seed", type=int, default=1337)

    def handle(self, *args, **options) -> None:
        User = get_user_model()

        if options["clear"]:
            n, _ = User.objects.filter(username=DEMO_USERNAME).delete()
            self.stdout.write(self.style.WARNING(f"Cleared demo user (cascade={n})"))

        with transaction.atomic():
            user, created = User.objects.get_or_create(
                username=DEMO_USERNAME,
                defaults={"email": "demo@local"},
            )
            # Always (re)set the demo password so it's predictable.
            user.set_password(DEMO_PASSWORD)
            user.save()
            token, _ = Token.objects.get_or_create(user=user)

            for s_idx in range(options["sessions"]):
                analyzer = PlaceholderAnalyzer(seed=options["seed"], exercise_type="squat")
                raw = _synthetic_frames()
                for f in raw:
                    analyzer.analyze_frame(AnalyzerFrame(**f))
                summary = analyzer.generate_summary()

                started = datetime.now(tz=timezone.utc) - timedelta(minutes=10 * (s_idx + 1))
                ended = started + timedelta(seconds=raw[-1]["timestamp_ms"] / 1000.0)

                session = Session.objects.create(
                    user=user,
                    exercise_type="squat",
                    started_at=started,
                    ended_at=ended,
                    rep_count=summary.rep_count,
                    overall_stability_score=summary.overall_stability_score,
                    quality_score=summary.quality_score,
                    progress_trend={
                        **summary.progress_trend,
                        "compensation_events": summary.compensation_events,
                    },
                    random_seed=options["seed"],
                    notes="Auto-generated demo session.",
                )
                buf = _persist_buffer(raw, persist_fps=15)
                TrajectoryData.objects.create(
                    session=session,
                    sample_rate_hz=15,
                    frames=buf,
                    frame_count=len(buf),
                )
                self.stdout.write(self.style.SUCCESS(
                    f"  • session #{session.id}: {summary.rep_count} reps, "
                    f"stability={summary.overall_stability_score:.2f}, "
                    f"quality={summary.quality_score:.2f}"
                ))

        self.stdout.write(self.style.SUCCESS(
            f"\nDemo user ready.\n"
            f"  username : {DEMO_USERNAME}\n"
            f"  password : {DEMO_PASSWORD}\n"
            f"  token    : {token.key}\n"
        ))
