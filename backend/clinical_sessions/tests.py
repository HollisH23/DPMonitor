from datetime import datetime, timezone

from django.contrib.auth import get_user_model
from django.test import TestCase

from clinical_sessions.models import Session, TrajectoryData


class SessionModelTests(TestCase):
    def setUp(self) -> None:
        User = get_user_model()
        self.user = User.objects.create_user(username="alpha", password="pw")

    def test_create_with_trajectory(self) -> None:
        s = Session.objects.create(
            user=self.user,
            exercise_type="squat",
            started_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            rep_count=5,
            overall_stability_score=0.8,
            quality_score=0.9,
            random_seed=1337,
        )
        TrajectoryData.objects.create(session=s, frame_count=3, frames=[1, 2, 3])
        self.assertEqual(s.trajectory.frame_count, 3)
        self.assertEqual(s.random_seed, 1337)
        self.assertEqual(self.user.sessions.count(), 1)
