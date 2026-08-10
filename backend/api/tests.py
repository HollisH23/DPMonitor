"""End-to-end REST + auth tests (Phase 2.1).

Run with:
    cd backend
    python manage.py test
"""
from __future__ import annotations

import math
import random
import unittest
from datetime import datetime, timedelta, timezone
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from analyzer import (
    AnalyzerFrame,
    PlaceholderAnalyzer,
    apply_global_seed,
    deterministic_context,
)
from clinical_sessions.models import Session, TrajectoryData

# The CTR-GCN analyzer requires torch; skip its tests when torch is absent so
# CI environments that haven't installed the heavy dep still get a green run.
try:
    import torch  # type: ignore  # noqa: F401
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

User = get_user_model()


def _synthetic_frames(n: int = 60) -> list[dict]:
    """Build a deterministic synthetic stream of MediaPipe-ish frames."""
    frames = []
    for i in range(n):
        y = 0.575 + 0.075 * math.sin(i / 30.0 * 2 * math.pi)
        frames.append(
            {
                "frame_index": i,
                "timestamp_ms": i * (1000.0 / 30.0),
                "points": {
                    "left_hip": [0.45, y, 0.0, 1.0],
                    "right_hip": [0.55, y, 0.0, 1.0],
                },
                "angles": {"left_knee": 95.0},
            }
        )
    return frames


def _synthetic_full_pose_frames(n: int = 80) -> list[dict]:
    """Build a fully-populated MediaPipe-33 stream for CTR-GCN inference.

    Every landmark is filled with a deterministic, slowly-varying sinusoid so
    the model receives a non-degenerate input and the tensor reshape exercises
    all 33 vertices rather than zero-padding most of them.
    """
    from analyzer.mediapipe_graph import LANDMARK_NAMES

    frames = []
    for i in range(n):
        phase = i / 30.0 * 2 * math.pi
        points = {}
        for j, name in enumerate(LANDMARK_NAMES):
            # Spread joints around the canvas with per-joint phase offsets so
            # the trajectory is recognisable but bounded inside [0, 1].
            x = 0.5 + 0.1 * math.sin(phase + j * 0.05)
            y = 0.5 + 0.1 * math.cos(phase + j * 0.05)
            z = 0.05 * math.sin(phase + j * 0.1)
            points[name] = [x, y, z, 1.0]
        frames.append(
            {
                "frame_index": i,
                "timestamp_ms": i * (1000.0 / 30.0),
                "points": points,
                "angles": {"left_knee": 95.0},
            }
        )
    return frames


def _auth_client(user) -> APIClient:
    token, _ = Token.objects.get_or_create(user=user)
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
    return client


# ---------------------------------------------------------------------------
# Auth surface
# ---------------------------------------------------------------------------


class AuthEndpointTests(TestCase):
    def setUp(self) -> None:
        self.client = APIClient()

    def test_register_returns_token_and_creates_user(self) -> None:
        resp = self.client.post(
            reverse("auth-register"),
            data={"username": "alice", "password": "topsecret"},
            format="json",
        )
        self.assertEqual(resp.status_code, 201, resp.content)
        self.assertIn("token", resp.data)
        self.assertEqual(resp.data["user"]["username"], "alice")
        self.assertTrue(User.objects.filter(username="alice").exists())

    def test_register_rejects_duplicate_username(self) -> None:
        User.objects.create_user(username="bob", password="pw1234")
        resp = self.client.post(
            reverse("auth-register"),
            data={"username": "bob", "password": "pw1234"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_register_rejects_short_password(self) -> None:
        resp = self.client.post(
            reverse("auth-register"),
            data={"username": "carol", "password": "123"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_login_returns_token(self) -> None:
        User.objects.create_user(username="dave", password="pw123456")
        resp = self.client.post(
            reverse("auth-login"),
            data={"username": "dave", "password": "pw123456"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("token", resp.data)

    def test_login_rejects_bad_credentials(self) -> None:
        User.objects.create_user(username="eve", password="pw123456")
        resp = self.client.post(
            reverse("auth-login"),
            data={"username": "eve", "password": "wrong"},
            format="json",
        )
        self.assertEqual(resp.status_code, 401)

    def test_me_requires_auth(self) -> None:
        resp = self.client.get(reverse("auth-me"))
        self.assertEqual(resp.status_code, 401)

    def test_me_returns_current_user(self) -> None:
        u = User.objects.create_user(username="frank", password="pw123456")
        c = _auth_client(u)
        resp = c.get(reverse("auth-me"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["user"]["username"], "frank")

    def test_logout_revokes_token(self) -> None:
        u = User.objects.create_user(username="gina", password="pw123456")
        c = _auth_client(u)
        self.assertEqual(c.post(reverse("auth-logout")).status_code, 200)
        # The same client still carries the (now-deleted) token; further
        # calls must 401.
        self.assertEqual(c.get(reverse("auth-me")).status_code, 401)


# ---------------------------------------------------------------------------
# Owner-scoped session endpoints
# ---------------------------------------------------------------------------


class SessionIngestTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username="ingestor", password="pw123456")
        self.client = _auth_client(self.user)

    def test_ingest_requires_auth(self) -> None:
        anon = APIClient()
        resp = anon.post(reverse("session-ingest"), data={}, format="json")
        self.assertEqual(resp.status_code, 401)

    def test_ingest_binds_session_to_request_user(self) -> None:
        frames = _synthetic_frames()
        payload = {
            "exercise_type": "squat",
            "started_at": "2026-05-01T12:00:00Z",
            "ended_at": "2026-05-01T12:01:00Z",
            "rep_count": 2,
            "overall_stability_score": 0.91,
            "quality_score": 0.95,
            "progress_trend": {"rom_avg": 0.82},
            "random_seed": 1337,
            "frames": frames,
            "sample_rate_hz": 15,
        }
        resp = self.client.post(reverse("session-ingest"), data=payload, format="json")
        self.assertEqual(resp.status_code, 201, resp.content)

        self.assertEqual(Session.objects.count(), 1)
        session = Session.objects.first()
        assert session is not None
        # CRITICAL: the FK was filled from request.user, not from any
        # client-supplied id (we sent none).
        self.assertEqual(session.user_id, self.user.id)
        self.assertEqual(session.trajectory.frame_count, len(frames))

    def test_ingest_ignores_any_user_field_in_payload(self) -> None:
        attacker = User.objects.create_user(username="evil", password="pw123456")
        payload = {
            "user": attacker.id,                # attempt to spoof
            "patient": attacker.id,             # legacy field — must not leak
            "exercise_type": "squat",
            "started_at": "2026-05-01T12:00:00Z",
            "rep_count": 1,
            "frames": [],
        }
        resp = self.client.post(reverse("session-ingest"), data=payload, format="json")
        self.assertEqual(resp.status_code, 201, resp.content)
        session = Session.objects.first()
        assert session is not None
        self.assertEqual(session.user_id, self.user.id)  # NOT attacker.id


class DataIsolationTests(TestCase):
    """The headline Phase 2.1 guarantee: User A cannot reach User B's data."""

    def setUp(self) -> None:
        self.user_a = User.objects.create_user(username="a_user", password="pw123456")
        self.user_b = User.objects.create_user(username="b_user", password="pw123456")
        self.session_a = Session.objects.create(
            user=self.user_a, exercise_type="squat",
            started_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            rep_count=3, quality_score=0.9, overall_stability_score=0.8,
        )
        self.session_b = Session.objects.create(
            user=self.user_b, exercise_type="lunge",
            started_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
            rep_count=5, quality_score=0.7, overall_stability_score=0.6,
        )

    def test_list_returns_only_callers_sessions(self) -> None:
        c = _auth_client(self.user_a)
        resp = c.get(reverse("session-list"))
        self.assertEqual(resp.status_code, 200)
        ids = {s["id"] for s in resp.data}
        self.assertEqual(ids, {self.session_a.id})

    def test_detail_404s_for_other_users_session(self) -> None:
        c = _auth_client(self.user_a)
        resp = c.get(reverse("session-detail", args=[self.session_b.id]))
        # Hidden as 404 to avoid leaking the existence of the row.
        self.assertEqual(resp.status_code, 404)

    def test_delete_404s_for_other_users_session(self) -> None:
        c = _auth_client(self.user_a)
        resp = c.delete(reverse("session-detail", args=[self.session_b.id]))
        self.assertEqual(resp.status_code, 404)
        # User B's session is still intact.
        self.assertTrue(Session.objects.filter(pk=self.session_b.id).exists())

    def test_anonymous_cannot_list(self) -> None:
        resp = APIClient().get(reverse("session-list"))
        self.assertEqual(resp.status_code, 401)


# ---------------------------------------------------------------------------
# Trend & health
# ---------------------------------------------------------------------------


class TrendEndpointTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username="trendy", password="pw123456")
        self.client = _auth_client(self.user)
        now = datetime.now(tz=timezone.utc)
        for i in range(3):
            Session.objects.create(
                user=self.user,
                exercise_type="squat",
                started_at=now - timedelta(days=i),
                rep_count=5,
                quality_score=0.8 + 0.05 * i,
                overall_stability_score=0.7,
            )

    def test_trend_aggregates_only_caller(self) -> None:
        other = User.objects.create_user(username="not_mine", password="pw123456")
        Session.objects.create(
            user=other, exercise_type="squat",
            started_at=datetime.now(tz=timezone.utc),
            rep_count=999,
        )
        resp = self.client.get(reverse("trend-last-seven"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["sessions"], 3)
        self.assertEqual(resp.data["total_reps"], 15)


class HealthEndpointTests(TestCase):
    def test_health_is_public(self) -> None:
        resp = APIClient().get(reverse("health"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["status"], "ok")
        self.assertTrue(resp.data["edge_computing"])


# ---------------------------------------------------------------------------
# Analyzer reproducibility (unchanged from Phase 1; still required)
# ---------------------------------------------------------------------------


class AnalyzerReproducibilityTests(TestCase):
    def _run(self, seed: int) -> tuple[list[dict], dict]:
        analyzer = PlaceholderAnalyzer(seed=seed, exercise_type="squat")
        frames = _synthetic_frames(90)
        per_frame: list[dict] = []
        for f in frames:
            per_frame.append(analyzer.analyze_frame(AnalyzerFrame(**f)).to_json())
        return per_frame, analyzer.generate_summary().to_json()

    def test_same_seed_produces_identical_results(self) -> None:
        a_frames, a_summary = self._run(seed=1337)
        b_frames, b_summary = self._run(seed=1337)
        self.assertEqual(a_frames, b_frames)
        self.assertEqual(a_summary, b_summary)

    def test_seed_isolation_via_context_manager(self) -> None:
        random.seed(42)
        before = [random.random() for _ in range(3)]
        with deterministic_context(1337):
            apply_global_seed(1337)
            _ = [random.random() for _ in range(5)]
        random.seed(42)
        after = [random.random() for _ in range(3)]
        self.assertEqual(before, after)

    def test_summary_fields_in_expected_range(self) -> None:
        _, summary = self._run(seed=1337)
        self.assertGreaterEqual(summary["overall_stability_score"], 0.0)
        self.assertLessEqual(summary["overall_stability_score"], 1.0)
        self.assertGreaterEqual(summary["quality_score"], 0.0)
        self.assertLessEqual(summary["quality_score"], 1.0)
        self.assertEqual(summary["random_seed"], 1337)


# ---------------------------------------------------------------------------
# seed_demo
# ---------------------------------------------------------------------------


class SeedDemoCommandTests(TestCase):
    def test_seed_demo_creates_user_and_sessions(self) -> None:
        out = StringIO()
        call_command("seed_demo", "--sessions", "2", stdout=out)
        self.assertEqual(User.objects.filter(username="demo").count(), 1)
        self.assertEqual(Session.objects.count(), 2)
        for s in Session.objects.all():
            self.assertTrue(hasattr(s, "trajectory"))
            self.assertGreater(s.trajectory.frame_count, 0)

    def test_seed_demo_is_idempotent_with_clear(self) -> None:
        call_command("seed_demo", "--sessions", "1", stdout=StringIO())
        call_command("seed_demo", "--clear", "--sessions", "1", stdout=StringIO())
        self.assertEqual(User.objects.filter(username="demo").count(), 1)
        self.assertEqual(Session.objects.count(), 1)


# ---------------------------------------------------------------------------
# CTR-GCN analyzer wrapper
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_TORCH, "torch not installed; skipping CTR-GCN tests")
class CTRGCNAnalyzerTests(TestCase):
    """Validate the analyzer's tensor pipeline and determinism guarantees.

    These tests exist because the most common way to break the integration
    is a silent shape mismatch (the model raises an obscure conv error far
    from the buffer-fill site) or a missed RNG seed (which surfaces as
    flaky quality_scores on the same input). Both are caught here.
    """

    WINDOW_SIZE = 16  # Small enough to keep the test snappy on CPU.

    def _make_analyzer(self, seed: int = 1337):
        # Import inside the test to keep the module import path light for
        # the torch-less environments above.
        from analyzer import CTRGCNAnalyzer

        return CTRGCNAnalyzer(
            seed=seed,
            exercise_type="squat",
            window_size=self.WINDOW_SIZE,
        )

    # --- Tensor shape ---------------------------------------------------

    def test_buffer_tensor_shape_matches_contract(self) -> None:
        from analyzer.mediapipe_graph import NUM_NODE

        analyzer = self._make_analyzer()
        shape = analyzer.buffer_tensor_shape()
        self.assertEqual(shape, (1, 3, self.WINDOW_SIZE, NUM_NODE, 1))

    def test_make_tensor_produces_NCTVM_shape(self) -> None:
        import numpy as np

        from analyzer.mediapipe_graph import NUM_NODE

        analyzer = self._make_analyzer()
        # (T, V, C) numpy fixture — what the buffer stacks into pre-reshape.
        arr = np.zeros((self.WINDOW_SIZE, NUM_NODE, 3), dtype=np.float32)
        tensor = analyzer._make_tensor(arr)
        self.assertEqual(
            tuple(tensor.shape),
            (1, 3, self.WINDOW_SIZE, NUM_NODE, 1),
            "CTR-GCN expects (N, C, T, V, M) = (1, 3, T, 33, 1)",
        )

    # --- Determinism ----------------------------------------------------

    def _run_window(self, seed: int) -> tuple[list[dict], list[float]]:
        """Feed exactly window_size frames and return the per-frame results +
        latest logits captured at window-fill."""
        analyzer = self._make_analyzer(seed=seed)
        frames = _synthetic_full_pose_frames(self.WINDOW_SIZE)
        per_frame = []
        for f in frames:
            per_frame.append(analyzer.analyze_frame(AnalyzerFrame(**f)).to_json())
        logits = (
            analyzer._latest_logits.tolist()
            if analyzer._latest_logits is not None
            else []
        )
        return per_frame, logits

    def test_same_seed_yields_identical_logits(self) -> None:
        a_frames, a_logits = self._run_window(seed=1337)
        b_frames, b_logits = self._run_window(seed=1337)
        # Logits captured at the moment the sliding window first fills.
        self.assertEqual(len(a_logits), len(b_logits))
        self.assertNotEqual(a_logits, [], "window should be full by end of test")
        for x, y in zip(a_logits, b_logits):
            self.assertAlmostEqual(x, y, places=5)

    def test_same_seed_yields_identical_quality_scores(self) -> None:
        a_frames, _ = self._run_window(seed=1337)
        b_frames, _ = self._run_window(seed=1337)
        self.assertEqual(
            [r["quality_score"] for r in a_frames],
            [r["quality_score"] for r in b_frames],
        )
        # And every other public field on the result, for good measure.
        self.assertEqual(a_frames, b_frames)

    def test_describe_reports_ctrgcn_identity(self) -> None:
        analyzer = self._make_analyzer()
        meta = analyzer.describe()
        self.assertEqual(meta["analyzer"], "ctrgcn-v1")
        self.assertEqual(meta["random_seed"], 1337)


class AnalyzerRegistryTests(TestCase):
    """The consumer uses ``get_analyzer`` rather than importing a concrete
    class; that indirection is what makes the analyzer swap a one-line
    config flip. Lock that contract in."""

    def test_placeholder_resolvable_without_torch(self) -> None:
        from analyzer import get_analyzer

        a = get_analyzer("placeholder", seed=1337, exercise_type="squat")
        self.assertEqual(a.name, "placeholder-v1")

    def test_unknown_name_raises(self) -> None:
        from analyzer import get_analyzer

        with self.assertRaises(ValueError):
            get_analyzer("not-a-real-model", seed=1337)

    @unittest.skipUnless(_HAS_TORCH, "torch not installed; skipping CTR-GCN registry test")
    def test_ctrgcn_resolvable_when_torch_present(self) -> None:
        from analyzer import get_analyzer

        a = get_analyzer(
            "ctrgcn", seed=1337, exercise_type="squat", window_size=8,
        )
        self.assertEqual(a.name, "ctrgcn-v1")


# ---------------------------------------------------------------------------
# MediaPipe → CTR-GCN graph mapping (no torch required)
# ---------------------------------------------------------------------------


class MediaPipeGraphMappingTests(TestCase):
    """The plan's verification asks specifically that the mapped MediaPipe
    landmarks stay inside the target graph's valid index range. Out-of-
    bounds edges show up as obscure indexing errors deep in the GCN
    forward pass, so we catch them here at the static-graph layer."""

    def test_all_edges_reference_valid_node_indices(self) -> None:
        from analyzer.mediapipe_graph import NUM_NODE, Graph

        g = Graph()
        for src, dst in g.inward + g.outward + g.self_link:
            self.assertGreaterEqual(src, 0)
            self.assertGreaterEqual(dst, 0)
            self.assertLess(src, NUM_NODE)
            self.assertLess(dst, NUM_NODE)

    def test_adjacency_matrix_shape(self) -> None:
        from analyzer.mediapipe_graph import NUM_NODE, Graph

        g = Graph()
        # CTR-GCN expects A to be (3, V, V) for spatial labeling: identity,
        # inward, outward.
        self.assertEqual(g.A.shape, (3, NUM_NODE, NUM_NODE))


# ---------------------------------------------------------------------------
# Normalization (pure numpy, no torch needed)
# ---------------------------------------------------------------------------


class NormalizationTests(TestCase):
    """Identical poses captured from different angles / distances should
    collapse to the same normalised representation. That's the whole
    point of the hip-centric + spine-scaled transform."""

    def _make_pose(self, *, shift=(0.0, 0.0, 0.0), scale=1.0) -> "np.ndarray":
        import numpy as np

        from analyzer.mediapipe_graph import LANDMARK_NAMES, NUM_NODE

        # Synthetic standing pose: hips at y≈0.6, shoulders at y≈0.4,
        # everything else hung off those. Coordinates start canonical
        # then get translated + scaled to mimic camera moves.
        pose = np.zeros((NUM_NODE, 3), dtype=np.float32)
        # Plant the hips and shoulders.
        pose[LANDMARK_NAMES.index("left_hip")]      = [0.45, 0.60, 0.0]
        pose[LANDMARK_NAMES.index("right_hip")]     = [0.55, 0.60, 0.0]
        pose[LANDMARK_NAMES.index("left_shoulder")] = [0.42, 0.40, 0.0]
        pose[LANDMARK_NAMES.index("right_shoulder")]= [0.58, 0.40, 0.0]
        # A few extremities to round out the test.
        pose[LANDMARK_NAMES.index("left_knee")]     = [0.46, 0.80, 0.0]
        pose[LANDMARK_NAMES.index("right_knee")]    = [0.54, 0.80, 0.0]
        pose[LANDMARK_NAMES.index("left_wrist")]    = [0.30, 0.55, 0.0]
        pose[LANDMARK_NAMES.index("right_wrist")]   = [0.70, 0.55, 0.0]
        pose = pose * scale + np.array(shift, dtype=np.float32)
        return pose

    def test_normalization_is_translation_invariant(self) -> None:
        import numpy as np

        from analyzer.normalization import normalize_pose

        a = normalize_pose(self._make_pose())
        b = normalize_pose(self._make_pose(shift=(0.2, -0.1, 0.05)))
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_normalization_is_scale_invariant(self) -> None:
        import numpy as np

        from analyzer.normalization import normalize_pose

        a = normalize_pose(self._make_pose(scale=1.0))
        b = normalize_pose(self._make_pose(scale=2.0))
        # Hip→shoulder distance is the unit, so after rescale the two
        # canvases collapse to the same shape regardless of pixel scale.
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_normalization_places_hip_midpoint_at_origin(self) -> None:
        import numpy as np

        from analyzer.mediapipe_graph import LANDMARK_NAMES
        from analyzer.normalization import normalize_pose

        normed = normalize_pose(self._make_pose())
        lh = normed[LANDMARK_NAMES.index("left_hip")]
        rh = normed[LANDMARK_NAMES.index("right_hip")]
        midpoint = 0.5 * (lh + rh)
        np.testing.assert_allclose(midpoint, np.zeros(3), atol=1e-6)


# ---------------------------------------------------------------------------
# Kinematic metrics (ROM + tremor)
# ---------------------------------------------------------------------------


class KinematicMetricsTests(TestCase):
    def test_tremor_zero_on_perfectly_smooth_sine(self) -> None:
        import math

        import numpy as np

        from analyzer.kinematics import tremor_metrics

        # Densely-sampled sine → small first/second differences.
        sig = np.array([math.sin(i / 30.0) for i in range(300)])
        m = tremor_metrics(sig)
        # Plan asks for "~0"; in practice the discrete approximation
        # gives ~3e-2 / ~1e-3 for these parameters — both safely small.
        self.assertLess(m["velocity_rms"], 0.1)
        self.assertLess(m["acceleration_rms"], 0.01)

    def test_tremor_large_on_random_noise(self) -> None:
        import numpy as np

        from analyzer.kinematics import tremor_metrics

        rng = np.random.default_rng(1337)
        sig = rng.normal(size=300)
        m = tremor_metrics(sig)
        # Pure noise: velocity RMS should be on the order of the signal
        # itself. Use a generous lower bound so we're not testing rng
        # internals, just that the function reacts.
        self.assertGreater(m["velocity_rms"], 0.5)

    def test_range_of_motion_captures_peak_angle(self) -> None:
        import math

        import numpy as np

        from analyzer.kinematics import joint_angle_series, range_of_motion
        from analyzer.mediapipe_graph import LANDMARK_NAMES, NUM_NODE

        # Build a (T, V, C) window where the knee opens and closes — i.e.
        # the angle at "left_knee" oscillates roughly between ~60° and
        # ~180°. Hip is anchored above the knee, ankle swings.
        T = 60
        window = np.zeros((T, NUM_NODE, 3), dtype=np.float32)
        ih = LANDMARK_NAMES.index("left_hip")
        ik = LANDMARK_NAMES.index("left_knee")
        ia = LANDMARK_NAMES.index("left_ankle")
        for t in range(T):
            theta = math.pi * 0.5 + 0.4 * math.sin(2 * math.pi * t / T)
            window[t, ih] = [0.0, -1.0, 0.0]  # hip above knee
            window[t, ik] = [0.0, 0.0, 0.0]   # knee at origin
            # Ankle rotates around the knee in the x-y plane.
            window[t, ia] = [math.sin(theta), math.cos(theta), 0.0]
        angles = joint_angle_series(
            window, LANDMARK_NAMES, ("left_hip", "left_knee", "left_ankle"),
        )
        rom = range_of_motion(angles)
        # Should swing through ~46° (2 * 0.4 rad ≈ 45.8°). Generous
        # bounds so we're not testing trig identity precision.
        self.assertGreater(rom["range_deg"], 30.0)
        self.assertLess(rom["range_deg"], 60.0)


# ---------------------------------------------------------------------------
# Cosine similarity (no torch required)
# ---------------------------------------------------------------------------


class SimilarityTests(TestCase):
    def test_identical_vectors_score_100(self) -> None:
        import numpy as np

        from analyzer.similarity import similarity_score

        v = np.array([1.0, 2.0, 3.0])
        self.assertEqual(similarity_score(v, v), 100.0)

    def test_orthogonal_vectors_score_0(self) -> None:
        import numpy as np

        from analyzer.similarity import similarity_score

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        self.assertEqual(similarity_score(a, b), 0.0)

    def test_no_reference_returns_none(self) -> None:
        import numpy as np

        from analyzer.similarity import similarity_score

        self.assertIsNone(similarity_score(np.array([1.0, 2.0]), None))


# ---------------------------------------------------------------------------
# CTR-GCN feature extraction (return_features=True)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_TORCH, "torch not installed; skipping feature tests")
class CTRGCNFeatureExtractionTests(TestCase):
    """Plan's Phase 2 Task 5 + verification: `return_features=True` must
    yield a dense pre-FC vector, not the logits."""

    def test_return_features_yields_dense_pre_fc_vector(self) -> None:
        import torch

        from analyzer import CTRGCNAnalyzer

        analyzer = CTRGCNAnalyzer(seed=1337, window_size=8)
        x = torch.zeros(*analyzer.buffer_tensor_shape(), dtype=torch.float32)
        with torch.no_grad():
            features, logits = analyzer._model(x, return_features=True)
        # CTR-GCN uses base_channel*4 = 256 as the pre-FC dimension.
        self.assertEqual(features.shape, (1, 256))
        # And the head still produces num_class logits.
        self.assertEqual(logits.shape, (1, analyzer.num_class))

    def test_extract_features_returns_none_before_window_fills(self) -> None:
        from analyzer import CTRGCNAnalyzer

        analyzer = CTRGCNAnalyzer(seed=1337, window_size=8, inference_stride=1)
        self.assertIsNone(analyzer.extract_features())


# ---------------------------------------------------------------------------
# CTR-GCN stride trigger + similarity integration
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_TORCH, "torch not installed; skipping CTR-GCN stride tests")
class CTRGCNStrideAndSimilarityTests(TestCase):
    WINDOW = 8
    STRIDE = 4

    def _drive(self, n_frames: int):
        from analyzer import CTRGCNAnalyzer

        analyzer = CTRGCNAnalyzer(
            seed=1337,
            window_size=self.WINDOW,
            inference_stride=self.STRIDE,
        )
        frames = _synthetic_full_pose_frames(n_frames)
        results = [analyzer.analyze_frame(AnalyzerFrame(**f)) for f in frames]
        return analyzer, results

    def test_inference_runs_only_on_stride(self) -> None:
        analyzer, _ = self._drive(self.WINDOW + 4 * self.STRIDE)
        # First inference at window-fill, then one per `stride` frames.
        # With 4 * stride extra frames after fill we expect 5 total calls
        # (one at fill, four on subsequent strides).
        self.assertEqual(analyzer._inference_calls, 5)

    def test_set_reference_feature_produces_score(self) -> None:
        import numpy as np

        analyzer, _ = self._drive(self.WINDOW)
        # Use the analyzer's own latest features as the reference: it
        # must then self-compare to a score of 100.
        ref = analyzer.extract_features()
        self.assertIsNotNone(ref)
        analyzer.set_reference_feature(ref)
        # Drive one more stride window to trigger another inference,
        # which will re-evaluate similarity against the reference.
        for f in _synthetic_full_pose_frames(self.STRIDE):
            analyzer.analyze_frame(AnalyzerFrame(**f))
        self.assertIsNotNone(analyzer._latest_similarity)
        # Inputs are identical, so similarity ≈ 100.
        self.assertGreater(analyzer._latest_similarity, 99.0)


# ---------------------------------------------------------------------------
# Post-workout synthesis (no torch needed — pure numpy)
# ---------------------------------------------------------------------------


class SynthesisTests(TestCase):
    """Phase 4 Task 8: the four headline metrics + chart series the
    Summary Box renders. These tests run without torch by feeding the
    synthesis layer a hand-built log."""

    def _oscillating_log(
        self, *, reps: int, samples_per_rep: int, joint: str = "left_knee",
        amplitude_deg: float = 40.0, center_deg: float = 110.0,
        with_similarity: bool = True,
    ) -> list[dict]:
        import math

        log: list[dict] = []
        idx = 0
        for r in range(reps):
            for s in range(samples_per_rep):
                # Start high, dip to center - amplitude, return to high.
                phase = 2 * math.pi * s / samples_per_rep
                angle = center_deg - amplitude_deg * (1 - math.cos(phase)) / 2
                log.append({
                    "frame_index": idx,
                    "t_ms": idx * (1000.0 / 30.0),
                    "similarity": 85.0 + 0.5 * r if with_similarity else None,
                    "angles": {joint: angle},
                })
                idx += 1
        return log

    def test_synthesis_empty_log_returns_empty_shell(self) -> None:
        from analyzer.synthesis import synthesize

        out = synthesize([])
        self.assertEqual(out["rep_count_by_angle"], 0)
        self.assertEqual(out["per_rep_rom"], [])
        self.assertIsNone(out["fatigue_index"])
        self.assertIsNone(out["primary_joint"])
        self.assertEqual(out["charts"]["rom_curve"], [])

    def test_rep_count_matches_oscillation_count(self) -> None:
        from analyzer.synthesis import synthesize

        log = self._oscillating_log(reps=5, samples_per_rep=30)
        out = synthesize(log)
        # Trough-based detector: 5 oscillations should produce 5 troughs.
        # Allow ±1 for boundary effects (the last trough's recovery
        # half-cycle may be incomplete).
        self.assertGreaterEqual(out["rep_count_by_angle"], 4)
        self.assertLessEqual(out["rep_count_by_angle"], 5)

    def test_per_rep_rom_captures_peak_amplitude(self) -> None:
        from analyzer.synthesis import synthesize

        log = self._oscillating_log(
            reps=4, samples_per_rep=30,
            amplitude_deg=50.0, center_deg=120.0,
        )
        out = synthesize(log)
        self.assertGreater(len(out["per_rep_rom"]), 0)
        # Every rep slice should swing through close to the configured
        # amplitude (50°). Generous lower bound accounts for the fact
        # that the first/last slice may be a half-cycle.
        ranges = [r["range_deg"] for r in out["per_rep_rom"]]
        self.assertGreater(max(ranges), 30.0)
        self.assertLess(max(ranges), 70.0)

    def test_overall_accuracy_is_mean_similarity_when_set(self) -> None:
        from analyzer.synthesis import synthesize

        log = self._oscillating_log(reps=3, samples_per_rep=20, with_similarity=True)
        out = synthesize(log)
        sims = [e["similarity"] for e in log]
        self.assertAlmostEqual(out["overall_accuracy"], sum(sims) / len(sims), places=2)

    def test_overall_accuracy_falls_back_to_quality_when_no_similarity(self) -> None:
        from analyzer.synthesis import synthesize

        log = self._oscillating_log(reps=2, samples_per_rep=20, with_similarity=False)
        # No similarity entries → fall back to mean(quality) × 100.
        out = synthesize(log, quality_samples=[0.9, 0.8, 0.85])
        # Mean(0.9, 0.8, 0.85) × 100 = 85.0
        self.assertAlmostEqual(out["overall_accuracy"], 85.0, places=2)

    def test_fatigue_index_higher_when_late_reps_jitter(self) -> None:
        import math

        from analyzer.synthesis import synthesize

        # Build a synthetic log where the first half is smooth and the
        # second half has accelerating jitter — late-session fatigue.
        log = []
        for i in range(120):
            phase = 2 * math.pi * (i % 30) / 30
            angle = 110.0 - 30.0 * (1 - math.cos(phase)) / 2
            if i >= 60:
                # Add high-frequency noise in the second half.
                angle += 8.0 * math.sin(i * 2.7)
            log.append({
                "frame_index": i,
                "t_ms": i * (1000.0 / 30.0),
                "similarity": None,
                "angles": {"left_knee": angle},
            })
        out = synthesize(log)
        self.assertIsNotNone(out["fatigue_index"])
        # Variance of per-rep tremor must be measurably > 0 since reps
        # in the second half jitter way more than reps in the first.
        self.assertGreater(out["fatigue_index"], 0.0)

    def test_rom_curve_subsampled_for_large_logs(self) -> None:
        from analyzer.synthesis import synthesize

        # 5000-frame log → must be sub-sampled to <= 400 points for the
        # canvas chart (otherwise we'd ship megabytes of points per
        # session).
        log = self._oscillating_log(reps=20, samples_per_rep=250)
        out = synthesize(log)
        self.assertLessEqual(len(out["charts"]["rom_curve"]), 400)

    def test_stability_trend_has_one_entry_per_rep(self) -> None:
        from analyzer.synthesis import synthesize

        log = self._oscillating_log(reps=5, samples_per_rep=30)
        out = synthesize(log)
        self.assertEqual(
            len(out["charts"]["stability_trend"]),
            len(out["per_rep_rom"]),
        )


# ---------------------------------------------------------------------------
# Analyzer-level integration: session log grows + summary carries summary_box
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_TORCH, "torch not installed; skipping log/summary tests")
class CTRGCNSessionLogTests(TestCase):
    """End-to-end: drive a CTR-GCN analyzer with synthetic frames and
    confirm the session log grows + the summary surfaces a summary_box
    payload the React component can consume."""

    def _drive(self, n_frames: int):
        from analyzer import CTRGCNAnalyzer

        a = CTRGCNAnalyzer(seed=1337, window_size=8, inference_stride=2)
        for f in _synthetic_full_pose_frames(n_frames):
            a.analyze_frame(AnalyzerFrame(**f))
        return a

    def test_session_log_grows_monotonically(self) -> None:
        a = self._drive(16)
        self.assertEqual(len(a._session_log), 16)
        # And every entry has the expected schema keys.
        for entry in a._session_log:
            self.assertIn("frame_index", entry)
            self.assertIn("t_ms", entry)
            self.assertIn("angles", entry)
            # angles dict is non-empty (the synthetic frames populate every
            # landmark, so every clinical joint angle is computable).
            self.assertGreater(len(entry["angles"]), 0)

    def test_summary_carries_summary_box_payload(self) -> None:
        a = self._drive(32)
        summary = a.generate_summary().to_json()
        self.assertIn("summary_box", summary["progress_trend"])
        sb = summary["progress_trend"]["summary_box"]
        # Schema contract checked by the SummaryBox component.
        for key in ("overall_accuracy", "rep_count_by_angle", "per_rep_rom",
                    "fatigue_index", "primary_joint", "charts"):
            self.assertIn(key, sb)
        self.assertIn("rom_curve", sb["charts"])
        self.assertIn("stability_trend", sb["charts"])


@unittest.skipUnless(_HAS_TORCH, "torch not installed; skipping CTR-GCN export tests")
class CTRGCNExportCompatibilityTests(TestCase):
    """Guard the ONNX/Core ML export path.

    ``CTRGC.forward`` originally contracted its adaptive topology with
    ``torch.einsum('ncuv,nctv->nctu', x1, x3)``. Several ONNX opsets cannot
    represent that op, and the Core ML converter inherits the limitation, so
    it was rewritten as ``matmul`` + ``permute``.

    These tests exist because the rewrite is the kind of change that is
    silently wrong: a transposed contraction still produces a correctly
    shaped tensor and a plausible-looking logit, and nothing downstream
    would complain. Pinning the equivalence numerically is the only way to
    notice.
    """

    def test_matmul_form_equals_einsum_form(self) -> None:
        import torch

        torch.manual_seed(1337)
        N, C, T, U, V = 2, 4, 6, 5, 5   # U == V for the square adjacency
        x1 = torch.randn(N, C, U, V)
        x3 = torch.randn(N, C, T, V)

        reference = torch.einsum("ncuv,nctv->nctu", x1, x3)
        rewritten = torch.matmul(x1, x3.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        self.assertEqual(tuple(rewritten.shape), tuple(reference.shape))
        self.assertLess(
            float((reference - rewritten).abs().max()),
            1e-5,
            "matmul rewrite must be numerically identical to the einsum it replaced",
        )

    def test_ctrgc_forward_contains_no_einsum(self) -> None:
        """A regression tripwire for anyone re-introducing ``einsum``.

        Inspects the AST rather than the raw source: the method carries a
        comment explaining which einsum the matmul replaced, and a plain
        substring search would trip over its own documentation.
        """
        import ast
        import inspect
        import textwrap

        from ctrgcn.ctrgcn import CTRGC

        tree = ast.parse(textwrap.dedent(inspect.getsource(CTRGC.forward)))
        called = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                called.add(func.attr)
            elif isinstance(func, ast.Name):
                called.add(func.id)

        self.assertNotIn(
            "einsum",
            called,
            "einsum in CTRGC.forward breaks ONNX/Core ML export — "
            "see scripts/export_coreml.py",
        )
        self.assertIn("matmul", called, "the matmul rewrite has gone missing")

    def test_model_traces_for_export(self) -> None:
        """The exact trace ``scripts/export_coreml.py`` performs.

        Tracing is where an unsupported op actually surfaces, so this is the
        cheapest possible early warning that the export has broken.
        """
        import torch

        from analyzer.mediapipe_graph import NUM_NODE
        from analyzer.seed import apply_global_seed
        from ctrgcn.ctrgcn import Model

        apply_global_seed(1337)
        model = Model(
            num_class=2,
            num_point=NUM_NODE,
            num_person=1,
            graph="analyzer.mediapipe_graph.Graph",
            graph_args={"labeling_mode": "spatial"},
            in_channels=3,
            drop_out=0.0,
            adaptive=True,
        )
        model.eval()

        # A short window keeps the trace fast; the op set is identical at 64.
        dummy = torch.zeros(1, 3, 16, NUM_NODE, 1, dtype=torch.float32)
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy, strict=False)
            traced_out = traced(dummy)
            eager_out = model(dummy)

        self.assertEqual(tuple(traced_out.shape), (1, 2))
        self.assertLess(float((traced_out - eager_out).abs().max()), 1e-5)


class CenteringTests(TestCase):
    """Framing assistant. Port of Final/centering_logic.py.

    Framing failures are the quietest kind of bad data: a patient half out
    of frame still produces landmarks, still fills the sliding window, and
    still yields a quality score. These tests exist because nothing
    downstream can tell the difference.

    No torch needed — this module is pure arithmetic.
    """

    @staticmethod
    def _pose(hip_x=0.5, shoulder_x=None, nose_y=0.10, hip_y=0.60,
              knee_y=0.80, nose_vis=0.9, drop=()):
        sx = hip_x if shoulder_x is None else shoulder_x
        w = 0.09
        pose = {
            "nose": [hip_x, nose_y, 0.0, nose_vis],
            "left_shoulder": [sx - w, 0.32, 0.0, 0.9],
            "right_shoulder": [sx + w, 0.32, 0.0, 0.9],
            "left_hip": [hip_x - w, hip_y, 0.0, 0.9],
            "right_hip": [hip_x + w, hip_y, 0.0, 0.9],
            "left_knee": [hip_x - w, knee_y, 0.0, 0.9],
            "right_knee": [hip_x + w, knee_y, 0.0, 0.9],
        }
        for name in drop:
            pose.pop(name, None)
        return pose

    def test_centered_pose_passes_every_check(self) -> None:
        from analyzer.centering import evaluate_centering

        r = evaluate_centering(self._pose())
        self.assertTrue(r.is_centered)
        self.assertEqual(r.status, "Patient is CENTERED")
        self.assertEqual(r.status_code, "centered")
        self.assertEqual(r.severity, "ok")

    def test_horizontal_bounds_are_inclusive(self) -> None:
        """0.30 and 0.70 must not themselves trigger a warning."""
        from analyzer.centering import evaluate_centering

        for x in (0.30, 0.70):
            self.assertTrue(evaluate_centering(self._pose(hip_x=x)).is_centered,
                            f"hip x={x} sits on the boundary and is allowed")
        self.assertEqual(
            evaluate_centering(self._pose(hip_x=0.2999)).status,
            "Patient too far LEFT")
        self.assertEqual(
            evaluate_centering(self._pose(hip_x=0.7001)).status,
            "Patient too far RIGHT")

    def test_correction_direction_opposes_displacement(self) -> None:
        from analyzer.centering import evaluate_centering

        self.assertEqual(evaluate_centering(self._pose(hip_x=0.1)).status_code,
                         "move_right")
        self.assertEqual(evaluate_centering(self._pose(hip_x=0.9)).status_code,
                         "move_left")

    def test_torso_ratio_is_nose_to_hip(self) -> None:
        """Regression guard for the easiest way to get this wrong.

        Measuring shoulder-to-hip instead of nose-to-hip roughly halves
        the ratio, which would push well-framed patients under the 0.12
        'too far' threshold while raising no error anywhere.
        """
        from analyzer.centering import evaluate_centering

        r = evaluate_centering(self._pose(nose_y=0.10, hip_y=0.60))
        self.assertAlmostEqual(r.torso_height_ratio, 0.50, places=4)

    def test_distance_thresholds(self) -> None:
        from analyzer.centering import evaluate_centering

        far = evaluate_centering(self._pose(nose_y=0.50, hip_y=0.58))
        self.assertEqual(far.status, "Patient is TOO FAR from camera")
        close = evaluate_centering(self._pose(nose_y=0.05, hip_y=0.80))
        self.assertEqual(close.status, "Patient is TOO CLOSE to camera")
        # Both boundaries are inclusive.
        self.assertTrue(
            evaluate_centering(self._pose(nose_y=0.48, hip_y=0.60)).is_centered)
        self.assertTrue(
            evaluate_centering(self._pose(nose_y=0.05, hip_y=0.75,
                                          knee_y=0.90)).is_centered)

    def test_head_visibility_gate_is_point_three(self) -> None:
        """Not 0.5 — the head check is coarser than occlusion carry-forward."""
        from analyzer.centering import evaluate_centering
        from analyzer.normalization import _OCCLUSION_THRESHOLD

        self.assertNotEqual(0.3, _OCCLUSION_THRESHOLD)
        self.assertTrue(evaluate_centering(self._pose(nose_vis=0.3)).is_centered)
        self.assertEqual(
            evaluate_centering(self._pose(nose_vis=0.29)).status,
            "Patient HEAD may be cut off")

    def test_clipping_checks(self) -> None:
        from analyzer.centering import evaluate_centering

        self.assertEqual(evaluate_centering(self._pose(nose_y=0.01)).status,
                         "Patient HEAD may be cut off")
        self.assertTrue(evaluate_centering(self._pose(nose_y=0.03)).is_centered)
        self.assertEqual(evaluate_centering(self._pose(knee_y=0.99)).status,
                         "Patient FEET may be cut off")
        self.assertTrue(evaluate_centering(self._pose(knee_y=0.97)).is_centered)

    def test_hip_issue_wins_the_headline_but_all_are_listed(self) -> None:
        from analyzer.centering import evaluate_centering

        r = evaluate_centering(
            self._pose(hip_x=0.05, nose_y=0.50, hip_y=0.58, knee_y=0.99))
        self.assertEqual(r.status, "Patient too far LEFT")
        self.assertIn("Patient FEET may be cut off", r.details)
        self.assertIn("Patient is TOO FAR from camera", r.details)
        self.assertTrue(r.details[-1].startswith("Hip center:"))

    def test_missing_landmarks_report_not_detected(self) -> None:
        from analyzer.centering import evaluate_centering

        self.assertEqual(evaluate_centering({}).status_code, "not_detected")
        for name in ("nose", "left_hip", "right_knee", "left_shoulder"):
            r = evaluate_centering(self._pose(drop=(name,)))
            self.assertEqual(r.status_code, "not_detected", f"missing {name}")
            self.assertEqual(r.status, "Patient has left the field of view")
            self.assertEqual(r.details, ["Pose not recognised."])
            self.assertIsNone(r.hip_center_x)

    def test_not_centered_never_reports_ok_severity(self) -> None:
        """The banner colour must never contradict the banner text."""
        from analyzer.centering import evaluate_centering

        for pose in (self._pose(hip_x=0.1), self._pose(knee_y=0.99),
                     self._pose(nose_y=0.0), self._pose(nose_y=0.5, hip_y=0.55),
                     self._pose(shoulder_x=0.05), {}):
            r = evaluate_centering(pose)
            self.assertFalse(r.is_centered)
            self.assertNotEqual(r.severity, "ok")
            self.assertNotEqual(r.status_code, "centered")
