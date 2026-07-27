// Live Monitor — Phase 2.1.
//
// Changes from Phase 1:
//   - No patient selector (sessions are owned by the authenticated user).
//   - Exercise is taken from `?exercise=` (set by the dashboard cards).
//   - Front-facing camera with mirror effect (CSS scaleX(-1)).
//   - Tutorial drawer surfaces cues for the selected exercise during
//     CALIBRATING and is collapsible during ACTIVE.
//   - Ingest payload drops `patient` — backend binds to request.user.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import { api } from '../api/client.js';
import { MonitorSocket } from '../api/monitorSocket.js';
import { useAuth } from '../auth/AuthContext.jsx';
import ControlBar from '../components/ControlBar.jsx';
import FeedbackList from '../components/FeedbackList.jsx';
import HUD from '../components/HUD.jsx';
import JointGauges from '../components/JointGauges.jsx';
import SkeletonOverlay from '../components/SkeletonOverlay.jsx';
import StabilityChart from '../components/StabilityChart.jsx';
import { usePoseDetection } from '../hooks/usePoseDetection.js';
import { STATES, useSessionMachine } from '../hooks/useSessionMachine.js';
import { EXERCISES, exerciseByKey } from '../lib/exercises.js';

const PERSIST_FPS = 15;
const PERSIST_INTERVAL_MS = 1000 / PERSIST_FPS;

export default function LiveMonitor() {
  const { token } = useAuth();
  const navigate = useNavigate();
  const [search, setSearch] = useSearchParams();
  const initialExercise = search.get('exercise') || 'squat';
  const [exerciseType, setExerciseType] = useState(initialExercise);
  const exercise = useMemo(() => exerciseByKey(exerciseType) || EXERCISES[0], [exerciseType]);

  const { state, transition } = useSessionMachine();
  const { videoRef, latest, error: camError } = usePoseDetection({
    enabled: state !== STATES.IDLE && state !== STATES.COMPLETED,
  });

  const [result, setResult] = useState({ count: 0, quality_score: 1.0, is_compensatory: false });
  const [feedbackItems, setFeedbackItems] = useState([]);
  const [qualitySamples, setQualitySamples] = useState([]);
  const [wsStatus, setWsStatus] = useState('closed');
  const [ready, setReady] = useState(false);
  const [seed] = useState(() => 1337);
  const [tutorialOpen, setTutorialOpen] = useState(true);
  const lastSummaryRef = useRef(null);

  const persistBufferRef = useRef([]);
  const lastPersistTsRef = useRef(0);
  const startedAtRef = useRef(null);
  const socketRef = useRef(null);

  // Sync the URL whenever the exercise changes — keeps reloads sane.
  useEffect(() => {
    if ((search.get('exercise') || '') !== exerciseType) {
      const next = new URLSearchParams(search);
      next.set('exercise', exerciseType);
      setSearch(next, { replace: true });
    }
  }, [exerciseType, search, setSearch]);

  // ---- WebSocket lifecycle -----------------------------------------------
  const openSocket = useCallback(() => {
    if (socketRef.current) return;
    if (!token) return; // Should never happen — route is guarded.
    const sock = new MonitorSocket({
      token,
      onStatus: (s) => {
        setWsStatus(s);
        if (s === 'unauthorized') {
          alert('Session expired. Please sign in again.');
          navigate('/login', { replace: true });
        }
      },
      onReady: () => setReady(true),
      onResult: (msg) => {
        setResult(msg);
        setQualitySamples((prev) => {
          const next = prev.concat([msg.quality_score]);
          return next.length > 900 ? next.slice(-900) : next;
        });
        if (msg.feedback?.length) {
          setFeedbackItems((prev) => prev.concat(msg.feedback).slice(-50));
        }
      },
      onSummary: (msg) => { lastSummaryRef.current = msg; },
      onError: (e) => console.error('[ws]', e),
    });
    sock.connect();
    sock.helloOnceOpen({ sessionSeed: seed, exerciseType });
    socketRef.current = sock;
  }, [token, navigate, seed, exerciseType]);

  const closeSocket = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    setReady(false);
  }, []);

  // ---- Frame pump --------------------------------------------------------
  useEffect(() => {
    if (!latest) return;
    if (state !== STATES.ACTIVE) return;

    if (socketRef.current?.isOpen()) {
      socketRef.current.sendFrame({
        frame_index: latest.frameIndex,
        timestamp_ms: latest.timestampMs,
        points: latest.points,
        angles: latest.angles,
      });
    }
    if (latest.timestampMs - lastPersistTsRef.current >= PERSIST_INTERVAL_MS) {
      lastPersistTsRef.current = latest.timestampMs;
      persistBufferRef.current.push({
        frame: latest.frameIndex,
        t: Math.round(latest.timestampMs),
        points: latest.points,
        angles: latest.angles,
      });
    }
  }, [latest, state]);

  // ---- State transitions -------------------------------------------------
  const handleStart = useCallback(() => {
    if (state === STATES.IDLE) {
      transition(STATES.CALIBRATING);
      openSocket();
      return;
    }
    if (state === STATES.CALIBRATING) {
      persistBufferRef.current = [];
      lastPersistTsRef.current = 0;
      startedAtRef.current = new Date().toISOString();
      setFeedbackItems([]);
      setQualitySamples([]);
      setResult({ count: 0, quality_score: 1.0, is_compensatory: false });
      lastSummaryRef.current = null;
      setTutorialOpen(false);
      transition(STATES.ACTIVE);
    }
  }, [state, transition, openSocket]);

  const handlePause = useCallback(() => transition(STATES.PAUSED), [transition]);
  const handleResume = useCallback(() => transition(STATES.ACTIVE), [transition]);
  const handleRecalibrate = useCallback(() => {
    if (state === STATES.ACTIVE) transition(STATES.PAUSED);
    if (socketRef.current) {
      socketRef.current.helloOnceOpen({ sessionSeed: seed, exerciseType });
    }
    persistBufferRef.current = [];
    lastPersistTsRef.current = 0;
    setFeedbackItems([]);
    setQualitySamples([]);
    setResult({ count: 0, quality_score: 1.0, is_compensatory: false });
  }, [state, transition, seed, exerciseType]);

  const handleFinish = useCallback(async () => {
    transition(STATES.COMPLETED);
    if (socketRef.current) socketRef.current.bye();
    await new Promise((r) => setTimeout(r, 250));
    const summary = lastSummaryRef.current || {
      rep_count: result.count,
      overall_stability_score: result.quality_score,
      quality_score: result.quality_score,
      progress_trend: {},
    };
    const progressTrend = {
      ...(summary.progress_trend || {}),
      compensation_events: summary.compensation_events ?? 0,
    };
    // No `patient` field — backend binds to request.user.
    const payload = {
      exercise_type: exerciseType,
      started_at: startedAtRef.current || new Date().toISOString(),
      ended_at: new Date().toISOString(),
      rep_count: summary.rep_count,
      overall_stability_score: summary.overall_stability_score,
      quality_score: summary.quality_score,
      progress_trend: progressTrend,
      random_seed: seed,
      notes: '',
      frames: persistBufferRef.current,
      sample_rate_hz: PERSIST_FPS,
    };
    closeSocket();
    try {
      const created = await api.ingestSession(payload);
      navigate(`/sessions/${created.id}`);
    } catch (e) {
      console.error(e);
      alert(`Failed to upload session: ${e.message}`);
      transition(STATES.IDLE);
    }
  }, [exerciseType, transition, result, seed, navigate, closeSocket]);

  useEffect(() => () => closeSocket(), [closeSocket]);

  // ---- Render ------------------------------------------------------------
  const exerciseSelector = (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="row-2">
        <div>
          <label>Exercise</label>
          <select value={exerciseType}
                  onChange={(e) => setExerciseType(e.target.value)}
                  disabled={state !== STATES.IDLE && state !== STATES.CALIBRATING}>
            {EXERCISES.map((x) => (
              <option key={x.key} value={x.key}>
                {x.icon} {x.name} — target {x.targetReps} reps
              </option>
            ))}
          </select>
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button className="ghost" onClick={() => setTutorialOpen((v) => !v)}>
            {tutorialOpen ? 'Hide tutorial' : 'Show tutorial'}
          </button>
        </div>
      </div>
      {tutorialOpen && (
        <div className="tutorial-drawer">
          <div className="card-title">{exercise.icon} {exercise.name}</div>
          <div className="muted" style={{ marginBottom: 8 }}>{exercise.summary}</div>
          <ol className="cue-list">
            {exercise.cues.map((c, i) => <li key={i}>{c}</li>)}
          </ol>
          <div className="contra">
            <span className="tag warn">Heads up</span> {exercise.contraindications}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div>
      {exerciseSelector}
      <div className="monitor">
        <div className="monitor-body">
          <div className="viewport">
            {/* Mirror the live view — patients are looking AT themselves. */}
            <video ref={videoRef} autoPlay playsInline muted className="mirrored" />
            <div className="overlay-mirror">
              <SkeletonOverlay
                points={latest?.points}
                qualityScore={result.quality_score}
                isCompensatory={result.is_compensatory}
              />
            </div>
            <HUD count={result.count} />
            <div className={`state-chip ${state}`}>
              {state}{state === STATES.CALIBRATING ? ' — get fully in frame' : ''}
            </div>
            {camError && (
              <div style={{
                position: 'absolute', inset: 'auto 16px 16px 16px', zIndex: 5,
                background: 'rgba(239,111,108,0.18)', border: '1px solid #ef6f6c',
                padding: '10px 12px', borderRadius: 10, color: '#fff',
              }}>
                Camera/MediaPipe error: {String(camError.message || camError)}
              </div>
            )}
          </div>
          <div className="sidebar">
            <JointGauges angles={latest?.angles || {}} exercise={exerciseType} />
            <FeedbackList items={feedbackItems} />
            <StabilityChart samples={qualitySamples} />
          </div>
        </div>
        <ControlBar
          state={state}
          ready={ready}
          wsStatus={wsStatus}
          seed={seed}
          onStart={handleStart}
          onPause={handlePause}
          onResume={handleResume}
          onRecalibrate={handleRecalibrate}
          onFinish={handleFinish}
        />
      </div>
    </div>
  );
}
