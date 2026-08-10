// Live Monitor — Phase 2.3.
//
// Changes from Phase 2.1:
//   - Added local video file upload mode alongside the existing webcam mode.
//   - Segmented control toggles between "Webcam" and "Video File" sources.
//   - Drag-and-drop or click-to-browse file upload zone for video files.
//   - Custom video player controls: play/pause, timeline seeker, speed,
//     mirror toggle, loop toggle.
//   - Session state transitions sync with video playback (play on ACTIVE,
//     pause on PAUSED, reset on recalibrate).
//   - Mirror is ON by default for webcam, OFF for uploaded videos.

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

// ---- Helpers ----------------------------------------------------------------

function formatTime(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const SPEED_OPTIONS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

// ---- Component --------------------------------------------------------------

export default function LiveMonitor() {
  const { token } = useAuth();
  const navigate = useNavigate();
  const [search, setSearch] = useSearchParams();
  const initialExercise = search.get('exercise') || 'squat';
  const [exerciseType, setExerciseType] = useState(initialExercise);
  const exercise = useMemo(() => exerciseByKey(exerciseType) || EXERCISES[0], [exerciseType]);

  const { state, transition } = useSessionMachine();

  // ---- Source mode state ---------------------------------------------------
  const [sourceMode, setSourceMode] = useState('webcam'); // 'webcam' | 'file'
  const [videoFile, setVideoFile] = useState(null);
  const [isMirrored, setIsMirrored] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  // ---- Video playback state (file mode) ------------------------------------
  const [videoState, setVideoState] = useState({
    playing: false,
    duration: 0,
    currentTime: 0,
    loop: false,
    speed: 1,
  });

  // ---- Pose detection ------------------------------------------------------
  const { videoRef, latest, error: camError } = usePoseDetection({
    enabled: state !== STATES.IDLE && state !== STATES.COMPLETED,
    videoFile: sourceMode === 'file' ? videoFile : null,
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

  // ---- Source mode switching ------------------------------------------------
  const handleSourceChange = useCallback((mode) => {
    if (mode === sourceMode) return;
    // Only allow switching when session is idle.
    if (state !== STATES.IDLE) return;
    setSourceMode(mode);
    setIsMirrored(mode === 'webcam');
    if (mode === 'webcam') {
      setVideoFile(null);
      setVideoState({ playing: false, duration: 0, currentTime: 0, loop: false, speed: 1 });
    }
  }, [sourceMode, state]);

  // ---- File handling -------------------------------------------------------
  const handleFileSelect = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      alert('Please select a video file (MP4, WebM, etc.).');
      return;
    }
    setVideoFile(file);
    setVideoState({ playing: false, duration: 0, currentTime: 0, loop: false, speed: 1 });
  }, []);

  const handleFileInputChange = useCallback((e) => {
    handleFileSelect(e.target.files?.[0]);
    // Reset so the same file can be re-selected.
    if (e.target) e.target.value = '';
  }, [handleFileSelect]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFileSelect(e.dataTransfer.files?.[0]);
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const clearFile = useCallback(() => {
    setVideoFile(null);
    setVideoState({ playing: false, duration: 0, currentTime: 0, loop: false, speed: 1 });
  }, []);

  // ---- Video event bindings (file mode) ------------------------------------
  useEffect(() => {
    const videoEl = videoRef.current;
    if (!videoEl || sourceMode !== 'file') return;

    const onPlay = () => setVideoState((v) => ({ ...v, playing: true }));
    const onPause = () => setVideoState((v) => ({ ...v, playing: false }));
    const onTimeUpdate = () => setVideoState((v) => ({ ...v, currentTime: videoEl.currentTime }));
    const onDurationChange = () => setVideoState((v) => ({ ...v, duration: videoEl.duration || 0 }));
    const onRateChange = () => setVideoState((v) => ({ ...v, speed: videoEl.playbackRate }));
    const onEnded = () => setVideoState((v) => ({ ...v, playing: false }));

    videoEl.addEventListener('play', onPlay);
    videoEl.addEventListener('pause', onPause);
    videoEl.addEventListener('timeupdate', onTimeUpdate);
    videoEl.addEventListener('durationchange', onDurationChange);
    videoEl.addEventListener('ratechange', onRateChange);
    videoEl.addEventListener('ended', onEnded);

    return () => {
      videoEl.removeEventListener('play', onPlay);
      videoEl.removeEventListener('pause', onPause);
      videoEl.removeEventListener('timeupdate', onTimeUpdate);
      videoEl.removeEventListener('durationchange', onDurationChange);
      videoEl.removeEventListener('ratechange', onRateChange);
      videoEl.removeEventListener('ended', onEnded);
    };
  }, [sourceMode, videoRef, state]);

  // ---- Video control handlers (file mode) ----------------------------------
  const togglePlayPause = useCallback(() => {
    const videoEl = videoRef.current;
    if (!videoEl) return;
    if (videoEl.paused) {
      videoEl.play().catch(() => {});
    } else {
      videoEl.pause();
    }
  }, [videoRef]);

  const handleSeek = useCallback((e) => {
    const videoEl = videoRef.current;
    if (!videoEl) return;
    videoEl.currentTime = Number(e.target.value);
  }, [videoRef]);

  const handleSpeedChange = useCallback((e) => {
    const videoEl = videoRef.current;
    if (!videoEl) return;
    const speed = Number(e.target.value);
    videoEl.playbackRate = speed;
    setVideoState((v) => ({ ...v, speed }));
  }, [videoRef]);

  const toggleLoop = useCallback(() => {
    const videoEl = videoRef.current;
    if (!videoEl) return;
    videoEl.loop = !videoEl.loop;
    setVideoState((v) => ({ ...v, loop: videoEl.loop }));
  }, [videoRef]);

  const toggleMirror = useCallback(() => setIsMirrored((v) => !v), []);

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

    // Branch B: send EMA-smoothed points to the backend GCN model.
    if (socketRef.current?.isOpen()) {
      socketRef.current.sendFrame({
        frame_index: latest.frameIndex,
        timestamp_ms: latest.timestampMs,
        points: latest.smoothedPoints,    // Branch B: smoothed for GCN
        angles: latest.angles,
      });
    }
    if (latest.timestampMs - lastPersistTsRef.current >= PERSIST_INTERVAL_MS) {
      lastPersistTsRef.current = latest.timestampMs;
      persistBufferRef.current.push({
        frame: latest.frameIndex,
        t: Math.round(latest.timestampMs),
        points: latest.smoothedPoints,    // Branch B: smoothed for persistence
        angles: latest.angles,
      });
    }
  }, [latest, state]);

  // ---- State transitions -------------------------------------------------
  const handleStart = useCallback(() => {
    if (state === STATES.IDLE) {
      // In file mode, block start if no file is selected.
      if (sourceMode === 'file' && !videoFile) return;
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

      // In file mode, start playback on transition to ACTIVE.
      if (sourceMode === 'file' && videoRef.current) {
        videoRef.current.currentTime = 0;
        videoRef.current.play().catch(() => {});
      }
    }
  }, [state, transition, openSocket, sourceMode, videoFile, videoRef]);

  const handlePause = useCallback(() => {
    transition(STATES.PAUSED);
    // In file mode, pause the video.
    if (sourceMode === 'file' && videoRef.current) {
      videoRef.current.pause();
    }
  }, [transition, sourceMode, videoRef]);

  const handleResume = useCallback(() => {
    transition(STATES.ACTIVE);
    // In file mode, resume playback.
    if (sourceMode === 'file' && videoRef.current) {
      videoRef.current.play().catch(() => {});
    }
  }, [transition, sourceMode, videoRef]);

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

    // In file mode, reset video to start and pause.
    if (sourceMode === 'file' && videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  }, [state, transition, seed, exerciseType, sourceMode, videoRef]);

  const handleFinish = useCallback(async () => {
    transition(STATES.COMPLETED);
    if (socketRef.current) socketRef.current.bye();

    // In file mode, pause the video.
    if (sourceMode === 'file' && videoRef.current) {
      videoRef.current.pause();
    }

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
  }, [exerciseType, transition, result, seed, navigate, closeSocket, sourceMode, videoRef]);

  useEffect(() => () => closeSocket(), [closeSocket]);

  // ---- Render ------------------------------------------------------------
  const isFileMode = sourceMode === 'file';
  const isIdle = state === STATES.IDLE;
  const canSwitchSource = isIdle;

  const exerciseSelector = (
    <div className="card" style={{ marginBottom: 16 }}>
      {/* Row 1: Source toggle + Exercise selector */}
      <div className="row-2" style={{ marginBottom: 12 }}>
        <div>
          <label>Input Source</label>
          <div className="segmented-control">
            <button
              type="button"
              className={`segmented-item${sourceMode === 'webcam' ? ' active' : ''}`}
              disabled={!canSwitchSource}
              onClick={() => handleSourceChange('webcam')}
            >
              📷 Webcam
            </button>
            <button
              type="button"
              className={`segmented-item${sourceMode === 'file' ? ' active' : ''}`}
              disabled={!canSwitchSource}
              onClick={() => handleSourceChange('file')}
            >
              📁 Video File
            </button>
          </div>
        </div>
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
      </div>

      {/* Row 2: File upload zone (only in file mode + idle) */}
      {isFileMode && isIdle && !videoFile && (
        <div
          className={`file-upload-zone${dragOver ? ' dragover' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="upload-icon">🎬</div>
          <div className="upload-text">
            Drag & drop a video here, or <strong>click to browse</strong>
          </div>
          <div className="upload-hint">Supports MP4, WebM, MOV, AVI</div>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileInputChange}
          />
        </div>
      )}

      {/* File info pill (file selected) */}
      {isFileMode && videoFile && (
        <div className="file-info-pill" style={{ marginBottom: 8 }}>
          <span>🎬</span>
          <span className="file-name">{videoFile.name}</span>
          <span className="file-size">{formatFileSize(videoFile.size)}</span>
          {isIdle && (
            <button className="ghost" onClick={clearFile}>✕</button>
          )}
        </div>
      )}

      {/* Tutorial toggle */}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button className="ghost" onClick={() => setTutorialOpen((v) => !v)}>
          {tutorialOpen ? 'Hide tutorial' : 'Show tutorial'}
        </button>
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

  // Custom video player controls (file mode only, rendered inside viewport)
  const videoControls = isFileMode && state !== STATES.IDLE && (
    <div className="video-player-controls">
      <div className="video-timeline">
        <input
          type="range"
          min={0}
          max={videoState.duration || 0}
          step={0.01}
          value={videoState.currentTime}
          onChange={handleSeek}
        />
      </div>
      <div className="video-controls-row">
        <button type="button" onClick={togglePlayPause} title={videoState.playing ? 'Pause' : 'Play'}>
          {videoState.playing ? '⏸' : '▶'}
        </button>
        <span className="video-time">
          {formatTime(videoState.currentTime)} / {formatTime(videoState.duration)}
        </span>
        <div className="video-spacer" />
        <select
          className="speed-select"
          value={videoState.speed}
          onChange={handleSpeedChange}
          title="Playback speed"
        >
          {SPEED_OPTIONS.map((s) => (
            <option key={s} value={s}>{s}×</option>
          ))}
        </select>
        <button
          type="button"
          className={videoState.loop ? 'active-toggle' : ''}
          onClick={toggleLoop}
          title="Loop"
        >
          🔁
        </button>
        <button
          type="button"
          className={isMirrored ? 'active-toggle' : ''}
          onClick={toggleMirror}
          title="Mirror"
        >
          🪞
        </button>
      </div>
    </div>
  );

  return (
    <div>
      {exerciseSelector}
      <div className="monitor">
        <div className="monitor-body">
          <div className="viewport">
            {/* Mirror the live view conditionally based on isMirrored. */}
            <video ref={videoRef} autoPlay={sourceMode === 'webcam'} playsInline muted
                   className={isMirrored ? 'mirrored' : ''} />
            <div className={isMirrored ? 'overlay-mirror' : ''} style={isMirrored ? {} : { position: 'absolute', inset: 0 }}>
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
                {sourceMode === 'file' ? 'Video' : 'Camera'}/MediaPipe error: {String(camError.message || camError)}
              </div>
            )}
            {videoControls}
          </div>
          <div className="sidebar">
            <JointGauges angles={latest?.angles || {}} exercise={exerciseType} />
            <FeedbackList items={feedbackItems} />
            <StabilityChart samples={qualitySamples} />
          </div>
        </div>
        <ControlBar
          state={state}
          ready={isFileMode ? (state === STATES.CALIBRATING && !!videoFile) || ready : ready}
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
