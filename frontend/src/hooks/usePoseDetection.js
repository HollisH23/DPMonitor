// usePoseDetection — webcam + MediaPipe Pose, two-branch output at 30 FPS.
//
// Returns: { videoRef, latest, isRunning, startCamera, stopCamera, error }
// where `latest` contains:
//   - points:          Raw MediaPipe coordinates (Branch A — UI overlay).
//   - angles:          Joint angles from raw points (Branch A — gauges).
//   - smoothedPoints:  EMA-filtered coordinates (Branch B — GCN model).
//
// Branch A uses raw coords for pixel-accurate body tracking with zero lag.
// Branch B uses EMA-smoothed coords to reduce jitter for the CTR-GCN model.
//
// Camera handling note (Phase 2.2 — FOV refinement):
// We deliberately bypass `@mediapipe/camera_utils` `Camera` helper because its
// constructor only forwards primitive width/height numbers to getUserMedia and
// does not let us pass full MediaTrackConstraints (e.g. `ideal` objects or an
// `aspectRatio` constraint). To enforce a widescreen 1080p user-facing capture
// we call `navigator.mediaDevices.getUserMedia` directly and drive MediaPipe
// from our own `requestAnimationFrame` loop.
//
// Phase 2.3 — Video file support:
// When `videoFile` is provided, we create a Blob URL and pipe its frames through
// the same MediaPipe Pose pipeline. Timestamps map to `videoEl.currentTime`.
// A `seeked` listener fires a single pose send so the skeleton overlay updates
// instantly when scrubbing while paused.

import { useCallback, useEffect, useRef, useState } from 'react';

import { computeAngles, landmarksToPoints } from '../lib/poseUtils.js';
import { createPoseSmoother } from '../lib/poseSmoothing.js';

export function usePoseDetection({ enabled, videoFile = null }) {
  const videoRef = useRef(null);
  const poseRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const sendingRef = useRef(false);
  const frameIndexRef = useRef(0);
  const startTimeRef = useRef(null);
  const blobUrlRef = useRef(null);
  // Track the current mode so cleanup knows which teardown path to take.
  const modeRef = useRef(null); // 'webcam' | 'file'
  // EMA smoother for Branch B (GCN model data). Raw points go to Branch A.
  const smootherRef = useRef(createPoseSmoother());

  const [latest, setLatest] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);

  const stopCamera = useCallback(() => {
    // Cancel the frame-pump first so no in-flight send() targets a dead video.
    if (rafRef.current != null) {
      try { cancelAnimationFrame(rafRef.current); } catch { /* no-op */ }
      rafRef.current = null;
    }
    sendingRef.current = false;

    // Stop every media track returned by getUserMedia (webcam mode only).
    if (streamRef.current) {
      try { streamRef.current.getTracks().forEach((t) => t.stop()); } catch { /* no-op */ }
      streamRef.current = null;
    }
    // Belt-and-braces: if the element still owns a stream, tear it down too.
    if (videoRef.current && videoRef.current.srcObject) {
      try { videoRef.current.srcObject.getTracks().forEach((t) => t.stop()); } catch { /* no-op */ }
      videoRef.current.srcObject = null;
    }

    // Revoke Blob URL if we created one (file mode).
    if (blobUrlRef.current) {
      try { URL.revokeObjectURL(blobUrlRef.current); } catch { /* no-op */ }
      blobUrlRef.current = null;
    }
    // Clear file-mode src so the element doesn't hold a stale reference.
    if (videoRef.current && modeRef.current === 'file') {
      videoRef.current.removeAttribute('src');
      videoRef.current.load(); // reset the element
    }

    // Reset the EMA smoother so the next session starts fresh.
    smootherRef.current.reset();

    modeRef.current = null;
    setIsRunning(false);
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    if (!videoRef.current) return;
    // Prevent double-starts — if already running, bail.
    if (streamRef.current || blobUrlRef.current) return;

    // Lazily import MediaPipe Pose so the dashboard doesn't pay the cost.
    let Pose;
    try {
      ({ Pose } = await import('@mediapipe/pose'));
    } catch (e) {
      setError(new Error(
        'MediaPipe failed to load. Run `npm install` and ensure @mediapipe/pose is present.'
      ));
      return;
    }

    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });
    pose.setOptions({
      modelComplexity: 2,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });
    pose.onResults((results) => {
      if (!results.poseLandmarks) {
        return;
      }
      // Branch A: raw coordinates for pixel-accurate UI overlay.
      const points = landmarksToPoints(results.poseLandmarks);
      const angles = computeAngles(points);

      // Branch B: EMA-smoothed coordinates for the GCN model.
      const smoothedPoints = smootherRef.current.smooth(points);

      let tMs;
      if (modeRef.current === 'file' && videoRef.current) {
        // For uploaded videos, map directly to the video's playback position.
        tMs = videoRef.current.currentTime * 1000;
      } else {
        const now = performance.now();
        if (startTimeRef.current == null) startTimeRef.current = now;
        tMs = now - startTimeRef.current;
      }

      const idx = frameIndexRef.current++;
      setLatest({
        frameIndex: idx,
        timestampMs: tMs,
        points,                // Branch A: raw for SkeletonOverlay / JointGauges
        angles,                // Branch A: raw angles for gauges
        smoothedPoints,        // Branch B: EMA-smoothed for WebSocket / GCN
        rawLandmarks: results.poseLandmarks,
      });
    });
    poseRef.current = pose;

    const videoEl = videoRef.current;

    // ------------------------------------------------------------------
    // FILE MODE — use Blob URL instead of getUserMedia
    // ------------------------------------------------------------------
    if (videoFile) {
      modeRef.current = 'file';
      const url = URL.createObjectURL(videoFile);
      blobUrlRef.current = url;

      videoEl.src = url;
      videoEl.muted = true;
      videoEl.playsInline = true;
      // Don't autoplay — let LiveMonitor control play/pause via session state.

      try {
        await new Promise((resolve, reject) => {
          const onLoaded = () => {
            videoEl.removeEventListener('loadedmetadata', onLoaded);
            videoEl.removeEventListener('error', onError);
            resolve();
          };
          const onError = (ev) => {
            videoEl.removeEventListener('loadedmetadata', onLoaded);
            videoEl.removeEventListener('error', onError);
            reject(ev?.error || new Error('Video element failed to load file.'));
          };
          videoEl.addEventListener('loadedmetadata', onLoaded);
          videoEl.addEventListener('error', onError);
        });
      } catch (e) {
        setError(e);
        setIsRunning(false);
        URL.revokeObjectURL(url);
        blobUrlRef.current = null;
        if (videoRef.current) videoRef.current.removeAttribute('src');
        return;
      }

      // Seeked handler — when user scrubs while paused, fire a single pose
      // send so the skeleton overlay updates immediately.
      const onSeeked = async () => {
        if (!poseRef.current || !videoRef.current) return;
        if (videoRef.current.readyState >= 2 && !sendingRef.current) {
          sendingRef.current = true;
          try {
            await poseRef.current.send({ image: videoRef.current });
          } catch { /* swallow */ } finally {
            sendingRef.current = false;
          }
        }
      };
      videoEl.addEventListener('seeked', onSeeked);

      // Frame pump for file mode — only pushes frames while video is playing.
      const pump = async () => {
        if (!blobUrlRef.current || !poseRef.current || !videoRef.current) {
          rafRef.current = null;
          return;
        }
        if (
          !sendingRef.current &&
          !videoRef.current.paused &&
          videoRef.current.readyState >= 2
        ) {
          sendingRef.current = true;
          try {
            await poseRef.current.send({ image: videoRef.current });
          } catch { /* swallow */ } finally {
            sendingRef.current = false;
          }
        }
        if (blobUrlRef.current) {
          rafRef.current = requestAnimationFrame(pump);
        } else {
          rafRef.current = null;
        }
      };
      rafRef.current = requestAnimationFrame(pump);
      setIsRunning(true);
      return;
    }

    // ------------------------------------------------------------------
    // WEBCAM MODE — existing getUserMedia flow
    // ------------------------------------------------------------------
    modeRef.current = 'webcam';

    // Phase 2.2 — request a widescreen 1080p user-facing stream. We pass full
    // MediaTrackConstraints (with `ideal` fields and `aspectRatio`) directly to
    // getUserMedia so the browser can negotiate the closest 16:9 mode the
    // hardware supports. The `@mediapipe/camera_utils` Camera helper cannot
    // forward these constraints, which is why we bypass it here.
    const constraints = {
      video: {
        facingMode: 'user',
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        aspectRatio: { ideal: 1.7777777778 }, // 16:9 aspect ratio
      },
      audio: false,
    };

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      setError(e);
      setIsRunning(false);
      return;
    }

    if (!videoEl) {
      // Component unmounted while we awaited getUserMedia — release the stream.
      try { stream.getTracks().forEach((t) => t.stop()); } catch { /* no-op */ }
      return;
    }

    streamRef.current = stream;
    videoEl.srcObject = stream;
    videoEl.muted = true;
    videoEl.playsInline = true;

    try {
      await new Promise((resolve, reject) => {
        const onLoaded = () => {
          videoEl.removeEventListener('loadedmetadata', onLoaded);
          videoEl.removeEventListener('error', onError);
          resolve();
        };
        const onError = (ev) => {
          videoEl.removeEventListener('loadedmetadata', onLoaded);
          videoEl.removeEventListener('error', onError);
          reject(ev?.error || new Error('Video element failed to load stream.'));
        };
        videoEl.addEventListener('loadedmetadata', onLoaded);
        videoEl.addEventListener('error', onError);
      });
      await videoEl.play();
    } catch (e) {
      setError(e);
      setIsRunning(false);
      // Roll back the stream so we don't leak the camera light.
      try { stream.getTracks().forEach((t) => t.stop()); } catch { /* no-op */ }
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      return;
    }

    // Custom frame-pump: on every animation frame, push the current video
    // frame into MediaPipe Pose. We guard with `sendingRef` so we never have
    // two `send()` calls in flight simultaneously (which would crash the WASM
    // runtime).
    const pump = async () => {
      // The pump may keep firing once after stopCamera() — bail if we've torn
      // down or if the video isn't ready to be sampled yet.
      if (!streamRef.current || !poseRef.current || !videoRef.current) {
        rafRef.current = null;
        return;
      }
      if (
        !sendingRef.current &&
        videoRef.current.readyState >= 2 /* HAVE_CURRENT_DATA */
      ) {
        sendingRef.current = true;
        try {
          await poseRef.current.send({ image: videoRef.current });
        } catch {
          // Swallow transient send errors; the next frame will retry.
        } finally {
          sendingRef.current = false;
        }
      }
      // Re-check after the await — stopCamera() may have run while we waited.
      if (streamRef.current) {
        rafRef.current = requestAnimationFrame(pump);
      } else {
        rafRef.current = null;
      }
    };
    rafRef.current = requestAnimationFrame(pump);
    setIsRunning(true);
  }, [videoFile]);

  // Auto-start when enabled, auto-stop when disabled / unmounted.
  useEffect(() => {
    if (enabled) startCamera();
    return () => { stopCamera(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, videoFile]);

  return { videoRef, latest, isRunning, error, startCamera, stopCamera };
}
