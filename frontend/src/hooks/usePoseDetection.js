// usePoseDetection — webcam + MediaPipe Pose, normalised output at 30 FPS.
//
// Returns: { videoRef, latest, isRunning, startCamera, stopCamera, error }
// where `latest` is { frameIndex, timestampMs, points, angles } updated on
// every MediaPipe callback. Callers can sample at whatever cadence they need
// (we downsample to 15 FPS for persistence in LiveMonitor).
//
// Camera handling note (Phase 2.2 — FOV refinement):
// We deliberately bypass `@mediapipe/camera_utils` `Camera` helper because its
// constructor only forwards primitive width/height numbers to getUserMedia and
// does not let us pass full MediaTrackConstraints (e.g. `ideal` objects or an
// `aspectRatio` constraint). To enforce a widescreen 1080p user-facing capture
// we call `navigator.mediaDevices.getUserMedia` directly and drive MediaPipe
// from our own `requestAnimationFrame` loop.

import { useCallback, useEffect, useRef, useState } from 'react';

import { computeAngles, landmarksToPoints } from '../lib/poseUtils.js';

export function usePoseDetection({ enabled }) {
  const videoRef = useRef(null);
  const poseRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);
  const sendingRef = useRef(false);
  const frameIndexRef = useRef(0);
  const startTimeRef = useRef(null);

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

    // Stop every media track returned by getUserMedia.
    if (streamRef.current) {
      try { streamRef.current.getTracks().forEach((t) => t.stop()); } catch { /* no-op */ }
      streamRef.current = null;
    }
    // Belt-and-braces: if the element still owns a stream, tear it down too.
    if (videoRef.current && videoRef.current.srcObject) {
      try { videoRef.current.srcObject.getTracks().forEach((t) => t.stop()); } catch { /* no-op */ }
      videoRef.current.srcObject = null;
    }
    setIsRunning(false);
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    if (!videoRef.current) return;
    if (streamRef.current) return; // already running

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
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    pose.onResults((results) => {
      if (!results.poseLandmarks) {
        return;
      }
      const points = landmarksToPoints(results.poseLandmarks);
      const angles = computeAngles(points);
      const now = performance.now();
      if (startTimeRef.current == null) startTimeRef.current = now;
      const tMs = now - startTimeRef.current;
      const idx = frameIndexRef.current++;
      setLatest({
        frameIndex: idx,
        timestampMs: tMs,
        points,
        angles,
        rawLandmarks: results.poseLandmarks,
      });
    });
    poseRef.current = pose;

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

    const videoEl = videoRef.current;
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
  }, []);

  // Auto-start when enabled, auto-stop when disabled / unmounted.
  useEffect(() => {
    if (enabled) startCamera();
    return () => { stopCamera(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);

  return { videoRef, latest, isRunning, error, startCamera, stopCamera };
}
