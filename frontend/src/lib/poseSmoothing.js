// poseSmoothing.js — Exponential Moving Average (EMA) temporal filter
// for the Branch B (GCN model) data pipeline.
//
// This smoother is NOT used by the UI overlay (Branch A), which renders
// raw MediaPipe coordinates for pixel-accurate body tracking. The EMA
// filter reduces high-frequency jitter in the data sent to the backend
// CTR-GCN model, where temporal stability matters more than frame-exact
// positional fidelity.
//
// Occlusion handling: when a joint's visibility drops below a threshold,
// the EMA alpha is reduced drastically so the smoothed position leans on
// the previous frame's value, preventing erratic jumps from low-confidence
// detections.

/**
 * Create a per-joint EMA smoother.
 *
 * @param {number} alpha      Base smoothing factor (0–1). Higher = more
 *                             responsive to new data, lower = smoother.
 *                             Default 0.6.
 * @param {number} occlusionThreshold  Visibility score below which a
 *                             joint is considered occluded. Default 0.5.
 * @param {number} occludedAlpha  Alpha used for occluded joints — much
 *                             lower so the position barely moves.
 *                             Default 0.15.
 * @returns {{ smooth, reset }}
 */
export function createPoseSmoother(
  alpha = 0.6,
  occlusionThreshold = 0.5,
  occludedAlpha = 0.15,
) {
  // Previous smoothed state: { jointName: [x, y, z, visibility] }
  let prev = null;

  /**
   * Smooth a full points dict, returning a new dict with the same shape.
   *
   * @param {Object} rawPoints  { name: [x, y, z, visibility], ... }
   * @returns {Object} smoothedPoints — same shape, smoothed coordinates.
   */
  function smooth(rawPoints) {
    if (!rawPoints) return rawPoints;

    // First frame — no previous state, just copy and store.
    if (prev === null) {
      prev = {};
      for (const name of Object.keys(rawPoints)) {
        const p = rawPoints[name];
        prev[name] = [p[0], p[1], p[2], p[3]];
      }
      return prev;
    }

    const out = {};
    for (const name of Object.keys(rawPoints)) {
      const raw = rawPoints[name];
      const prevJoint = prev[name];

      if (!prevJoint) {
        // New joint that didn't exist last frame — take raw.
        out[name] = [raw[0], raw[1], raw[2], raw[3]];
      } else {
        const visibility = raw[3];
        // Use reduced alpha for occluded joints so they hold position.
        const a = visibility < occlusionThreshold ? occludedAlpha : alpha;

        out[name] = [
          a * raw[0] + (1 - a) * prevJoint[0],
          a * raw[1] + (1 - a) * prevJoint[1],
          a * raw[2] + (1 - a) * prevJoint[2],
          // Smooth visibility itself to prevent flicker.
          a * raw[3] + (1 - a) * prevJoint[3],
        ];
      }
    }

    // Update previous state for next frame.
    prev = out;
    return out;
  }

  /**
   * Reset the smoother state. Call on session start/stop/recalibrate.
   */
  function reset() {
    prev = null;
  }

  return { smooth, reset };
}
