// Helpers shared between the MediaPipe runtime and the rendering layer.
//
// MediaPipe Pose returns 33 landmarks in a fixed order. We translate them
// to a stable, human-readable dict before sending across the wire — this
// matches the backend's expected `points` shape and is robust against
// future model upgrades that reshuffle landmark order.

export const POSE_LANDMARKS = [
  'nose',
  'left_eye_inner', 'left_eye', 'left_eye_outer',
  'right_eye_inner', 'right_eye', 'right_eye_outer',
  'left_ear', 'right_ear',
  'mouth_left', 'mouth_right',
  'left_shoulder', 'right_shoulder',
  'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist',
  'left_pinky', 'right_pinky',
  'left_index', 'right_index',
  'left_thumb', 'right_thumb',
  'left_hip', 'right_hip',
  'left_knee', 'right_knee',
  'left_ankle', 'right_ankle',
  'left_heel', 'right_heel',
  'left_foot_index', 'right_foot_index',
];

// Stick-figure edges (indices into POSE_LANDMARKS).
export const POSE_EDGES = [
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
  ['left_ankle', 'left_heel'], ['left_heel', 'left_foot_index'],
  ['right_ankle', 'right_heel'], ['right_heel', 'right_foot_index'],
];

export function landmarksToPoints(landmarks) {
  // landmarks: array of 33 {x, y, z, visibility} in [0,1].
  const out = {};
  for (let i = 0; i < POSE_LANDMARKS.length && i < landmarks.length; i++) {
    const lm = landmarks[i];
    out[POSE_LANDMARKS[i]] = [
      clamp01(lm.x), clamp01(lm.y), Number(lm.z) || 0, Number(lm.visibility) || 0,
    ];
  }
  return out;
}

function clamp01(v) { v = Number(v) || 0; return v < 0 ? 0 : v > 1 ? 1 : v; }

// Joint angle (in degrees) at vertex `b` formed by segments b→a and b→c.
// Uses x,y only (sufficient for sagittal-plane exercises like squats).
export function jointAngleDeg(a, b, c) {
  if (!a || !b || !c) return null;
  const v1x = a[0] - b[0], v1y = a[1] - b[1];
  const v2x = c[0] - b[0], v2y = c[1] - b[1];
  const dot = v1x * v2x + v1y * v2y;
  const m1 = Math.hypot(v1x, v1y);
  const m2 = Math.hypot(v2x, v2y);
  if (m1 === 0 || m2 === 0) return null;
  const cos = Math.min(1, Math.max(-1, dot / (m1 * m2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

export function computeAngles(points) {
  // Returns commonly-watched angles. Missing landmarks → angle omitted.
  const out = {};
  const lk = jointAngleDeg(points.left_hip, points.left_knee, points.left_ankle);
  if (lk != null) out.left_knee = lk;
  const rk = jointAngleDeg(points.right_hip, points.right_knee, points.right_ankle);
  if (rk != null) out.right_knee = rk;
  const ls = jointAngleDeg(points.left_elbow, points.left_shoulder, points.left_hip);
  if (ls != null) out.left_shoulder = ls;
  const rs = jointAngleDeg(points.right_elbow, points.right_shoulder, points.right_hip);
  if (rs != null) out.right_shoulder = rs;
  const back = jointAngleDeg(points.left_shoulder, points.left_hip, points.left_knee);
  if (back != null) out.back = back;
  return out;
}
