// Joint-angle thresholds per exercise type.
//
// Each entry describes how to render and grade a joint:
//   range : [min, max]            slider range for the gauge bar
//   ok    : [low, high]           values inside this band → green
//   target: number (optional)     a "good form" marker
//   min   : number (optional)     warn band start
//
// We expose this as a registry so the UI can swap the gauge set when the
// clinician picks a different exercise, and so the editor can persist
// per-exercise overrides to localStorage.

export const EXERCISES_WITH_GAUGES = [
  'squat',
  'lunge',
  'shoulder_raise',
  'knee_extension',
  'custom',
];

export const BUILTIN_THRESHOLDS = {
  squat: {
    left_knee:  { range: [0, 180], ok: [70, 130], min: 70, target: 90 },
    right_knee: { range: [0, 180], ok: [70, 130], min: 70, target: 90 },
    back:       { range: [90, 180], ok: [150, 180], min: 150 },
  },
  lunge: {
    left_knee:  { range: [0, 180], ok: [80, 110], target: 90 },
    right_knee: { range: [0, 180], ok: [80, 110], target: 90 },
  },
  shoulder_raise: {
    left_shoulder:  { range: [0, 180], ok: [80, 160], target: 130 },
    right_shoulder: { range: [0, 180], ok: [80, 160], target: 130 },
  },
  knee_extension: {
    left_knee:  { range: [0, 180], ok: [160, 180], target: 175 },
    right_knee: { range: [0, 180], ok: [160, 180], target: 175 },
  },
  custom: {
    left_knee:  { range: [0, 180], ok: [60, 150] },
    right_knee: { range: [0, 180], ok: [60, 150] },
    left_shoulder:  { range: [0, 180], ok: [30, 170] },
    right_shoulder: { range: [0, 180], ok: [30, 170] },
  },
};

const LS_KEY_PREFIX = 'rehab.thresholds.';

export function loadThresholds(exercise) {
  try {
    const raw = localStorage.getItem(LS_KEY_PREFIX + exercise);
    if (!raw) return cloneDefaults(exercise);
    const parsed = JSON.parse(raw);
    return mergeWithDefaults(exercise, parsed);
  } catch {
    return cloneDefaults(exercise);
  }
}

export function saveThresholds(exercise, thresholds) {
  try {
    localStorage.setItem(LS_KEY_PREFIX + exercise, JSON.stringify(thresholds));
  } catch { /* private mode, full quota, etc. — best-effort. */ }
}

export function resetThresholds(exercise) {
  try { localStorage.removeItem(LS_KEY_PREFIX + exercise); } catch { /* no-op */ }
  return cloneDefaults(exercise);
}

function cloneDefaults(exercise) {
  return JSON.parse(JSON.stringify(BUILTIN_THRESHOLDS[exercise] || BUILTIN_THRESHOLDS.custom));
}

function mergeWithDefaults(exercise, override) {
  const defaults = cloneDefaults(exercise);
  // Only keep keys that exist in defaults — prevents stale joints from a
  // previous schema lingering forever.
  const out = {};
  for (const key of Object.keys(defaults)) {
    out[key] = { ...defaults[key], ...(override[key] || {}) };
  }
  return out;
}
