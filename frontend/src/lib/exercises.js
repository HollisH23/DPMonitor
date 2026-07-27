// Phase 2.1 — prescribed exercise catalogue.
//
// In a real deployment this would come from a `Prescription` API tied to
// the user's clinician. For MVP we ship a static list that every user
// sees, but the consumer (Dashboard, LiveMonitor) only ever reads from
// this file — so swapping it for an API call later is a one-place change.

export const EXERCISES = [
  {
    key: 'squat',
    name: 'Bodyweight Squat',
    targetReps: 12,
    icon: '🦵',
    summary: 'Hinge at the hips, drive knees out, lower until thighs are parallel.',
    cues: [
      'Feet shoulder-width apart, toes slightly out.',
      'Hinge at the hips first, then bend the knees.',
      'Keep chest tall — eyes forward, not down.',
      'Drive through the heels to stand back up.',
    ],
    contraindications: 'Stop if you feel sharp knee or lower-back pain.',
  },
  {
    key: 'chest_expansion',
    name: 'Chest Expansion',
    targetReps: 10,
    icon: '🌅',
    summary: 'Open the chest by reaching the arms wide and back. Great for posture.',
    cues: [
      'Stand tall, arms straight out in front, palms together.',
      'Inhale and pull both arms wide and slightly back.',
      'Squeeze the shoulder blades down and together.',
      'Exhale and return slowly to start.',
    ],
    contraindications: 'Avoid if you have an unstable shoulder.',
  },
  {
    key: 'lunge',
    name: 'Stationary Lunge',
    targetReps: 10,
    icon: '🚶',
    summary: 'Step forward, lower into a 90-90 lunge, drive back up.',
    cues: [
      'Step one foot forward into a long stance.',
      'Lower straight down — front knee tracks over the foot.',
      'Both knees aim for ~90°.',
      'Drive through the front heel to return.',
    ],
    contraindications: 'Hold a wall for balance if needed.',
  },
  {
    key: 'shoulder_raise',
    name: 'Shoulder Raise',
    targetReps: 12,
    icon: '💪',
    summary: 'Lift the arms in front of you to shoulder height with control.',
    cues: [
      'Stand tall, arms by your sides.',
      'Raise both arms slowly to shoulder height.',
      'Pause at the top, then lower with control.',
      'Avoid shrugging or arching the lower back.',
    ],
    contraindications: 'Use a slow tempo if you have impingement.',
  },
  {
    key: 'knee_extension',
    name: 'Seated Knee Extension',
    targetReps: 15,
    icon: '🦴',
    summary: 'Straighten the knee to near-lockout, control the lowering phase.',
    cues: [
      'Sit tall in a chair, both feet flat.',
      'Slowly straighten one knee until almost locked.',
      'Hold for one breath at the top.',
      'Lower slowly under control.',
    ],
    contraindications: 'Skip if you have post-op restrictions on knee extension.',
  },
];

export function exerciseByKey(key) {
  return EXERCISES.find((e) => e.key === key) || null;
}
