// Joint Gauges — current angles for clinically interesting joints, with
// per-exercise threshold settings that can be edited inline. Each gauge
// is a "standardized component slot" — the set is fully driven by the
// thresholds dict, so adding a joint is just a config change.

import { useEffect, useState } from 'react';

import {
  BUILTIN_THRESHOLDS,
  loadThresholds,
  resetThresholds,
  saveThresholds,
} from '../lib/thresholds.js';

const LABELS = {
  left_knee: 'L Knee',
  right_knee: 'R Knee',
  left_shoulder: 'L Shoulder',
  right_shoulder: 'R Shoulder',
  back: 'Back',
};

export default function JointGauges({ angles, exercise = 'custom' }) {
  const [thresholds, setThresholds] = useState(() => loadThresholds(exercise));
  const [editing, setEditing] = useState(false);

  // When the exercise changes (e.g. clinician switches drill), reload the
  // stored thresholds for that exercise.
  useEffect(() => { setThresholds(loadThresholds(exercise)); }, [exercise]);

  function update(joint, bound, value) {
    setThresholds((prev) => {
      const next = { ...prev, [joint]: { ...prev[joint] } };
      const ok = [...(prev[joint].ok || [0, 180])];
      if (bound === 'okLow')  ok[0] = Number(value);
      if (bound === 'okHigh') ok[1] = Number(value);
      next[joint].ok = ok;
      saveThresholds(exercise, next);
      return next;
    });
  }

  function reset() {
    setThresholds(resetThresholds(exercise));
  }

  const keys = Object.keys(thresholds);

  return (
    <div className="gauges-wrap">
      <div className="gauges">
        {keys.map((k) => {
          const angle = angles?.[k];
          const t = thresholds[k];
          const pct = angle != null && t.range
            ? Math.max(0, Math.min(100, ((angle - t.range[0]) / (t.range[1] - t.range[0])) * 100))
            : 0;
          let tone = '';
          if (angle != null && t.ok) {
            if (angle < t.ok[0] || angle > t.ok[1]) tone = 'bad';
            else if (t.min != null && angle < t.min + 10) tone = 'warn';
          }
          return (
            <div key={k} className={`gauge ${tone}`}>
              <div className="label">{LABELS[k] || k}</div>
              <div className="value">{angle != null ? `${angle.toFixed(0)}°` : '—'}</div>
              <div className="bar"><div className="fill" style={{ width: `${pct}%` }} /></div>
              {t.ok && <div className="band-hint">ok: {t.ok[0]}°–{t.ok[1]}°</div>}
            </div>
          );
        })}
      </div>
      <div className="gauges-toolbar">
        <button className="ghost small" onClick={() => setEditing((v) => !v)}>
          {editing ? 'Done' : 'Edit thresholds'}
        </button>
        {editing && (
          <button className="ghost small" onClick={reset} title="Restore plan defaults">
            Reset to defaults
          </button>
        )}
      </div>

      {editing && (
        <div className="threshold-editor">
          <div className="muted small" style={{ marginBottom: 8 }}>
            Editing thresholds for <b>{exercise}</b>. Saved locally — never
            leaves this machine.
          </div>
          {keys.map((k) => {
            const t = thresholds[k];
            const builtin = (BUILTIN_THRESHOLDS[exercise] || {})[k];
            const isCustom = builtin && (
              builtin.ok?.[0] !== t.ok?.[0] || builtin.ok?.[1] !== t.ok?.[1]
            );
            return (
              <div className="threshold-row" key={k}>
                <div className="threshold-row-label">
                  {LABELS[k] || k} {isCustom && <span className="tag warn">custom</span>}
                </div>
                <label>OK band low (°)</label>
                <input type="number" min={t.range?.[0] ?? 0} max={t.range?.[1] ?? 180}
                       value={t.ok?.[0] ?? 0}
                       onChange={(e) => update(k, 'okLow', e.target.value)} />
                <label>OK band high (°)</label>
                <input type="number" min={t.range?.[0] ?? 0} max={t.range?.[1] ?? 180}
                       value={t.ok?.[1] ?? 180}
                       onChange={(e) => update(k, 'okHigh', e.target.value)} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
