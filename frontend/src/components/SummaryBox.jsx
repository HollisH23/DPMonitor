// Phase 4 Task 9: native "Summary Box" rendered at the bottom of the
// session-records view. It consumes the analyzer's post-workout
// synthesis (carried on `session.progress_trend.summary_box`) and lays
// out the four headline metrics plus two charts:
//
//   1. ROM Curve         — joint angle vs. time, with rep boundaries.
//   2. Stability Trend   — per-rep stability so fatigue is visible at a
//                          glance (declining bars ≡ muscle fatigue).
//
// Pure-canvas drawing keeps the bundle dep-free (matches the project's
// existing MultiLineChart / StabilityChart approach). When the synthesis
// payload is absent (older sessions or the placeholder analyzer), the
// component degrades to a friendly empty state rather than crashing.

import { useEffect, useRef } from 'react';

function fmtPercent(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return `${v.toFixed(1)}%`;
}

function fmtDegrees(v) {
  if (v == null || Number.isNaN(v)) return '—';
  return `${v.toFixed(1)}°`;
}

function fmtFatigue(v) {
  // Variance numbers are scale-free; round to a sensible precision and
  // attach a qualitative band so clinicians don't have to interpret
  // raw variances.
  if (v == null || Number.isNaN(v)) return '—';
  const value = v.toFixed(3);
  let band = 'Low';
  if (v > 0.10) band = 'High';
  else if (v > 0.03) band = 'Moderate';
  return `${value} (${band})`;
}

// ---------------------------------------------------------------------
// ROM curve canvas — joint angle vs. time, troughs marked.
// ---------------------------------------------------------------------
function ROMCurve({ curve, joint, height = 200 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    const { clientWidth: w, clientHeight: h } = c;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    ctx.clearRect(0, 0, w, h);

    if (!curve || curve.length < 2) {
      ctx.fillStyle = '#7d8898';
      ctx.font = '12px ui-monospace, monospace';
      ctx.fillText('No ROM data captured.', 10, 20);
      return;
    }

    const angles = curve.map((p) => p.angle);
    const minA = Math.min(...angles);
    const maxA = Math.max(...angles);
    const spanA = (maxA - minA) || 1;
    const t0 = curve[0].t_ms;
    const tEnd = curve[curve.length - 1].t_ms;
    const spanT = (tEnd - t0) || 1;
    const pad = 8;

    // Gridlines.
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 5; i++) {
      const y = (h * i) / 5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Gradient fill under the curve.
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(102,224,194,0.30)');
    grad.addColorStop(1, 'rgba(102,224,194,0)');
    ctx.beginPath();
    curve.forEach((p, i) => {
      const x = ((p.t_ms - t0) / spanT) * w;
      const y = pad + ((maxA - p.angle) / spanA) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line.
    ctx.beginPath();
    curve.forEach((p, i) => {
      const x = ((p.t_ms - t0) / spanT) * w;
      const y = pad + ((maxA - p.angle) / spanA) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#66e0c2';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Mark rep troughs with vertical dotted lines so the clinician can
    // see exactly when each rep boundary was detected.
    ctx.strokeStyle = 'rgba(245,183,64,0.55)';
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    curve.forEach((p) => {
      if (!p.is_trough) return;
      const x = ((p.t_ms - t0) / spanT) * w;
      ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, h - pad); ctx.stroke();
    });
    ctx.setLineDash([]);

    // Axis annotations.
    ctx.fillStyle = '#7d8898';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(`${maxA.toFixed(0)}°`, 6, pad + 12);
    ctx.fillText(`${minA.toFixed(0)}°`, 6, h - pad - 2);
    if (joint) {
      const lbl = `joint: ${joint}`;
      ctx.fillText(lbl, w - ctx.measureText(lbl).width - 8, pad + 12);
    }
  }, [curve, joint]);
  return <canvas ref={ref} style={{ height, width: '100%' }} />;
}

// ---------------------------------------------------------------------
// Stability trend canvas — per-rep stability bars.
// ---------------------------------------------------------------------
function StabilityTrend({ trend, height = 180 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    const { clientWidth: w, clientHeight: h } = c;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    ctx.clearRect(0, 0, w, h);

    if (!trend || trend.length === 0) {
      ctx.fillStyle = '#7d8898';
      ctx.font = '12px ui-monospace, monospace';
      ctx.fillText('Not enough reps to chart stability.', 10, 20);
      return;
    }

    const pad = 10;
    const barW = Math.max(8, (w - pad * 2) / trend.length - 4);
    const drawH = h - pad * 2;

    // Gridlines (25/50/75/100%).
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    for (let i = 1; i < 4; i++) {
      const y = pad + (drawH * i) / 4;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Bars: green when stability ≥ 0.8, amber 0.6–0.8, red below.
    trend.forEach((row, i) => {
      const v = Math.max(0, Math.min(1, row.stability));
      const x = pad + i * (barW + 4);
      const barH = v * drawH;
      const y = pad + (drawH - barH);
      ctx.fillStyle =
        v >= 0.8 ? '#66e0c2' : v >= 0.6 ? '#f5b740' : '#ef6f6c';
      ctx.fillRect(x, y, barW, barH);
      // Rep number under each bar.
      ctx.fillStyle = '#7d8898';
      ctx.font = '11px ui-monospace, monospace';
      const label = `${row.rep}`;
      const tx = x + (barW - ctx.measureText(label).width) / 2;
      ctx.fillText(label, tx, h - 2);
    });

    // Y-axis hint.
    ctx.fillStyle = '#7d8898';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('stability', 6, pad + 10);
  }, [trend]);
  return <canvas ref={ref} style={{ height, width: '100%' }} />;
}

// ---------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------
export default function SummaryBox({ summary }) {
  if (!summary) {
    return (
      <div className="card full">
        <div className="card-h"><div className="card-title">Summary</div></div>
        <div className="muted">
          No summary metrics available for this session. Sessions recorded
          before the CTR-GCN pipeline was wired in won't have this block.
        </div>
      </div>
    );
  }

  const {
    overall_accuracy: overallAccuracy,
    rep_count_by_angle: repCountByAngle,
    per_rep_rom: perRepRom = [],
    fatigue_index: fatigueIndex,
    primary_joint: primaryJoint,
    charts = {},
  } = summary;

  // Peak ROM across the whole session is the largest single-rep range —
  // this is what physios watch as the milestone metric.
  let peakROM = null;
  for (const r of perRepRom) {
    if (r.range_deg != null && (peakROM == null || r.range_deg > peakROM)) {
      peakROM = r.range_deg;
    }
  }

  return (
    <div className="card full">
      <div className="card-h">
        <div className="card-title">Summary</div>
        <span className="muted">
          {primaryJoint
            ? `Primary joint: ${primaryJoint}`
            : 'Aggregate session metrics'}
        </span>
      </div>

      {/* Four headline metrics. Mirrors the dashboard's metric tiles. */}
      <div className="report" style={{ marginBottom: 8 }}>
        <div className="metric">
          <span className="label">Overall Accuracy</span>
          <span className="value">{fmtPercent(overallAccuracy)}</span>
        </div>
        <div className="metric">
          <span className="label">Reps (angle-detected)</span>
          <span className="value">{repCountByAngle ?? 0}</span>
        </div>
        <div className="metric">
          <span className="label">Peak ROM</span>
          <span className="value">{fmtDegrees(peakROM)}</span>
        </div>
        <div className="metric">
          <span className="label">Fatigue</span>
          <span className="value">{fmtFatigue(fatigueIndex)}</span>
        </div>
      </div>

      {/* Two charts side-by-side on wide screens, stacked on narrow ones. */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div className="card">
          <div className="card-h">
            <div className="card-title">ROM Curve</div>
            <span className="muted">angle vs. time · troughs mark reps</span>
          </div>
          <ROMCurve curve={charts.rom_curve || []} joint={primaryJoint} />
        </div>
        <div className="card">
          <div className="card-h">
            <div className="card-title">Stability Trend</div>
            <span className="muted">per-rep stability (1 − jitter)</span>
          </div>
          <StabilityTrend trend={charts.stability_trend || []} />
        </div>
      </div>

      {/* Per-rep ROM table — secondary detail, collapsed visually with
          minimal styling so it doesn't overwhelm the headline tiles. */}
      {perRepRom.length > 0 && (
        <details style={{ marginTop: 12 }}>
          <summary className="muted">Per-rep ROM detail</summary>
          <table style={{ width: '100%', marginTop: 8, fontSize: 12 }}>
            <thead>
              <tr style={{ color: '#7d8898', textAlign: 'left' }}>
                <th>Rep</th><th>Min°</th><th>Max°</th><th>Range°</th>
              </tr>
            </thead>
            <tbody>
              {perRepRom.map((r) => (
                <tr key={r.rep}>
                  <td>{r.rep}</td>
                  <td>{fmtDegrees(r.min_deg)}</td>
                  <td>{fmtDegrees(r.max_deg)}</td>
                  <td>{fmtDegrees(r.range_deg)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  );
}
