import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';

import { api } from '../api/client.js';
import MultiLineChart from '../components/MultiLineChart.jsx';
import SummaryBox from '../components/SummaryBox.jsx';

function pct(v) { return v == null ? '—' : `${Math.round(v * 100)}%`; }
function fmtDate(iso) { return iso ? new Date(iso).toLocaleString() : '—'; }

// Trajectory chart: plots hip-y over time so the clinician can scan
// for smoothness, depth, and tempo at a glance.
function TrajectoryChart({ frames }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    const { clientWidth: w, clientHeight: h } = c;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    ctx.clearRect(0, 0, w, h);

    if (!frames || frames.length < 2) return;
    const ys = frames.map((f) => {
      const p = f.points || {};
      const lh = p.left_hip, rh = p.right_hip;
      if (lh && rh) return 0.5 * (lh[1] + rh[1]);
      if (lh) return lh[1];
      if (rh) return rh[1];
      return null;
    }).filter((v) => v != null);
    if (ys.length < 2) return;

    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const span = (maxY - minY) || 1e-3;

    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    for (let i = 1; i < 5; i++) {
      const y = (h * i) / 5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(102,224,194,0.30)');
    grad.addColorStop(1, 'rgba(102,224,194,0)');
    ctx.beginPath();
    ys.forEach((v, i) => {
      const x = (i / (ys.length - 1)) * w;
      const y = ((v - minY) / span) * (h - 8) + 4;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    ctx.beginPath();
    ys.forEach((v, i) => {
      const x = (i / (ys.length - 1)) * w;
      const y = ((v - minY) / span) * (h - 8) + 4;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#66e0c2'; ctx.lineWidth = 2; ctx.stroke();
  }, [frames]);
  return <canvas ref={ref} />;
}

export default function ReportView() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [session, setSession] = useState(null);
  const [err, setErr] = useState(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    api.getSession(sessionId).then(setSession).catch((e) => setErr(e));
  }, [sessionId]);

  const frames = session?.trajectory?.frames || [];

  const duration = useMemo(() => {
    if (!session?.started_at || !session?.ended_at) return null;
    return (new Date(session.ended_at) - new Date(session.started_at)) / 1000;
  }, [session]);

  // Build angle series for every joint that appears in any frame.
  const angleSeries = useMemo(() => {
    if (!frames.length) return [];
    const joints = new Set();
    for (const f of frames) Object.keys(f.angles || {}).forEach((k) => joints.add(k));
    const colors = ['#66e0c2', '#f5b740', '#ef6f6c', '#8ab4ff', '#d09cff'];
    let i = 0;
    return Array.from(joints).map((joint) => ({
      label: joint,
      color: colors[i++ % colors.length],
      data: frames.map((f) => (f.angles && f.angles[joint] != null ? Number(f.angles[joint]) : null))
                  .filter((v) => v != null),
      yMin: 0, yMax: 180,
    }));
  }, [frames]);

  // Adherence breakdown: time spent within OK band (per joint) vs out of band.
  const adherence = useMemo(() => {
    if (!frames.length || !angleSeries.length) return [];
    const rows = [];
    for (const s of angleSeries) {
      // Default "ok" band: 60°–150° unless we know the exercise better.
      const okLow = 60, okHigh = 150;
      const inBand = s.data.filter((v) => v >= okLow && v <= okHigh).length;
      const total = s.data.length || 1;
      rows.push({
        joint: s.label,
        inBandPct: inBand / total,
        color: s.color,
      });
    }
    return rows;
  }, [angleSeries, frames]);

  async function handleDelete() {
    if (!session) return;
    if (!window.confirm('Delete this session and its trajectory permanently?')) return;
    setDeleting(true);
    try {
      await api.deleteSession(session.id);
      navigate('/dashboard');
    } catch (e) {
      alert(`Delete failed: ${e.message}`);
      setDeleting(false);
    }
  }

  if (err) return <div className="card">Failed to load: {err.message}</div>;
  if (!session) return <div className="card">Loading…</div>;

  const compensationEvents = session.progress_trend?.compensation_events ?? null;

  return (
    <div>
      <div className="page-h">
        <div>
          <h1>Session Report</h1>
          <div className="sub">
            {session.patient_name} · {session.exercise_type} · {fmtDate(session.started_at)}
            {session.random_seed != null && <> · seed <span className="tag">{session.random_seed}</span></>}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <Link to="/dashboard"><button>← Back</button></Link>
          <button className="danger" onClick={handleDelete} disabled={deleting}>
            {deleting ? 'Deleting…' : 'Delete Session'}
          </button>
        </div>
      </div>

      <div className="report">
        <div className="metric">
          <span className="label">Reps</span>
          <span className="value">{session.rep_count}</span>
        </div>
        <div className="metric">
          <span className="label">Overall Stability</span>
          <span className="value">{pct(session.overall_stability_score)}</span>
        </div>
        <div className="metric">
          <span className="label">Quality Score</span>
          <span className="value">{pct(session.quality_score)}</span>
        </div>
        <div className="metric">
          <span className="label">Duration</span>
          <span className="value">{duration != null ? `${duration.toFixed(0)}s` : '—'}</span>
        </div>

        <div className="card full">
          <div className="card-h">
            <div className="card-title">Trajectory Stability (hip midpoint)</div>
            <span className="muted">
              {frames.length} frames @ {session.trajectory?.sample_rate_hz || 15} FPS
            </span>
          </div>
          <TrajectoryChart frames={frames} />
        </div>

        <div className="card full">
          <div className="card-h">
            <div className="card-title">Joint Angle Traces</div>
            <span className="muted">degrees over session timeline</span>
          </div>
          {angleSeries.length > 0
            ? <MultiLineChart series={angleSeries} yLabel="degrees" />
            : <div className="muted">No joint-angle data was captured.</div>}
        </div>

        <div className="card">
          <div className="card-h"><div className="card-title">Adherence Breakdown</div></div>
          {adherence.length === 0 && <div className="muted">No data captured.</div>}
          {adherence.map((row) => (
            <div key={row.joint} style={{ marginBottom: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span>
                  <span style={{
                    display: 'inline-block', width: 8, height: 8, borderRadius: 2,
                    background: row.color, marginRight: 6,
                  }} />
                  {row.joint}
                </span>
                <span className="num">{Math.round(row.inBandPct * 100)}% in OK band</span>
              </div>
              <div className="bar" style={{ height: 6, background: 'var(--bg-3)', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${row.inBandPct * 100}%`,
                  background: row.inBandPct >= 0.8
                    ? 'var(--accent)'
                    : row.inBandPct >= 0.6 ? 'var(--warn)' : 'var(--bad)',
                }} />
              </div>
            </div>
          ))}
        </div>

        <div className="card">
          <div className="card-h"><div className="card-title">Progress Trend</div></div>
          {session.progress_trend && Object.keys(session.progress_trend).length > 0 ? (
            <ul style={{ paddingLeft: 18, margin: 0 }}>
              {Object.entries(session.progress_trend).map(([k, v]) => (
                <li key={k}>
                  <span className="muted">{k}:</span>{' '}
                  <code>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</code>
                </li>
              ))}
            </ul>
          ) : <div className="muted">No trend data captured.</div>}
        </div>

        <div className="card full">
          <div className="card-h"><div className="card-title">Clinical Adherence</div></div>
          <div className="muted" style={{ lineHeight: 1.7 }}>
            Session ran for {duration != null ? `${duration.toFixed(0)}s` : '—'} with{' '}
            <span className="tag">{session.rep_count}</span> rep(s) completed at an
            overall quality of <span className="tag">{pct(session.quality_score)}</span>.
            {compensationEvents != null && (
              <> The analyzer flagged <span className="tag warn">{compensationEvents}</span> compensation event(s).</>
            )}
            {' '}Stability and quality scores above 80% indicate clean form;
            below 60% suggests compensatory movement or insufficient ROM and
            warrants clinician review.
          </div>
        </div>

        {/* Phase 4 Task 9: the native Summary Box. Sits at the bottom of
            the records view so it's the last thing the clinician reads
            before deciding next steps. Reads `summary_box` synthesized
            by the CTR-GCN analyzer at session end. */}
        <SummaryBox summary={session.progress_trend?.summary_box} />
      </div>
    </div>
  );
}
