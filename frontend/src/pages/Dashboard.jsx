// Phase 2.1 — Patient self-service Dashboard.
//
// Two modules only:
//   1) Exercise Guides & Prompts (replaces "Recent Sessions").
//   2) Last-7 Trend Chart (retained, scoped to the authenticated user).
// All clinician-side patient list / management UI is gone.

import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

import { api } from '../api/client.js';
import { useAuth } from '../auth/AuthContext.jsx';
import ExerciseCard from '../components/ExerciseCard.jsx';
import { EXERCISES } from '../lib/exercises.js';

function pct(v) { return v == null ? '—' : `${Math.round(v * 100)}%`; }
function fmtDate(iso) { return iso ? new Date(iso).toLocaleString() : '—'; }

export default function Dashboard() {
  const { user } = useAuth();
  const [trend, setTrend] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    api.trend7().then(setTrend).catch(() => setTrend(null));
    api.listSessions().then(setSessions).catch(() => setSessions([]));
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  // Per-exercise completion count this week, used to label each card.
  const completionByExercise = (() => {
    const m = new Map();
    for (const row of trend?.by_exercise || []) {
      m.set(row.exercise_type, row.count);
    }
    return m;
  })();

  return (
    <div>
      <div className="page-h">
        <div>
          <h1>Welcome back{user?.username ? `, ${user.username}` : ''}</h1>
          <div className="sub">
            Your prescribed exercises for today.
            {health && <> · Seed <span className="tag">{health.random_seed}</span></>}
          </div>
        </div>
      </div>

      {/* === Exercise Guides & Prompts ============================== */}
      <section style={{ marginBottom: 24 }}>
        <div className="section-h">
          <h2>Today's Training</h2>
          <span className="muted">{EXERCISES.length} exercises prescribed</span>
        </div>
        <div className="ex-grid">
          {EXERCISES.map((ex) => (
            <ExerciseCard
              key={ex.key}
              exercise={ex}
              completedThisWeek={completionByExercise.get(ex.key) || 0}
            />
          ))}
        </div>
      </section>

      {/* === Last-7 Trend (RETAINED) ================================ */}
      <section>
        <div className="section-h">
          <h2>Your Last 7 Days</h2>
          <span className="muted">user-scoped progress</span>
        </div>
        <div className="dash-grid">
          <div className="card">
            <div className="card-h">
              <div className="card-title">Trend Metrics</div>
            </div>
            <div className="dash-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
              <div className="metric">
                <span className="label">Sessions</span>
                <span className="value">{trend?.sessions ?? 0}</span>
              </div>
              <div className="metric">
                <span className="label">Total Reps</span>
                <span className="value">{trend?.total_reps ?? 0}</span>
              </div>
              <div className="metric">
                <span className="label">Avg Stability</span>
                <span className="value">{pct(trend?.avg_stability)}</span>
              </div>
              <div className="metric">
                <span className="label">Avg Quality</span>
                <span className="value">{pct(trend?.avg_quality)}</span>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-h">
              <div className="card-title">Recent Sessions</div>
              <span className="muted">your history</span>
            </div>
            <div className="list">
              {sessions.length === 0 && (
                <div className="muted">No sessions yet — click any exercise card above to start.</div>
              )}
              {sessions.slice(0, 7).map((s) => (
                <Link key={s.id} to={`/sessions/${s.id}`} style={{ textDecoration: 'none' }}>
                  <div className="list-row" style={{ gridTemplateColumns: '1.2fr 1fr 0.8fr 0.8fr auto' }}>
                    <div>
                      <div>{s.exercise_type}</div>
                      <div className="small">{fmtDate(s.started_at)}</div>
                    </div>
                    <div className="num">Reps {s.rep_count}</div>
                    <div className="num">Stab {pct(s.overall_stability_score)}</div>
                    <div>
                      <span className={`tag ${s.quality_score >= 0.8 ? 'ok' : s.quality_score >= 0.6 ? 'warn' : 'bad'}`}>
                        Q {pct(s.quality_score)}
                      </span>
                    </div>
                    <div className="muted">›</div>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
