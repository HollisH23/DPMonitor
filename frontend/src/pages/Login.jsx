// Phase 2.1 — Login / Register entry point.
//
// Tabbed UI: Sign in (default) and Create account. On success the
// AuthContext fills with token+user and the user is redirected to the
// dashboard (or to wherever they were headed before being bounced).

import { useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';

import { useAuth } from '../auth/AuthContext.jsx';

export default function Login() {
  const { isAuthenticated, login, register } = useAuth();
  const location = useLocation();
  const redirectTo = location.state?.from?.pathname || '/dashboard';

  const [mode, setMode] = useState('login'); // 'login' | 'register'
  const [form, setForm] = useState({ username: '', password: '', email: '' });
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  if (isAuthenticated) return <Navigate to={redirectTo} replace />;

  function set(k, v) { setForm((f) => ({ ...f, [k]: v })); }

  async function onSubmit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      if (mode === 'login') await login(form.username.trim(), form.password);
      else await register(form.username.trim(), form.password, form.email.trim());
    } catch (e) {
      setError(e.detail?.detail || e.detail || e.message || 'Something went wrong.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-screen">
      <div className="auth-card card">
        <div className="auth-brand">
          <div className="brand-mark" aria-hidden>◉</div>
          <div>
            <div className="brand-title">Rehab Monitor</div>
            <div className="brand-sub">Local · Private · Deterministic</div>
          </div>
        </div>
        <div className="auth-tabs">
          <button type="button"
                  className={mode === 'login' ? 'primary' : 'ghost'}
                  onClick={() => { setMode('login'); setError(null); }}>
            Sign in
          </button>
          <button type="button"
                  className={mode === 'register' ? 'primary' : 'ghost'}
                  onClick={() => { setMode('register'); setError(null); }}>
            Create account
          </button>
        </div>
        <form className="form" onSubmit={onSubmit}>
          <div>
            <label>Username</label>
            <input required autoFocus autoComplete="username"
                   value={form.username}
                   onChange={(e) => set('username', e.target.value)} />
          </div>
          <div>
            <label>Password</label>
            <input type="password" required minLength={6}
                   autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                   value={form.password}
                   onChange={(e) => set('password', e.target.value)} />
          </div>
          {mode === 'register' && (
            <div>
              <label>Email (optional)</label>
              <input type="email" autoComplete="email"
                     value={form.email}
                     onChange={(e) => set('email', e.target.value)} />
            </div>
          )}
          {error && <div className="tag bad" style={{ alignSelf: 'flex-start' }}>{String(error)}</div>}
          <button className="primary" type="submit" disabled={submitting}>
            {submitting
              ? (mode === 'login' ? 'Signing in…' : 'Creating account…')
              : (mode === 'login' ? 'Sign in' : 'Create account')}
          </button>
        </form>
        <div className="auth-foot edge-pill">
          <span className="edge-dot" /> All data stays on your machine
        </div>
      </div>
    </div>
  );
}
