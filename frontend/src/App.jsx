// Phase 2.1 — App shell with authenticated routing.
//
// All clinician-side patient management has been removed. The router
// has a single public route (/login) and an authenticated group that
// renders the patient-self-service shell.

import { NavLink, Navigate, Outlet, Route, Routes, useNavigate } from 'react-router-dom';

import { AuthProvider, useAuth } from './auth/AuthContext.jsx';
import RequireAuth from './auth/RequireAuth.jsx';
import Dashboard from './pages/Dashboard.jsx';
import LiveMonitor from './pages/LiveMonitor.jsx';
import Login from './pages/Login.jsx';
import ReportView from './pages/ReportView.jsx';

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<RequireAuth />}>
          <Route element={<AuthedShell />}>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/monitor" element={<LiveMonitor />} />
            <Route path="/sessions/:sessionId" element={<ReportView />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Route>
        </Route>
      </Routes>
    </AuthProvider>
  );
}

function AuthedShell() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  async function onLogout() {
    await logout();
    navigate('/login', { replace: true });
  }

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden>◉</div>
          <div className="brand-text">
            <div className="brand-title">Rehab Monitor</div>
            <div className="brand-sub">Local · Private · Deterministic</div>
          </div>
        </div>
        <nav className="nav">
          <NavLink to="/dashboard" className="nav-link">Dashboard</NavLink>
          <NavLink to="/monitor" className="nav-link">Live Monitor</NavLink>
        </nav>
        <div className="sidebar-footer">
          {user && (
            <div className="user-pill">
              <div className="user-avatar" aria-hidden>{(user.username || '?').slice(0, 1).toUpperCase()}</div>
              <div className="user-meta">
                <div className="user-name">{user.username}</div>
                <button className="ghost small" onClick={onLogout}>Sign out</button>
              </div>
            </div>
          )}
          <span className="edge-pill">
            <span className="edge-dot" /> Local Data Processing
          </span>
        </div>
      </aside>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
