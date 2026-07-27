// Token-aware REST client (Phase 2.1).
//
// Holds a singleton token state. The AuthContext is responsible for
// calling `api.setToken(...)` whenever the user logs in/out. On any 401
// from the backend, we fire `api.onUnauthorized()` so the AuthContext
// can clear its state and the router can bounce back to /login.

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api';

let _token = null;

async function request(method, path, body, { auth = true } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (auth && _token) headers.Authorization = `Token ${_token}`;

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail;
    try { detail = await res.json(); } catch { detail = await res.text(); }
    const err = new Error(
      `HTTP ${res.status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`
    );
    err.status = res.status;
    err.detail = detail;
    if (res.status === 401 && typeof api.onUnauthorized === 'function') {
      // Defer to next tick so callers see the rejection first.
      setTimeout(() => api.onUnauthorized && api.onUnauthorized(), 0);
    }
    throw err;
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  // Mutable token slot. AuthContext keeps this in sync.
  setToken(t)  { _token = t || null; },
  getToken()   { return _token; },
  // Optional hook used by AuthContext for global 401 handling.
  onUnauthorized: null,

  // ---- auth ----
  health:   ()                       => request('GET',  '/health/', undefined, { auth: false }),
  register: (username, password, email) =>
    request('POST', '/auth/register/', { username, password, email }, { auth: false }),
  login:    (username, password)     => request('POST', '/auth/login/',  { username, password }, { auth: false }),
  logout:   ()                       => request('POST', '/auth/logout/'),
  me:       ()                       => request('GET',  '/auth/me/'),

  // ---- sessions (owner-scoped server-side) ----
  listSessions:   ()        => request('GET',    '/sessions/'),
  getSession:     (id)      => request('GET',    `/sessions/${id}/`),
  ingestSession:  (payload) => request('POST',   '/sessions/ingest/', payload),
  deleteSession:  (id)      => request('DELETE', `/sessions/${id}/`),

  // ---- dashboard support ----
  trend7: () => request('GET', '/trend/'),
};

export const API_BASE_URL = API_BASE;
