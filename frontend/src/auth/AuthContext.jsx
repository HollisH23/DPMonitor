// Phase 2.1 — App-wide auth context.
//
// Holds the token + user info. The token is persisted to localStorage so
// a page reload keeps the user signed in. On 401 from any REST call, the
// API client triggers a session reset which clears state here and the
// router redirects back to /login.

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

import { api } from '../api/client.js';

const STORAGE_KEY = 'rehab.auth.v1';

const AuthCtx = createContext(null);

function readStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}
function writeStored(v) {
  try {
    if (v) localStorage.setItem(STORAGE_KEY, JSON.stringify(v));
    else   localStorage.removeItem(STORAGE_KEY);
  } catch { /* private mode */ }
}

export function AuthProvider({ children }) {
  // { token, user } | null
  const [auth, setAuth] = useState(() => readStored());
  const [hydrating, setHydrating] = useState(!!readStored());

  // Push the current token into the API client so subsequent calls carry it.
  useEffect(() => {
    api.setToken(auth?.token || null);
    writeStored(auth);
  }, [auth]);

  // On boot, if we have a stored token, validate it via /auth/me/ — that
  // way we drop stale tokens (e.g. user revoked from another tab) early.
  useEffect(() => {
    let cancelled = false;
    async function verify() {
      if (!auth?.token) { setHydrating(false); return; }
      try {
        const { user } = await api.me();
        if (!cancelled) setAuth((a) => a ? { ...a, user } : a);
      } catch (e) {
        if (!cancelled && e.status === 401) setAuth(null);
      } finally {
        if (!cancelled) setHydrating(false);
      }
    }
    verify();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When ANY REST call comes back 401, drop the token so the router can
  // bounce the user back to /login.
  useEffect(() => {
    api.onUnauthorized = () => setAuth(null);
    return () => { api.onUnauthorized = null; };
  }, []);

  const login = useCallback(async (username, password) => {
    const { token, user } = await api.login(username, password);
    setAuth({ token, user });
  }, []);

  const register = useCallback(async (username, password, email) => {
    const { token, user } = await api.register(username, password, email);
    setAuth({ token, user });
  }, []);

  const logout = useCallback(async () => {
    try { await api.logout(); } catch { /* best-effort */ }
    setAuth(null);
  }, []);

  const value = useMemo(() => ({
    token: auth?.token || null,
    user: auth?.user || null,
    isAuthenticated: !!auth?.token,
    hydrating,
    login,
    register,
    logout,
  }), [auth, hydrating, login, register, logout]);

  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>;
}

export function useAuth() {
  const v = useContext(AuthCtx);
  if (!v) throw new Error('useAuth must be used inside <AuthProvider>');
  return v;
}
