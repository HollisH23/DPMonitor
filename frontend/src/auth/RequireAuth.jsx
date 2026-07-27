// Route guard. Wraps protected routes; renders <Outlet /> when the user
// is authenticated, else redirects to /login (preserving the intended URL
// so we can bounce them back on successful login).
import { Navigate, Outlet, useLocation } from 'react-router-dom';

import { useAuth } from './AuthContext.jsx';

export default function RequireAuth() {
  const { isAuthenticated, hydrating } = useAuth();
  const location = useLocation();
  if (hydrating) {
    return <div className="card" style={{ margin: 40 }}>Verifying session…</div>;
  }
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return <Outlet />;
}
