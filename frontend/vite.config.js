import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Default dev server on :5173 (matches backend CORS_ALLOWED_ORIGINS).
// REST and WebSocket calls hit http://localhost:8000 by default, overridable
// via VITE_API_BASE / VITE_WS_BASE env vars.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false
  }
});
