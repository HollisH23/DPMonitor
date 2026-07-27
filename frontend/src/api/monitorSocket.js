// MonitorSocket (Phase 2.1) — token-authenticated WebSocket client.
//
// Browsers cannot set custom headers on `new WebSocket(...)`, so the
// token is passed as a query-string parameter. The backend's
// `TokenAuthMiddleware` validates it and rejects unauthenticated
// connections with close code 4401.
//
// Lifecycle:
//   const sock = new MonitorSocket({ token, onResult, onSummary, onReady, onError, onStatus });
//   sock.connect();
//   sock.helloOnceOpen({ sessionSeed, exerciseType });
//   sock.sendFrame({ frame_index, timestamp_ms, points, angles });
//   sock.bye();
//   sock.close();

const WS_BASE = import.meta.env.VITE_WS_BASE || 'ws://localhost:8000/ws';

export class MonitorSocket {
  constructor({ token, onResult, onSummary, onReady, onError, onStatus } = {}) {
    if (!token) throw new Error('MonitorSocket requires a token.');
    this.url = `${WS_BASE}/monitor/?token=${encodeURIComponent(token)}`;
    this.onResult = onResult || (() => {});
    this.onSummary = onSummary || (() => {});
    this.onReady = onReady || (() => {});
    this.onError = onError || (() => {});
    this.onStatus = onStatus || (() => {}); // 'connecting' | 'open' | 'closed' | 'unauthorized'
    this.ws = null;
  }

  connect() {
    this.onStatus('connecting');
    try {
      this.ws = new WebSocket(this.url);
    } catch (e) {
      this.onError(e);
      this.onStatus('closed');
      return;
    }
    this.ws.addEventListener('open',  () => this.onStatus('open'));
    this.ws.addEventListener('close', (ev) => {
      // 4401 = token rejected by middleware (see consumers.py).
      this.onStatus(ev.code === 4401 ? 'unauthorized' : 'closed');
    });
    this.ws.addEventListener('error', (e) => this.onError(e));
    this.ws.addEventListener('message', (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === 'ready')   this.onReady(msg);
      else if (msg.type === 'result')  this.onResult(msg);
      else if (msg.type === 'summary') this.onSummary(msg);
      else if (msg.type === 'error')   this.onError(new Error(msg.detail));
    });
  }

  helloOnceOpen({ sessionSeed, exerciseType }) {
    const payload = { type: 'hello', session_seed: sessionSeed, exercise_type: exerciseType };
    if (this.isOpen()) { this._send(payload); return; }
    if (!this.ws) return;
    const onOpen = () => {
      this.ws.removeEventListener('open', onOpen);
      this._send(payload);
    };
    this.ws.addEventListener('open', onOpen);
  }

  sendFrame(payload) { this._send({ type: 'frame', ...payload }); }
  bye()              { this._send({ type: 'bye' }); }
  close()            { if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.close(); }
  isOpen()           { return !!this.ws && this.ws.readyState === WebSocket.OPEN; }

  _send(obj) {
    if (this.isOpen()) this.ws.send(JSON.stringify(obj));
  }
}
