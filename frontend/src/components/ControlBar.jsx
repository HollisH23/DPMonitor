// Bottom control bar: action buttons (Start/Pause, Recalibrate, Finish) +
// status badges (WebSocket, Random Seed, Local Edge Computing).
import { STATES } from '../hooks/useSessionMachine.js';

export default function ControlBar({
  state,
  onStart,
  onPause,
  onResume,
  onRecalibrate,
  onFinish,
  wsStatus,    // 'connecting' | 'open' | 'closed'
  seed,
  ready,
}) {
  const wsTone =
    wsStatus === 'open' ? 'ok' :
    wsStatus === 'connecting' ? 'warn' : 'bad';

  return (
    <div className="controlbar">
      <div className="actions">
        {state === STATES.IDLE && (
          <button className="primary" onClick={onStart}>Start Session</button>
        )}
        {state === STATES.CALIBRATING && (
          <button className="primary" disabled={!ready} onClick={onStart}>
            Begin Recording
          </button>
        )}
        {state === STATES.ACTIVE && (
          <button onClick={onPause}>Pause</button>
        )}
        {state === STATES.PAUSED && (
          <button className="primary" onClick={onResume}>Resume</button>
        )}
        <button className="ghost" onClick={onRecalibrate}
                disabled={state === STATES.IDLE || state === STATES.COMPLETED}>
          Recalibrate
        </button>
        <button className="primary" onClick={onFinish}
                disabled={state !== STATES.ACTIVE && state !== STATES.PAUSED}>
          Finish &amp; Generate Report
        </button>
      </div>
      <div className="status">
        <span className="status-item">
          <span className={`status-dot ${wsTone}`} /> WebSocket {wsStatus}
        </span>
        <span className="status-item">
          <span className="status-dot ok" /> Seed {seed}
        </span>
        <span className="status-item" title="All processing happens on this machine.">
          <span className="status-dot ok" /> Local Edge Computing
        </span>
      </div>
    </div>
  );
}
