// useSessionMachine — strict five-state machine for the Live Monitor.
//
//   IDLE → CALIBRATING → ACTIVE ↔ PAUSED → COMPLETED
//
// Transitions are guarded so that, e.g., you can't go ACTIVE → IDLE without
// passing through COMPLETED. The state name doubles as a CSS class for the
// chip rendered over the video.

import { useCallback, useState } from 'react';

export const STATES = Object.freeze({
  IDLE: 'IDLE',
  CALIBRATING: 'CALIBRATING',
  ACTIVE: 'ACTIVE',
  PAUSED: 'PAUSED',
  COMPLETED: 'COMPLETED',
});

const ALLOWED = {
  IDLE: ['CALIBRATING'],
  CALIBRATING: ['ACTIVE', 'IDLE'],
  ACTIVE: ['PAUSED', 'COMPLETED'],
  PAUSED: ['ACTIVE', 'COMPLETED'],
  COMPLETED: ['IDLE'],
};

export function useSessionMachine(initial = STATES.IDLE) {
  const [state, setState] = useState(initial);

  const transition = useCallback((next) => {
    setState((current) => {
      if (current === next) return current;
      const allowed = ALLOWED[current] || [];
      if (!allowed.includes(next)) {
        // Silently ignore disallowed transitions — UI should make them
        // unreachable via disabled buttons anyway.
        console.warn(`[session] illegal transition ${current} → ${next}`);
        return current;
      }
      return next;
    });
  }, []);

  return { state, transition };
}
