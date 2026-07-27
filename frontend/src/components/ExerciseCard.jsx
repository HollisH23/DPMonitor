// Interactive Exercise Card.
//
// Clicking the card opens the Live Monitoring View for the selected
// exercise. A "Tutorial" affordance opens a modal with text/icon cues so
// the patient can review correct form before starting — fulfilling the
// plan's "standardized exercise tutorials" requirement without needing
// real video assets at MVP.

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function ExerciseCard({ exercise, completedThisWeek = 0 }) {
  const navigate = useNavigate();
  const [showTutorial, setShowTutorial] = useState(false);

  function start() {
    navigate(`/monitor?exercise=${exercise.key}`);
  }

  return (
    <div className="ex-card">
      <button className="ex-card-body" onClick={start} aria-label={`Start ${exercise.name}`}>
        <div className="ex-icon" aria-hidden>{exercise.icon}</div>
        <div className="ex-text">
          <div className="ex-name">{exercise.name}</div>
          <div className="ex-summary">{exercise.summary}</div>
          <div className="ex-meta">
            <span className="tag">Target {exercise.targetReps} reps</span>
            <span className={`tag ${completedThisWeek > 0 ? 'ok' : ''}`}>
              {completedThisWeek > 0
                ? `${completedThisWeek}× this week`
                : 'Not yet this week'}
            </span>
          </div>
        </div>
      </button>
      <div className="ex-actions">
        <button className="ghost small" onClick={() => setShowTutorial(true)}>Tutorial</button>
        <button className="primary small" onClick={start}>Start</button>
      </div>

      {showTutorial && (
        <div className="modal-backdrop" onClick={() => setShowTutorial(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-h">
              <div className="card-title">{exercise.icon} {exercise.name} — How to perform</div>
              <button className="ghost small" onClick={() => setShowTutorial(false)}>Close</button>
            </div>
            <div className="muted small">{exercise.summary}</div>
            <ol className="cue-list">
              {exercise.cues.map((c, i) => <li key={i}>{c}</li>)}
            </ol>
            <div className="contra">
              <span className="tag warn">Heads up</span>{' '}{exercise.contraindications}
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button className="ghost" onClick={() => setShowTutorial(false)}>Got it</button>
              <button className="primary" onClick={() => { setShowTutorial(false); start(); }}>
                Start session
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
