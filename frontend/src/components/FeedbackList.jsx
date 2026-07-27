// Scrolling feed of analyzer-generated correction cues.
import { useEffect, useRef } from 'react';

function classify(cue) {
  const s = cue.toLowerCase();
  if (s.includes('rep counted') || s.includes('target')) return 'ok';
  if (s.includes('keep') || s.includes('slow')) return 'warn';
  if (s.includes('collapse') || s.includes('compensat')) return 'bad';
  return '';
}

export default function FeedbackList({ items }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [items]);

  return (
    <div className="feedback" ref={ref}>
      <h4>Live Feedback</h4>
      {items.length === 0 && <div className="muted">Awaiting movement…</div>}
      {items.map((cue, i) => (
        <div key={i} className={`cue ${classify(cue)}`}>{cue}</div>
      ))}
    </div>
  );
}
