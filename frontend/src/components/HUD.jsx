// Large, high-contrast rep counter pinned to the viewport's top-left.
export default function HUD({ count }) {
  return (
    <div className="hud">
      <div className="count-label">Reps</div>
      <div className="count">{String(count).padStart(2, '0')}</div>
    </div>
  );
}
