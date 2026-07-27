// StabilityChart — tiny rolling line chart of the movement smoothness
// waveform (quality_score over time). Pure canvas, no charting deps.
import { useEffect, useRef } from 'react';

export default function StabilityChart({ samples, max = 180 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    const { clientWidth: w, clientHeight: h } = c;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    ctx.clearRect(0, 0, w, h);

    // baseline grid
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const y = (h * i) / 4;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const data = samples.slice(-max);
    if (data.length < 2) return;

    // Gradient under the curve.
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(102,224,194,0.35)');
    grad.addColorStop(1, 'rgba(102,224,194,0)');

    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (max - 1)) * w;
      const y = h - v * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line.
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (max - 1)) * w;
      const y = h - v * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#66e0c2';
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [samples, max]);

  return (
    <div className="stability">
      <h4>Stability</h4>
      <canvas ref={ref} />
    </div>
  );
}
