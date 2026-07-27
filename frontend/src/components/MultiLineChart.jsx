// MultiLineChart — generic multi-series time chart for the Clinical Report.
// Each series is { label, color, data:number[], yMin, yMax }. yRange is taken
// from the first series's yMin/yMax (or auto-computed). Pure canvas.

import { useEffect, useRef } from 'react';

export default function MultiLineChart({ series, height = 220, yLabel }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    const { clientWidth: w, clientHeight: h } = c;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    ctx.clearRect(0, 0, w, h);

    const all = series.flatMap((s) => s.data || []);
    if (all.length < 2) return;
    let yMin = series[0]?.yMin ?? Math.min(...all);
    let yMax = series[0]?.yMax ?? Math.max(...all);
    if (yMax === yMin) yMax = yMin + 1;
    const pad = 8;
    const drawH = h - pad * 2;

    // Gridlines.
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 5; i++) {
      const y = (h * i) / 5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    for (const s of series) {
      const data = s.data || [];
      if (data.length < 2) continue;
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = (i / (data.length - 1)) * w;
        const y = pad + ((yMax - v) / (yMax - yMin)) * drawH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = s.color || '#66e0c2';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Legend.
    let lx = 10, ly = 14;
    ctx.font = '11px ui-monospace, monospace';
    for (const s of series) {
      ctx.fillStyle = s.color || '#66e0c2';
      ctx.fillRect(lx, ly - 8, 10, 10);
      ctx.fillStyle = '#aab4c2';
      ctx.fillText(s.label, lx + 14, ly);
      lx += ctx.measureText(s.label).width + 36;
    }
    if (yLabel) {
      ctx.fillStyle = '#7d8898';
      ctx.fillText(yLabel, w - ctx.measureText(yLabel).width - 8, 14);
    }
  }, [series, height, yLabel]);
  return <canvas ref={ref} style={{ height }} />;
}
