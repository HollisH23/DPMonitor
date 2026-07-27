import { useEffect, useRef } from 'react';

import { POSE_EDGES } from '../lib/poseUtils.js';

// Renders the skeletal overlay (joints + bones) on top of the video.
// Color reflects movement quality: green = standard, amber = caution,
// red = compensatory. The video itself is rendered by the parent.
export default function SkeletonOverlay({ points, qualityScore, isCompensatory }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Match canvas pixel size to its on-screen size for crisp lines.
    const { clientWidth: w, clientHeight: h } = canvas;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx.clearRect(0, 0, w, h);

    if (!points) return;

    let color = '#66e0c2'; // ok
    if (isCompensatory) color = '#ef6f6c';
    else if (qualityScore != null && qualityScore < 0.7) color = '#f5b740';

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;

    // Bones.
    for (const [a, b] of POSE_EDGES) {
      const pa = points[a]; const pb = points[b];
      if (!pa || !pb) continue;
      if (pa[3] < 0.3 || pb[3] < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(pa[0] * w, pa[1] * h);
      ctx.lineTo(pb[0] * w, pb[1] * h);
      ctx.stroke();
    }
    // Joints.
    ctx.shadowBlur = 12;
    for (const k of Object.keys(points)) {
      const p = points[k];
      if (p[3] < 0.3) continue;
      ctx.beginPath();
      ctx.arc(p[0] * w, p[1] * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [points, qualityScore, isCompensatory]);

  return <canvas ref={canvasRef} className="overlay" aria-hidden />;
}
