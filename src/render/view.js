/**
 * Renderer. DIE EINZIGE DOM-abhängige Schicht.
 *
 * src/physics, src/world, src/body und src/sim sind DOM-frei (tools/harness.js
 * prüft das als Check 9). Deshalb läuft derselbe Kern unverändert im Browser,
 * im Worker und headless — der Worker-Split ist kein Retrofit.
 */
import { tipPosition, BODY } from '../body/kraken.js';

export function createView(canvas, sim) {
  const ctx = canvas.getContext('2d');
  let scale = 1, ox = 0, oy = 0;

  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    scale = Math.min(canvas.width / sim.map.width, canvas.height / sim.map.height);
    ox = (canvas.width - sim.map.width * scale) / 2;
    oy = (canvas.height - sim.map.height * scale) / 2;
  }
  resize();
  window.addEventListener('resize', resize);

  function draw() {
    const w = sim.world;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#000409';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(scale, 0, 0, scale, ox, oy);

    // Karte
    ctx.fillStyle = '#0a1a12';
    for (const r of sim.map.rects) ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeStyle = '#123a26'; ctx.lineWidth = 1 / scale;
    for (const r of sim.map.rects) ctx.strokeRect(r.x, r.y, r.w, r.h);

    // Kraken
    for (const c of sim.creatures) {
      drawCreature(ctx, w, c);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  function drawCreature(ctx, w, c) {
    const col = c.index === 0 ? '#00ffb3' : `hsl(${(c.index * 67) % 360},90%,60%)`;
    // Arme als geglättete, sich verjüngende Züge
    for (const arm of c.arms) {
      const anchored = !arm.grip.broken;
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      for (let i = 0; i < arm.segs.length - 1; i++) {
        const t = i / (arm.segs.length - 1);
        ctx.strokeStyle = anchored ? 'rgba(255,221,0,0.95)' : col;
        ctx.globalAlpha = anchored ? 0.9 : 0.75;
        ctx.lineWidth = BODY.segThickProx + (BODY.segThickDist - BODY.segThickProx) * t;
        ctx.beginPath();
        ctx.moveTo(w.px[arm.segs[i]], w.py[arm.segs[i]]);
        ctx.lineTo(w.px[arm.segs[i + 1]], w.py[arm.segs[i + 1]]);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      // Saugnapf
      const tp = tipPosition(w, arm);
      if (anchored) {
        const load = Math.min(1, (arm.grip.force || 0) / Math.max(1, arm.grip.breakForce));
        ctx.strokeStyle = `rgba(255,${Math.floor(221 - load * 180)},0,0.9)`;
        ctx.lineWidth = 1.5 / scale * 1.5;
        ctx.beginPath(); ctx.arc(tp.x, tp.y, 5 + load * 4, 0, Math.PI * 2); ctx.stroke();
        // Ankerlinie
        ctx.strokeStyle = 'rgba(255,221,0,0.25)'; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(tp.x, tp.y); ctx.lineTo(arm.grip.lbx, arm.grip.lby); ctx.stroke();
        ctx.setLineDash([]);
      } else {
        ctx.fillStyle = 'rgba(0,255,179,0.35)';
        ctx.beginPath(); ctx.arc(tp.x, tp.y, 3, 0, Math.PI * 2); ctx.fill();
      }
    }
    // Mantel
    const m = c.mantle;
    const g = ctx.createRadialGradient(w.px[m], w.py[m], 1, w.px[m], w.py[m], BODY.mantleRadius);
    g.addColorStop(0, col);
    g.addColorStop(0.6, 'rgba(0,60,45,0.95)');
    g.addColorStop(1, '#00120c');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(w.px[m], w.py[m], BODY.mantleRadius, 0, Math.PI * 2); ctx.fill();
    // Blickrichtung (Augen liegen bei lokalem -y)
    const fa = w.ang[m] - Math.PI / 2;
    ctx.fillStyle = '#000';
    for (const s of [-6, 6]) {
      const ex = w.px[m] + Math.cos(w.ang[m]) * s - Math.sin(w.ang[m]) * -5;
      const ey = w.py[m] + Math.sin(w.ang[m]) * s + Math.cos(w.ang[m]) * -5;
      ctx.beginPath(); ctx.arc(ex, ey, 2.6, 0, Math.PI * 2); ctx.fill();
    }
    void fa;
  }

  return { draw, resize };
}
