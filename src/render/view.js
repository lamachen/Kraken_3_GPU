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

  // ── Kamera & Spur (Optionen, per API gesetzt) ──────────────────────
  // setFollow(idx, zoom): Kamera zentriert auf Tier idx und zoomt. Ohne das
  // ist die Drift des Platzhalter-Gangs (~4–6 px/s, tools/demogait-probe.mjs)
  // auf der 1440-px-Karte kaum sichtbar — das Tier wirkt wie ein Kreisel.
  // setTrail(true): Mantel-Spur + Spawn-Marke, damit Netto-Bewegung gegen
  // den Startpunkt sichtbar wird (Pfadlänge vs. Luftlinie).
  let followIdx = -1, followZoom = 1, trailOn = false;
  const TRAIL_MAX = 900;             // Punkte pro Tier (≈15 s bei 60 fps)
  const trails = new Map();          // Tier → flaches [x,y,x,y,…]-Array
  let cur = 1;                       // wirksamer Transform-Scale im draw()

  function setFollow(idx, zoom) { followIdx = (idx === null || idx === undefined) ? -1 : idx | 0; followZoom = zoom || 1; }
  function setTrail(on) { trailOn = !!on; if (!on) trails.clear(); }

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

  /** Spur-Punkte nachführen; Teleports (Respawn) löschen die Spur des Tiers. */
  function updateTrails(w) {
    const many = sim.creatures.length > 8;
    for (const c of sim.creatures) {
      if (many && c.index !== 0 && c.index !== followIdx) { trails.delete(c); continue; }
      const x = w.px[c.mantle], y = w.py[c.mantle];
      let t = trails.get(c);
      if (!t) { t = []; trails.set(c, t); }
      const n = t.length;
      if (n >= 2 && Math.hypot(x - t[n - 2], y - t[n - 1]) > 250) t.length = 0;  // Respawn
      t.push(x, y);
      if (t.length > TRAIL_MAX * 2) t.splice(0, t.length - TRAIL_MAX * 2);
    }
  }

  function drawTrails(w) {
    for (const c of sim.creatures) {
      const t = trails.get(c);
      if (!t || t.length < 4) continue;
      const col = c.index === 0 ? '0,255,179' : `${hslRgb(c.index)}`;
      // alte Spur dunkel, jüngstes Zehntel hell → Bewegungsrichtung lesbar
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.strokeStyle = `rgba(${col},0.16)`; ctx.lineWidth = 2 / cur;
      ctx.beginPath(); ctx.moveTo(t[0], t[1]);
      for (let i = 2; i < t.length; i += 2) ctx.lineTo(t[i], t[i + 1]);
      ctx.stroke();
      const from = Math.max(0, t.length - Math.floor(t.length / 5) * 2);
      ctx.strokeStyle = `rgba(${col},0.55)`;
      ctx.beginPath(); ctx.moveTo(t[from], t[from + 1]);
      for (let i = from + 2; i < t.length; i += 2) ctx.lineTo(t[i], t[i + 1]);
      ctx.stroke();
      // Spawn-Marke: kleines Kreuz — der Bezugspunkt der Netto-Drift
      const sp = c.spawn;
      if (sp) {
        ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1 / cur;
        ctx.beginPath();
        ctx.moveTo(sp.x - 6, sp.y); ctx.lineTo(sp.x + 6, sp.y);
        ctx.moveTo(sp.x, sp.y - 6); ctx.lineTo(sp.x, sp.y + 6);
        ctx.stroke();
      }
    }
  }
  // hsl() aus drawCreature für die Spur als RGB-Tripel (Canvas-rgba braucht das)
  function hslRgb(index) {
    const h = ((index * 67) % 360) / 360, s = 0.9, l = 0.6;
    const f = (n) => {
      const k = (n + h * 12) % 12;
      const a = s * Math.min(l, 1 - l);
      return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, Math.min(9 - k, 1)))));
    };
    return `${f(0)},${f(8)},${f(4)}`;
  }

  function draw() {
    const w = sim.world;
    // wirksame Kamera: Übersicht oder Follow
    let sc = scale, tx = ox, ty = oy;
    if (followIdx >= 0) {
      const fc = sim.creatures[followIdx];
      if (fc) {
        sc = scale * followZoom;
        tx = canvas.width / 2 - w.px[fc.mantle] * sc;
        ty = canvas.height / 2 - w.py[fc.mantle] * sc;
      }
    }
    cur = sc;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#000409';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(sc, 0, 0, sc, tx, ty);
    if (trailOn) { updateTrails(w); drawTrails(w); }

    // Karte
    ctx.fillStyle = '#0a1a12';
    for (const r of sim.map.rects) ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeStyle = '#123a26'; ctx.lineWidth = 1 / cur;
    for (const r of sim.map.rects) ctx.strokeRect(r.x, r.y, r.w, r.h);

    // Lichtquellen (Sehen ist intermittierend — die Quellen wandern)
    ctx.globalCompositeOperation = 'screen';
    for (const l of sim.map.lights || []) {
      const g = ctx.createRadialGradient(l.x, l.y, 0, l.x, l.y, l.r * 0.35);
      g.addColorStop(0, `rgba(${l.c[0]},${l.c[1]},${l.c[2]},0.16)`);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(l.x, l.y, l.r * 0.35, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';

    // Entitäten (Beute): Körper + Duft-Aureole
    for (const e of sim.entities || []) {
      if (!e.alive) continue;
      ctx.strokeStyle = 'rgba(255,105,25,0.18)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(e.x, e.y, 26, 0, Math.PI * 2); ctx.stroke();
      if (e.noise > 0.02) {   // hörbar → Ring, Radius ~ Lautstärke
        ctx.strokeStyle = `rgba(120,180,255,${(0.12 + e.noise * 0.35).toFixed(3)})`;
        ctx.beginPath(); ctx.arc(e.x, e.y, e.radius + 4 + e.noise * 22, 0, Math.PI * 2); ctx.stroke();
      }
      const rg = ctx.createRadialGradient(e.x - 3, e.y - 3, 1, e.x, e.y, e.radius);
      rg.addColorStop(0, '#ffcc44'); rg.addColorStop(0.5, '#ff6600'); rg.addColorStop(1, '#2a0800');
      ctx.fillStyle = rg;
      ctx.beginPath(); ctx.arc(e.x, e.y, e.radius, 0, Math.PI * 2); ctx.fill();
    }

    // Kraken
    for (const c of sim.creatures) {
      drawEcho(ctx, c);
      drawCreature(ctx, w, c);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  /**
   * Echolot visualisieren: die Strahlen, deren Echo bereits EINGETROFFEN ist.
   * Bewusst nicht die ausgesendeten Strahlen — das Tier sieht die Vergangenheit,
   * gestaffelt nach Entfernung. Genau das ist die Einschränkung, die das Netz
   * lernen muss zu kompensieren.
   */
  function drawEcho(ctx, c) {
    const e = c.echo;
    const h = c.heading !== undefined ? c.heading : (w_ang(c) - Math.PI / 2);
    for (let i = 0; i < e.rays; i++) {
      if (e.range_[i] <= 0) continue;
      const t = e.rays > 1 ? i / (e.rays - 1) : 0.5;
      const a = h + (t - 0.5) * e.fan;
      const d = e.range_[i] * e.range;
      const x = sim.creatures[0] === c ? c.x : c.x, y = c.y;
      const ex = x + Math.cos(a) * d, ey = y + Math.sin(a) * d;
      const alpha = 0.12 + e.intensity[i] * 0.5;
      ctx.strokeStyle = `rgba(192,80,255,${alpha.toFixed(3)})`;
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.fillStyle = `rgba(220,140,255,${Math.min(0.95, alpha + 0.25).toFixed(3)})`;
      ctx.beginPath(); ctx.arc(ex, ey, 1.6 + e.intensity[i] * 2.4, 0, Math.PI * 2); ctx.fill();
    }
    // Aussendungs-Blitz
    if (c.lastEchoEmit !== undefined && sim.time - c.lastEchoEmit < 0.07) {
      ctx.strokeStyle = 'rgba(220,160,255,0.75)'; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(c.x, c.y, 20 + (sim.time - c.lastEchoEmit) * 900, 0, Math.PI * 2); ctx.stroke();
    }
  }
  function w_ang(c) { return sim.world.ang[c.mantle]; }

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
        ctx.lineWidth = 1.5 / cur * 1.5;
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

  return { draw, resize, setFollow, setTrail };
}
