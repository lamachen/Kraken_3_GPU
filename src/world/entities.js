/**
 * Entitäten: alles, was nicht der Krake ist und wahrgenommen werden kann.
 *
 * Bewusst minimal gehalten — Milestone 2 braucht etwas, das man riechen und
 * hören kann. Milestone 4 (Gegner) baut darauf auf: ein Bewohner-Agent ist
 * dieselbe Struktur mit einer Policy statt eines Fluchtverhaltens.
 *
 * Jede Entität emittiert:
 *   scent  — Duftstärke (Chemorezeption, träge und langlebig)
 *   noise  — Geräusch (passives Gehör, skaliert mit Tempo)
 *   light  — ob sie selbst leuchtet (Photorezeption)
 *
 * ── Die gegenseitige Wahrnehmung, die in v2 gebaut, aber nie benutzt wurde ──
 * Dort emittierte der Spieler `CHEM.dep(…,0.016)` (4× stärker als die Krake),
 * `emitSound(…,0.25·v)` proportional zum Tempo und `emitSound(mx,my,2.0)` beim
 * Angriff — und `stepPhysics` hatte null Referenzen auf den Spieler. Die ganze
 * Mechanik lief mit, stand aber unter keinem Selektionsdruck. Hier sind die
 * Emissionen Teil der Entitäts-Schnittstelle, damit sie nicht wieder verloren
 * gehen können.
 */

export const KIND_PREY = 0;
export const KIND_INTRUDER = 1;

export function makeEntity(kind, x, y, opts = {}) {
  // Determinismus: heading/seed MÜSSEN aus einem benannten RNG kommen. Mit
  // Math.random() waren zwei createSim-Aufrufe mit gleichem Seed verschieden
  // (gefangen vom gputrain-passthrough-Check: Beute driftete ab Schritt 85,
  // sobald sie vom deterministischen Flucht- ins Zufalls-Wandern wechselte).
  // Der Math.random-Fallback bleibt nur für Aufrufer ohne Sim-Kontext.
  const r = opts.rnd || Math.random;
  return {
    kind,
    x, y, vx: 0, vy: 0,
    radius: opts.radius !== undefined ? opts.radius : 9,
    speed: opts.speed !== undefined ? opts.speed : 60,
    panic: 0,
    scent: opts.scent !== undefined ? opts.scent : 1.0,
    noise: 0,
    emitsLight: !!opts.emitsLight,
    lightRadius: opts.lightRadius || 0,
    seed: r() * 100,
    wanderT: 0,
    heading: r() * Math.PI * 2,
    alive: true,
  };
}

/**
 * Entitäten fortschreiben. `threats` sind Positionen, vor denen geflohen wird.
 *
 * Tempo erzeugt Geräusch — genau wie in v2: ein sprintender Spieler ist hörbar,
 * ein schleichender nicht. Das ist der Trade-off, der Tempo zu einer
 * Entscheidung macht statt zu einer reinen Fortbewegungsgröße.
 */
export function tickEntity(e, dt, width, height, threats, rnd) {
  if (!e.alive) return;
  let fx = 0, fy = 0;
  for (const t of threats || []) {
    if (!t) continue;
    const dx = e.x - t.x, dy = e.y - t.y;
    const d = Math.hypot(dx, dy);
    if (d < 1 || d > 260) continue;
    e.panic = Math.min(1, e.panic + dt * 1.6);
    fx += dx / d; fy += dy / d;
  }
  e.panic = Math.max(0, e.panic - dt * 0.4);

  e.wanderT -= dt;
  if (e.wanderT <= 0) {
    e.heading += (rnd() - 0.5) * 2.2;
    e.wanderT = 0.4 + rnd() * 1.2;
  }
  const spd = e.speed * (e.panic > 0.3 ? 2.1 : 1);
  let dx = Math.cos(e.heading), dy = Math.sin(e.heading);
  if (fx || fy) { const l = Math.hypot(fx, fy) || 1; dx = fx / l; dy = fy / l; }
  e.vx = dx * spd; e.vy = dy * spd;
  e.x += e.vx * dt; e.y += e.vy * dt;

  const pad = e.radius + 16;
  if (e.x < pad) { e.x = pad; e.heading = Math.PI - e.heading; }
  if (e.x > width - pad) { e.x = width - pad; e.heading = Math.PI - e.heading; }
  if (e.y < pad) { e.y = pad; e.heading = -e.heading; }
  if (e.y > height - pad) { e.y = height - pad; e.heading = -e.heading; }

  // Geräusch skaliert mit dem Tempo: Schleichen ist leise, Sprinten ist laut.
  const v = Math.hypot(e.vx, e.vy);
  e.noise = Math.min(1, (v / 240) * (0.35 + e.panic * 0.65));
}

/** Duftbeitrag einer Entität an einem Punkt, mit 1/r²-Abfall und Sättigung. */
export function scentAt(e, x, y, range) {
  if (!e.alive || !e.scent) return 0;
  const d2 = (e.x - x) * (e.x - x) + (e.y - y) * (e.y - y);
  const r2 = range * range;
  if (d2 > r2) return 0;
  return e.scent * Math.max(0, 1 - d2 / r2);
}

/**
 * Geräuschbeitrag einer Entität an einem Punkt.
 *
 * QUADRATISCHER Abfall `(1 - d/range)²`, nicht linear. Grund: Richtungshören
 * funktioniert über die DIFFERENZ zwischen gegenüberliegenden Rezeptoren, und
 * bei linearem Abfall über 320 px Reichweite und 20 px Rezeptorabstand war
 * diese Differenz praktisch null (gemessen: Quadranten 0.43/0.45/0.47/0.45,
 * Peilung dadurch wertlos). Mit (1-d/range)² ist das Verhältnis zwischen nahem
 * und fernem Rezeptor ≈ 1.9 — auflösbar.
 */
export function noiseAt(e, x, y, range) {
  if (!e.alive || !e.noise) return 0;
  const d = Math.hypot(e.x - x, e.y - y);
  if (d > range) return 0;
  const f = 1 - d / range;
  return e.noise * f * f;
}
