/**
 * ECHOLOT — SDF-Raymarching mit Laufzeit.
 *
 * ── Warum nicht das Wellenfeld aus Kraken_v2 ──────────────────────
 * v2 hatte eine echte 2D-Wellengleichung auf einem Gitter (Zwei-Puffer-Ping-Pong,
 * Dämpfung 0.92, Wandreflexion über ein walls-Raster). Konzeptuell stark, aber
 * fürs Training unbezahlbar: jedes Tier braucht ein EIGENES Feld, weil es sein
 * eigenes Echo hört. 2048 Tiere × 240×135 Zellen = 66 Mio. Zellen pro Schritt.
 *
 * Raymarching gegen das SDF kostet stattdessen E Strahlen × R Rechtecke pro
 * Aussendung — reine Arithmetik, kein Gitter, und auf der GPU ein trivialer
 * Dispatch.
 *
 * ── Warum Laufzeit und nicht einfach Distanz ──────────────────────
 * Ohne Verzögerung wäre das Echolot ein Entfernungssensor. Der Punkt an
 * Echolot ist aber, dass das Tier die Welt NUR über die Zeit zwischen Aussenden
 * und Echo erfährt — es hört also die Vergangenheit, und zwar eine nach
 * Entfernung gestaffelte. Ein sich bewegendes Hindernis erscheint dort, wo es
 * beim Aussenden war. Diese Latenz ist eine echte Einschränkung und damit genau
 * die Sorte "Stärke mit eingebauter Schwäche", um die es im Projekt geht.
 *
 * Zusätzlich: Wer ortet, verrät sich. Die Aussendung ist für jeden anderen
 * hörbaren Akteur in der Welt sichtbar (siehe entities.js / PASSIVE_HEAR). Im
 * Training mit unabhängigen Lanes spielt das keine Rolle, im Spiel schon —
 * deshalb ist es hier als `loudness` bereits Teil der Schnittstelle.
 *
 * ── Schnittstelle ─────────────────────────────────────────────────
 * Diese Datei implementiert das `EchoSense`-Interface. Das Spiel wird später
 * eine zweite Implementierung desselben Interfaces auf dem Wellenfeld bekommen
 * (gegenseitiges Hören, echte Beugung). Bewusste Divergenz zwischen Trainings-
 * und Spiel-Sim, gleiche Schnittstelle — siehe README Milestone 2.
 */

export const ECHO_RAYS = 12;          // Strahlen pro Aussendung
export const ECHO_FAN = Math.PI * 0.85;   // Öffnungswinkel des Fächers (Sichtfeld)
export const ECHO_RANGE = 340;         // px — maximale Reichweite
export const SOUND_SPEED = 900;        // px/s
export const ECHO_INTERVAL = 0.42;     // s — minimale Pause zwischen Aussendungen
export const ECHO_LOUDNESS = 1.0;      // wie weit die Aussendung für andere hörbar ist

/** Pro Ray: [0] normierte Distanz (1 = nichts in Reichweite), [1] Echo-Intensität. */
export const ECHO_DIM = ECHO_RAYS * 2;

export function createEcho(opts = {}) {
  const rays = opts.rays || ECHO_RAYS;
  const fan = opts.fan || ECHO_FAN;
  const range = opts.range || ECHO_RANGE;
  const speed = opts.speed || SOUND_SPEED;
  return {
    rays, fan, range, speed,
    interval: opts.interval !== undefined ? opts.interval : ECHO_INTERVAL,
    cooldown: 0,
    // Ausstehende Echos: {at, dir, dist, intensity} — at = Sim-Zeit des Eintreffens
    pending: [],
    // Letzter gemessener Satz (was das Netz gerade sieht)
    range_: new Float64Array(rays),
    intensity: new Float64Array(rays),
    lastEmit: -1,
    emitCount: 0,
  };
}

/**
 * Echolot aussenden. Gibt die Loudness zurück (für passive Hörer).
 * `heading` ist die Blickrichtung des Mantels.
 */
export function emit(echo, map, x, y, heading, time) {
  if (echo.cooldown > 0) return 0;
  echo.cooldown = echo.interval;
  echo.lastEmit = time;
  echo.emitCount++;
  const n = echo.rays;
  for (let i = 0; i < n; i++) {
    const t = n > 1 ? i / (n - 1) : 0.5;
    const a = heading + (t - 0.5) * echo.fan;
    const d = raymarch(map.sdf, x, y, a, echo.range);
    if (d >= echo.range) continue;               // nichts getroffen → kein Echo
    // Intensität fällt mit dem Quadrat der Distanz (Kugelwelle, 1/r²)
    const intensity = Math.min(1, (echo.range * 0.22) / Math.max(8, d * d / echo.range));
    echo.pending.push({ at: time + (2 * d) / echo.speed, i, dist: d, intensity });
  }
  // Ankunftszeit sortiert halten — billig, und das Netz sieht die Reihenfolge
  echo.pending.sort((p, q) => p.at - q.at);
  return ECHO_LOUDNESS;
}

/**
 * Strahl gegen das analytische SDF marschieren lassen.
 *
 * Schrittweite adaptiv: jeder Schritt darf so weit sein wie die aktuelle
 * Distanz zur nächsten Oberfläche. Das ist exakt, braucht keine Voxel und ist
 * dieselbe Arithmetik, die später im WGSL-Shader steht.
 */
export function raymarch(sdf, x, y, angle, maxDist) {
  const dx = Math.cos(angle), dy = Math.sin(angle);
  let t = 0;
  // Erster Schritt klein, damit wir nicht durch dünne Wände tunneln
  let step = 2;
  for (let i = 0; i < 96; i++) {
    if (t >= maxDist) return maxDist;
    const px = x + dx * t, py = y + dy * t;
    const hit = sdf(px, py);
    if (!hit) return maxDist;
    if (hit.d <= 0) return t;                 // inside → Oberfläche erreicht
    t += Math.max(1, hit.d * 0.85);
    step = hit.d;
    if (step < 0.5) return t;
  }
  return maxDist;
}

/**
 * Einen Sim-Schritt fortschreiben: Cooldown laufen lassen und eingetroffene
 * Echos in die Ausgabepuffer schreiben.
 *
 * Ein Puffer-Eintrag bleibt stehen, bis ein neueres Echo desselben Strahls
 * eintrifft — das Tier "erinnert" sich also an das letzte Echo pro Richtung,
 * genau wie ein echtes Gehör, das zwischen den Klicks auskommen muss.
 */
export function tick(echo, dt, time) {
  if (echo.cooldown > 0) echo.cooldown -= dt;
  while (echo.pending.length && echo.pending[0].at <= time) {
    const e = echo.pending.shift();
    echo.range_[e.i] = e.dist / echo.range;
    echo.intensity[e.i] = e.intensity;
  }
}

/** In den Eingangsvektor schreiben (2 Werte pro Strahl). */
export function write(echo, out, offset) {
  for (let i = 0; i < echo.rays; i++) {
    out[offset + i * 2] = echo.range_[i];
    out[offset + i * 2 + 1] = echo.intensity[i];
  }
  return offset + echo.rays * 2;
}

/** Richtung und Stärke des nächsten Hindernisses — für Diagnose und HUD. */
export function nearest(echo, heading) {
  let best = Infinity, bi = -1;
  for (let i = 0; i < echo.rays; i++) {
    if (echo.range_[i] > 0 && echo.range_[i] < best) { best = echo.range_[i]; bi = i; }
  }
  if (bi < 0) return null;
  const t = echo.rays > 1 ? bi / (echo.rays - 1) : 0.5;
  return { dist: best * echo.range, angle: heading + (t - 0.5) * echo.fan, intensity: echo.intensity[bi] };
}
