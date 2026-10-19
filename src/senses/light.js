/**
 * PHOTOREZEPTION — zwei Rezeptoren (links/rechts) plus Gesamtlicht.
 *
 * Die Lichtquellen der Karte bewegen sich, Sehen ist damit konstruktionsbedingt
 * intermittierend. Das ist gewollt: eine Kreatur, die nur im Licht etwas
 * wahrnimmt, hat eine echte Schwäche, die der Spieler ausnutzen kann — und die
 * nicht als Regel programmiert ist, sondern aus der Kartenphysik folgt.
 */
export const LIGHT_DIM = 3;   // links, rechts, gesamt

export function createLight(opts = {}) {
  return { spread: opts.spread || 26, left: 0, right: 0, total: 0, ambient: opts.ambient || 0.05 };
}

export function sample(light, map, x, y, heading) {
  const perp = heading + Math.PI / 2;
  const lx = x + Math.cos(perp) * light.spread, ly = y + Math.sin(perp) * light.spread;
  const rx = x - Math.cos(perp) * light.spread, ry = y - Math.sin(perp) * light.spread;
  light.left = sampleAt(map, lx, ly, light.ambient);
  light.right = sampleAt(map, rx, ry, light.ambient);
  light.total = sampleAt(map, x, y, light.ambient);
}

function sampleAt(map, x, y, ambient) {
  let v = ambient;
  for (const l of map.lights || []) {
    const d = Math.hypot(x - l.x, y - l.y);
    v += Math.max(0, 1 - d / l.r) * l.ir;
  }
  return Math.min(1, v);
}

export function write(light, out, offset) {
  out[offset] = light.left; out[offset + 1] = light.right; out[offset + 2] = light.total;
  return offset + LIGHT_DIM;
}
