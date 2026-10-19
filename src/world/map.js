/**
 * Karte als Rechteckliste + analytisches Signed-Distance-Field.
 *
 * Analytisch statt gebackener Textur, weil:
 *   - exakte Distanzen und Normalen (kein Treppeneffekt an Kanten)
 *   - kein Baking-Schritt; die Karte kann pro Generation neu gewürfelt werden
 *   - portiert als reine Arithmetik 1:1 nach WGSL (min über N Rechtecke)
 */

/** Deterministischer PRNG — ohne ihn sind Messreihen nicht reproduzierbar. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeMap(width, height, opts = {}) {
  const o = { wallThickness: 14, corridorCols: 4, corridorRows: 3, density: 0.48, seed: 1, ...opts };
  const rects = [];
  const T = o.wallThickness;

  rects.push({ x: 0, y: 0, w: width, h: T });
  rects.push({ x: 0, y: height - T, w: width, h: T });
  rects.push({ x: 0, y: 0, w: T, h: height });
  rects.push({ x: width - T, y: 0, w: T, h: height });

  const rnd = mulberry32(o.seed);
  const cw = width / o.corridorCols, ch = height / o.corridorRows;
  for (let col = 1; col < o.corridorCols; col++) {
    const xl = col * cw;
    for (let row = 0; row < o.corridorRows; row++) {
      if (rnd() > o.density) continue;
      const ys = row * ch;
      const gSz = Math.max(60, Math.min(ch * (0.28 + rnd() * 0.38), height * 0.4));
      const gStart = ys + rnd() * Math.max(1, ch - gSz);
      const wt = 10 + Math.floor(rnd() * 10);
      if (gStart > ys + 14) rects.push({ x: xl - wt / 2, y: ys, w: wt, h: gStart - ys });
      const gEnd = gStart + gSz;
      if (gEnd < ys + ch - 14) rects.push({ x: xl - wt / 2, y: gEnd, w: wt, h: ys + ch - gEnd });
    }
  }
  for (let row = 1; row < o.corridorRows; row++) {
    const yl = row * ch;
    for (let col = 0; col < o.corridorCols; col++) {
      if (rnd() > o.density) continue;
      const xs = col * cw;
      const gSz = Math.max(60, Math.min(cw * (0.28 + rnd() * 0.38), width * 0.3));
      const gStart = xs + rnd() * Math.max(1, cw - gSz);
      const wt = 10 + Math.floor(rnd() * 10);
      if (gStart > xs + 14) rects.push({ x: xs, y: yl - wt / 2, w: gStart - xs, h: wt });
      const gEnd = gStart + gSz;
      if (gEnd < xs + cw - 14) rects.push({ x: gEnd, y: yl - wt / 2, w: xs + cw - gEnd, h: wt });
    }
  }
  return { width, height, rects, sdf: makeSdf(rects) };
}

/**
 * Signierte Distanz + outward Normal über einer Rechteckliste.
 * d < 0 heißt "innerhalb". Gibt null zurück, wenn die Liste leer ist.
 */
export function makeSdf(rects) {
  return function sdf(x, y) {
    let best = Infinity, bnx = 0, bny = 0;
    for (let i = 0; i < rects.length; i++) {
      const r = rects[i];
      const cx = Math.max(r.x, Math.min(x, r.x + r.w));
      const cy = Math.max(r.y, Math.min(y, r.y + r.h));
      const dx = x - cx, dy = y - cy;
      const d2 = dx * dx + dy * dy;
      if (d2 > 1e-12) {
        if (d2 < best * best) { const d = Math.sqrt(d2); best = d; bnx = dx / d; bny = dy / d; }
      } else {
        const dl = x - r.x, dr = r.x + r.w - x, dt = y - r.y, db = r.y + r.h - y;
        const m = Math.min(dl, dr, dt, db);
        const d = -m;
        if (d < best) {
          best = d;
          if (m === dl) { bnx = -1; bny = 0; }
          else if (m === dr) { bnx = 1; bny = 0; }
          else if (m === dt) { bnx = 0; bny = -1; }
          else { bnx = 0; bny = 1; }
        }
      }
    }
    if (!Number.isFinite(best)) return null;
    return { d: best, nx: bnx, ny: bny };
  };
}

export function findSpawn(map, clearance, rnd) {
  for (let i = 0; i < 500; i++) {
    const x = clearance + rnd() * Math.max(1, map.width - clearance * 2);
    const y = clearance + rnd() * Math.max(1, map.height - clearance * 2);
    const hit = map.sdf(x, y);
    if (hit && hit.d >= clearance) return { x, y };
  }
  return { x: map.width * 0.5, y: map.height * 0.5 };
}

/** Anteil freier Positionen, von denen aus ein Anker in Reichweite liegt. */
export function coverageAt(map, reach, samples, rnd) {
  let ok = 0, tried = 0, sum = 0;
  for (let i = 0; i < samples; i++) {
    const x = 20 + rnd() * (map.width - 40);
    const y = 20 + rnd() * (map.height - 40);
    const h = map.sdf(x, y);
    if (!h || h.d < 20) continue;
    tried++;
    let best = Infinity;
    for (const r of map.rects) {
      const cx = Math.max(r.x, Math.min(x, r.x + r.w));
      const cy = Math.max(r.y, Math.min(y, r.y + r.h));
      const d = Math.hypot(x - cx, y - cy);
      if (d < best) best = d;
    }
    sum += best;
    if (best <= reach) ok++;
  }
  return { coverage: tried ? ok / tried : 0, meanDist: tried ? sum / tried : Infinity, tried };
}
