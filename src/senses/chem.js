/**
 * CHEMOREZEPTION — Rezeptorbogen aus Punktquellen.
 *
 * v2 hatte ein echtes Diffusionsgitter (`ChemGrid`, 40 px Zellen, Diffusion alle
 * 12 Schritte, Zerfall 0.992/Schritt). Das war konzeptionell schön, aber pro
 * Tier ein eigenes Gitter — bei 2048 Lanes unbezahlbar, und die Diffusion war
 * im Trainingsmodus ohnehin wirkungslos, weil `CHEM.diffuse()` durch den
 * `stepCount`-NaN-Bug nie aufgerufen wurde.
 *
 * Hier: Punktquellen mit 1/r²-Abfall über einen Rezeptorbogen. Dieselbe
 * Wahrnehmungsqualität (Richtung + Stärke einer Duftspur), O(Quellen × Sensoren)
 * statt O(Gitterzellen), und auf der GPU ein Dispatch.
 *
 * Für ein echtes Spur-Folgen (Duft, der liegen bleibt) braucht es ein Feld —
 * das bleibt als Spiel-Implementierung dieses Interfaces vorbehalten, genau wie
 * das Wellenfeld beim Echolot.
 */
import { scentAt } from '../world/entities.js';

export const CHEM_ARC = Math.PI * 1.6;   // Öffnung des Rezeptorbogens
export const CHEM_SENSORS = 6;
export const CHEM_RANGE = 190;           // px
export const CHEM_DIM = CHEM_SENSORS;

export function createChem(opts = {}) {
  return {
    sensors: opts.sensors || CHEM_SENSORS,
    arc: opts.arc || CHEM_ARC,
    range: opts.range || CHEM_RANGE,
    values: new Float64Array(opts.sensors || CHEM_SENSORS),
    peak: 0,
    smoothed: new Float64Array(opts.sensors || CHEM_SENSORS),
    smooth: opts.smooth !== undefined ? opts.smooth : 0.35,   // Duft ist träge
  };
}

/**
 * Abtasten. `heading` ist die Blickrichtung; der Bogen liegt davor.
 * Bewusst träge geglättet: ein Duftgradient ändert sich langsam, und die
 * Glättung gibt dem Netz ein zeitlich stabiles Signal statt Frame-Rauschen.
 */
export function sample(chem, entities, x, y, heading) {
  let peak = 0;
  for (let i = 0; i < chem.sensors; i++) {
    const t = chem.sensors > 1 ? i / (chem.sensors - 1) : 0.5;
    const a = heading + (t - 0.5) * chem.arc;
    // Drei Abtasttiefen entlang des Strahls, damit auch eine Quelle SEITLICH
    // hinter der Vorderkante erfasst wird, nicht nur genau auf der Linie.
    let s = 0;
    for (let k = 1; k <= 3; k++) {
      const d = chem.range * (k / 3);
      const px = x + Math.cos(a) * d, py = y + Math.sin(a) * d;
      for (const e of entities) s += scentAt(e, px, py, chem.range * 0.6);
    }
    const v = Math.min(1, s / 3);
    chem.values[i] = v;
    chem.smoothed[i] += (v - chem.smoothed[i]) * chem.smooth;
    if (v > peak) peak = v;
  }
  chem.peak = peak;
}

export function write(chem, out, offset) {
  for (let i = 0; i < chem.sensors; i++) out[offset + i] = chem.smoothed[i];
  return offset + chem.sensors;
}

/** Richtung des stärksten Dufts relativ zur Blickrichtung — für Diagnose/Fitness. */
export function gradientDirection(chem) {
  let best = -1, bi = 0;
  for (let i = 0; i < chem.sensors; i++) if (chem.smoothed[i] > best) { best = chem.smoothed[i]; bi = i; }
  if (best <= 0.001) return null;
  const t = chem.sensors > 1 ? bi / (chem.sensors - 1) : 0.5;
  return { angle: (t - 0.5) * chem.arc, strength: best };
}
