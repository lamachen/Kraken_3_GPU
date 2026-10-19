/**
 * PASSIVES GEHÖR — vier Rezeptoren im Quadrat, Differenzbildung über Paare.
 *
 * v2 machte das mit zwei Abtastpunkten 6 px auseinander und bildete die
 * Differenz — ein Gradientendetektor, also primitives Richtungshören. Dasselbe
 * Prinzip hier, aber mit vier Punkten um den Körper, damit die Richtung über
 * den vollen Kreis aufgelöst wird statt nur vor/zurück.
 *
 * Das ist der Sinn, der im Spiel "wer ortet, verrät sich" trägt: eine
 * Echolot-Aussendung ist für jeden passiven Hörer ein lautes Ereignis an einer
 * bekannten Position. Im Training mit unabhängigen Lanes gibt es keine anderen
 * Kreaturen — die Schnittstelle ist trotzdem hier, damit Milestone 4 (Gegner)
 * sie nur noch befüllen muss.
 */
import { noiseAt } from '../world/entities.js';

export const HEAR_DIM = 4;
// Rezeptoren auf der Mantelfläche (Radius 18), Reichweite bewusst kürzer als
// beim Duft: Richtungshören braucht einen messbaren Gradienten über den Kopf,
// und der verschwindet, wenn die Reichweite groß gegen den Rezeptorabstand ist.
export const HEAR_RANGE = 240;
export const HEAR_RADIUS = 20;
export const HEAR_GAIN = 6;      // Peilungsstärke: die Quadrantendifferenz ist naturgemäß klein

export function createHear(opts = {}) {
  return { range: opts.range || HEAR_RANGE, radius: opts.radius || HEAR_RADIUS, v: new Float64Array(4), peak: 0 };
}

export function sample(hear, entities, x, y) {
  let peak = 0;
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2;
    const px = x + Math.cos(a) * hear.radius, py = y + Math.sin(a) * hear.radius;
    let s = 0;
    for (const e of entities) s += noiseAt(e, px, py, hear.range);
    hear.v[i] = Math.min(1, s);
    if (hear.v[i] > peak) peak = hear.v[i];
  }
  hear.peak = peak;
}

export function write(hear, out, offset) {
  for (let i = 0; i < 4; i++) out[offset + i] = hear.v[i];
  return offset + 4;
}

/** Geschätzte Peilung zur lautesten Quelle, relativ zur Körperachse. */
export function bearing(hear, heading) {
  // Vektor-Differenz der gegenüberliegenden Quadranten als grobe Peilung
  const vx = hear.v[0] - hear.v[2], vy = hear.v[1] - hear.v[3];
  const m = Math.hypot(vx, vy);
  if (m * HEAR_GAIN < 0.02) return null;
  const a = Math.atan2(vy, vx);
  let d = a - heading;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return { angle: d, strength: Math.min(1, m * HEAR_GAIN) };
}
