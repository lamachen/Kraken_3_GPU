/**
 * Neuronales Netz — MLP mit rekurrenter Rückkopplung der eigenen Ausgaben.
 *
 * ── Warum rekurrent ────────────────────────────────────────────────
 * Ohne Gedächtnis kann das Netz keinen Rhythmus erzeugen. Da die handgeschriebene
 * CPG-Gangkurve bewusst entfernt wurde (gemessen: Trend auf die Netto-Distanz
 * −9.34 px/Generation MIT Kurve gegen +5.14 ohne), muss der Takt aus dem Netz
 * selbst kommen — und dafür braucht es Zustand. Die 20 Ausgaben von t−1 sind
 * dieser Zustand.
 *
 * ── Warum batched ──────────────────────────────────────────────────
 * Alle Gewichte aller Tiere liegen in EINEM flachen Float32Array, alle Eingänge
 * in einem zweiten. Die CPU-Schleife hier ist strukturell identisch zu dem, was
 * der GPU-Dispatch in Milestone 3b macht: ein batched Matmul pro Schicht.
 * Wer das pro Tier als Objektgraph baut, kann es später nicht mechanisch
 * übersetzen.
 */

export const HIDDEN = 24;

/** Ausgaben: 1 Tonus + 6 Arme × (Schulter, Krümmung, Haftung) + 1 Echolot. */
export const OUT_DIM = 1 + 6 * 3 + 1;

/** Rückkopplung der eigenen Ausgaben zählt als zusätzlicher Eingang. */
export function inputDim(proprioDim, senseDim) {
  return proprioDim + senseDim + OUT_DIM;
}

/** Gewichteanzahl pro Genom. */
export function weightCount(nIn, hidden = HIDDEN, nOut = OUT_DIM) {
  return nIn * hidden + hidden + hidden * hidden + hidden + hidden * nOut + nOut;
}

/**
 * Ein Genom zufällig initialisieren.
 *
 * Skalierung 1/sqrt(fan_in) statt v2's (rnd()-0.5)*1.6: bei 105 Eingängen wäre
 * die Voraktivations-Varianz ~25× zu groß, jedes tanh sättigt sofort und die
 * Gradientenfreiheit ist weg — alle Ausgaben ±1, kein Selektionsgradient.
 */
export function randomGenome(nIn, hidden = HIDDEN, nOut = OUT_DIM, rnd = Math.random) {
  const w = new Float32Array(weightCount(nIn, hidden, nOut));
  const s1 = 1 / Math.sqrt(nIn), s2 = 1 / Math.sqrt(hidden);
  let i = 0;
  for (; i < nIn * hidden; i++) w[i] = gauss(rnd) * s1;
  for (; i < nIn * hidden + hidden; i++) w[i] = 0;                 // Bias h1
  const b2 = i;
  for (; i < b2 + hidden * hidden; i++) w[i] = gauss(rnd) * s2;
  for (; i < b2 + hidden * hidden + hidden; i++) w[i] = 0;         // Bias h2
  const b3 = i;
  for (; i < b3 + hidden * nOut; i++) w[i] = gauss(rnd) * s2;
  for (; i < w.length; i++) w[i] = 0;                              // Bias out
  return w;
}

/** Normalverteiltes Rauschen (Box-Muller) — für ES-Perturbationen und Init. */
export function gauss(rnd = Math.random) {
  let u = 0, v = 0;
  while (u === 0) u = rnd();
  while (v === 0) v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}

/**
 * Vorwärtsdurchlauf für EIN Tier mit explizitem Scratch-Puffer.
 *
 * `scratch` muss Länge 2*hidden tragen (h1 | h2). Geteilte Puffer statt
 * Allokation pro Aufruf: v2 allokierte 3 Float32Arrays pro forward() und lief
 * 8× pro Sim-Schritt = ~1440 Objekte/s, was den GC spürbar belastete.
 */
export function forwardInto(w, x, out, scratch, nIn, hidden = HIDDEN, nOut = OUT_DIM) {
  const h1 = 0, h2 = hidden;
  const b1 = nIn * hidden;
  const b2 = b1 + hidden;
  const b3 = b2 + hidden * hidden;
  const b4 = b3 + hidden;
  const b5 = b4 + hidden * nOut;

  for (let j = 0; j < hidden; j++) {
    let s = w[b1 + j];
    const row = j * nIn;
    for (let i = 0; i < nIn; i++) s += x[i] * w[row + i];
    scratch[h1 + j] = s > 20 ? 1 : (s < -20 ? -1 : Math.tanh(s));
  }
  for (let j = 0; j < hidden; j++) {
    let s = w[b2 + j];
    const row = b3 - hidden * hidden + j * hidden;
    for (let i = 0; i < hidden; i++) s += scratch[h1 + i] * w[row + i];
    scratch[h2 + j] = s > 20 ? 1 : (s < -20 ? -1 : Math.tanh(s));
  }
  for (let j = 0; j < nOut; j++) {
    let s = w[b5 + j];
    const row = b4 + j * hidden;
    for (let i = 0; i < hidden; i++) s += scratch[h2 + i] * w[row + i];
    out[j] = s > 20 ? 1 : (s < -20 ? -1 : Math.tanh(s));
  }
  return out;
}

/**
 * Ausgaben in die Steuerstruktur übersetzen.
 *
 * out[0]                → globaler Muskeltonus 0..1
 * out[1 + a*3 + 0]      → Schulter-Abweichung Arm a  (−1..1 × Limit)
 * out[1 + a*3 + 1]      → Krümmung Arm a             (−1..1 × Limit)
 * out[1 + a*3 + 2]      → Haftstärke Arm a           (0..1, nur positiv = klebrig)
 * out[OUT_DIM-1]        → Echolot aussenden (Schwellwert > 0.3)
 *
 * Die Haftstärke ist bewusst NUR im positiven Bereich aktiv: ein Zufallsgenom
 * mit Nullpunkt 0.5 wäre im Schnitt halb klebrig, und bei Haftung überall auf
 * dem Untergrund klebten dann alle sechs Arme fest. In Kraken_v2 führte genau
 * das zum Deadlock (90 % verankert, Netto-Trend −3.82 px/Generation).
 */
export function decode(out, control, arms) {
  control.tone = 0.5 + 0.5 * out[0];
  for (let a = 0; a < arms; a++) {
    const o = 1 + a * 3;
    control.arms[a].shoulder = out[o];
    control.arms[a].curl = out[o + 1];
    control.arms[a].adhesion = out[o + 2] > 0 ? out[o + 2] : 0;
  }
  control.emitEcho = out[OUT_DIM - 1] > 0.3;
  return control;
}
