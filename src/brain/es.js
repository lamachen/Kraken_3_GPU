/**
 * Natural Evolution Strategies (OpenAI-ES) mit gespiegelten Perturbationen.
 *
 * ── Warum nicht CMA-ES ──────────────────────────────────────────────
 * Bei 3644 Gewichten wäre die Kovarianzmatrix 13.3 Mio. Einträge = 53 MB,
 * plus Eigenzerlegung pro Generation. Bei v2's 5307 Gewichten wären es 112 MB.
 * Natural ES braucht gar keine Kovarianz: der Gradient wird direkt aus den
 * Fitness-Differenzen gespiegelter Paare geschätzt.
 *
 * ── Warum gespiegelt ────────────────────────────────────────────────
 * Für jedes Rauschen ε wird sowohl θ+σε als auch θ−σε evaluiert. Die Differenz
 * der beiden Fitnesswerte halbiert die Varianz der Gradientenschätzung und
 * eliminiert den Bias durch die Baseline. Bei knappen Evaluationsbudgets — und
 * genau das ist hier das Thema — ist das der größte einzelne Gewinn.
 *
 * ── Zustand ─────────────────────────────────────────────────────────
 *   theta   Float32Array(nW)   aktuelles Genom
 *   m, v    Float32Array(nW)   Adam-Momente
 *   eps     Float32Array(nW)   das aktuelle Perturbationsvektor-Paar
 */

export function createES(nWeights, opts = {}) {
  const o = {
    // KALIBRIERT an einer bekannten Zielfunktion (64-dim Paraboloid,
    // tools/harness.js Check N2): lr 0.25 schwingt um das Optimum ohne zu
    // konvergieren, lr ≤ 0.10 erreicht ~9 % Restabstand. 0.030 liegt sicher
    // im stabilen Bereich; sigma muss zur Skala der Gewichte passen, nicht
    // zur Zieldistanz.
    sigma: 0.06,          // Perturbationsstärke
    lr: 0.030,            // Lernrate
    popSize: 64,          // MUSS gerade sein (gespiegelte Paare)
    eliteFrac: 1.0,       // 1.0 = alle Paare tragen bei (gewichtet über Advantage)
    minSigma: 0.004,
    sigmaDecay: 0.999,
    adamBeta1: 0.9,
    adamBeta2: 0.999,
    adamEps: 1e-8,
    clip: 2.5,            // Advantage-Clipping gegen Ausreißer
    ...opts,
  };
  if (o.popSize % 2 !== 0) o.popSize++;
  return {
    nWeights, opts: o,
    theta: new Float32Array(nWeights),
    m: new Float32Array(nWeights),
    v: new Float32Array(nWeights),
    eps: new Float32Array(nWeights),
    // ε pro Paar SPEICHERN statt einen replaybaren RNG-Stream zu verlangen.
    // Der erste Entwurf zog ε im Update nochmal mit demselben Seed — das ist
    // fragil (jede Änderung an genomeAt() bricht die Korrelation still) und
    // kostet 32 × 3644 Floats = 466 KB, was gegenüber dem Risiko nichts ist.
    epsBank: new Float32Array((o.popSize >> 1) * nWeights),
    t: 0,
    history: [],
    bestEver: -Infinity,
    bestTheta: null,
  };
}

/**
 * i-tes Genom der aktuellen Generation schreiben.
 *
 * Paar-Index p = i >> 1, Vorzeichen = (i & 1) ? −1 : +1.
 * Das Perturbationsvektor wird beim ERSTEN Aufruf eines Paars frisch gezogen
 * und für das Spiegelbild wiederverwendet.
 */
export function genomeAt(es, i, out, rnd) {
  const p = i >> 1;
  const sign = (i & 1) ? -1 : 1;
  if (sign > 0) {
    const off = p * es.nWeights;
    for (let k = 0; k < es.nWeights; k++) {
      const e = gauss01(rnd);
      es.epsBank[off + k] = e;
      es.eps[k] = e;
    }
  }
  const s = es.opts.sigma * sign;
  for (let k = 0; k < es.nWeights; k++) out[k] = es.theta[k] + es.eps[k] * s;
  return out;
}

/** Gauss-Näherung ohne Box-Muller-Logarithmus: Summe von 4 Uniformen (Irwin-Hall). */
function gauss01(rnd) {
  return ((rnd() + rnd() + rnd() + rnd()) - 2) * 1.7320508;
}

/**
 * Update aus den Fitnesswerten aller Individuen.
 * `fitness` hat Länge popSize, Index i gehört zu genomeAt(es, i).
 */
export function update(es, fitness) {
  const o = es.opts;
  const half = o.popSize >> 1;

  // Advantage = Fitness-Differenz im gespiegelten Paar, normalisiert über die
  // Generation. Normalisierung macht die Lernrate unabhängig vom Fitness-Maßstab
  // — wichtig, weil die Fitness über die Curriculum-Phasen stark skaliert.
  const diffs = new Float64Array(half);
  let mean = 0;
  for (let p = 0; p < half; p++) {
    diffs[p] = (fitness[p * 2] - fitness[p * 2 + 1]) / 2;
    mean += fitness[p * 2] + fitness[p * 2 + 1];
  }
  mean /= o.popSize;
  let sd = 0;
  for (let p = 0; p < half; p++) sd += diffs[p] * diffs[p];
  sd = Math.sqrt(sd / Math.max(1, half)) || 1;

  // Gradient: ∇θ ≈ (1 / (half·σ)) Σ (diff_p / sd) · ε_p
  // ε wird für das Update neu gezogen — identisch zur Paar-Reihenfolge von
  // genomeAt, deshalb hier derselbe deterministische Stream nötig. Der Aufrufer
  // übergibt ihn über `es.replayRnd`.
  const inv = 1 / (half * o.sigma * sd);
  const g = new Float32Array(es.nWeights);
  for (let p = 0; p < half; p++) {
    const off = p * es.nWeights;
    const d = diffs[p];
    for (let k = 0; k < es.nWeights; k++) g[k] += d * es.epsBank[off + k];
  }
  for (let k = 0; k < es.nWeights; k++) {
    let a = g[k] * inv;
    if (a > o.clip) a = o.clip; else if (a < -o.clip) a = -o.clip;
    g[k] = a;
  }

  // Adam
  es.t++;
  const b1 = o.adamBeta1, b2 = o.adamBeta2;
  const c1 = 1 - Math.pow(b1, es.t), c2 = 1 - Math.pow(b2, es.t);
  for (let k = 0; k < es.nWeights; k++) {
    es.m[k] = b1 * es.m[k] + (1 - b1) * g[k];
    es.v[k] = b2 * es.v[k] + (1 - b2) * g[k] * g[k];
    const step = o.lr * (es.m[k] / c1) / (Math.sqrt(es.v[k] / c2) + o.adamEps);
    es.theta[k] += Number.isFinite(step) ? step : 0;
  }
  o.sigma = Math.max(o.minSigma, o.sigma * o.sigmaDecay);

  let best = -Infinity, bestI = 0;
  for (let i = 0; i < fitness.length; i++) if (fitness[i] > best) { best = fitness[i]; bestI = i; }
  if (best > es.bestEver) {
    es.bestEver = best;
    if (!es.bestTheta) es.bestTheta = new Float32Array(es.nWeights);
    es.bestTheta.set(es.theta);
  }
  es.history.push({ gen: es.t, best, mean, sd: sd * 2, sigma: o.sigma, bestIdx: bestI });
  if (es.history.length > 4000) es.history.shift();
  return { best, mean, sd: sd * 2, sigma: o.sigma };
}
