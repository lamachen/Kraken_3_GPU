// Injektions-Probe: CPU vs CPU, aber B bekommt pro Schritt eine kleine
// Zustandsstörung auf ALLE Körper — Proxy für den systematischen
// Jacobi-f32-vs-Gauss-Seidel-f64-Solverrest. Kalibriert aus G3: 0.0105 px
// Differenz über ~100 Schritte glatter Zug → ~1e-4 px/Schritt unter Last.
// Frage: Reproduziert Injektion dieser Größe die G7-Zahlen
// (max 32.69, mean 7.89, 293/7200 Grip-Mismatches, pathLen -4.0 %)?
// Skalen: 1e-5, 1e-4, 1e-3 px/Schritt.
import { createTrainer } from '../src/sim/train.js';
import { gpuTrainStep, beginGenerationGpu } from '../src/sim/gputrain.js';

const OPTS = { popSize: 8, stepsPerGen: 150, seed: 7, prey: 2 };

// deterministischer PRNG für die Injektion (reproduzierbar)
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function run(eps) {
  const A = createTrainer(OPTS);
  const B = createTrainer(OPTS);
  beginGenerationGpu(A, null);
  beginGenerationGpu(B, null);
  const rnd = mulberry32(123);
  let maxD = 0, sumD = 0, gripMism = 0, gripTot = 0, firstFlip = -1;
  for (let s = 0; s < 150; s++) {
    if (eps > 0) {
      const w = B.sim.world;
      for (let i = 0; i < w.n; i++) {
        w.px[i] += (rnd() * 2 - 1) * eps;
        w.py[i] += (rnd() * 2 - 1) * eps;
      }
    }
    gpuTrainStep(A, null);
    gpuTrainStep(B, null);
    for (let i = 0; i < 8; i++) {
      const ca = A.sim.creatures[i], cb = B.sim.creatures[i];
      const d = Math.hypot(ca.x - cb.x, ca.y - cb.y);
      if (d > maxD) maxD = d;
      sumD += d;
      for (let a = 0; a < 6; a++) {
        gripTot++;
        const fa = ca.arms[a].grip.broken, fb = cb.arms[a].grip.broken;
        if (fa !== fb) { gripMism++; if (firstFlip < 0) firstFlip = s; }
      }
    }
  }
  const plA = A.sim.creatures.reduce((s, c) => s + c.pathLen, 0) / 8;
  const plB = B.sim.creatures.reduce((s, c) => s + c.pathLen, 0) / 8;
  return { eps, maxD, meanD: sumD / 1200, gripMism, gripTot, firstFlip, plA, plB };
}

console.log('Injektions-Probe (CPU vs CPU + Rauschen, seed 7, pop 8, prey 2, 150 Schritte)');
console.log('Referenz (echte GPU in G7): max 32.69 · mean 7.888 · 293/7200 (4.07 %) · pathLen 27.04→25.96 (-4.0 %)');
console.log('    eps   |   max Δ   |  mean Δ  | Grip-Mism | 1.Flip | pathLen A → B    (Δ%)');
for (const eps of [0, 1e-5, 1e-4, 1e-3]) {
  const r = run(eps);
  console.log(
    `  ${eps.toExponential(0).padStart(6)}  | ${r.maxD.toFixed(4).padStart(9)} | ${r.meanD.toFixed(4).padStart(8)} |` +
    ` ${String(r.gripMism + '/' + r.gripTot).padStart(9)} | ${String(r.firstFlip).padStart(6)} |` +
    ` ${r.plA.toFixed(2)} → ${r.plB.toFixed(2)}  (${(100 * (r.plB - r.plA) / Math.max(r.plA, 1e-9)).toFixed(1)} %)`);
}
