// Lyapunov-Probe: CPU vs CPU mit 1e-7 px Anfangsstörung — dieselben Genome,
// dieselbe Steuerlogik wie in G7 (seed 7, pop 8, prey 2, 150 NN-Schritte).
// Frage: Wächst eine Störung unter f32-ulp in 150 Schritten auf O(10) px?
// Wenn ja, ist die G7-Abweichung GPU vs CPU chaotische Amplifikation
// (Jacobi-f32 vs Gauss-Seidel-f64 als systematische Störung), kein Pfad-Bug.
import { createTrainer } from '../src/sim/train.js';
import { gpuTrainStep, beginGenerationGpu } from '../src/sim/gputrain.js';

const OPTS = { popSize: 8, stepsPerGen: 150, seed: 7, prey: 2 };
const A = createTrainer(OPTS);
const B = createTrainer(OPTS);
beginGenerationGpu(A, null);
beginGenerationGpu(B, null);

// Störung: 1e-7 px auf den Mantel von Tier 0 (weit unter f32-ulp ≈ 1.7e-4 px)
const w = B.sim.world, m0 = B.sim.creatures[0].mantle;
w.px[m0] += 1e-7;

let maxD = 0, sumD = 0, gripMism = 0, gripTot = 0;
const curve = [];
const MARKS = new Set([1, 5, 10, 25, 50, 75, 100, 150]);
for (let s = 0; s < 150; s++) {
  gpuTrainStep(A, null);
  gpuTrainStep(B, null);
  let stepMax = 0;
  for (let i = 0; i < 8; i++) {
    const ca = A.sim.creatures[i], cb = B.sim.creatures[i];
    const d = Math.hypot(ca.x - cb.x, ca.y - cb.y);
    if (d > maxD) maxD = d;
    if (d > stepMax) stepMax = d;
    sumD += d;
    for (let a = 0; a < 6; a++) {
      gripTot++;
      if (ca.arms[a].grip.broken !== cb.arms[a].grip.broken) gripMism++;
    }
  }
  if (MARKS.has(s + 1)) curve.push([s + 1, stepMax]);
}
const plA = A.sim.creatures.reduce((s, c) => s + c.pathLen, 0) / 8;
const plB = B.sim.creatures.reduce((s, c) => s + c.pathLen, 0) / 8;
console.log('Lyapunov-Probe (CPU f64 vs CPU f64, 1e-7 px Störung, seed 7, prey 2):');
console.log('  max Δpos:', maxD.toFixed(4), 'px · mean:', (sumD / 1200).toFixed(4));
console.log('  Grip-Mismatches:', gripMism + '/' + gripTot, '=', (100 * gripMism / gripTot).toFixed(2), '%');
console.log('  Pfadlänge: A=' + plA.toFixed(3), 'B=' + plB.toFixed(3),
  'Δ=' + (100 * Math.abs(plA - plB) / Math.max(plA, 1e-9)).toFixed(2) + '%');
console.log('  Wachstumskurve (Schritt, maxΔ px):');
for (const [s, d] of curve) console.log('    ' + String(s).padStart(3) + ': ' + d.toExponential(3));
