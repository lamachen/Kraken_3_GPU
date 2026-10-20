// Break-Timing-Probe: CPU vs CPU, aber B's _postGripBreak bekommt auf f
// einen relativen Rauschfaktor — direkter Proxy für den Jacobi-vs-GS-
// Unterschied in λ (Größenordnung aus G3: 0.0105 px / 70.3 px ≈ 1.5e-4 rel).
// These: Nicht Positionsdrift treibt G7 auseinander (Injektions-Probe:
// kontraktiv), sondern verschobene Bruch-ZEITPUNKTE — diskrete Topologie-
// Änderungen, die permanent zwei verschiedene Dynamiken erzeugen.
// Referenz (echte GPU in G7): max 32.69 · mean 7.888 · 293/7200 · pathLen -4.0 %
import { createTrainer } from '../src/sim/train.js';
import { gpuTrainStep, beginGenerationGpu } from '../src/sim/gputrain.js';

const OPTS = { popSize: 8, stepsPerGen: 150, seed: 7, prey: 2 };

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function run(REL) {
  const A = createTrainer(OPTS);
  const B = createTrainer(OPTS);
  beginGenerationGpu(A, null);
  beginGenerationGpu(B, null);
  const rnd = mulberry32(999);
  if (REL > 0) {
    const wB = B.sim.world;
    wB._postGripBreak = function () {
      const inv = 1 / this.h;
      for (const c of this.constraints) {
        if (c.type !== 2 || c.broken) continue;
        let f = Math.hypot(c.lambdaX, c.lambdaY) * inv;
        f *= 1 + (rnd() * 2 - 1) * REL;
        c.force = f;
        if (f > c.breakForce) { c.broken = true; this.stats.broken++; }
      }
    };
  }
  let maxD = 0, sumD = 0, gripMism = 0, gripTot = 0, firstFlip = -1;
  let brkA = 0, brkB = 0;
  for (let s = 0; s < 150; s++) {
    gpuTrainStep(A, null);
    gpuTrainStep(B, null);
    for (let i = 0; i < 8; i++) {
      const ca = A.sim.creatures[i], cb = B.sim.creatures[i];
      const d = Math.hypot(ca.x - cb.x, ca.y - cb.y);
      if (d > maxD) maxD = d;
      sumD += d;
      for (let a = 0; a < 6; a++) {
        gripTot++;
        if (ca.arms[a].grip.broken !== cb.arms[a].grip.broken) {
          gripMism++; if (firstFlip < 0) firstFlip = s;
        }
      }
    }
  }
  brkA = A.sim.world.stats.broken; brkB = B.sim.world.stats.broken;
  const plA = A.sim.creatures.reduce((s, c) => s + c.pathLen, 0) / 8;
  const plB = B.sim.creatures.reduce((s, c) => s + c.pathLen, 0) / 8;
  return { REL, maxD, meanD: sumD / 1200, gripMism, gripTot, firstFlip, brkA, brkB, plA, plB };
}

console.log('Break-Timing-Probe (λ-Rauschen auf f im Bruchtest, seed 7, pop 8, prey 2)');
console.log('Referenz (echte GPU in G7): max 32.69 · mean 7.888 · 293/7200 (4.07 %) · pathLen -4.0 %');
console.log('    REL   |   max Δ   |  mean Δ  | Grip-Mism | 1.Flip | Brüche A/B | pathLen A → B    (Δ%)');
for (const REL of [0, 1.5e-4, 1e-3, 1e-2]) {
  const r = run(REL);
  console.log(
    `  ${r.REL.toExponential(1).padStart(6)}  | ${r.maxD.toFixed(4).padStart(9)} | ${r.meanD.toFixed(4).padStart(8)} |` +
    ` ${String(r.gripMism + '/' + r.gripTot).padStart(9)} | ${String(r.firstFlip).padStart(6)} |` +
    ` ${String(r.brkA + '/' + r.brkB).padStart(10)} | ${r.plA.toFixed(2)} → ${r.plB.toFixed(2)}  (${(100 * (r.plB - r.plA) / Math.max(r.plA, 1e-9)).toFixed(1)} %)`);
}
