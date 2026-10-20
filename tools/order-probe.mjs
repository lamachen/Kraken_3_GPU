// Ordnungs-Probe: Löst die Constraint-Konkurrenz (Muskel vs. Gelenkgrenze vs.
// Grip vs. Wand) unter farbiger Reihenfolge anders als unter Listen-Reihenfolge?
//
// Die GPU dispatched pro Farbe sequentiell (Farbe 0..15, innerhalb stabil in
// Listenordnung) — eine stabile Sortierung von world.constraints nach c.color
// emuliert exakt diese Ordnung auf der CPU. Variante B2 rundet zusätzlich den
// Zustand nach jedem Substep auf f32 (die GPU hält f32-Puffer).
//
// Referenz (echte GPU in G7): max 32.69 · mean 7.888 · 293/7200 (4.07 %)
//                              pathLen 27.04 → 25.96 (-4.0 %)
import { createTrainer } from '../src/sim/train.js';
import { gpuTrainStep, beginGenerationGpu } from '../src/sim/gputrain.js';
import { ensureColored } from '../src/physics/xpbd.js';

const OPTS = { popSize: 8, stepsPerGen: 150, seed: 7, prey: 2 };

function roundState(w) {
  for (let i = 0; i < w.n; i++) {
    w.px[i] = Math.fround(w.px[i]); w.py[i] = Math.fround(w.py[i]);
    w.ppx[i] = Math.fround(w.ppx[i]); w.ppy[i] = Math.fround(w.ppy[i]);
    w.vx[i] = Math.fround(w.vx[i]); w.vy[i] = Math.fround(w.vy[i]);
    w.ang[i] = Math.fround(w.ang[i]); w.angP[i] = Math.fround(w.angP[i]);
    w.av[i] = Math.fround(w.av[i]);
  }
  const invH = 1 / w.h;
  for (const c of w.constraints) {
    if (c.type !== 2) continue;
    c.lambdaX = Math.fround(c.lambdaX); c.lambdaY = Math.fround(c.lambdaY);
    if (!c.broken) c.force = Math.hypot(c.lambdaX, c.lambdaY) * invH;
  }
}

function run(mode) {
  const A = createTrainer(OPTS);
  const B = createTrainer(OPTS);
  beginGenerationGpu(A, null);
  beginGenerationGpu(B, null);

  if (mode !== 'list') {
    const wB = B.sim.world;
    ensureColored(wB);
    // stabil nach Farbe sortieren = GPU-Dispatch-Ordnung
    wB.constraints.sort((x, y) => (x.color | 0) - (y.color | 0));
    if (mode === 'color+f32') {
      const orig = wB.substep;
      wB.substep = function () { orig.call(this); roundState(this); };
    }
  }

  let maxD = 0, sumD = 0, gripMism = 0, gripTot = 0, firstFlip = -1;
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
        if (ca.arms[a].grip.broken !== cb.arms[a].grip.broken) {
          gripMism++; if (firstFlip < 0) firstFlip = s;
        }
      }
    }
    if (MARKS.has(s + 1)) curve.push([s + 1, stepMax]);
  }
  const plA = A.sim.creatures.reduce((s, c) => s + c.pathLen, 0) / 8;
  const plB = B.sim.creatures.reduce((s, c) => s + c.pathLen, 0) / 8;
  return { maxD, meanD: sumD / 1200, gripMism, gripTot, firstFlip,
    brkA: A.sim.world.stats.broken, brkB: B.sim.world.stats.broken, plA, plB, curve };
}

console.log('Ordnungs-Probe (seed 7, pop 8, prey 2, 150 NN-Schritte)');
console.log('Referenz (echte GPU in G7): max 32.69 · mean 7.888 · 293/7200 (4.07 %) · pathLen -4.0 %');
console.log('   Modus    |   max Δ   |  mean Δ  | Grip-Mism | 1.Flip | Brüche A/B | pathLen A → B    (Δ%)');
for (const mode of ['list', 'color', 'color+f32']) {
  const r = run(mode);
  console.log(
    `  ${mode.padEnd(10)} | ${r.maxD.toFixed(4).padStart(9)} | ${r.meanD.toFixed(4).padStart(8)} |` +
    ` ${String(r.gripMism + '/' + r.gripTot).padStart(9)} | ${String(r.firstFlip).padStart(6)} |` +
    ` ${String(r.brkA + '/' + r.brkB).padStart(10)} | ${r.plA.toFixed(2)} → ${r.plB.toFixed(2)}  (${(100 * (r.plB - r.plA) / Math.max(r.plA, 1e-9)).toFixed(1)} %)`);
  if (mode !== 'list') {
    console.log('    Kurve (Schritt: maxΔ px):',
      r.curve.map(([s, d]) => `${s}:${d < 0.01 ? d.toExponential(1) : d.toFixed(2)}`).join('  '));
  }
}
