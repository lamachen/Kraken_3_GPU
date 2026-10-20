/**
 * f32-Paritäts-Probe — der headlose G7-Ersatz.
 *
 * A = f64-Oracle (Listen-==Farbordnung, Gauss-Seidel)
 * B = f32ify(Aufbau) — DIESELBE Ordnung, dieselbe Steuerlogik, nur die
 *     Solver-Präzision ist f32 (Klasse der gültigen GPU-Implementierungen).
 *
 * Referenz (echte RTX 3070 Ti, G7, Build fb622c8):
 *   max Δpos 32.6865 · mean 7.887651 · Grip-Mismatches 293/7200 (4.07 %)
 *   pathLen 27.0382 → 25.9602 (-3.99 %) · NaN 0
 *
 * Erwartung, wenn die These stimmt: B liegt in derselben Größenordnung —
 * dann ist die G7-Abweichung die chaotische Amplifikation von Solver-
 * Internika (f32-Arithmetik) über diskrete Bruch-/Slip-Ereignisse und
 * KEIN Verdrahtungs-Bug. Bleibt B < 0.01 px, wäre f32 allein zu klein
 * und die GPU-Zahl müsste eine andere Ursache haben.
 */
import { createTrainer } from '../src/sim/train.js';
import { gpuTrainStep, beginGenerationGpu } from '../src/sim/gputrain.js';
import { f32ify } from './f32solver.mjs';

function run(seed) {
  const OPTS = { popSize: 8, stepsPerGen: 150, seed, prey: 2 };
  const A = createTrainer(OPTS);
  const B = createTrainer(OPTS);
  beginGenerationGpu(A, null);
  beginGenerationGpu(B, null);
  const emu = f32ify(B.sim.world, B.sim.map);
  emu.roundState();   // = upload(): Anfangszustand geht als f32 in die Puffer

  let maxD = 0, sumD = 0, gripMism = 0, gripTot = 0, firstFlip = -1, onset = -1;
  let onlyA = 0, onlyB = 0, maxPreyD = 0;
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
      if (d > 0.1 && onset < 0) onset = s;
      sumD += d;
      for (let a = 0; a < 6; a++) {
        gripTot++;
        const fa = ca.arms[a].grip.broken, fb = cb.arms[a].grip.broken;
        if (fa !== fb) {
          gripMism++; if (firstFlip < 0) firstFlip = s;
          if (fa) onlyA++; else onlyB++;
        }
      }
    }
    for (let e = 0; e < A.sim.entities.length; e++) {
      const ea = A.sim.entities[e], eb = B.sim.entities[e];
      maxPreyD = Math.max(maxPreyD, Math.hypot(ea.x - eb.x, ea.y - eb.y));
    }
    if (MARKS.has(s + 1)) curve.push([s + 1, stepMax]);
  }
  const plA = A.sim.creatures.reduce((s, c) => s + c.pathLen, 0) / 8;
  const plB = B.sim.creatures.reduce((s, c) => s + c.pathLen, 0) / 8;
  return { maxD, meanD: sumD / 1200, gripMism, gripTot, firstFlip, onset,
    onlyA, onlyB, maxPreyD,
    brkA: A.sim.world.stats.broken, brkB: B.sim.world.stats.broken,
    nanA: A.sim.world.stats.nan, nanB: B.sim.world.stats.nan,
    plA, plB, curve };
}

console.log('f32-Paritäts-Probe: f64-Oracle vs f32-Solver-Emulation (gleiche Ordnung)');
console.log('Referenz echte GPU (G7, fb622c8): max 32.6865 · mean 7.8877 · 293/7200 (4.07 %) · pathLen -3.99 %');
console.log('');
for (const seed of [7, 13, 21]) {
  const r = run(seed);
  console.log(`seed ${String(seed).padStart(2)}: max ${r.maxD.toFixed(4).padStart(8)} px · mean ${r.meanD.toFixed(4).padStart(7)} · ` +
    `Mism ${String(r.gripMism + '/' + r.gripTot).padStart(8)} (${(100 * r.gripMism / r.gripTot).toFixed(2)} %, A-only ${r.onlyA}, B-only ${r.onlyB}) · ` +
    `1.Flip@${r.firstFlip} · onset(Δ>0.1px)@${r.onset}`);
  console.log(`         Brüche ${r.brkA}/${r.brkB} · NaN ${r.nanA}/${r.nanB} · Beute-Δ max ${r.maxPreyD.toFixed(3)} px · ` +
    `pathLen ${r.plA.toFixed(3)} → ${r.plB.toFixed(3)} (${(100 * (r.plB - r.plA) / Math.max(r.plA, 1e-9)).toFixed(2)} %)`);
  console.log('         Kurve:', r.curve.map(([s, d]) => `${s}:${d < 0.01 ? d.toExponential(1) : d.toFixed(2)}`).join('  '));
}
