#!/usr/bin/env node
/**
 * Spawn-Lärm vs. Policy-Signal — warum stagniert die ES?
 *
 * Hypothese: `resetCreatures` zieht JEDE Generation frische Zufalls-Spawns aus
 * dem laufenden rnd-Stream (sim.spawnCache wird nirgends gesetzt). Damit werden
 * sogar die beiden Hälften eines Spiegelpaars (θ+σε, θ−σε) auf UNTERSCHIEDLICHEN
 * Spawns evaluiert — der Paardifferenz-Schätzer misst dann vor allem Spawn-
 * Lotterie statt Policy-Unterschied. v2 ist an genau dieser Lotterie gestorben
 * (Score-Streuung 150…571 ohne Trend, siehe fitness.js).
 *
 * Messung (je Bedingung ein frischer Trainer, identischer Seed):
 *   A: ALLE 64 Genome identisch (zufällig), Spawns wie im alten Code
 *      (crnSpawns:false) → sd der Paardifferenzen = reiner Spawn-Lärm.
 *   B: echte ES-Generation (θ=0, σ=0.06), ohne CRN
 *      → sd der Paardifferenzen = sqrt(Lärm² + Signal²); Signal = var(B)−var(A).
 *   C: identisches Genom MIT CRN (crnSpawns:true, Paar teilt Spawn+Heading)
 *      → Paardifferenz MUSS exakt 0 sein (deterministische, unabhängige Lanes).
 *   D: echte ES-Generation MIT CRN → Paardifferenz-sd = reines Signal.
 *
 *   node tools/spawnnoise-probe.mjs [--pop 64] [--steps 500] [--seed 1]
 */
import { createTrainer, beginGeneration, trainStep, N_IN, N_W } from '../src/sim/train.js';
import { score } from '../src/sim/fitness.js';
import { HIDDEN, OUT_DIM, randomGenome } from '../src/brain/net.js';
import { mulberry32 } from '../src/world/map.js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? Number(argv[i + 1]) : d; };
const POP = arg('pop', 64), STEPS = arg('steps', 500), SEED = arg('seed', 1);

function runGen(tr, genomeOverride) {
  beginGeneration(tr);
  if (genomeOverride) for (let i = 0; i < POP; i++) tr.genomes.set(genomeOverride, i * N_W);
  for (let s = 0; s < STEPS; s++) trainStep(tr);
  const f = new Float64Array(POP);
  for (let i = 0; i < tr.sim.creatures.length; i++) f[i] = score(tr.sim.creatures[i], tr.opts);
  return f;
}

function stats(f) {
  const n = f.length;
  let mean = 0; for (const v of f) mean += v; mean /= n;
  let vr = 0; for (const v of f) vr += (v - mean) * (v - mean); vr /= Math.max(1, n - 1);
  const diffs = [];
  for (let p = 0; p < n >> 1; p++) diffs.push((f[p * 2] - f[p * 2 + 1]) / 2);
  let dm = 0; for (const d of diffs) dm += d; dm /= diffs.length;
  let dv = 0; for (const d of diffs) dv += (d - dm) * (d - dm); dv /= Math.max(1, diffs.length - 1);
  return { mean, sd: Math.sqrt(vr), diffSd: Math.sqrt(dv), min: Math.min(...f), max: Math.max(...f) };
}

console.log(`Spawn-Lärm-Probe: pop ${POP}, ${STEPS} Schritte, seed ${SEED}`);

// Bedingung A: identisches Zufalls-Genom für alle — Unterschiede sind reine Spawn-Lotterie.
// g0 bekommt einen EIGENEN RNG-Stream, damit A denselben rnd-Verlauf nimmt wie B
// (epsBank-Züge in beginGeneration identisch ⇒ identische Spawn-Sets in A und B).
const g0 = randomGenome(N_IN, HIDDEN, OUT_DIM, mulberry32(SEED + 999));
const trA = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: SEED, prey: 0, crnSpawns: false });
const fA = runGen(trA, g0);
const sA = stats(fA);

// Bedingung B: echte ES-Generation (θ=0, σ=0.06), derselbe Seed, ohne CRN.
const trB = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: SEED, prey: 0, crnSpawns: false });
const fB = runGen(trB, null);
const sB = stats(fB);

// Bedingung C: identisches Genom MIT CRN (Paar teilt Spawn+Heading)
// → Paardifferenzen müssen exakt 0 sein (deterministische, unabhängige Lanes).
const trC = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: SEED, prey: 0 });
const fC = runGen(trC, g0);
const sC = stats(fC);

// Bedingung D: echte ES-Generation MIT CRN → Paardifferenz-sd = reines Signal.
const trD = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: SEED, prey: 0 });
const fD = runGen(trD, null);
const sD = stats(fD);

const noiseVar = sA.diffSd * sA.diffSd;
const totVar = sB.diffSd * sB.diffSd;
const signalVar = Math.max(0, totVar - noiseVar);
const snr = noiseVar > 0 ? signalVar / noiseVar : Infinity;

console.log(`\n── ohne CRN (Alter Stand: frische Zufalls-Spawns pro Generation und Slot) ──`);
console.log(`A  identisches Genom, Spawn-Lotterie : Score Ø ${sA.mean.toFixed(1)} (sd ${sA.sd.toFixed(1)}, ${sA.min.toFixed(0)}…${sA.max.toFixed(0)})`);
console.log(`   Paardifferenz-sd (NOISE-FLOOR)    : ${sA.diffSd.toFixed(2)} Punkte`);
console.log(`B  ES-Generation (θ=0, σ=0.06)       : Score Ø ${sB.mean.toFixed(1)} (sd ${sB.sd.toFixed(1)}, ${sB.min.toFixed(0)}…${sB.max.toFixed(0)})`);
console.log(`   Paardifferenz-sd (Lärm+Signal)    : ${sB.diffSd.toFixed(2)} Punkte`);
console.log(`   Signal-Varianz = var(B) − var(A)  : ${signalVar.toFixed(1)}  (Signal-sd ${Math.sqrt(signalVar).toFixed(2)})`);
console.log(`   SNR pro Paar (Signal/Lärm)        : ${snr.toFixed(3)}   Lärm-Anteil: ${(100 * noiseVar / Math.max(1e-9, totVar)).toFixed(1)} %`);

console.log(`\n── mit CRN (Paar teilt Spawn+Heading, über Generationen fix) ──`);
console.log(`C  identisches Genom                 : Score Ø ${sC.mean.toFixed(1)} (sd ${sC.sd.toFixed(1)}, ${sC.min.toFixed(0)}…${sC.max.toFixed(0)})`);
console.log(`   Paardifferenz-sd (NOISE-FLOOR)    : ${sC.diffSd.toFixed(6)} Punkte   ${sC.diffSd === 0 ? '✓ exakt 0' : '✗ NICHT null!'}`);
console.log(`D  ES-Generation (θ=0, σ=0.06)       : Score Ø ${sD.mean.toFixed(1)} (sd ${sD.sd.toFixed(1)}, ${sD.min.toFixed(0)}…${sD.max.toFixed(0)})`);
console.log(`   Paardifferenz-sd (= SIGNAL)       : ${sD.diffSd.toFixed(2)} Punkte`);

console.log(`\n⇒ Noise-Floor ${sA.diffSd.toFixed(1)} → ${sC.diffSd.toFixed(1)} Punkte; ` +
  `σ=0.06-Signal ohne CRN ${Math.sqrt(signalVar).toFixed(1)} (im Lärm nicht messbar) vs. mit CRN ${sD.diffSd.toFixed(1)} Punkte.`);
if (snr < 0.25) console.log('⇒ Ohne CRN misst der Gradientenschätzer fast nur Spawn-Lotterie — Such-Flaschenhals bestätigt.');
process.exitCode = (sC.diffSd === 0 && snr < 0.25) ? 0 : 1;
