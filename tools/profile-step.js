/**
 * Wo steckt die Zeit in einem Tier-Schritt?
 *
 * Zerlegt den trainStep in seine Phasen (Sinne, Netz, Steuerung, Physik,
 * Sinn-Nachführung) und misst µs pro Tier·Schritt. Die Antwort entscheidet
 * die GPU-Trainingsarchitektur: Ist die Physik der Brocken, reicht der
 * Hybrid (Physik GPU, Rest CPU). Sind die Sinne der Brocken, muss das
 * Echolot mit auf die GPU.
 *
 * Nutzung: node tools/profile-step.js [popSize] [steps]
 */
import { createTrainer, N_IN, N_W, beginGeneration } from '../src/sim/train.js';
import { readProprio, readSenses, applyControl, preStep, postStep, tickSenses } from '../src/sim/core.js';
import { forwardInto, decode, HIDDEN, OUT_DIM } from '../src/brain/net.js';
import { PROPRIO_DIM, SENSE_DIM } from '../src/sim/core.js';
import { tickMapLights } from '../src/world/map.js';
import { tickEntity } from '../src/world/entities.js';

const POP = Number(process.argv[2] || 32);
const STEPS = Number(process.argv[3] || 500);

const tr = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: 1, prey: 2, lights: 4 });
beginGeneration(tr);
const { sim, genomes, inputs, outputs, scratch, controls } = tr;
const n = sim.creatures.length;

const T = { proprio: 0, senses: 0, net: 0, control: 0, physics: 0, post: 0, world: 0, tickSenses: 0 };
let now = process.hrtime.bigint();
const us = (t0) => Number(process.hrtime.bigint() - t0) / 1000;

// Aufwärmen (JIT)
for (let s = 0; s < 50; s++) runStep(false);
for (const k in T) T[k] = 0;
now = process.hrtime.bigint();
const wall0 = now;
for (let s = 0; s < STEPS; s++) runStep(true);
const wall = us(wall0);

function runStep(measure) {
  for (let i = 0; i < n; i++) {
    const c = sim.creatures[i];
    const x = inputs.subarray(i * N_IN, (i + 1) * N_IN);
    let t0 = process.hrtime.bigint();
    readProprio(c, x);
    if (measure) { T.proprio += us(t0); t0 = process.hrtime.bigint(); }
    readSenses(c, x, sim.entities);
    if (measure) { T.senses += us(t0); t0 = process.hrtime.bigint(); }
    const oPrev = outputs.subarray(i * OUT_DIM, (i + 1) * OUT_DIM);
    for (let k = 0; k < OUT_DIM; k++) x[PROPRIO_DIM + SENSE_DIM + k] = oPrev[k];
    forwardInto(genomes.subarray(i * N_W, (i + 1) * N_W), x, oPrev,
      scratch.subarray(i * 2 * HIDDEN, (i + 1) * 2 * HIDDEN), N_IN, HIDDEN, OUT_DIM);
    decode(oPrev, controls[i], c.arms.length);
    if (measure) { T.net += us(t0); t0 = process.hrtime.bigint(); }
    applyControl(c, controls[i]);
    if (measure) T.control += us(t0);
  }
  // stepSim, auseinandergenommen
  let t0 = process.hrtime.bigint();
  for (const c of sim.creatures) preStep(c);
  if (measure) { T.post += us(t0); t0 = process.hrtime.bigint(); }
  sim.world.step();
  if (measure) { T.physics += us(t0); t0 = process.hrtime.bigint(); }
  for (const c of sim.creatures) postStep(c);
  if (measure) T.post += us(t0);
  sim.time += sim.opts.h * sim.opts.substeps;
  sim.stepCount++;
  const dtS = sim.opts.h * sim.opts.substeps;
  t0 = process.hrtime.bigint();
  tickMapLights(sim.map, dtS);
  if (sim.entities.length) {
    const threats = sim.creatures.map(c => ({ x: c.x, y: c.y }));
    for (const e of sim.entities) tickEntity(e, dtS, sim.map.width, sim.map.height, threats, sim.rnd);
  }
  if (measure) { T.world += us(t0); t0 = process.hrtime.bigint(); }
  for (const c of sim.creatures) tickSenses(c, sim);
  if (measure) T.tickSenses += us(t0);
}

const perAnimalStep = (v) => v / (n * STEPS);
const rows = Object.entries(T).map(([k, v]) => [k, perAnimalStep(v)]);
const sum = rows.reduce((a, r) => a + r[1], 0);
console.log(`Profil: pop=${POP} steps=${STEPS} prey=2  (µs pro Tier·Schritt)`);
for (const [k, v] of rows.sort((a, b) => b[1] - a[1])) {
  console.log(`  ${k.padEnd(11)} ${v.toFixed(1).padStart(8)} µs  ${(100 * v / sum).toFixed(1).padStart(5)} %`);
}
console.log(`  ${'SUMME'.padEnd(11)} ${sum.toFixed(1).padStart(8)} µs`);
console.log(`  Wandzeit  ${perAnimalStep(wall).toFixed(1)} µs (Overhead Messung: ${(100 * (perAnimalStep(wall) / sum - 1)).toFixed(1)} %)`);
console.log(`  → CPU-Schritte/s bei pop=${POP}: ${(n / (wall / 1e6 / STEPS)).toFixed(0)} Tier-Schritte/s`);
