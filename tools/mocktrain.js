#!/usr/bin/env node
/**
 * Hybrid-Trainer headless — ohne WebGPU-Device (Sandbox/CI).
 *
 * Die GPU selbst kann hier nicht laufen (kein Vulkan/Render-Node), aber die
 * GESAMTE Trainer-Verdrahtung schon. Zwei Backends:
 *
 *   passthrough — Physik auf der CPU, Spiegel-Aufrufe werden gezählt.
 *                 Muss bit-identisch zum CPU-Trainer sein (Harness-Check
 *                 gputrain-identical pinnt das); hier als Laufzeitbasis.
 *   f32mock     — Physik auf der CPU, aber jeder Schritt rundet den
 *                 gespiegelten Zustand auf f32, genau wie mirrorInto() es
 *                 nach einem echten GPU-Schritt tut. Obergut-Schätzung der
 *                 GPU-Divergenz (echte GPU: Jacobi statt Gauss-Seidel,
 *                 darum nur Indikation, kein Ersatz für gpu.html G7).
 *
 * Am Ende steht eine PROJEKTION für die echte GPU: CPU-Kosten minus Physik
 * (Profil-Konstante, tools/profile-step.js) plus GPU-Physik (aus G6,
 * --gpu-rate) plus geschätzter Sync-Overhead (--sync-ms pro Schritt).
 * Projektion ≠ Messung — die echte Zahl liefert G8 im Browser.
 *
 *   node tools/mocktrain.js [--gens 8] [--pop 32] [--steps 500]
 *                           [--backend f32mock] [--prey 0] [--seed 7]
 *                           [--gpu-rate 455000] [--sync-ms 2]
 */
import { createTrainer, endGeneration } from '../src/sim/train.js';
import { gpuTrainStep, beginGenerationGpu, passthroughBackend } from '../src/sim/gputrain.js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? Number(argv[i + 1]) : d; };
const GENS = arg('gens', 8), POP = arg('pop', 32), STEPS = arg('steps', 500);
const PREY = arg('prey', 0), SEED = arg('seed', 7);
const GPU_RATE = arg('gpu-rate', 455000);   // Tier-Schritte/s Physik, G6 auf RTX 3070 Ti
const SYNC_MS = arg('sync-ms', 2);          // writeBuffer + mapAsync-Roundtrips pro Schritt (Schätzung)
const BACKEND = argv.includes('--backend') ? argv[argv.indexOf('--backend') + 1] : 'f32mock';
const PHYS_US = 302;                        // µs/Tier·Schritt CPU-Physik (profile-step.js, pop=32)

const tr = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: SEED, prey: PREY });
const w = tr.sim.world;

function f32mockBackend(world) {
  return {
    kind: 'f32mock',
    upload() {}, uploadControls() {},
    step() {
      world.step();
      for (let i = 0; i < world.n; i++) {
        world.px[i] = Math.fround(world.px[i]); world.py[i] = Math.fround(world.py[i]);
        world.ppx[i] = Math.fround(world.ppx[i]); world.ppy[i] = Math.fround(world.ppy[i]);
        world.vx[i] = Math.fround(world.vx[i]); world.vy[i] = Math.fround(world.vy[i]);
        world.ang[i] = Math.fround(world.ang[i]); world.angP[i] = Math.fround(world.angP[i]);
        world.av[i] = Math.fround(world.av[i]);
      }
      const invH = 1 / world.h;
      for (const c of world.constraints) {
        if (c.type !== 2) continue;
        c.lambdaX = Math.fround(c.lambdaX); c.lambdaY = Math.fround(c.lambdaY);
        if (!c.broken) c.force = Math.hypot(c.lambdaX, c.lambdaY) * invH;
      }
    },
    download() {},
  };
}

const counters = {};
const backend = BACKEND === 'passthrough' ? passthroughBackend(tr.sim, counters) : f32mockBackend(w);

console.log(`mocktrain: ${GENS} Gen × ${POP} Tiere × ${STEPS} Schritte, Backend=${backend.kind}, prey=${PREY}, seed=${SEED}`);
console.log(' gen |   best |   mean | Anker | Zug-px | Zyklen | Schritte | gerade | s/Gen');
console.log('-----+--------+--------+-------+--------+--------+----------+--------+------');
const t00 = Date.now();
let lastGen = 0;
for (let g = 0; g < GENS; g++) {
  beginGenerationGpu(tr, backend);
  const t0 = Date.now();
  for (let s = 0; s < STEPS; s++) await gpuTrainStep(tr, backend);
  const r = endGeneration(tr);
  lastGen = (Date.now() - t0) / 1000;
  console.log(
    String(g).padStart(4) + ' | ' + r.best.toFixed(2).padStart(6) + ' | ' + r.mean.toFixed(2).padStart(6)
    + ' | ' + (r.anchor * 100).toFixed(0).padStart(4) + '% | ' + r.gripPull.toFixed(0).padStart(6)
    + ' | ' + r.cycles.toFixed(2).padStart(6) + ' | ' + r.strides.toFixed(1).padStart(8) + ' | ' + r.straight.toFixed(2).padStart(6)
    + ' | ' + lastGen.toFixed(1).padStart(5));
}
const total = (Date.now() - t00) / 1000;
if (backend.kind === 'passthrough') {
  console.log(`\nAufrufe: upload=${counters.upload} (Soll ${GENS}), uploadControls/step/download=${counters.step} (Soll ${GENS * STEPS})`);
}
const animalSteps = POP * STEPS;
const physCpuS = animalSteps * PHYS_US / 1e6;
const physGpuS = animalSteps / GPU_RATE;
const projGen = lastGen - physCpuS + physGpuS + STEPS * SYNC_MS / 1000;
console.log(`\nGemessen (Mock, CPU-Physik): ${lastGen.toFixed(1)} s/Gen, gesamt ${total.toFixed(0)} s`);
console.log(`PROJEKTION echte GPU (Annahmen: CPU-Physik ${PHYS_US} µs/Tier·Schritt, GPU ${GPU_RATE} Tier-Schritte/s, Sync ${SYNC_MS} ms/Schritt):`);
console.log(`  ≈ ${projGen.toFixed(1)} s/Gen bei pop=${POP}  →  ${(animalSteps / projGen).toFixed(0)} Tier-Schritte/s`);
console.log(`  Zum Vergleich M3a (CPU-Trainer, pop=32): ~4.5 s/Gen, ~2840 Tier-Schritte/s`);
console.log('  Die echte Messung macht gpu.html G8 im Browser — Projektion ≠ Messung.');
