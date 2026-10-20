/**
 * Hybrid-Trainer (Milestone 3c): Physik auf der GPU, Sinne/Netz/ES auf der CPU.
 *
 * Architektur — gemessen, nicht geraten (tools/profile-step.js, pop=32):
 *   Physik       301.9 µs/Tier·Schritt  (91.3 %)   → GPU (G6: 90.8× auf RTX 3070 Ti)
 *   Netz          13.7 µs               ( 4.1 %)   → CPU (batchbar, Milestone 3d)
 *   Proprio       6.2 µs  Sinne 6.3 µs  Rest 2.7 µs → CPU
 *
 * Der entscheidende Trick: Nach jedem GPU-Schritt spiegelt mirrorInto() den
 * kompletten Zustand (alle Körper px/py/vx/vy/ang/av, alle Constraint-λ,
 * Grip-broken, Grip-force) in dieselbe CPU-XpbdWorld zurück. Damit laufen
 * readProprio, readSenses, tickSenses, applyControl (inkl. Saugnapf-Reflex!),
 * postStep und score() UNVERÄNDERT — keine duplizierte Sinn-Logik, die
 * auseinanderlaufen könnte. Die CPU bleibt das Oracle; gpuTrainStep() ist
 * strukturell derselbe Ablauf wie trainStep() in train.js.
 *
 * Parität wird zweistufig bewiesen:
 *   1. headless (tools/harness.js): Passthrough-Backend → Fitness
 *      BIT-IDENTISCH zum CPU-Trainer (gputrain-passthrough-identical).
 *   2. Browser (gpu.html G7): echte f32-GPU im Lockstep gegen das f64-Oracle.
 */
import { createTrainer, beginGeneration, endGeneration, trainStep, N_IN, N_W } from './train.js';
import { readProprio, readSenses, applyControl, preStep, postStep, tickSenses, PROPRIO_DIM, SENSE_DIM } from './core.js';
import { forwardInto, decode, HIDDEN, OUT_DIM } from '../brain/net.js';
import { tickMapLights } from '../world/map.js';
import { tickEntity } from '../world/entities.js';
import { upload, uploadControls, step as gpuStep, download } from '../gpu/gpuworld.js';

export { N_IN, N_W, createTrainer, endGeneration, trainStep };

/**
 * Backend = die Physik-Seite des Trainers. Vier Methoden, keine Annahmen
 * darüber, was darunterliegt:
 *   upload()          voller Zustand CPU→GPU (nach Respawn)
 *   uploadControls()  Steuerfelder CPU→GPU (Compliance, tgt, Anker, Adhäsion, broken)
 *   step()            ein Sim-Schritt Physik   (async erlaubt)
 *   download()        Zustand GPU→CPU spiegeln (async erlaubt)
 */
export function gpuBackend(g) {
  return {
    kind: 'gpu',
    upload: () => upload(g),
    uploadControls: () => uploadControls(g),
    step: () => gpuStep(g),
    download: () => download(g),
  };
}

/**
 * Passthrough-Backend für headless Tests: Physik läuft auf der CPU, die
 * Spiegel-Pfade werden als Aufrufe gezählt. gpuTrainStep() mit diesem Backend
 * MUSS bit-identische Fitness zum reinen CPU-Trainer liefern — das pinnt die
 * gesamte Verdrahtung (Reihenfolge, Controls, Respawn, ES) ohne Device.
 */
export function passthroughBackend(sim, counters = {}) {
  const c = counters;
  c.upload = c.upload || 0; c.uploadControls = 0; c.step = 0; c.download = 0;
  return {
    kind: 'passthrough', counters: c,
    upload: () => { c.upload++; },
    uploadControls: () => { c.uploadControls++; },
    step: () => { c.step++; sim.world.step(); },
    download: () => { c.download++; },
  };
}

/**
 * Ein Trainingsschritt — exakt die Reihenfolge aus trainStep()/stepSim(),
 * nur dass die Physik über das Backend läuft. backend === null → reine
 * CPU-Physik (identisch zu trainStep).
 */
export async function gpuTrainStep(tr, backend) {
  const { sim, genomes, inputs, outputs, scratch, controls } = tr;
  const n = sim.creatures.length;

  // 1) Wahrnehmen → Netz → Steuern (identisch zu trainStep)
  for (let i = 0; i < n; i++) {
    const c = sim.creatures[i];
    const x = inputs.subarray(i * N_IN, (i + 1) * N_IN);
    readProprio(c, x);
    readSenses(c, x, sim.entities);
    const oPrev = outputs.subarray(i * OUT_DIM, (i + 1) * OUT_DIM);
    for (let k = 0; k < OUT_DIM; k++) x[PROPRIO_DIM + SENSE_DIM + k] = oPrev[k];
    forwardInto(genomes.subarray(i * N_W, (i + 1) * N_W), x, oPrev,
      scratch.subarray(i * 2 * HIDDEN, (i + 1) * 2 * HIDDEN), N_IN, HIDDEN, OUT_DIM);
    decode(oPrev, controls[i], c.arms.length);
    applyControl(c, controls[i]);
  }

  // 2) Messwerte vor der Physik sichern (identisch zu stepSim)
  for (const c of sim.creatures) preStep(c);

  // 3) Physik
  if (backend) {
    backend.uploadControls();
    await backend.step();
    await backend.download();
  } else {
    sim.world.step();
  }

  // 4) Messwerte nach der Physik (identisch zu stepSim)
  for (const c of sim.creatures) postStep(c);
  sim.time += sim.opts.h * sim.opts.substeps;
  sim.stepCount++;

  // 5) Welt-Update + Sinne nachführen (identisch zu stepSim)
  const dtS = sim.opts.h * sim.opts.substeps;
  tickMapLights(sim.map, dtS);
  if (sim.entities.length) {
    const threats = sim.creatures.map(c => ({ x: c.x, y: c.y }));
    for (const e of sim.entities) tickEntity(e, dtS, sim.map.width, sim.map.height, threats, sim.rnd);
  }
  for (const c of sim.creatures) tickSenses(c, sim);

  // 6) Beute-Kontaktdistanz (identisch zu trainStep)
  if (sim.entities.length) {
    for (let i = 0; i < n; i++) {
      const c = sim.creatures[i];
      for (const e of sim.entities) {
        if (!e.alive) continue;
        const d = Math.hypot(e.x - c.x, e.y - c.y);
        if (d < c.preyMinDist) c.preyMinDist = d;
      }
    }
  }
  tr.stepInGen++;
}

/** Generation beginnen: Genome mischen, Tiere respawnen, Backend neu laden. */
export function beginGenerationGpu(tr, backend) {
  beginGeneration(tr);
  if (backend) backend.upload();
}

/**
 * Kompletter Trainingslauf.
 * @param onGen  optionaler Callback (record, gen) nach jeder Generation.
 */
export async function runGpuTraining(tr, backend, gens, stepsPerGen, onGen) {
  const hist = [];
  for (let g = 0; g < gens; g++) {
    beginGenerationGpu(tr, backend);
    for (let s = 0; s < stepsPerGen; s++) await gpuTrainStep(tr, backend);
    const r = endGeneration(tr);
    hist.push(r);
    if (onGen) onGen(r, g);
  }
  return hist;
}
