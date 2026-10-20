/**
 * Trainings-Loop: verbindet Simulation, Netz, Fitness und Evolution.
 *
 * DOM-frei — läuft unverändert im Browser, im Worker und headless.
 * `tools/harness.js` fährt denselben Code, den später der GPU-Dispatch fährt:
 * ein flacher Gewichts-Block für alle Tiere, ein flacher Eingangs-Block,
 * Vorwärtsdurchlauf pro Tier, Fitness, ES-Update.
 */
import { createSim, stepSim, applyControl, readState, readProprio, readSenses, PROPRIO_DIM, SENSE_DIM, INPUT_DIM } from './core.js';
import { score } from './fitness.js';
import { HIDDEN, OUT_DIM, weightCount, randomGenome, forwardInto, decode } from '../brain/net.js';
import { createES, genomeAt, update as esUpdate } from '../brain/es.js';

export const N_IN = INPUT_DIM + OUT_DIM;              // + rekurrente Ausgaben
export const N_W = weightCount(N_IN, HIDDEN, OUT_DIM);

export function createTrainer(opts = {}) {
  const o = {
    popSize: 64,
    stepsPerGen: 900,
    seed: 1,
    creatures: 0,          // 0 → popSize
    prey: 0,
    sigma: 0.06,
    lr: 0.030,
    ...opts,
  };
  const pop = o.popSize;
  const sim = createSim({ ...o, creatures: pop, prey: o.prey });
  const es = createES(N_W, { popSize: pop, sigma: o.sigma, lr: o.lr });

  // Flache Blöcke — dieselbe Form, die der GPU-Buffer hat
  const genomes = new Float32Array(pop * N_W);
  const inputs = new Float32Array(pop * N_IN);
  const outputs = new Float32Array(pop * OUT_DIM);
  const scratch = new Float32Array(pop * 2 * HIDDEN);
  const controls = sim.creatures.map(c => ({
    tone: 0.5, emitEcho: false,
    arms: c.arms.map(() => ({ shoulder: 0, curl: 0, adhesion: 0 })),
  }));
  const fitness = new Float64Array(pop);
  const rndState = { s: (o.seed * 2654435761) >>> 0 };
  const rnd = () => {
    rndState.s = (rndState.s + 0x6D2B79F5) >>> 0;
    let t = rndState.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const tr = {
    sim, es, opts: o, rnd, genomes, inputs, outputs, scratch, controls, fitness,
    gen: 0, stepInGen: 0, history: [],
  };
  seedPopulation(tr);
  return tr;
}

function seedPopulation(tr, baseGenome) {
  const { popSize } = tr.opts;
  for (let i = 0; i < popSize; i++) {
    const g = baseGenome
      ? mutate(baseGenome, tr.genomes.subarray(i * N_W, (i + 1) * N_W), tr.rnd, 0.08, 0.25)
      : randomGenome(N_IN, HIDDEN, OUT_DIM, tr.rnd);
    tr.genomes.set(g.subarray ? g : Float32Array.from(g), i * N_W);
  }
}

function mutate(src, dst, rnd, rate, mag) {
  for (let k = 0; k < src.length; k++) {
    let v = src[k];
    if (rnd() < rate) v += (rnd() * 2 - 1) * mag;
    if (rnd() < 0.002) v = (rnd() * 2 - 1) * 1.5;
    dst[k] = v;
  }
  return dst;
}

/** Eine Generation beginnen: Genome in die ES-Paare schreiben. */
export function beginGeneration(tr) {
  const { popSize } = tr.opts;
  for (let i = 0; i < popSize; i++) {
    genomeAt(tr.es, i, tr.genomes.subarray(i * N_W, (i + 1) * N_W), tr.rnd);
  }
  // Sim zurücksetzen, damit jedes Genom unter gleichen Bedingungen startet
  resetCreatures(tr);
  tr.stepInGen = 0;
}

function resetCreatures(tr) {
  const { sim, rnd } = tr;
  for (let i = 0; i < sim.creatures.length; i++) {
    const c = sim.creatures[i];
    const sp = sim.map && sim.spawnCache ? sim.spawnCache[i] : null;
    const x = sp ? sp.x : 60 + rnd() * (sim.map.width - 120);
    const y = sp ? sp.y : 60 + rnd() * (sim.map.height - 120);
    respawnCreature(sim, c, x, y, rnd() * Math.PI * 2);
  }
}

/** Ein Tier an neuer Position neu aufsetzen, ohne die Welt neu zu bauen. */
import { spawnKraken, ARM_REACH } from '../body/kraken.js';
import { findSpawn } from '../world/map.js';
import { measureReach } from './fitness.js';
function respawnCreature(sim, c, x, y, angle) {
  const w = sim.world;
  // Alle Körper des Tiers an die neue Pose setzen: Mantel + Arme neu verkettet.
  const arms = c.arms;
  w.px[c.mantle] = x; w.py[c.mantle] = y;
  w.ppx[c.mantle] = x; w.ppy[c.mantle] = y;
  w.vx[c.mantle] = 0; w.vy[c.mantle] = 0;
  w.ang[c.mantle] = angle; w.angP[c.mantle] = angle; w.av[c.mantle] = 0;
  for (let a = 0; a < arms.length; a++) {
    const arm = arms[a];
    const relAng = (a / arms.length) * Math.PI * 2;
    const th0 = angle + relAng;
    let jx = x + Math.cos(th0) * 24, jy = y + Math.sin(th0) * 24;
    let th = th0;
    for (let i = 0; i < arm.segs.length; i++) {
      const b = arm.segs[i];
      const ux = Math.cos(th), uy = Math.sin(th);
      const cx = jx + ux * 13.5 * 0.5, cy = jy + uy * 13.5 * 0.5;
      w.px[b] = cx; w.py[b] = cy; w.ppx[b] = cx; w.ppy[b] = cy;
      w.vx[b] = 0; w.vy[b] = 0;
      w.ang[b] = th; w.angP[b] = th; w.av[b] = 0;
      jx = cx + ux * 13.5 * 0.5; jy = cy + uy * 13.5 * 0.5;
      th += 0.28;
    }
    // Motor-Ziele auf die Spawn-Pose (Fixpunkt) und Grip lösen
    arm.motors[0].target = 0;
    for (let j = 1; j < arm.motors.length; j++) arm.motors[j].target = 0;
    arm.grip.broken = true; arm.grip.force = 0; arm.grip.lambdaX = 0; arm.grip.lambdaY = 0;
  }
  // Messwerte zurücksetzen
  c.x = x; c.y = y; c.spawn = { x, y, angle };
  c.pathLen = 0; c.maxDisplacement = 0; c.angularAccum = 0; c.forwardProg = 0;
  c.gripPullProg = 0; c.anchoredSteps = 0; c.anchoredSum = 0; c.gripCycles = 0;
  c.freeMove = 0; c.energySpent = 0; c.hits = 0; c.preyMinDist = Infinity;
  c.wasAnchored = false; c.prevSpawnDist = 0; c.steps = 0;
  c.prevVx = 0; c.prevVy = 0; c.contact.fill(0);
  for (const s of [c.echo]) { s.range_.fill(0); s.intensity.fill(0); s.pending.length = 0; s.cooldown = 0; }
  c.chem.smoothed.fill(0); c.chem.values.fill(0); c.chem.peak = 0;
  c.hear.v.fill(0); c.hear.peak = 0;
  c.spawnReach = measureReach(sim.map, x, y);
}

/** Ein Trainingsschritt: alle Tiere wahrnehmen → Netz → steuern → Physik. */
export function trainStep(tr) {
  const { sim, genomes, inputs, outputs, scratch, controls } = tr;
  const n = sim.creatures.length;
  for (let i = 0; i < n; i++) {
    const c = sim.creatures[i];
    const x = inputs.subarray(i * N_IN, (i + 1) * N_IN);
    readProprio(c, x);
    readSenses(c, x, sim.entities);
    // rekurrente Eingänge: die Ausgaben des vorigen Schritts
    const oPrev = outputs.subarray(i * OUT_DIM, (i + 1) * OUT_DIM);
    for (let k = 0; k < OUT_DIM; k++) x[PROPRIO_DIM + SENSE_DIM + k] = oPrev[k];

    forwardInto(genomes.subarray(i * N_W, (i + 1) * N_W), x, oPrev,
      scratch.subarray(i * 2 * HIDDEN, (i + 1) * 2 * HIDDEN), N_IN, HIDDEN, OUT_DIM);
    decode(oPrev, controls[i], c.arms.length);
    applyControl(c, controls[i]);
  }
  stepSim(sim);
  // Beute-Kontakt: dieselbe Mechanik wie der Zug — die Beute ist frei und
  // leichter, also wird SIE herangezogen, nicht die Krake hingebracht.
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

/** Generation auswerten und die Population fortschreiben. */
export function endGeneration(tr) {
  const { sim, fitness } = tr;
  for (let i = 0; i < sim.creatures.length; i++) fitness[i] = score(sim.creatures[i], tr.opts);
  const r = esUpdate(tr.es, fitness);

  // Metriken für die Telemetrie
  const n = sim.creatures.length;
  const avg = (f) => { let s = 0; for (const c of sim.creatures) s += f(c); return s / n; };
  const rec = {
    gen: tr.gen,
    best: r.best, mean: r.mean, sigma: r.sigma,
    bestGenomeIdx: r.bestIdx,
    gripPull: avg(c => c.gripPullProg),
    anchor: avg(c => c.steps ? c.anchoredSteps / c.steps : 0),
    disp: avg(c => c.maxDisplacement),
    straight: avg(c => c.pathLen > 1 ? c.maxDisplacement / c.pathLen : 0),
    cycles: avg(c => c.gripCycles),
    spin: avg(c => c.angularAccum / (Math.PI * 2)),
    eff: avg(c => c.forwardProg / Math.max(0.5, c.energySpent)),
  };
  tr.history.push(rec);
  if (tr.history.length > 4000) tr.history.shift();
  tr.gen++;
  return rec;
}
