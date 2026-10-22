#!/usr/bin/env node
/**
 * θ-Transfer-Messung: ein gespeichertes Genom auf einer Karte, die es beim
 * Training NIE gesehen hat (neuer Seed ⇒ neues Terrain, frische Ring-Spawns).
 *
 * Die θ-Endmessung in tools/trainrun.js prüft Generalisierung über Spawns auf
 * DER Trainingskarte. Hier wird zusätzlich das Terrain getauscht — die Frage
 * ist, ob die ES eine Gangart gelernt hat oder nur die Nischen einer Karte.
 *
 *   node tools/theta-eval.mjs --theta exp/theta-crn-s1.json --seed 777
 *                             [--pop 64] [--steps 500]
 */
import { readFileSync } from 'node:fs';
import { createTrainer, beginGeneration, trainStep, rerollSpawns, N_W } from '../src/sim/train.js';
import { score } from '../src/sim/fitness.js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? Number(argv[i + 1]) : d; };
const strArg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const THETA = strArg('theta', ''), SEED = arg('seed', 777), POP = arg('pop', 64), STEPS = arg('steps', 500);

if (!THETA) { console.error('--theta <datei.json> fehlt'); process.exit(2); }
const theta = Float32Array.from(JSON.parse(readFileSync(THETA, 'utf8')));
if (theta.length !== N_W) { console.error(`Genom hat ${theta.length} Gewichte, erwartet ${N_W}`); process.exit(2); }

const tr = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: SEED, prey: 0 });
rerollSpawns(tr, false);                 // POP eigene frische Ring-Spawns
beginGeneration(tr);
for (let i = 0; i < POP; i++) tr.genomes.set(theta, i * N_W);
for (let s = 0; s < STEPS; s++) trainStep(tr);

const C = tr.sim.creatures;
const avg = (f) => C.reduce((a, c) => a + f(c), 0) / C.length;
const disp = avg(c => c.maxDisplacement);
const net = avg(c => Math.hypot(c.x - c.spawn.x, c.y - c.spawn.y));
const straight = avg(c => c.pathLen > 1 ? c.maxDisplacement / c.pathLen : 0);
const anchor = avg(c => c.steps ? c.anchoredSteps / c.steps : 0);
const strides = avg(c => c.strides);
const spin = avg(c => c.angularAccum / (Math.PI * 2));
const best = Math.max(...C.map(c => score(c, tr.opts)));

console.log(
  `${THETA.split('/').pop()} auf Karte seed ${SEED} (${STEPS} Schr., ${POP} Spawns): ` +
  `Ø netto ${disp.toFixed(1)} px · Ende ${net.toFixed(1)} px · gerade ${(straight * 100).toFixed(0)} % · ` +
  `Anker ${(anchor * 100).toFixed(0)} % · Griffe ${strides.toFixed(0)} · Spin ${spin.toFixed(2)} · best ${best.toFixed(0)}`
);
