#!/usr/bin/env node
/**
 * Trainingslauf: die einzige Messung, die wirklich zählt.
 *
 *   node tools/trainrun.js [--gens 20] [--pop 32] [--steps 500] [--prey 0] [--seed 7]
 */
import { createTrainer, beginGeneration, trainStep, endGeneration, rerollSpawns, N_IN, N_W } from '../src/sim/train.js';
import { score } from '../src/sim/fitness.js';
import { OUT_DIM, HIDDEN } from '../src/brain/net.js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? Number(argv[i + 1]) : d; };
const strArg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? argv[i + 1] : d; };
const GENS = arg('gens', 20), POP = arg('pop', 32), STEPS = arg('steps', 500), PREY = arg('prey', 0), SEED = arg('seed', 7);
const CRN = arg('crn', 1), SIGMA = arg('sigma', 0.06), LR = arg('lr', 0.030);
const OUT = strArg('out', '');

console.log(`Training: ${GENS} Generationen, Population ${POP} (=${POP >> 1} gespiegelte Paare), ${STEPS} Schritte/Gen, ${PREY} Beute, Seed ${SEED}`);
console.log(`ES: sigma ${SIGMA}, lr ${LR} · CRN-Spawns: ${CRN ? 'an (Paar teilt Spawn+Heading, über Generationen fix)' : 'aus (frische Zufalls-Spawns — Noise-Floor 46.3 Punkte/Paar, gemessen)'}`);
console.log(`Netz: ${N_IN} Eingänge → ${HIDDEN} → ${HIDDEN} → ${OUT_DIM}  =  ${N_W} Gewichte`);
console.log(`Evaluationsbudget: ${POP} Genome/Gen bei ${N_W} Gewichten = ${(N_W / POP).toFixed(0)}:1   (Kraken_v2: 55:1)\n`);
console.log(' gen |   best |   mean | sigma | Zug-px | Anker | netto | gerade | Zyklen | Schritte | Umdreh |  Eff  | s/Gen');
console.log('-----+--------+--------+-------+--------+-------+-------+--------+--------+----------+--------+-------+------');

const tr = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: SEED, prey: PREY, crnSpawns: !!CRN, sigma: SIGMA, lr: LR });
const hist = [];
const t00 = Date.now();
for (let g = 0; g < GENS; g++) {
  beginGeneration(tr);
  const t0 = Date.now();
  for (let s = 0; s < STEPS; s++) trainStep(tr);
  const r = endGeneration(tr);
  const dt = (Date.now() - t0) / 1000;
  hist.push(r);
  console.log(
    String(r.gen).padStart(4) + ' | ' + r.best.toFixed(0).padStart(6) + ' | ' + r.mean.toFixed(0).padStart(6) +
    ' | ' + r.sigma.toFixed(4) + ' | ' + r.gripPull.toFixed(0).padStart(6) +
    ' | ' + (r.anchor * 100).toFixed(0).padStart(4) + '% | ' + r.disp.toFixed(0).padStart(5) +
    ' | ' + (r.straight * 100).toFixed(0).padStart(5) + '% | ' + r.cycles.toFixed(0).padStart(6) +
    ' | ' + r.strides.toFixed(1).padStart(8) +
    ' | ' + r.spin.toFixed(2).padStart(6) + ' | ' + r.eff.toFixed(2).padStart(5) + ' | ' + dt.toFixed(1));
}
const total = (Date.now() - t00) / 1000;

const slope = (arr) => {
  const n = arr.length; if (n < 3) return 0;
  const mx = (n - 1) / 2, my = arr.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (i - mx) * (arr[i] - my); den += (i - mx) * (i - mx); }
  return den ? num / den : 0;
};
const first = hist.slice(0, Math.max(1, Math.floor(GENS / 4)));
const last = hist.slice(-Math.max(1, Math.floor(GENS / 4)));
const avg = (a, k) => a.reduce((s, x) => s + x[k], 0) / a.length;

console.log('\n── Trend (lineare Regression über alle ' + GENS + ' Generationen) ──');
for (const [k, label, unit] of [['best', 'Score (best)', ''], ['mean', 'Score (mean)', ''],
    ['gripPull', 'Zug-Strecke', 'px'], ['disp', 'Netto-Distanz', 'px'], ['anchor', 'Anker-Rate', ''],
    ['straight', 'Geradlinigkeit', ''], ['cycles', 'Zugzyklen', ''], ['strides', 'Wechselgriffe', ''], ['spin', 'Umdrehungen', '']]) {
  const s = slope(hist.map(h => h[k]));
  const scale = (k === 'anchor' || k === 'straight') ? 100 : 1;
  const u = (k === 'anchor' || k === 'straight') ? '%' : unit;
  console.log('  ' + label.padEnd(20) + (s >= 0 ? '+' : '') + (s * scale).toFixed(3).padStart(9) + ' ' + u + '/Generation');
}
console.log('\n── Ø erstes vs. letztes Viertel ──');
for (const [k, label] of [['best', 'Score'], ['gripPull', 'Zug-px'], ['disp', 'netto-px'], ['anchor', 'Anker'], ['straight', 'gerade']]) {
  const f = avg(first, k), l = avg(last, k);
  const sc = (k === 'anchor' || k === 'straight') ? 100 : 1;
  const u = (k === 'anchor' || k === 'straight') ? '%' : '';
  const rel = f !== 0 ? ((l - f) / Math.abs(f) * 100) : 0;
  console.log('  ' + label.padEnd(10) + (f * sc).toFixed(1).padStart(9) + u + '  →  ' + (l * sc).toFixed(1).padStart(9) + u +
    '   (' + (rel >= 0 ? '+' : '') + rel.toFixed(0) + ' %)');
}
console.log('\nGesamtzeit: ' + total.toFixed(1) + ' s für ' + GENS + ' Generationen = ' +
  (total / GENS).toFixed(1) + ' s/Gen  (' + (POP * STEPS * GENS / total).toFixed(0) + ' Kraken-Schritte/s)');

// ── θ-Endmessung auf FRISCHEN Spawns ──────────────────────────────────────
// Die ES optimiert θ auf den festen Trainings-Spawns (CRN). Was zählt, ist,
// ob das Ergebnis auf Spawns generalisiert, die es beim Training nie gesehen
// hat. rerollSpawns(pairShared=false) gibt jedem der POP Slots einen eigenen,
// ring-validierten Spawn auf derselben Karte → POP unabhängige Stichproben.
// Referenz: die besten getakteten Skript-Gänge (tools/demogait-probe.mjs).
{
  const EVAL_STEPS = STEPS;
  const evTr = createTrainer({ popSize: POP, stepsPerGen: EVAL_STEPS, seed: SEED, prey: PREY });
  rerollSpawns(evTr, false);                       // frische Spawns, kein Paar-Sharing
  beginGeneration(evTr);
  const theta = tr.es.theta;
  for (let i = 0; i < POP; i++) evTr.genomes.set(theta, i * N_W);
  for (let s = 0; s < EVAL_STEPS; s++) trainStep(evTr);
  const C = evTr.sim.creatures;
  const avg = (f) => C.reduce((a, c) => a + f(c), 0) / C.length;
  const disp = avg(c => c.maxDisplacement);
  const net = avg(c => Math.hypot(c.x - c.spawn.x, c.y - c.spawn.y));
  const straight = avg(c => c.pathLen > 1 ? c.maxDisplacement / c.pathLen : 0);
  const anchor = avg(c => c.steps ? c.anchoredSteps / c.steps : 0);
  const strides = avg(c => c.strides);
  const spin = avg(c => c.angularAccum / (Math.PI * 2));
  const bestScore = Math.max(...C.map(c => score(c, evTr.opts)));
  console.log(`\n── θ-Endmessung (${EVAL_STEPS} Schritte, ${POP} FRISCHE Spawns auf Karte seed ${SEED}) ──`);
  console.log(`  Ø Netto-Disp ${disp.toFixed(1)} px · Ø End-Distanz ${net.toFixed(1)} px · Geradlinigkeit ${(straight * 100).toFixed(0)} %`);
  console.log(`  Anker ${(anchor * 100).toFixed(0)} % · Wechselgriffe ${strides.toFixed(1)} · Spin ${spin.toFixed(2)} Umdr · bester Score ${bestScore.toFixed(0)}`);
  console.log(`  Referenz Skript-Gänge (Ring-Spawns, 500 Schr., Karte seed 1/2): sync T=32 57.4/46.7 px · wave T=32 46.4/39.8 px · legacy 36.1/35.6 px`);
  if (OUT) {
    const { writeFileSync } = await import('node:fs');
    writeFileSync(OUT, JSON.stringify(Array.from(theta)));
    console.log(`  θ (${theta.length} Gewichte) → ${OUT}`);
  }
}
