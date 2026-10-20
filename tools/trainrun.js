#!/usr/bin/env node
/**
 * Trainingslauf: die einzige Messung, die wirklich zählt.
 *
 *   node tools/trainrun.js [--gens 20] [--pop 32] [--steps 500] [--prey 0]
 */
import { createTrainer, beginGeneration, trainStep, endGeneration, N_IN, N_W } from '../src/sim/train.js';
import { OUT_DIM, HIDDEN } from '../src/brain/net.js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? Number(argv[i + 1]) : d; };
const GENS = arg('gens', 20), POP = arg('pop', 32), STEPS = arg('steps', 500), PREY = arg('prey', 0);

console.log(`Training: ${GENS} Generationen, Population ${POP} (=${POP >> 1} gespiegelte Paare), ${STEPS} Schritte/Gen, ${PREY} Beute`);
console.log(`Netz: ${N_IN} Eingänge → ${HIDDEN} → ${HIDDEN} → ${OUT_DIM}  =  ${N_W} Gewichte`);
console.log(`Evaluationsbudget: ${POP} Genome/Gen bei ${N_W} Gewichten = ${(N_W / POP).toFixed(0)}:1   (Kraken_v2: 55:1)\n`);
console.log(' gen |   best |   mean | sigma | Zug-px | Anker | netto | gerade | Zyklen | Umdreh |  Eff  | s/Gen');
console.log('-----+--------+--------+-------+--------+-------+-------+--------+--------+--------+-------+------');

const tr = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: 7, prey: PREY });
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
    ['straight', 'Geradlinigkeit', ''], ['cycles', 'Zugzyklen', ''], ['spin', 'Umdrehungen', '']]) {
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
