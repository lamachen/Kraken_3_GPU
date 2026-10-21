/**
 * Kollisions-Probe: Was hat der Schreib-Race in solveCollisions physikalisch
 * angerichtet? (Diagnose zum Fix in 36bafa5.)
 *
 * Läuft bewusst im G7-Szenario (Trainer, NN-gesteuert, pop 32, seed 7, 2
 * Beute, 150 Schritte). Mit Zufallssteuerung und 4 Tieren misst man hier
 * nämlich NULL Kontakte — gemessen, nicht geraten: die Tiere wandern dann in
 * der offenen Kartenmitte und die Kollision ist tot. Im Trainer-Szenario sind
 * es 60 162 wirksame Korrekturen, also ~0.44 pro Körper und Schritt: Boden-
 * und Wandkontakt ist der Normalfall, nicht die Ausnahme.
 *
 * Gemessen wird, einmal mit CPU-/Fix-Semantik (Punkte eines Körpers
 * sequentiell, kumulativ) und einmal mit der Race-Semantik des Shader-Standes
 * 529fdc7 (alle Punkte lesen denselben Zustand, letzter Schreiber gewinnt):
 *
 *   wirksame Korrekturen  Punkt-Korrekturen über der Totzone
 *   angewendete Schreiben wie viele davon tatsächlich im Körper landen
 *   Rest-Penetration      wie tief Körper NACH dem Solve noch im Untergrund
 *                         stecken (Ø pro Punkt und max pro Körper) — die
 *                         Größe, die über Slip und Grip-Bruch entscheidet
 *
 * Aufruf: node tools/collprobe.mjs [--pop 32] [--steps 150] [--seed 7] [--prey 2]
 */
import { createTrainer } from '../src/sim/train.js';
import { gpuTrainStep, beginGenerationGpu } from '../src/sim/gputrain.js';
import { score } from '../src/sim/fitness.js';
import { groupCollPoints } from '../src/gpu/gpuworld.js';
import { SLACK_POS } from '../src/physics/xpbd.js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? Number(argv[i + 1]) : d; };
const OPTS = { popSize: arg('pop', 32), stepsPerGen: arg('steps', 150), seed: arg('seed', 7), prey: arg('prey', 2) };

function run(mode) {
  const tr = createTrainer(OPTS);
  beginGenerationGpu(tr, null);
  const w = tr.sim.world, map = tr.sim.map;
  const { order, range } = groupCollPoints(w.collPts, w.n);
  const nB = w.n;
  let wouldApply = 0, applied = 0;

  // Penetration + Normale eines Punktes aus einem festen Körper-Ursprung
  const contact = (b, cp, ox, oy) => {
    const ca = Math.cos(w.angP[b]), sa = Math.sin(w.angP[b]);
    const hit = map.sdf(ox + cp.lx * ca - cp.ly * sa, oy + cp.lx * sa + cp.ly * ca);
    if (!hit) return null;
    const pen = cp.r - hit.d;
    if (pen <= SLACK_POS) return null;
    return { pen, nx: hit.nx, ny: hit.ny };
  };
  const residual = () => {
    let sum = 0, n = 0, max = 0;
    for (let b = 0; b < nB; b++) {
      if (w.im[b] === 0) continue;
      let s = 0;
      const cnt = range[nB + b];
      for (let q = 0; q < cnt; q++) {
        const c = contact(b, w.collPts[order[range[b] + q]], w.ppx[b], w.ppy[b]);
        if (c) { s += c.pen; n++; }
      }
      sum += s; if (s > max) max = s;
    }
    return { sum, max, n };
  };

  w._solveCollisions = function () {
    if (mode === 'seq') {
      for (let b = 0; b < nB; b++) {
        if (w.im[b] === 0) continue;
        const cnt = range[nB + b];
        for (let q = 0; q < cnt; q++) {
          const c = contact(b, w.collPts[order[range[b] + q]], w.ppx[b], w.ppy[b]);
          if (!c) continue;
          wouldApply++;
          w.ppx[b] += c.nx * c.pen; w.ppy[b] += c.ny * c.pen;   // kumulativ
          applied++; w.stats.collisions++;   // applied = Schreibvorgänge auf bodies[]
        }
      }
      return;
    }
    for (let b = 0; b < nB; b++) {
      if (w.im[b] === 0) continue;
      const cnt = range[nB + b];
      if (!cnt) continue;
      const ox = w.ppx[b], oy = w.ppy[b];          // alle Punkte sehen DAS
      let dx = 0, dy = 0, hits = 0;
      for (let q = 0; q < cnt; q++) {
        const c = contact(b, w.collPts[order[range[b] + q]], ox, oy);
        if (!c) continue;
        wouldApply++; hits++;
        dx = c.nx * c.pen; dy = c.ny * c.pen;      // überschreibt statt zu summieren
      }
      if (hits) { w.ppx[b] = ox + dx; w.ppy[b] = oy + dy; applied++; w.stats.collisions += hits; }
    }
  };

  let resSum = 0, resMax = 0, resN = 0, steps = 0;
  for (let s = 0; s < OPTS.stepsPerGen; s++) {
    gpuTrainStep(tr, null);
    const r = residual();
    resSum += r.sum; if (r.max > resMax) resMax = r.max; resN += r.n; steps++;
  }
  const mean = v => v.reduce((a, b) => a + b, 0) / v.length;
  return {
    mode, wouldApply, applied,
    resPerStep: resSum / steps, resPerPoint: resSum / Math.max(1, resN), resMax,
    pathLen: mean(tr.sim.creatures.map(c => c.pathLen)),
    fit: mean(tr.sim.creatures.map(c => score(c, OPTS))),
    anchor: mean(tr.sim.creatures.map(c => c.steps ? c.anchoredSteps / c.steps : 0)),
    brk: w.stats.broken,
    nB, nP: w.collPts.length,
  };
}

const seq = run('seq'), race = run('race');
const f = (v, d = 4) => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(d)) : String(v));
const pad = (v, n, d) => f(v, d).padStart(n);
console.log(`Kollisions-Probe im G7-Szenario: pop ${OPTS.popSize}, ${OPTS.stepsPerGen} Schritte, seed ${OPTS.seed}, ${OPTS.prey} Beute`);
console.log(`${seq.nB} Körper und ${seq.nP} Kollisionspunkte gesamt (pro Tier 43 Körper, 92 Punkte: Mantel 8, Armsegment 2)`);
console.log('');
console.log('                              sequentiell (CPU/Fix)   Race (Stand 529fdc7)');
console.log(`wirksame Korrekturen          ${pad(seq.wouldApply, 23)}   ${pad(race.wouldApply, 20)}`);
console.log(`Schreibvorgänge auf bodies[]  ${pad(seq.applied, 23)}   ${pad(race.applied, 20)}`);
console.log(`davon verworfen               ${pad(0, 23)}   ${pad(race.wouldApply - race.applied, 20)}  = ${(100 * (race.wouldApply - race.applied) / Math.max(1, race.wouldApply)).toFixed(1)} %`);
console.log(`Rest-Pen. pro Schritt (Σ px)  ${pad(seq.resPerStep, 23, 3)}   ${pad(race.resPerStep, 20, 3)}  (${(race.resPerStep / Math.max(1e-12, seq.resPerStep)).toFixed(2)}×)`);
console.log(`Rest-Pen. pro Punkt (Ø px)    ${pad(seq.resPerPoint, 23, 5)}   ${pad(race.resPerPoint, 20, 5)}  (${(race.resPerPoint / Math.max(1e-12, seq.resPerPoint)).toFixed(2)}×)`);
console.log(`max Rest-Pen. pro Körper      ${pad(seq.resMax, 23, 4)}   ${pad(race.resMax, 20, 4)}  (${(race.resMax / Math.max(1e-12, seq.resMax)).toFixed(2)}×)`);
console.log(`Grip-Brüche                   ${pad(seq.brk, 23)}   ${pad(race.brk, 20)}`);
console.log(`Anker-Rate                    ${pad(100 * seq.anchor, 22, 1)} %  ${pad(100 * race.anchor, 19, 1)} %`);
console.log(`Ø pathLen                     ${pad(seq.pathLen, 23, 4)}   ${pad(race.pathLen, 20, 4)}  (${(100 * (race.pathLen - seq.pathLen) / Math.max(1e-9, seq.pathLen)).toFixed(2)} %)`);
console.log(`Ø Fitness score()             ${pad(seq.fit, 23, 2)}   ${pad(race.fit, 20, 2)}  (${(100 * (race.fit - seq.fit) / Math.max(1e-9, Math.abs(seq.fit))).toFixed(2)} %)`);
console.log('');
console.log('Lesart: Der Race findet NICHT dieselben Kontakte, sondern MEHR — ein zu wenig');
console.log('herausgeschobener Körper steht in der nächsten Iteration und im nächsten');
console.log('Substep weiterhin im Untergrund und meldet denselben Kontakt immer wieder.');
console.log(`${(100 * (race.wouldApply - seq.wouldApply) / Math.max(1, seq.wouldApply)).toFixed(1)} % mehr Meldungen, aber ${(100 * (race.wouldApply - race.applied) / Math.max(1, race.wouldApply)).toFixed(1)} % davon verworfen.`);
console.log('Netto bleibt pro Schritt die ' + (race.resPerStep / Math.max(1e-12, seq.resPerStep)).toFixed(2) + '-fache Rest-Penetration stehen (Σ über alle Punkte).');
console.log('');
console.log('Vergleich mit der echten Hardware (G7, RTX 3070 Ti, Build 529fdc7):');
console.log('  pathLen  hier ' + (100 * (race.pathLen - seq.pathLen) / seq.pathLen).toFixed(2) + ' %   Hardware -14.80 %   race-freie f32-Emulation +0.01 %');
console.log('  Fitness  hier ' + (100 * (race.fit - seq.fit) / seq.fit).toFixed(2) + ' %    race-freie f32-Emulation -0.00 %');
console.log('Der CPU-Nachbau des Races ist damit ein Prädiktor für das, was die GPU tut —');
console.log('die G7-Abweichung war messbar, bevor der Shader je auf echtem Silizium lief.');
