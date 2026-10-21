#!/usr/bin/env node
/**
 * Schritt-Kopfraum: was kann der Körper — und sieht score() es?
 *
 * `tools/gaitprobe.mjs` hat gezeigt: die ES findet einen Wechselgang (79 %
 * Wechselgriffe, 4.8 Greif-Ereignisse/Tier/Episode), aber der gait-Term der
 * Fitness ist strukturell blind (1.4 % des Scores, 1.7 % seines Deckels).
 * Offene Frage:
 *
 *   Liegt die niedrige Gangfrequenz an der PHYSIK (Muskelzug reißt den Grip
 *   nicht, Drag frisst den Vorschub) oder an der SELEKTION (kein Term trägt
 *   die Schrittrate)?
 *
 * Messstand: GETAKTETE Skript-Controller treten gegen das trainierte Netz an —
 * in JEWEILS FRISCHEN, aber IDENTISCHEN Welten. Frischer Trainer pro
 * Konfiguration mit gleichem Seed ⇒ `createTrainer` und `beginGeneration`
 * ziehen dieselbe Zufallsfolge (Genome zuerst, dann Spawns), also stehen alle
 * 32 Tiere jeder Konfiguration an exakt denselben Stellen, mit denselben
 * `spawnReach`-Werten und derselben Beute. Nur der Controller unterscheidet
 * sich. (Ein Einzel-Tier-Vergleich war der erste Entwurf und ist durchgefallen:
 * das trainierte Genom griff an EINER fremden Spawn-Pose kein einziges Mal —
 * 0.0 px, Score 28. Ein einzelner Spawn ist kein Messstand, sondern eine
 * Lotterie; 32 Spawn-Punkte mitteln das heraus.)
 *
 * Das ist ein Messstand, kein Bauplan: die handgeschriebene Gangkurve bleibt
 * aus dem Produkt draussen (Kraken_v2: Trend auf Netto-Distanz −9.34 px/Gen
 * MIT Kurve gegen +5.14 ohne).
 *
 *   node tools/strideprobe.mjs [--steps 500] [--pop 32] [--seed 7] [--prey 0]
 *                              [--gens 6] [--train-prey 2] [--json] [--no-evolved]
 */
import { stepSim, applyControl, readProprio, readSenses, PROPRIO_DIM, SENSE_DIM } from '../src/sim/core.js';
import { score, scoreTerms, TERM_ORDER } from '../src/sim/fitness.js';
import { HIDDEN, OUT_DIM, forwardInto, decode } from '../src/brain/net.js';
import { createTrainer, beginGeneration, trainStep, endGeneration, N_IN, N_W } from '../src/sim/train.js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? Number(argv[i + 1]) : d; };
const STEPS = arg('steps', 500), POP = arg('pop', 32), SEED = arg('seed', 7);
const PREY = arg('prey', 0), GENS = arg('gens', 6), TRAIN_PREY = arg('train-prey', 2);
const JSON_OUT = argv.includes('--json');
const NO_EVOLVED = argv.includes('--no-evolved');

// ── Skript-Controller: (c, s, ctl) → ctl ───────────────────────────
/**
 * Getakteter Gang. Standphase zieht den Rumpf, Schwungphase setzt um.
 * @param {number} T        Periodenlänge in Sim-Schritten
 * @param {string} pattern  'wave' (metachron) | 'tripod' (Wechseldreibein) | 'sync' (alles gleichzeitig)
 * @param {number} amp      Schulteramplitude in rad
 * @param {number} duty     Anteil der Standphase
 */
function timedGait(T, pattern, amp = 0.8, duty = 0.5) {
  return (c, s, ctl) => {
    ctl.tone = 0.9;
    for (let a = 0; a < c.arms.length; a++) {
      let off = 0;
      if (pattern === 'wave') off = a / c.arms.length;
      else if (pattern === 'tripod') off = (a % 2) * 0.5;
      const u = (((s / T) + off) % 1 + 1) % 1;
      const stance = u < duty;
      const p = stance ? u / duty : (u - duty) / (1 - duty);
      const k = ctl.arms[a];
      k.shoulder = stance ? amp * (1 - 2 * p) : amp * (2 * p - 1);
      k.curl = stance ? -0.2 : 0.7;
      k.adhesion = stance ? 1.0 : 0.0;
    }
    return ctl;
  };
}

const CONFIGS = [
  ['halten (Adhäsion 1, keine Bewegung)', (c, s, ctl) => {
    ctl.tone = 0.5;
    for (const k of ctl.arms) { k.shoulder = 0; k.curl = 0; k.adhesion = 1.0; }
    return ctl;
  }],
  ['zappeln (Adhäsion 1, Arme schwingen)', (c, s, ctl) => {
    ctl.tone = 0.9;
    for (let a = 0; a < c.arms.length; a++) {
      const ph = 2 * Math.PI * (s / 40) + a * 1.047;
      ctl.arms[a].shoulder = 0.9 * Math.sin(ph);
      ctl.arms[a].curl = 0.5 * Math.sin(ph * 2);
      ctl.arms[a].adhesion = 1.0;
    }
    return ctl;
  }],
];
for (const T of [16, 32, 64, 128]) {
  CONFIGS.push([`wave   T=${T}`, timedGait(T, 'wave')]);
  CONFIGS.push([`tripod T=${T}`, timedGait(T, 'tripod')]);
  CONFIGS.push([`sync   T=${T}`, timedGait(T, 'sync')]);
}

// ── Messstand: eine Konfiguration über POP identische Spawns fahren ──
function evalConfig(runCreature, steps) {
  const tr = createTrainer({ popSize: POP, stepsPerGen: steps, seed: SEED, prey: PREY });
  beginGeneration(tr);                        // ⇒ identische Spawns für jede Konfiguration
  const cs = tr.sim.creatures, N = cs.length;
  const A = cs[0].arms.length;
  const prev = new Uint8Array(N * A);
  for (let i = 0; i < N; i++) for (let a = 0; a < A; a++) prev[i * A + a] = cs[i].arms[a].grip.broken ? 1 : 0;
  const ev = { attach: 0, forceBreak: 0, volRelease: 0, handover: 0, zeroSteps: 0, anchoredSum: 0 };

  for (let s = 0; s < steps; s++) {
    for (let i = 0; i < N; i++) {
      const out = runCreature(cs[i], s, i, tr);
      applyControl(cs[i], out);
      tr.controls[i] = out;                   // für die Bruch-/Löse-Klassifikation
    }
    stepSim(tr.sim);
    for (let i = 0; i < N; i++) {
      let anchored = 0;
      for (let a = 0; a < A; a++) {
        const wasBroken = prev[i * A + a] === 1, isBroken = cs[i].arms[a].grip.broken;
        if (wasBroken && !isBroken) {
          ev.attach++;
          let others = 0;
          for (let b = 0; b < A; b++) if (b !== a && !cs[i].arms[b].grip.broken) others++;
          if (others > 0) ev.handover++;
        } else if (!wasBroken && isBroken) {
          if (tr.controls[i].arms[a].adhesion < 0.04) ev.volRelease++; else ev.forceBreak++;
        }
        if (!isBroken) anchored++;
        prev[i * A + a] = isBroken ? 1 : 0;
      }
      ev.anchoredSum += anchored;
      if (anchored === 0) ev.zeroSteps++;
    }
  }

  const opts = tr.sim.opts;
  const per = cs.map(c => ({
    disp: c.maxDisplacement, pathLen: c.pathLen, gripPull: c.gripPullProg, cycles: c.gripCycles,
    anchor: c.steps ? c.anchoredSteps / c.steps : 0,
    straight: c.pathLen > 1 ? c.maxDisplacement / c.pathLen : 0,
    strides: c.strides, attaches: c.attachEvents,
    ratchetOk: c.gripPullProg <= c.maxDisplacement + 1e-9,
    score: score(c, opts), terms: scoreTerms(c, opts),
  }));
  const mean = (f) => per.reduce((s, r) => s + f(r), 0) / per.length;
  const meanTerms = {};
  for (const k of TERM_ORDER) meanTerms[k] = mean(r => r.terms[k]);
  let bi = 0;
  for (let i = 1; i < per.length; i++) if (per[i].score > per[bi].score) bi = i;
  return {
    disp: mean(r => r.disp), dispBest: per[bi].disp, pathLen: mean(r => r.pathLen),
    strides: mean(r => r.strides), ratchetOk: per.every(r => r.ratchetOk),
    gripPull: mean(r => r.gripPull), cycles: mean(r => r.cycles), anchor: mean(r => r.anchor),
    straight: mean(r => r.straight), scoreMean: mean(r => r.score), scoreBest: per[bi].score,
    anchoredMean: ev.anchoredSum / (N * steps), attachPerAnimal: ev.attach / N,
    stepsPerAttach: ev.attach > 0 ? (N * steps) / ev.attach : Infinity,
    handoverFrac: ev.attach > 0 ? ev.handover / ev.attach : 0,
    zeroFrac: ev.zeroSteps / (N * steps),
    forceBreakPerAnimal: ev.forceBreak / N, volReleasePerAnimal: ev.volRelease / N,
    pxPerAttach: ev.attach > 0 ? per.reduce((s, r) => s + r.disp, 0) / ev.attach : 0,
    meanTerms, movers: per.filter(r => r.disp > 5).length, N,
  };
}

// ── Netz-Controller (trainiertes Genom) ────────────────────────────
function netRunner(genome, adhFloor = -1) {
  return (c, s, i, tr) => {
    const x = tr.inputs.subarray(i * N_IN, (i + 1) * N_IN);
    const o = tr.outputs.subarray(i * OUT_DIM, (i + 1) * OUT_DIM);
    const scratch = tr.scratch.subarray(i * 2 * HIDDEN, (i + 1) * 2 * HIDDEN);
    const ctl = tr.controls[i];
    readProprio(c, x);
    readSenses(c, x, tr.sim.entities);
    for (let k = 0; k < OUT_DIM; k++) x[PROPRIO_DIM + SENSE_DIM + k] = o[k];
    forwardInto(genome, x, o, scratch, N_IN, HIDDEN, OUT_DIM);
    decode(o, ctl, c.arms.length);
    if (adhFloor >= 0) for (const a of ctl.arms) if (a.adhesion < adhFloor) a.adhesion = adhFloor;
    return ctl;
  };
}

function bestEvolvedGenome() {
  const tr = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: SEED, prey: TRAIN_PREY });
  let genome = null, rec = null, inWorld = null;
  for (let g = 0; g < GENS; g++) {
    beginGeneration(tr);
    for (let s = 0; s < STEPS; s++) trainStep(tr);
    rec = endGeneration(tr);
    genome = Float32Array.from(tr.genomes.subarray(rec.bestGenomeIdx * N_W, (rec.bestGenomeIdx + 1) * N_W));
    // Referenz IN DER TRAININGSWELT: dieselbe Generation, dieselben Spawns,
    // dieselbe Beute — der einzige Vergleich, der keine Verteilungsverschiebung
    // enthält. (Das Genom in eine fremde Welt zu stellen ist keine Messung:
    // die Sinne bekommen dann Eingänge, die es im Training nie gesehen hat.)
    const cs = tr.sim.creatures, opts = tr.sim.opts;
    const mean = (f) => cs.reduce((a, c) => a + f(c), 0) / cs.length;
    const meanTerms = {};
    for (const k of TERM_ORDER) {
      const ts = cs.map(c => scoreTerms(c, opts));
      meanTerms[k] = ts.reduce((a, t) => a + t[k], 0) / ts.length;
    }
    inWorld = {
      meanTerms, strides: mean(c => c.strides), ratchetOk: cs.every(c => c.gripPullProg <= c.maxDisplacement + 1e-9),
      disp: mean(c => c.maxDisplacement), pathLen: mean(c => c.pathLen),
      gripPull: mean(c => c.gripPullProg), cycles: mean(c => c.gripCycles),
      anchor: mean(c => c.steps ? c.anchoredSteps / c.steps : 0),
      straight: mean(c => c.pathLen > 1 ? c.maxDisplacement / c.pathLen : 0),
      scoreMean: rec.mean, scoreBest: rec.best,
    };
    if (!JSON_OUT) process.stderr.write(`  [trainiert] Gen ${g}: best ${rec.best.toFixed(1)} mean ${rec.mean.toFixed(1)} Zyklen ${rec.cycles.toFixed(2)} Anker ${(rec.anchor * 100).toFixed(0)} %\n`);
  }
  return { genome, inWorld, gens: GENS };
}

// ── Durchlauf ──────────────────────────────────────────────────────
if (!JSON_OUT) console.log(`strideprobe: ${STEPS} Schritte × ${POP} Tiere pro Konfiguration, seed ${SEED}, prey ${PREY} (identische Spawns)\n`);
const results = [];
for (const [name, fn] of CONFIGS) {
  const ctlProto = { tone: 0.5, emitEcho: false, arms: Array.from({ length: 6 }, () => ({ shoulder: 0, curl: 0, adhesion: 0 })) };
  const r = evalConfig((c, s) => fn(c, s, ctlProto), STEPS);
  r.name = name; r.kind = 'script';
  results.push(r);
  if (!JSON_OUT) process.stderr.write(`  · ${name}\n`);
}
let evolved = null, evolvedForced = null, trainWorld = null;
if (!NO_EVOLVED) {
  if (!JSON_OUT) console.log(`\nTrainiere Vergleichs-Genom (${GENS} Gen × ${POP} Tiere, prey ${TRAIN_PREY}, seed ${SEED}) …`);
  const { genome, inWorld } = bestEvolvedGenome();
  evolved = evalConfig(netRunner(genome), STEPS); evolved.name = `EVOLVIERT (bestes Genom, ${GENS} Gen)`; evolved.kind = 'evolved';
  evolvedForced = evalConfig(netRunner(genome, 0.3), STEPS);
  evolvedForced.name = 'EVOLVIERT + Adhäsion ≥ 0.3 erzwungen'; evolvedForced.kind = 'evolved-forced';
  trainWorld = { ...inWorld, name: `EVOLVIERT in der Trainingswelt (Ref)`, kind: 'evolved-ref',
                 anchoredMean: NaN, attachPerAnimal: NaN, stepsPerAttach: NaN, handoverFrac: NaN, movers: NaN };
  results.push(evolved, evolvedForced, trainWorld);
}

const num = (v, dash = '—') => Number.isFinite(v) ? v.toFixed(v >= 10 ? 0 : (dash === '∞' ? 0 : 2)) : dash;
const byDisp = results.slice().sort((a, b) => b.disp - a.disp);
const byScore = results.slice().sort((a, b) => b.scoreMean - a.scoreMean);
const bestScript = byDisp.filter(r => r.kind === 'script')[0];

if (!JSON_OUT) {
  console.log('\n── Gemessen (Mittel über ' + POP + ' identische Spawns), sortiert nach Netto-Distanz ──');
  console.log(' Konfiguration                             netto-px | Weg-px | Zug-px | Anker | ØArme | attach/Tier | Schritte/attach | Wechsel | Schritte | Zyklen | Score Ø');
  console.log('-------------------------------------------+---------+--------+--------+-------+-------+-------------+-----------------+---------+----------+--------+--------');
  for (const r of byDisp) {
    console.log(' ' + r.name.padEnd(42) + ' | ' +
      r.disp.toFixed(1).padStart(7) + ' | ' + r.pathLen.toFixed(0).padStart(6) + ' | ' + r.gripPull.toFixed(1).padStart(6) + ' | ' +
      (r.anchor * 100).toFixed(0).padStart(4) + '% | ' + num(r.anchoredMean).padStart(5) + ' | ' +
      num(r.attachPerAnimal).padStart(11) + ' | ' +
      num(r.stepsPerAttach, '∞').padStart(15) + ' | ' +
      (Number.isFinite(r.handoverFrac) ? (r.handoverFrac * 100).toFixed(0) : '—').padStart(6) + '% | ' +
      num(r.strides).padStart(8) + ' | ' + r.cycles.toFixed(2).padStart(6) + ' | ' + r.scoreMean.toFixed(1).padStart(7) +
      (r.kind.startsWith('evolved') ? '  ◄' : ''));
  }

  const ratchetBad = results.filter(r => r.ratchetOk === false);
  console.log('\n── Invariante gripPullProg ≤ maxDisplacement (Ratchet gegen Rattern) ──');
  console.log('  ' + (ratchetBad.length === 0
    ? '✅ in allen ' + results.length + ' Konfigurationen gehalten — vor dem Fix holte wave T=16 126.9 px „Zug" aus 28.3 px Netto (4.5×)'
    : '❌ verletzt in: ' + ratchetBad.map(r => r.name).join(', ')));

  console.log('\n── Kopfraum ──');
  const ref = trainWorld || evolved;           // fairer Bezug: Trainingswelt
  console.log('  Bester Skript-Gang:      ' + bestScript.name + ' → ' + bestScript.disp.toFixed(1) +
    ' px netto (Ø), Score ' + bestScript.scoreMean.toFixed(1));
  if (evolved) {
    console.log('  Trainiertes Netz (Ref.):  ' + ref.disp.toFixed(1) + ' px netto (Ø), Score ' +
      ref.scoreMean.toFixed(1) + ' · Anker ' + (ref.anchor * 100).toFixed(0) + ' %  ← letzte Trainings-Generation, eigene Welt');
    console.log('  … in dieser Sonden-Welt:  ' + evolved.disp.toFixed(1) + ' px, Score ' + evolved.scoreMean.toFixed(1) +
      ' · Tiere >5 px: ' + evolved.movers + '/' + POP +
      (evolved.disp < ref.disp * 0.5 ? '  ← Verteilungsverschiebung (andere Beute/Sinne), kein Verhalten' : ''));
    console.log('  … mit Adhäsion ≥ 0.3:     ' + evolvedForced.disp.toFixed(1) + ' px, Score ' + evolvedForced.scoreMean.toFixed(1) +
      ' · attach/Tier ' + evolvedForced.attachPerAnimal.toFixed(1) + ' · Ø ' + evolvedForced.anchoredMean.toFixed(2) + ' Arme');
    const rel = ref.disp > 1e-9 ? bestScript.disp / ref.disp : Infinity;
    console.log('  Faktor Distanz (Skript / trainiertes Netz): ' + (Number.isFinite(rel) ? rel.toFixed(2) + '×' : '∞') +
      '  ·  Faktor Score: ' + (ref.scoreMean > 1e-9 ? (bestScript.scoreMean / ref.scoreMean).toFixed(2) + '×' : '—'));
    if (bestScript.disp > ref.disp * 1.5) {
      console.log('  ⇒ KOPFRAUM VORHANDEN: Ein getakteter Gang kommt deutlich weiter als das');
      console.log('    trainierte Netz. Die Physik begrenzt den Gang also NICHT. Seit der');
      console.log('    Fitness-Revision ist auch die Belohnungs-Rangfolge nicht mehr der');
      console.log('    Engpass (siehe Konkordanz unten) — was bleibt, ist die SUCHE: die ES');
      console.log('    muss die Phasenversetzung zwischen sechs Armen erst entdecken.');
    } else if (bestScript.disp < ref.disp) {
      console.log('  ⇒ KEIN KOPFRAUM: kein Skript-Gang schlägt die ES. Dann ist die Gangfrequenz');
      console.log('    physikalisch/strukturell begrenzt und der blinde gait-Term kostet nur');
      console.log('    Telemetrie, nicht Leistung.');
    } else {
      console.log('  ⇒ Skript und Netz liegen gleichauf — der Kopfraum ist klein.');
    }
    console.log('\n  Term-Beiträge (Ø über ' + POP + ' Tiere):');
    console.log('    ' + 'Term'.padEnd(13) + 'Netz(Ref)'.padStart(10) + 'Netz fremd'.padStart(11) + 'Netz+Adh'.padStart(10) + ('Skript ' + bestScript.name.trim()).slice(0, 14).padStart(15));
    for (const k of TERM_ORDER) {
      console.log('    ' + k.padEnd(13) + ref.meanTerms[k].toFixed(2).padStart(10) +
        evolved.meanTerms[k].toFixed(2).padStart(11) + evolvedForced.meanTerms[k].toFixed(2).padStart(10) +
        bestScript.meanTerms[k].toFixed(2).padStart(15));
    }
    console.log('\n  Rangfolge nach SCORE Ø — bevorzugt die Fitness überhaupt den weiteren Gang?');
    for (const r of byScore.slice(0, 7)) {
      console.log('    ' + r.scoreMean.toFixed(1).padStart(7) + '  ' + r.name.padEnd(42) + ' (' + r.disp.toFixed(1) + ' px netto)');
    }
    const rankScore = byScore.map(r => r.name), rankDisp = byDisp.map(r => r.name);
    const scripts = results.filter(r => r.kind === 'script');
    let concordant = 0;
    for (const r of scripts) if (rankScore.indexOf(r.name) - rankDisp.indexOf(r.name) === 0) concordant++;
    console.log('\n  Rang-Konkordanz Score↔Distanz über ' + scripts.length + ' Skript-Gänge: ' +
      concordant + ' exakt gleiche Plätze.');
    console.log('  Referenz (Fitness VOR der Revision): Platz 1 nach Score war `wave T=16` mit');
    console.log('  28.3 px netto — ein Ratterer, dessen gripPull-Term 330 von 454 Punkten trug.');
    console.log('  Außerdem: „halten" 104.4 und „zappeln" 255.8 gegen 149.8 für das trainierte');
    console.log('  Netz. Beide Fehlanreize sind jetzt weg (unten in der Tabelle nachlesbar).');
    const stat = results.find(r => r.name.startsWith('halten'));
    const zap = results.find(r => r.name.startsWith('zappeln'));
    if (stat && zap) {
      console.log('  Stillstands-Boden: halten ' + stat.scoreMean.toFixed(1) + ' · zappeln ' + zap.scoreMean.toFixed(1) +
        ' · bester Gang ' + byScore[0].scoreMean.toFixed(1) + ' → Faktor ' + (byScore[0].scoreMean / Math.max(1e-9, stat.scoreMean)).toFixed(1) + '×');
    }
  }
} else {
  console.log(JSON.stringify({ opts: { STEPS, POP, SEED, PREY, GENS, TRAIN_PREY }, results }, null, 1));
}
