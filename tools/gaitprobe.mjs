#!/usr/bin/env node
/**
 * Gang-Sonde: WIE greifen die Tiere tatsächlich — und was sieht die Fitness?
 *
 * Ausgangspunkt ist eine offene Behauptung im README:
 *   „Bekannte Grenze: Zugzyklen ≈ 1. Die ES findet sofort ‚einmal festhalten
 *    und nie mehr loslassen'; höhere gait-Gewichte ändern das nicht. Der Grund
 *    ist dynamisch: Loslassen erfordert Adhäsion < 0.04 auf allen sechs Armen
 *    gleichzeitig …"
 * Diese Sonde trennt zwei Ursachen, die dieselbe Zahl produzieren:
 *
 *   (A) VERHALTEN statisch  — die Arme halten wirklich dauerhaft, es gibt
 *       praktisch keine Greif-/Löse-Ereignisse pro Arm.
 *   (B) METRIK blind        — pro Arm wird sehr wohl gesetzt und gelöst
 *       (Wechselgang), aber `gripCycles` zählt nur den Übergang
 *       „ALLE Arme lose" → „mindestens einer hält". Bei sechs Armen und
 *       gestaffelten Brüchen kommt dieser Übergang nie vor: die Metrik ist
 *       strukturell unerreichbar, nicht das Verhalten falsch.
 *
 * Gemessen wird deshalb beides nebeneinander: Ereignisse PRO ARM (attach,
 * Kraft-Bruch, freiwilliges Lösen, Haltezeit) und die Größe, die score()
 * tatsächlich sieht (`gripCycles`, `anchoredSteps`).
 *
 *   node tools/gaitprobe.mjs [--gens 4] [--pop 32] [--steps 500]
 *                            [--seed 7] [--prey 2] [--json]
 */
import { createTrainer, beginGeneration, trainStep, endGeneration } from '../src/sim/train.js';
import { score, scoreTerms, TERM_ORDER, FIT_WEIGHTS } from '../src/sim/fitness.js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? Number(argv[i + 1]) : d; };
const GENS = arg('gens', 4), POP = arg('pop', 32), STEPS = arg('steps', 500);
const SEED = arg('seed', 7), PREY = arg('prey', 2);
const JSON_OUT = argv.includes('--json');

const tr = createTrainer({ popSize: POP, stepsPerGen: STEPS, seed: SEED, prey: PREY });
const N = tr.sim.creatures.length;
const A = tr.sim.creatures[0].arms.length;

// Vorzustand der Grip-Flags (vor trainStep) — daraus werden die Ereignisse
// des Schritts klassifiziert. Die Adhäsion desselben Schritts steht danach in
// tr.controls[i].arms[a].adhesion, denn applyControl() läuft VOR stepSim().
const prev = new Uint8Array(N * A);
const held = new Int32Array(N * A);          // Haltezeit in Schritten
const holdHist = new Int32Array(13);         // 0..9, 10-19, 20-49, ≥50
const anchorHist = new Int32Array(A + 1);    // Schritte mit k haftenden Armen

function bucketHold(n) {
  if (n < 10) return n;
  if (n < 20) return 10;
  if (n < 50) return 11;
  return 12;
}

const gens = [];
for (let g = 0; g < GENS; g++) {
  beginGeneration(tr);
  held.fill(0); holdHist.fill(0); anchorHist.fill(0);
  // prev MUSS den echten Zustand nach dem Respawn tragen (alle Grips gelöst) —
  // sonst zählt Schritt 0 einen Bruch, wo keiner war, und übersieht den ersten
  // Griff. 6 Tiere × 1 Phantom-Ereignis pro Generation verfälscht genau die
  // Größe, um die es hier geht.
  for (let i = 0; i < N; i++) {
    const arms = tr.sim.creatures[i].arms;
    for (let a = 0; a < A; a++) prev[i * A + a] = arms[a].grip.broken ? 1 : 0;
  }
  const st = {
    attach: 0, forceBreak: 0, volRelease: 0, handover: 0, zeroSteps: 0,
    anchoredSum: 0, adhSum: 0, adhN: 0, adhLo: 0, adhMid: 0, adhHi: 0,
    holdSum: 0, holdN: 0, holdMax: 0,
  };

  for (let s = 0; s < STEPS; s++) {
    for (let i = 0; i < N; i++) {
      const arms = tr.sim.creatures[i].arms;
      for (let a = 0; a < A; a++) prev[i * A + a] = arms[a].grip.broken ? 1 : 0;
    }
    trainStep(tr);
    for (let i = 0; i < N; i++) {
      const c = tr.sim.creatures[i], arms = c.arms, ctl = tr.controls[i].arms;
      let anchored = 0;
      for (let a = 0; a < A; a++) {
        const wasBroken = prev[i * A + a] === 1;
        const isBroken = arms[a].grip.broken;
        const adh = ctl[a].adhesion;
        st.adhSum += adh; st.adhN++;
        if (adh < 0.04) st.adhLo++; else if (adh <= 0.08) st.adhMid++; else st.adhHi++;

        if (wasBroken && !isBroken) {
          st.attach++;
          // Wechselgriff: gesetzt, während ein ANDERER Arm noch hält.
          let others = 0;
          for (let b = 0; b < A; b++) if (b !== a && !arms[b].grip.broken) others++;
          if (others > 0) st.handover++;
        } else if (!wasBroken && isBroken) {
          // Ursache: Adhäsion unter 0.04 = vom Netz gewollt, sonst Kraft-Bruch.
          if (adh < 0.04) st.volRelease++; else st.forceBreak++;
          const t = held[i * A + a];
          holdHist[bucketHold(t)]++;
          st.holdSum += t; st.holdN++; if (t > st.holdMax) st.holdMax = t;
          held[i * A + a] = 0;
        }
        if (!isBroken) { anchored++; held[i * A + a]++; }
      }
      st.anchoredSum += anchored;
      anchorHist[anchored]++;
      if (anchored === 0) st.zeroSteps++;
    }
  }
  const r = endGeneration(tr);
  // Term-Zerlegung: „Score 418" sagt nichts darüber, WAS die Selektion trägt.
  const terms = tr.sim.creatures.map(c => scoreTerms(c, tr.opts));
  const fits = tr.sim.creatures.map(c => score(c, tr.opts));
  let bi = 0;
  for (let i = 1; i < fits.length; i++) if (fits[i] > fits[bi]) bi = i;
  const meanTerms = {};
  for (const k of TERM_ORDER) {
    let s = 0; for (const t of terms) s += t[k];
    meanTerms[k] = s / terms.length;
  }
  const sorted = fits.slice().sort((x, y) => y - x);
  const stridesEngine = tr.sim.creatures.reduce((a, c) => a + (c.strides || 0), 0) / tr.sim.creatures.length;
  gens.push({ gen: g, ...r, ...st, stridesEngine, bestScored: sorted[0], medianScored: sorted[sorted.length >> 1],
              bestTerms: terms[bi], bestIdx: bi, meanTerms });
}

const last = gens[gens.length - 1];
const per = (v) => v / N;                     // pro Tier und Episode
const pct = (v) => (100 * v / (N * STEPS)).toFixed(1);

if (!JSON_OUT) {
  console.log(`gaitprobe: ${GENS} Gen × ${POP} Tiere × ${STEPS} Schritte, seed ${SEED}, prey ${PREY}, ${A} Arme`);
  console.log('Frage: ist „Zugzyklen ≈ 1" ein statisches VERHALTEN (A) oder eine blinde METRIK (B)?\n');
  console.log(' gen |  best | Anker | 0-Anker | attach/Tier | Bruch/Tier | lösen/Tier | Wechsel | Ø Haltezeit | Zyklen | Schritte');
  console.log('-----+-------+-------+---------+-------------+------------+------------+---------+-------------+--------+---------');
  for (const r of gens) {
    console.log(
      String(r.gen).padStart(4) + ' | ' + r.best.toFixed(0).padStart(5) + ' | ' +
      (r.anchor * 100).toFixed(0).padStart(4) + '% | ' + pct(r.zeroSteps).padStart(5) + '% | ' +
      per(r.attach).toFixed(1).padStart(11) + ' | ' + per(r.forceBreak).toFixed(1).padStart(10) + ' | ' +
      per(r.volRelease).toFixed(1).padStart(10) + ' | ' + (100 * r.handover / Math.max(1, r.attach)).toFixed(0).padStart(6) + '% | ' +
      (r.holdN ? (r.holdSum / r.holdN).toFixed(1) : '—').padStart(11) + ' | ' +
      r.cycles.toFixed(2).padStart(6) + ' | ' + (r.stridesEngine || 0).toFixed(1).padStart(8));
  }

  console.log('\n── Letzte Generation im Detail ──');
  console.log('  Verteilung „haftende Arme" über alle ' + (N * STEPS) + ' Tier-Schritte:');
  for (let k = 0; k <= A; k++) {
    const n = anchorHist[k];
    if (!n) continue;
    console.log('    ' + String(k).padStart(2) + ' Arme: ' + (100 * n / (N * STEPS)).toFixed(1).padStart(5) + ' %  ' + '█'.repeat(Math.round(50 * n / (N * STEPS))));
  }
  console.log('  Haltezeit eines Grips bis zum Lösen (Schritte):');
  const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10-19', '20-49', '≥50'];
  for (let k = 0; k < holdHist.length; k++) {
    if (!holdHist[k]) continue;
    console.log('    ' + labels[k].padStart(6) + ': ' + String(holdHist[k]).padStart(6) + '  ' + '█'.repeat(Math.min(60, Math.round(60 * holdHist[k] / Math.max(1, last.holdN)))));
  }
  console.log('  Adhäsions-Ausgaben des Netzes: Ø ' + (last.adhSum / Math.max(1, last.adhN)).toFixed(3) +
    ' · <0.04 (lösen) ' + (100 * last.adhLo / Math.max(1, last.adhN)).toFixed(1) + ' %' +
    ' · 0.04…0.08 (Tastzone, kein Re-Grip) ' + (100 * last.adhMid / Math.max(1, last.adhN)).toFixed(1) + ' %' +
    ' · >0.08 (greifen) ' + (100 * last.adhHi / Math.max(1, last.adhN)).toFixed(1) + ' %');
  console.log('  Ereignisse gesamt: ' + last.attach + ' attach, ' + last.forceBreak + ' Kraft-Brüche, ' +
    last.volRelease + ' freiwillige Lösungen, ' + last.handover + ' davon Wechselgriffe (' +
    (100 * last.handover / Math.max(1, last.attach)).toFixed(0) + ' %)');
  console.log('  score() sieht: Ø Zyklen ' + last.cycles.toFixed(2) + ' (maximal erreichbar bei ' + STEPS +
    ' Schritten: ' + Math.min(120, Math.floor(STEPS / 2)) + '), Ø Anker-Rate ' + (last.anchor * 100).toFixed(1) + ' %');

  // ── Fitness-Zerlegung: welche Terme tragen den Score wirklich? ──
  const bt = last.bestTerms, mt = last.meanTerms;
  const sumOf = (t) => TERM_ORDER.reduce((s, k) => s + t[k], 0);
  console.log('\n  Fitness-Zerlegung (Beiträge in Punkten, Tier ' + last.bestIdx + ' = bestes der letzten Generation):');
  console.log('    Term          Gewicht    bestes Tier    Ø Population   Anteil');
  console.log('    ------------  ---------  ------------  -------------  -------');
  const totB = sumOf(bt);
  for (const k of TERM_ORDER) {
    const share = Math.abs(totB) > 1e-9 ? 100 * bt[k] / totB : 0;
    console.log('    ' + k.padEnd(13) + String(FIT_WEIGHTS[k] !== undefined ? FIT_WEIGHTS[k] : '—').padStart(7) +
      bt[k].toFixed(2).padStart(13) + mt[k].toFixed(2).padStart(14) +
      (share.toFixed(1) + ' %').padStart(9));
  }
  console.log('    ' + 'SUMME'.padEnd(13) + ''.padStart(7) + totB.toFixed(2).padStart(13) + sumOf(mt).toFixed(2).padStart(14));
  const progBase = bt.gripPull + bt.reach;
  console.log('    → stride-Term: ' + bt.stride.toFixed(2) + ' von ' + totB.toFixed(0) + ' Punkten = ' +
    (100 * bt.stride / Math.max(1e-9, totB)).toFixed(1) + ' % des Scores');
  console.log('      (multiplikativ: Fortschrittsbasis ' + progBase.toFixed(1) + ' × strideBonus ' +
    FIT_WEIGHTS.strideBonus + ' × min(Schritte,' + FIT_WEIGHTS.strideCap + ')/' + FIT_WEIGHTS.strideCap +
    ' → Deckel hier ' + (progBase * FIT_WEIGHTS.strideBonus).toFixed(1) + ' Punkte)');
  console.log('    → Zähler-Vergleich: Sonde ' + last.handover + ' Wechselgriffe (anderer Arm am Schritt-ENDE haftend),');
  console.log('      postStep() ' + Math.round(last.stridesEngine * N) + ' strides (anderer Arm ÜBER den Schritt hinweg haftend — strenger,');
  console.log('      zählt gleichzeitiges Loslassen-und-Wiedergreifen nicht als Schritt).');

  const perArmCycles = per(last.attach);
  console.log('\n── Diagnose ──');
  const zeroFrac = last.zeroSteps / (N * STEPS);
  const perArmPerEp = last.attach / N / A;         // Zyklen pro Arm und Episode
  const gaitCap = (bt.gripPull + bt.reach) * FIT_WEIGHTS.strideBonus;   // Deckel = Fortschrittsbasis × Bonus
  console.log('  Verhalten: ' + perArmCycles.toFixed(1) + ' Greif-Ereignisse/Tier/Episode = ' +
    perArmPerEp.toFixed(2) + ' pro Arm (' + STEPS + ' Schritte), ' +
    (100 * last.handover / Math.max(1, last.attach)).toFixed(0) + ' % davon als Wechselgriff,');
  console.log('             Ø ' + (last.anchoredSum / (N * STEPS)).toFixed(2) + ' von ' + A +
    ' Armen gleichzeitig haftend, Ø Haltezeit ' + (last.holdN ? (last.holdSum / last.holdN).toFixed(1) : '∞') +
    ' Schritte, längste ' + last.holdMax + '.');
  console.log('  Metrik:    Schritte mit NULL Anker ' + (100 * zeroFrac).toFixed(2) + ' % → score() zählt Ø ' +
    last.cycles.toFixed(2) + ' Zyklen,');
  console.log('             der stride-Term trägt ' + bt.stride.toFixed(1) + ' von möglichen ' + gaitCap.toFixed(0) +
    ' Punkten (' + (100 * bt.stride / gaitCap).toFixed(1) + ' % des Deckels, ' +
    (100 * bt.stride / Math.max(1e-9, totB)).toFixed(1) + ' % des Scores).');
  if (zeroFrac < 0.02 && perArmCycles >= 1) {
    console.log('  ⇒ (B) METRIK blind: Es WIRD umgesetzt (Wechselgriff ' +
      (100 * last.handover / Math.max(1, last.attach)).toFixed(0) + ' %), aber `gripCycles` verlangt den');
    console.log('     Vollverlust aller ' + A + ' Anker in einem Schritt. Bei gestaffelten Brüchen kommt das');
    console.log('     praktisch nie vor (' + (100 * zeroFrac).toFixed(2) + ' % der Schritte) — der alte gait-Term war ein toter');
    console.log('     Gradient, nicht das Verhalten falsch. Die README-Behauptung „einmal festhalten');
    console.log('     und nie mehr loslassen" ist damit FALSCH: es wird gelöst, nur nie alles gleichzeitig.');
  } else if (perArmCycles < 1) {
    console.log('  ⇒ (A) VERHALTEN statisch: unter einem Greif-Ereignis pro Arm und Episode.');
    console.log('     Die Arme halten wirklich dauerhaft — hier ist die Metrik nicht das Problem.');
  } else {
    console.log('  ⇒ Gemischt: ' + perArmCycles.toFixed(1) + ' Ereignisse/Tier bei ' + (100 * zeroFrac).toFixed(1) +
      ' % Schritten ohne Anker — beides spielt eine Rolle.');
  }
  console.log('  Offene Frage: ist der Takt PHYSIKALISCH begrenzt (Muskelzug reißt den Grip nicht,');
  console.log('  Reichweite der Arme) oder SELEKTIONS-seitig (kein Term belohnt schnelleres Umsetzen)?');
  console.log('  → tools/strideprobe.mjs misst den Kopfraum mit getakteten Skript-Controllern.');
} else {
  console.log(JSON.stringify({ opts: { GENS, POP, STEPS, SEED, PREY, arms: A }, gens, holdHist: Array.from(holdHist), anchorHist: Array.from(anchorHist) }, null, 1));
}
