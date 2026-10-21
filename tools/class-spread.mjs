/**
 * Klassen-Spreizungs-Probe — die Messung, die das G7-Pfadlängen-Tor ersetzt.
 *
 * Anlass (Hardware-Messung RTX 3070 Ti, Build 529fdc7, pop 32, seed 7):
 *   Kontrolle (Ordnung gedreht, f64): pathLen -7.9 %
 *   GPU:                              pathLen -14.8 %  → Festwert-Tor ±10 % FAIL
 * Das Tor war enger als die Streuung EINES einzelnen Realisierungs-Paares der
 * Fehlerklasse. Ein Festwert-Tor auf einem chaotischen Aggregat ist nicht
 * kalibrierbar — die SPREIZUNG über viele Klassen-Mitglieder aber messbar.
 *
 * Pro Seed und Variante gemessen: maxΔ Position, Grip-Flag-Übereinstimmung,
 * Ø pathLen-Abweichung und Ø FITNESS-Abweichung (score() — die eigentliche
 * Selektionsgröße; pathLen ist nur ihr Rohbaustein).
 *
 * Varianten (alles gültige Mitglieder derselben Fehlerklasse):
 *   ctrl   Solve-Ordnung komplett gedreht, reines f64  (gröbster Fall)
 *   wpermK K-te Zufallsordnung INNERHALB der Farbklassen, reines f64 —
 *          genau die Ordnungsfreiheit, die die GPU hat (pro Farbe ein
 *          paralleler Dispatch, Farben aufsteigend wie auf der CPU)
 *   f32    f32-Emulation per-op, gleiche Ordnung       (GPU-nächste Arithmetik)
 *
 * Aufruf:  node tools/class-spread.mjs [--pop 32] [--seeds 7,13,21] [--steps 150]
 *                                      [--variants ctrl,f32] [--perms 6] [--json]
 */
import { createTrainer } from '../src/sim/train.js';
import { gpuTrainStep, beginGenerationGpu } from '../src/sim/gputrain.js';
import { score } from '../src/sim/fitness.js';
import { f32ify } from './f32solver.mjs';

const args = process.argv.slice(2);
const arg = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const POP = +arg('pop', 32);
const STEPS = +arg('steps', 150);
const SEEDS = String(arg('seeds', '7,13,21,33,48')).split(',').map(Number);
const PREY = +arg('prey', 2);
const PERMS = +arg('perms', 0);
const AS_JSON = args.includes('--json');
const VARIANTS = [...String(arg('variants', 'ctrl,f32')).split(',')];
for (let k = 0; k < PERMS; k++) VARIANTS.push('wperm' + k);

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const mean = v => v.reduce((s, x) => s + x, 0) / v.length;
const sd = v => { const m = mean(v); return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, v.length - 1)); };

function runVariant(seed, variant) {
  const OPTS = { popSize: POP, stepsPerGen: STEPS, seed, prey: PREY };
  const A = createTrainer(OPTS);
  const B = createTrainer(OPTS);
  beginGenerationGpu(A, null);
  beginGenerationGpu(B, null);

  let emu = null;
  if (variant === 'ctrl') {
    B.sim.world.constraints.reverse();
  } else if (variant.startsWith('f32')) {
    // f32race = Kollisions-Race der GPU (letzter Punkt gewinnt), f32raceLo =
    // erster gewinnt, f32jacobi = alle Deltas summiert aus dem Ursprungs-Zustand
    const collMode = variant === 'f32' ? 'seq'
      : variant === 'f32race' ? 'race'
      : variant === 'f32raceLo' ? 'raceLo'
      : variant === 'f32jacobi' ? 'jacobi'
      : variant === 'f32coll' ? 'collRev' : 'seq';
    emu = f32ify(B.sim.world, B.sim.map, { collMode });
    emu.roundState();
  } else if (variant.startsWith('wperm')) {
    const rnd = mulberry32(2000 + (+variant.slice(5)));
    const w = B.sim.world, arr = w.constraints;
    for (let c = 0; c < w.colorOff.length; c++) {
      const o = w.colorOff[c], n = w.colorCnt[c];
      for (let i = n - 1; i > 0; i--) {
        const j = (rnd() * (i + 1)) | 0;
        const t = arr[o + i]; arr[o + i] = arr[o + j]; arr[o + j] = t;
      }
    }
  }

  let maxD = 0, sumD = 0, mism = 0, tot = 0, brkA = 0, brkB = 0;
  for (let s = 0; s < STEPS; s++) {
    gpuTrainStep(A, null);
    gpuTrainStep(B, null);
    if (emu) emu.roundState();
    for (let i = 0; i < POP; i++) {
      const ca = A.sim.creatures[i], cb = B.sim.creatures[i];
      const d = Math.hypot(ca.x - cb.x, ca.y - cb.y);
      if (d > maxD) maxD = d;
      sumD += d;
      for (let a = 0; a < 6; a++) {
        tot++;
        const fa = ca.arms[a].grip.broken, fb = cb.arms[a].grip.broken;
        brkA += fa ? 1 : 0; brkB += fb ? 1 : 0;
        if (fa !== fb) mism++;
      }
    }
  }
  const plA = A.sim.creatures.map(c => c.pathLen), plB = B.sim.creatures.map(c => c.pathLen);
  const fA = A.sim.creatures.map(c => score(c, OPTS)), fB = B.sim.creatures.map(c => score(c, OPTS));
  const plD = plB.map((v, i) => v - plA[i]), fD = fB.map((v, i) => v - fA[i]);
  return {
    seed, variant, maxD, meanD: sumD / (STEPS * POP),
    flags: 100 * (1 - mism / tot), brkA, brkB,
    plA: mean(plA), plDev: 100 * (mean(plB) - mean(plA)) / Math.max(1e-9, mean(plA)),
    plSemRel: 100 * sd(plD) / Math.sqrt(POP) / Math.max(1e-9, mean(plA)),
    fA: mean(fA), fDev: 100 * (mean(fB) - mean(fA)) / Math.max(1e-9, Math.abs(mean(fA))),
    fSemRel: 100 * sd(fD) / Math.sqrt(POP) / Math.max(1e-9, Math.abs(mean(fA))),
    fSpreadA: 100 * sd(fA) / Math.max(1e-9, Math.abs(mean(fA))),
  };
}

const rows = [];
for (const seed of SEEDS) {
  for (const variant of VARIANTS) {
    const t0 = Date.now();
    const r = runVariant(seed, variant);
    r.ms = Date.now() - t0;
    rows.push(r);
    if (!AS_JSON) console.log(
      `seed ${String(seed).padStart(3)} ${variant.padEnd(6)} | maxΔ ${r.maxD.toFixed(2).padStart(7)} px | flags ${r.flags.toFixed(2).padStart(6)} % | ` +
      `pathLen ${r.plA.toFixed(2).padStart(7)} dev ${r.plDev.toFixed(2).padStart(7)} % (±${r.plSemRel.toFixed(2)} SEM) | ` +
      `fit ${r.fA.toFixed(1).padStart(7)} dev ${r.fDev.toFixed(2).padStart(7)} % (±${r.fSemRel.toFixed(2)} SEM, Tier-SD ${r.fSpreadA.toFixed(1)} %) | ${r.ms} ms`);
  }
}

if (AS_JSON) { console.log(JSON.stringify(rows, null, 1)); }
else {
  console.log('');
  for (const variant of VARIANTS) {
    const rs = rows.filter(r => r.variant === variant);
    if (!rs.length) continue;
    const f = (k, u) => {
      const v = rs.map(r => r[k]).sort((a, b) => a - b);
      return `min ${v[0].toFixed(2)} · med ${v[v.length >> 1].toFixed(2)} · max ${v[v.length - 1].toFixed(2)} ${u}`;
    };
    console.log(`${variant.padEnd(6)} (${rs.length} Seeds)  pathLen-Dev  ${f('plDev', '%')}`);
    console.log(`${''.padEnd(6)}             Fitness-Dev  ${f('fDev', '%')}`);
    console.log(`${''.padEnd(6)}             maxΔ         ${f('maxD', 'px')}`);
  }
  const all = rows.filter(r => r.variant !== 'f32');
  if (all.length > 1) {
    console.log(`\nALLE Ordnungs-Varianten (${all.length} Läufe, ohne f32):`);
    console.log(`  pathLen-Dev  Ø ${mean(all.map(r => r.plDev)).toFixed(2)} % · SD ${sd(all.map(r => r.plDev)).toFixed(2)} pp · Bereich [${Math.min(...all.map(r => r.plDev)).toFixed(2)}, ${Math.max(...all.map(r => r.plDev)).toFixed(2)}] %`);
    console.log(`  Fitness-Dev  Ø ${mean(all.map(r => r.fDev)).toFixed(2)} % · SD ${sd(all.map(r => r.fDev)).toFixed(2)} pp · Bereich [${Math.min(...all.map(r => r.fDev)).toFixed(2)}, ${Math.max(...all.map(r => r.fDev)).toFixed(2)}] %`);
    console.log(`  Ø SEM pro Lauf: pathLen ${mean(all.map(r => r.plSemRel)).toFixed(2)} %, Fitness ${mean(all.map(r => r.fSemRel)).toFixed(2)} %`);
  }
}
