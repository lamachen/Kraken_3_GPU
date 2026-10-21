#!/usr/bin/env node
/**
 * Messstand für den Viewer-Gang: warum sieht der Benutzer einen Kreisel?
 *
 * Ausgangslage (Benutzerbeobachtung, Build 57da2d4): der Platzhalter-Controller
 * aus src/render/demo.js produziert eine schnell rotierende Krake OHNE
 * Fortbewegung. Diese Beobachtung wird hier zur Zahl gemacht und gegen
 * getaktete Kandidaten-Gänge gestellt — derselbe Messstand wie
 * tools/strideprobe.mjs: frischer Trainer pro Konfiguration mit gleichem Seed
 * ⇒ identische Spawns, nur der Controller unterscheidet sich.
 *
 *   node tools/demogait-probe.mjs [--steps 500] [--pop 16] [--seed 7]
 *                                 [--long 1500] [--longpop 8] [--json]
 *
 * Ausgabe: netto |d| (Distanz Spawn→Ende), Pfadlänge, Umdrehungen
 * (angularAccum/2π), Ankeranteil — Mittel über alle Tiere.
 */
import { stepSim, applyControl } from '../src/sim/core.js';
import { createTrainer, beginGeneration } from '../src/sim/train.js';
import { demoControl } from '../src/render/demo.js';

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf('--' + n); return i >= 0 ? Number(argv[i + 1]) : d; };
const STEPS = arg('steps', 500), POP = arg('pop', 16), SEED = arg('seed', 7);
const LONG = arg('long', 1500), LONGPOP = arg('longpop', 8);
const JSON_OUT = argv.includes('--json');

// ── Kandidaten ──────────────────────────────────────────────────────
// legacy: der bisherige Platzhalter (Phasenversatz pro Arm, Sinus-Schulter)
const CONFIGS = [
  ['legacy demoControl (Platzhalter)', (c, s) => demoControl(c, s)],
];

// getakteter Greif-Zug-Gang aus tools/strideprobe.mjs (dort gemessen:
// Skript-Gänge schlagen das evolvierte Netz deutlich)
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
for (const T of [16, 32, 64]) {
  CONFIGS.push([`wave   T=${T}`, timedGait(T, 'wave')]);
  CONFIGS.push([`tripod T=${T}`, timedGait(T, 'tripod')]);
}
CONFIGS.push(['sync   T=32', timedGait(32, 'sync')]);

// ── Messlauf: eine Konfiguration über POP identische Spawns ─────────
function run(name, fn, steps, pop) {
  const tr = createTrainer({ popSize: pop, stepsPerGen: steps, seed: SEED, prey: 0 });
  beginGeneration(tr);                                   // ⇒ identische Spawns
  const cs = tr.sim.creatures;
  const ctls = cs.map(c => ({
    tone: 0.5, emitEcho: false,
    arms: c.arms.map(() => ({ shoulder: 0, curl: 0, adhesion: 0 })),
  }));
  for (let s = 0; s < steps; s++) {
    for (let i = 0; i < cs.length; i++) {
      const out = fn(cs[i], s, ctls[i]);
      applyControl(cs[i], out);
    }
    stepSim(tr.sim);
  }
  const n = cs.length;
  const mean = (f) => cs.reduce((s, c) => s + f(c), 0) / n;
  return {
    name, steps, pop,
    net: mean(c => Math.hypot(c.x - c.spawn.x, c.y - c.spawn.y)),
    path: mean(c => c.pathLen),
    spin: mean(c => c.angularAccum / (Math.PI * 2)),
    anchor: mean(c => (c.steps ? c.anchoredSteps / c.steps : 0)) * 100,
  };
}

const LONGONLY = argv.includes('--longonly');
const LONG_NAMES = ['legacy demoControl (Platzhalter)', 'wave   T=32', 'wave   T=64', 'sync   T=32'];
let rows, best;
if (LONGONLY) {
  rows = [];
  best = { name: LONG_NAMES[1] };
} else {
  rows = CONFIGS.map(([n, f]) => run(n, f, STEPS, POP));
  best = rows.slice(1).reduce((a, b) => (b.net > a.net ? b : a), rows[1]);
}
const longRuns = (LONGONLY
  ? LONG_NAMES.map(n => [n, CONFIGS.find(([m]) => m === n)[1]])
  : [[rows[0].name, CONFIGS[0][1]], [best.name, CONFIGS.find(([n]) => n === best.name)[1]]]
).map(([n, f]) => run(n, f, LONG, LONGPOP));
if (LONGONLY) best = longRuns.slice(1).reduce((a, b) => (b.net > a.net ? b : a), longRuns[1]);

if (JSON_OUT) {
  console.log(JSON.stringify({ rows, longRuns, best: best.name }, null, 2));
} else {
  console.log(`Demogait-Messstand — pop ${POP}, seed ${SEED}, ${STEPS} Schritte (identische Spawns)`);
  console.log('  Konfiguration                    netto|d| px   Pfad px   Umdreh.   Anker %');
  for (const r of rows) {
    console.log(`  ${r.name.padEnd(32)} ${r.net.toFixed(1).padStart(8)} ${r.path.toFixed(1).padStart(10)} ${r.spin.toFixed(2).padStart(9)} ${r.anchor.toFixed(1).padStart(9)}`);
  }
  console.log(`  Beste Skript-Konfiguration: ${best.name}`);
  console.log(`  Langlauf (${LONG} Schritte, pop ${LONGPOP}):`);
  for (const r of longRuns) {
    console.log(`  ${r.name.padEnd(32)} ${r.net.toFixed(1).padStart(8)} ${r.path.toFixed(1).padStart(10)} ${r.spin.toFixed(2).padStart(9)} ${r.anchor.toFixed(1).padStart(9)}`);
  }
}
