#!/usr/bin/env node
/**
 * Abnahme-Harness für Milestone 1 — CPU-Referenzlöser.
 *
 * Dieser Harness ist das ORAKEL für die WebGPU-Portierung: dieselben Checks
 * müssen später gegen den WGSL-Solver bestehen, sonst ist ein Shader-Fehler
 * nicht von einem Designproblem zu unterscheiden.
 *
 * Die Check-Liste ist direkt aus den gemessenen Fehlern von Kraken_v2 abgeleitet.
 * Jeder Check nennt die Zahl, die dort falsch war.
 *
 *   node tools/harness.js
 *   node tools/harness.js --only passive,grip
 */
import { XpbdWorld } from '../src/physics/xpbd.js';
import { makeMap, findSpawn, mulberry32, coverageAt } from '../src/world/map.js';
import {
  spawnKraken, tipPosition, attachGrip, releaseGrip, armCurl, armExtension,
  toneToCompliance, adhesionToBreakForce, BODY, ARM_REACH,
} from '../src/body/kraken.js';
import { createSim, stepSim, readState, readProprio, applyControl, PROPRIO_DIM } from '../src/sim/core.js';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const argv = process.argv.slice(2);
const onlyIdx = argv.indexOf('--only');
const only = onlyIdx >= 0 ? argv[onlyIdx + 1].split(',') : null;

const results = [];
function check(id, name, pass, detail) {
  results.push({ id, name, pass, detail });
  console.log(`  ${pass ? '✅' : '❌'} ${id.padEnd(11)} ${name.padEnd(44)} ${detail}`);
}
function section(t) { console.log('\n' + '═'.repeat(78) + '\n ' + t + '\n' + '═'.repeat(78)); }
const want = (id) => !only || only.includes(id);

// ══════════════════════════════════════════════════════════════
section('CHECK 1 — PASSIVES GLEICHGEWICHT (der wichtigste Test)');
if (want('passive')) {
  // Kraken_v2: ein komplett toter Aktuator-Satz produzierte 964 px Pfad und
  // max|v| = 69 px/step im ersten Frame (matter-js angularStiffness=0 injizierte
  // Winkelgeschwindigkeit über `bodyA.angle -= p` ohne anglePrev mitzunehmen).
  const sim = createSim({ seed: 7, creatures: 1 });
  const c = sim.creatures[0];
  for (let i = 0; i < 960; i++) stepSim(sim);
  const disp = Math.hypot(c.x - c.spawn.x, c.y - c.spawn.y);
  check('passive', 'Passiv exakt in Ruhe (8 s, kein Antrieb)',
    c.pathLen < 1e-9 && disp < 1e-9 && sim.world.stats.nan === 0,
    `Pfad=${c.pathLen.toExponential(2)}px  Verschiebung=${disp.toExponential(2)}px  max|ω|=${sim.world.stats.maxAngSpeed.toExponential(2)}  (v2: 964 px)`);
}

// ══════════════════════════════════════════════════════════════
section('CHECK 2 — SPAWN OHNE CONSTRAINT-VERLETZUNG');
if (want('spawn')) {
  // Kraken_v2: Schulter 56.94 px / Segmente 26.99 px Verletzung, weil die Pose
  // NACH Constraint.create() gesetzt wurde und Matter pointA nur inkrementell
  // ab dem eingefrorenen angleA rotiert.
  const world = new XpbdWorld();
  const map = makeMap(1440, 810, { seed: 3 });
  world.sdf = map.sdf;
  const k = spawnKraken(world, 720, 405, 0.7);
  let worst = 0, worstAng = 0, n = 0;
  for (const c of world.constraints) {
    if (c.broken) continue;
    const A = c.a, B = c.b;
    const ca = Math.cos(world.ang[A]), sa = Math.sin(world.ang[A]);
    const gax = world.px[A] + c.lax * ca - c.lay * sa;
    const gay = world.py[A] + c.lax * sa + c.lay * ca;
    if (c.type === 'angle') continue;
    if (c.compliance > 0 && c.type !== 0 && c.type !== 2) continue;
    if (c.type === 1) continue;   // Winkel-Constraints haben keine Punkt-Verletzung
    let gbx, gby;
    if (B >= 0) {
      const cb = Math.cos(world.ang[B]), sb = Math.sin(world.ang[B]);
      gbx = world.px[B] + c.lbx * cb - c.lby * sb;
      gby = world.py[B] + c.lbx * sb + c.lby * cb;
    } else { gbx = c.lbx; gby = c.lby; }
    const v = Math.hypot(gax - gbx, gay - gby);
    if (v > worst) worst = v;
    n++;
    // Winkel-Targets gegen die Ist-Pose
  }
  // separately: Abweichung der Motor-Ziele von der Ist-Pose
  for (const arm of k.arms) {
    for (const m of arm.motors) {
      const A = m.a, B = m.b;
      const rel = B >= 0 ? Math.atan2(Math.sin(world.ang[B] - world.ang[A]), Math.cos(world.ang[B] - world.ang[A])) : world.ang[A];
      const dev = Math.atan2(Math.sin(rel - m.rest), Math.cos(rel - m.rest));
      const d = Math.abs(dev - m.target);
      if (d > worstAng) worstAng = d;
    }
  }
  check('spawn', 'Pin-Constraints verletzungsfrei bei t=0', worst < 1e-9,
    `max=${worst.toExponential(3)} px über ${n} Gelenke  (v2: 56.94 px)`);
  check('spawn-motor', 'Muskelziele = Spawn-Pose (Fixpunkt)', worstAng < 1e-9,
    `max=${worstAng.toExponential(3)} rad  (v2: Arme wurden beim Spawn zusammengefaltet)`);
}

// ══════════════════════════════════════════════════════════════
section('CHECK 3 — GRIP-AND-PULL: zieht ein verankerter Arm den Mantel?');
if (want('grip')) {
  // Der einzig vorgesehene Antrieb. Kalibrierung in Kraken_v2: Zugweite war von
  // Muskelmoment (0.004…1.2) und Rumpfmasse (8…40) praktisch unabhängig,
  // 109…118 px = 92…99 % der Reichweite. Hier muss dasselbe gelten.
  const rows = [];
  for (const tone of [0.3, 0.65, 1.0]) {
    const sim = createSim({ seed: 11, creatures: 1 });
    const c = sim.creatures[0];
    const arm = c.arms[0];
    // Spitze verankern (Haftung am Untergrund, überall möglich)
    attachGrip(sim.world, arm, adhesionToBreakForce(1.0));
    const d0 = Math.hypot(c.x - c.spawn.x, c.y - c.spawn.y);
    const anchor = { ...tipPosition(sim.world, arm) };
    const startDist = Math.hypot(sim.world.px[c.mantle] - anchor.x, sim.world.py[c.mantle] - anchor.y);
    let minDist = startDist, broke = false;
    for (let i = 0; i < 600; i++) {
      // Arm vollständig einknicken, alle anderen Arme schlaff
      applyControl(c, {
        tone,
        arms: c.arms.map((_, j) => j === 0
          ? { shoulder: 0, curl: BODY.jointLimit * 0.98, adhesion: 1 }
          : { shoulder: 0, curl: 0, adhesion: 0 }),
      });
      stepSim(sim);
      if (arm.grip.broken) { broke = true; break; }
      const d = Math.hypot(sim.world.px[c.mantle] - anchor.x, sim.world.py[c.mantle] - anchor.y);
      if (d < minDist) minDist = d;
    }
    rows.push({ tone, startDist, pulled: startDist - minDist, pct: (startDist - minDist) / startDist * 100, broke, netDisp: Math.hypot(c.x - c.spawn.x, c.y - c.spawn.y) });
  }
  console.log('    Tonus | Mantel→Anker | herangezogen | Anteil | abgerissen | Mantel netto');
  for (const r of rows) {
    console.log('    ' + r.tone.toFixed(2).padStart(5) + ' | ' + r.startDist.toFixed(1).padStart(12) + 'px | ' +
      r.pulled.toFixed(1).padStart(10) + 'px | ' + r.pct.toFixed(0).padStart(5) + '% | ' +
      (r.broke ? '     JA' : '    nein').padStart(10) + ' | ' + r.netDisp.toFixed(1).padStart(9) + 'px');
  }
  const best = rows.reduce((a, b) => (b.pct > a.pct ? b : a));
  check('grip', 'Verankerter Arm zieht den Mantel (>60 % der Distanz)',
    best.pct > 60,
    `bestes Ergebnis ${best.pulled.toFixed(1)} px = ${best.pct.toFixed(0)} % bei Tonus ${best.tone}  (v2: 92…99 %)`);
}

// ══════════════════════════════════════════════════════════════
section('CHECK 4 — BEGRENZTE HALTEKRAFT: reißt der Grip bei echter Kraft?');
if (want('break')) {
  // Gewünscht: Saugnapf löst sich, wenn eine Kraft den Körper in eine andere
  // Richtung zerrt. Der Bruch wird über den Lagrange-Multiplikator geprüft,
  // also über die echte Kraft — nicht über eine Dehnungsschwelle (v2: 22 px).
  const sim = createSim({ seed: 13, creatures: 1 });
  const c = sim.creatures[0];
  const arm = c.arms[0];
  attachGrip(sim.world, arm, adhesionToBreakForce(0.05));   // schwach gestellt
  const fWeak = adhesionToBreakForce(0.05);
  let brokeWeak = false, stepsWeak = 0;
  for (let i = 0; i < 400; i++) {
    // Andere Arme ziehen kräftig in eine andere Richtung
    applyControl(c, {
      tone: 1.0,
      arms: c.arms.map((_, j) => j === 0
        ? { shoulder: 0, curl: BODY.jointLimit * 0.9, adhesion: 0.05 }
        : { shoulder: (j % 2 ? 1 : -1) * BODY.shoulderLimit * 0.8, curl: BODY.jointLimit * 0.9, adhesion: 0 }),
    });
    stepSim(sim); stepsWeak++;
    if (arm.grip.broken) { brokeWeak = true; break; }
  }
  const peakForce = arm.grip.force || 0;
  check('break', 'Schwach gestellter Grip reißt unter Zug', brokeWeak,
    `Grenz=${fWeak.toFixed(0)} N, Spitze=${peakForce.toFixed(0)} N, nach ${stepsWeak} Schritten`);

  const sim2 = createSim({ seed: 13, creatures: 1 });
  const c2 = sim2.creatures[0];
  const arm2 = c2.arms[0];
  attachGrip(sim2.world, arm2, adhesionToBreakForce(1.0));
  const fStrong = adhesionToBreakForce(1.0);
  for (let i = 0; i < 400; i++) {
    applyControl(c2, {
      tone: 1.0,
      arms: c2.arms.map((_, j) => j === 0
        ? { shoulder: 0, curl: BODY.jointLimit * 0.9, adhesion: 1 }
        : { shoulder: (j % 2 ? 1 : -1) * BODY.shoulderLimit * 0.8, curl: BODY.jointLimit * 0.9, adhesion: 0 }),
    });
    stepSim(sim2);
  }
  check('break-hold', 'Voll gestellter Grip hält unter demselben Zug', !arm2.grip.broken,
    `Grenz=${fStrong.toFixed(0)} N, Spitze=${(arm2.grip.force || 0).toFixed(0)} N`);
}

// ══════════════════════════════════════════════════════════════
section('CHECK 5 — DAUERLAUF: NaN, Grenzen, Kartenrand');
if (want('long')) {
  const sim = createSim({ seed: 17, creatures: 4 });
  const N = 1200;
  let peakV = 0, peakAV = 0, out = 0, nan = 0;
  const ctrl = { tone: 1, arms: sim.creatures[0].arms.map(() => ({ shoulder: 0, curl: 0, adhesion: 0.5 })) };
  for (let i = 0; i < N; i++) {
    for (const c of sim.creatures) {
      // zufällig angesteuert, damit alle Pfade belastet werden
      const rnd = Math.sin(i * 0.37 + c.index * 2.1);
      applyControl(c, {
        tone: 0.5 + 0.5 * rnd,
        arms: c.arms.map((_, j) => ({
          shoulder: Math.sin(i * 0.11 + j) * BODY.shoulderLimit,
          curl: Math.sin(i * 0.23 + j * 1.7) * BODY.jointLimit,
          adhesion: Math.max(0, Math.sin(i * 0.07 + j * 0.9)),
        })),
      });
    }
    stepSim(sim);
    const w = sim.world;
    if (w.stats.maxSpeed > peakV) peakV = w.stats.maxSpeed;
    if (w.stats.maxAngSpeed > peakAV) peakAV = w.stats.maxAngSpeed;
    nan += w.stats.nan;
    for (let b = 0; b < w.n; b++) {
      if (!Number.isFinite(w.px[b]) || !Number.isFinite(w.py[b])) { nan++; continue; }
      if (w.px[b] < -60 || w.px[b] > 1500 || w.py[b] < -60 || w.py[b] > 870) out++;
    }
  }
  check('long-nan', 'Kein NaN über ' + N + ' Schritte', nan === 0, `NaN-Ereignisse: ${nan}`);
  check('long-oob', 'Nichts fliegt aus der Karte', out === 0, `außerhalb: ${out}  (v2: 129)`);
  check('long-v', 'Geschwindigkeit im Limit', peakV <= sim.world.maxSpeed * 1.001, `peak|v|=${peakV.toFixed(1)} px/s, Limit ${sim.world.maxSpeed}  (v2: 4188 px/s)`);
  check('long-av', 'Winkelgeschwindigkeit im Limit', peakAV <= sim.world.maxAngSpeed * 1.001, `peak|ω|=${peakAV.toFixed(2)} rad/s, Limit ${sim.world.maxAngSpeed}  (v2: 3918 rad/step)`);
}

// ══════════════════════════════════════════════════════════════
section('CHECK 6 — MASSENVERHÄLTNIS (Solver-Stabilität)');
if (want('mass')) {
  // v2: Rumpf 26 / Spitze 0.48 mit degenerierter Geometrie = 250:1 → instabil.
  const sim = createSim({ seed: 5, creatures: 1 });
  const w = sim.world;
  const mantleMass = 1 / w.im[sim.creatures[0].mantle];
  let minSeg = Infinity, maxSeg = 0;
  for (const arm of sim.creatures[0].arms) for (const b of arm.segs) {
    const m = 1 / w.im[b]; if (m < minSeg) minSeg = m; if (m > maxSeg) maxSeg = m;
  }
  const ratio = mantleMass / minSeg;
  check('mass', 'Rumpf : distales Segment ≤ 12:1', ratio <= 12,
    `${mantleMass.toFixed(1)} : ${minSeg.toFixed(2)} = ${ratio.toFixed(1)}:1  (v2: 250:1)`);
  check('mass-taper', 'Taper entlang des Arms ≤ 2:1', maxSeg / minSeg <= 2.001,
    `${maxSeg.toFixed(2)} → ${minSeg.toFixed(2)} = ${(maxSeg / minSeg).toFixed(2)}:1`);
}

// ══════════════════════════════════════════════════════════════
section('CHECK 7 — DETERMINISMUS (Voraussetzung für Messreihen)');
if (want('determinism')) {
  const run = (seed) => {
    const sim = createSim({ seed, creatures: 2 });
    for (let i = 0; i < 300; i++) {
      for (const c of sim.creatures) {
        applyControl(c, {
          tone: 0.8,
          arms: c.arms.map((_, j) => ({ shoulder: Math.sin(i * 0.1 + j) * 0.6, curl: Math.sin(i * 0.2 + j) * 0.7, adhesion: 0.6 })),
        });
      }
      stepSim(sim);
    }
    return sim.creatures.map(c => [c.x, c.y, c.angle]).flat().join('|');
  };
  const a = run(42), b = run(42), c = run(43);
  check('determinism', 'Gleicher Seed → bit-identisches Ergebnis', a === b, a === b ? 'reproduzierbar' : 'DIVERGENT');
  check('determinism-seed', 'Anderer Seed → anderes Ergebnis', a !== c, 'Seed wirkt');
}

// ══════════════════════════════════════════════════════════════
section('CHECK 8 — PROPRIOZEPTION vollständig und belegt');
if (want('proprio')) {
  const sim = createSim({ seed: 23, creatures: 1 });
  const c = sim.creatures[0];
  const p = new Float64Array(PROPRIO_DIM);
  // Arme bewegen, damit nicht alles im Ruhezustand gemessen wird
  for (let i = 0; i < 120; i++) {
    applyControl(c, { tone: 0.9, arms: c.arms.map((_, j) => ({ shoulder: Math.sin(i * 0.15 + j) * 0.8, curl: Math.sin(i * 0.3 + j) * 0.9, adhesion: 0 })) });
    readProprio(c, p);
    stepSim(sim);
  }
  let nz = 0, nonfinite = 0, outOfRange = 0;
  for (let i = 0; i < p.length; i++) {
    if (!Number.isFinite(p[i])) nonfinite++;
    if (Math.abs(p[i]) > 1e-9) nz++;
    if (Math.abs(p[i]) > 1.5) outOfRange++;
  }
  check('proprio-fill', `Propriozeption belegt (${PROPRIO_DIM} Eingänge)`, nz >= PROPRIO_DIM * 0.6,
    `${nz}/${PROPRIO_DIM} Werte ≠ 0  (v2: 64 Eingänge, davon 0 Gelenkwinkel)`);
  check('proprio-finite', 'Keine NaN/Inf in den Eingängen', nonfinite === 0, `${nonfinite} ungültig`);
  check('proprio-range', 'Werte normiert (|x| ≤ 1.5)', outOfRange === 0, `${outOfRange} außerhalb`);
}

// ══════════════════════════════════════════════════════════════
section('CHECK 9 — KERN IST DOM-FREI (Voraussetzung für Worker + GPU)');
if (want('domfree')) {
  const dirs = ['src/physics', 'src/world', 'src/body', 'src/sim'];
  const hits = [];
  for (const d of dirs) {
    const abs = join(ROOT, d);
    let files = [];
    try { files = readdirSync(abs).filter(f => f.endsWith('.js')); } catch { continue; }
    for (const f of files) {
      const src = readFileSync(join(abs, f), 'utf8');
      for (const pat of ['document', 'window', 'canvas', 'requestAnimationFrame', 'localStorage', 'getContext']) {
        const re = new RegExp('\\b' + pat + '\\b');
        // Kommentare ignorieren
        const stripped = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
        if (re.test(stripped)) hits.push(`${d}/${f}: ${pat}`);
      }
    }
  }
  check('domfree', 'physics/world/body/sim ohne jede DOM-Referenz', hits.length === 0,
    hits.length ? hits.join(', ') : 'rein — läuft unverändert in Worker und Headless');
}

// ══════════════════════════════════════════════════════════════
section('CHECK 10 — PERFORMANCE');
if (want('perf')) {
  let cps1 = 0, cps64 = 0;   // Kraken-Schritte/s (nicht Welt-Schritte/s!)
  for (const n of [1, 8, 64]) {
    const sim = createSim({ seed: 31, creatures: n });
    const ctrl = sim.creatures[0].arms.map(() => ({ shoulder: 0.3, curl: 0.4, adhesion: 0.3 }));
    const t0 = Date.now();
    let steps = 0;
    while (Date.now() - t0 < 1500) {
      for (const c of sim.creatures) applyControl(c, { tone: 0.7, arms: ctrl });
      stepSim(sim); steps++;
    }
    const ms = Date.now() - t0;
    const sps = steps / (ms / 1000);
    if (n === 1) cps1 = sps * n;
    if (n === 64) cps64 = sps * n;
    console.log(`    ${String(n).padStart(3)} Kraken: ${sps.toFixed(0).padStart(6)} Sim-Schritte/s  = ${(sps * n).toFixed(0).padStart(8)} Kraken-Schritte/s   (${(ms / steps).toFixed(2)} ms/Schritt)`);
  }
  check('perf-single', '≥ 500 Kraken-Schritte/s bei 1 Tier (v2: 101)', cps1 >= 500, `${cps1.toFixed(0)} steps/s = ${(cps1/101).toFixed(1)}× Kraken_v2`);
  // ACHTUNG — hier wird LINEARE Skalierung geprüft, nicht superlineare.
  // Auf der CPU kann Batching den Durchsatz nicht vervielfachen; es amortisiert
  // nur Fixkosten und — entscheidend — erzwingt dasselbe Datenlayout, das der
  // GPU-Dispatch braucht. Der superlineare Gewinn kommt erst aus der räumlichen
  // Parallelität des Shaders. Was hier garantiert sein muss: die Kosten PRO TIER
  // dürfen mit der Batch-Größe nicht wachsen. Der erste Entwurf war O(n²)
  // (w.step() pro Krake statt pro Sim-Schritt) und lieferte 59 Kraken-Schritte/s
  // bei 64 Tieren statt 3622 — das hätte die gesamte GPU-These zerstört.
  check('perf-linear', 'Kosten pro Tier wachsen nicht mit dem Batch (≥ 50 % Effizienz bei 64×)',
    cps64 >= cps1 * 0.5,
    `${cps64.toFixed(0)} vs ${cps1.toFixed(0)} Kraken-Schritte/s = ${(cps64 / cps1 * 100).toFixed(0)} % Effizienz bei 64× Batch`);
}

section('CHECK 11 — KARTEN-ABDECKUNG / SPAWN');
if (want('map')) {
  const map = makeMap(1440, 810, { seed: 7 });
  const rnd = mulberry32(99);
  let ok = 0;
  for (let s = 0; s < 20; s++) {
    const m = makeMap(1440, 810, { seed: s + 1 });
    const sp = findSpawn(m, ARM_REACH + 20, mulberry32(s + 500));
    const h = m.sdf(sp.x, sp.y);
    if (h && h.d >= ARM_REACH) ok++;
  }
  check('map-spawn', 'Spawn mit voller Armfreiheit in 20 Karten', ok >= 18, `${ok}/20 Karten (Clearance ${ARM_REACH.toFixed(1)} px)`);
  const cov = coverageAt(map, ARM_REACH, 3000, mulberry32(5));
  console.log(`    Wände in Armreichweite: ${(cov.coverage * 100).toFixed(1)} %  (Ø Abstand ${cov.meanDist.toFixed(1)} px)`);
  console.log(`    Hinweis: irrelevant für den Halt — der Saugnapf haftet am Untergrund überall,`);
  console.log(`             nicht nur an Wänden. Wände sind Hindernisse, keine Greifpunkte.`);
  check('map-nogrip', 'Keine dedizierten Greifpunkte in der Karte', !/GRIP_SPACING|gripPoints|nub/i.test(readFileSync(join(ROOT, 'src/world/map.js'), 'utf8')),
    'nur Rand + zufälliger Korridorraster');
}

// ══════════════════════════════════════════════════════════════
section('ZUSAMMENFASSUNG');
const pass = results.filter(r => r.pass).length;
for (const r of results) if (!r.pass) console.log(`  ❌ ${r.id}: ${r.name} — ${r.detail}`);
console.log(`\n  ${pass}/${results.length} Checks bestanden.`);
console.log(`  Körper: ${BODY.arms} Arme × ${BODY.segsPerArm} Segmente = ${1 + BODY.arms * BODY.segsPerArm} Körper/Krake, Reichweite ${ARM_REACH.toFixed(1)} px`);
process.exit(pass === results.length ? 0 : 2);
