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
import { createSim, stepSim, readState, readProprio, readSenses, applyControl, tickSenses, PROPRIO_DIM, SENSE_DIM, INPUT_DIM } from '../src/sim/core.js';
import { createEcho, emit as echoEmit, tick as echoTick, raymarch, ECHO_RAYS, SOUND_SPEED } from '../src/senses/echo.js';
import { createChem, sample as chemSample, gradientDirection } from '../src/senses/chem.js';
import { createLight, sample as lightSample } from '../src/senses/light.js';
import { createHear, sample as hearSample, bearing as hearBearing } from '../src/senses/hear.js';
import { makeEntity, tickEntity, KIND_PREY } from '../src/world/entities.js';
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
  const dirs = ['src/physics', 'src/world', 'src/body', 'src/sim', 'src/senses'];
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

section('CHECK S1 — ECHOLOT: Distanz, Laufzeit, Cooldown');
if (want('echo')) {
  const map = makeMap(1440, 810, { seed: 7 });
  // (a) Distanzgenauigkeit gegen die echte SDF-Distanz
  const wall = map.rects[0];                       // obere Randwand
  const y = wall.y + wall.h + 60;
  const trueD = 60;
  const measured = raymarch(map.sdf, 720, y, -Math.PI / 2, 340);
  check('echo-dist', 'Echolot-Distanz stimmt mit der Geometrie überein', Math.abs(measured - trueD) < 3,
    `gemessen ${measured.toFixed(1)} px, tatsächlich ${trueD} px (Fehler ${(measured - trueD).toFixed(2)} px)`);

  // (b) Laufzeit: das Echo darf NICHT im selben Schritt da sein
  const e = createEcho();
  echoEmit(e, map, 720, y, -Math.PI / 2, 0);
  echoTick(e, 1 / 60, 1 / 60);
  const early = e.range_.filter(v => v > 0).length;
  const eta = e.pending.length ? e.pending[0].at : 0;
  const expectEta = (2 * measured) / SOUND_SPEED;
  let t = 0, arrived = 0;
  while (t < 0.5 && e.pending.length) { t += 1 / 60; echoTick(e, 1 / 60, t); }
  arrived = e.range_.filter(v => v > 0).length;
  check('echo-latency', 'Echo kommt verzögert an (2·d/c), nicht sofort', early === 0 && arrived > 0,
    `nach 1 Frame: ${early} Strahlen; ETA ${eta.toFixed(4)} s, erwartet ${expectEta.toFixed(4)} s; nach ${t.toFixed(2)} s: ${arrived}/${ECHO_RAYS}`);

  // (c) Cooldown: eine SOFORTIGE Neu-Aussendung muss blockiert sein.
  // (Der erste Entwurf maß das nach Ablauf der Wartezeit — dann ist der
  //  Cooldown korrekt frei und der Test war sinnlos.)
  const ec = createEcho();
  echoEmit(ec, map, 720, y, -Math.PI / 2, 0);
  const n1 = ec.emitCount;
  echoEmit(ec, map, 720, y, -Math.PI / 2, 0.01);          // 10 ms später
  const n2 = ec.emitCount;
  for (let tt = 1 / 60; tt <= 0.5; tt += 1 / 60) echoTick(ec, 1 / 60, tt);   // Refraktärzeit abwarten
  echoEmit(ec, map, 720, y, -Math.PI / 2, 0.5);
  const n3 = ec.emitCount;
  check('echo-cooldown', 'Aussendung hat Refraktärzeit, die danach wieder frei ist',
    n2 === n1 && n3 === n1 + 1,
    `sofort: ${n1}→${n2} (blockiert), nach ${ec.interval} s: →${n3} (frei)`);

  // (d) Wer ortet, verrät sich — Loudness ist Teil der Schnittstelle
  const e2 = createEcho();
  const loud = echoEmit(e2, map, 720, y, -Math.PI / 2, 0);
  check('echo-loud', 'Aussendung ist hörbar (loudness > 0)', loud > 0,
    `loudness ${loud.toFixed(2)} — im Spiel hört das jeder passive Hörer`);
}

section('CHECK S2 — PASSIVES GEHÖR: Richtungsauflösung');
if (want('hear')) {
  const prey = makeEntity(KIND_PREY, 0, 0, {});
  prey.noise = 0.9;
  let maxErr = 0, n = 0;
  for (const [dx, dy] of [[60, 0], [0, -60], [-60, 0], [0, 60], [42, 42], [-42, 42]]) {
    const h = createHear();
    hearSample(h, [prey], 700, 400);
    prey.x = 700 + dx; prey.y = 400 + dy;
    hearSample(h, [prey], 700, 400);
    const b = hearBearing(h, 0);
    if (!b) continue;
    const want = Math.atan2(dy, dx);
    let err = b.angle - want;
    while (err > Math.PI) err -= Math.PI * 2;
    while (err < -Math.PI) err += Math.PI * 2;
    maxErr = Math.max(maxErr, Math.abs(err) * 180 / Math.PI); n++;
  }
  check('hear-bearing', 'Peilung zeigt zur Quelle (≤ 50° Fehler bei 4 Quadranten)', maxErr <= 50,
    `${n} Richtungen getestet, größter Fehler ${maxErr.toFixed(0)}°`);

  // Abfall mit der Distanz
  const near = createHear(), far = createHear();
  prey.x = 760; prey.y = 400; hearSample(near, [prey], 700, 400);
  prey.x = 840; hearSample(far, [prey], 700, 400);
  check('hear-falloff', 'Intensität fällt mit der Distanz', near.peak > far.peak * 1.3,
    `60px → ${near.peak.toFixed(3)}, 140px → ${far.peak.toFixed(3)}`);
}

section('CHECK S3 — CHEMOREZEPTION: Gradient zeigt zur Quelle');
if (want('chem')) {
  const prey = makeEntity(KIND_PREY, 780, 400, { scent: 1.0 });
  let ok = 0, n = 0;
  for (const hdg of [0, Math.PI / 2, Math.PI, -Math.PI / 2]) {
    const c = createChem();
    for (let i = 0; i < 80; i++) chemSample(c, [prey], 700, 400, hdg);
    const g = gradientDirection(c);
    if (!g) continue;
    n++;
    // Der Gradient ist relativ zur Blickrichtung; die Quelle liegt bei Weltwinkel 0
    let err = (hdg + g.angle) - 0;
    while (err > Math.PI) err -= Math.PI * 2;
    while (err < -Math.PI) err += Math.PI * 2;
    if (Math.abs(err) < 0.9) ok++;
  }
  check('chem-gradient', 'Duftgradient zeigt zur Quelle (±0.9 rad bei 6 Sensoren)', ok >= 3,
    `${ok}/${n} Blickrichtungen korrekt aufgelöst`);
  const far = createChem();
  const farPrey = makeEntity(KIND_PREY, 1300, 700, { scent: 1.0 });
  chemSample(far, [farPrey], 700, 400, 0);
  check('chem-range', 'Außerhalb der Reichweite → kein Signal', far.peak < 0.01,
    `peak ${far.peak.toFixed(4)} bei ~700 px Abstand`);
}

section('CHECK S4 — PHOTOREZEPTION fällt mit dem Abstand ab');
if (want('light')) {
  const map = makeMap(1440, 810, { seed: 7 });
  map.lights = [{ x: 400, y: 300, r: 260, ir: 0.8, c: [255, 255, 255], ph: 0, sp: 0.1 }];
  const a = createLight(); lightSample(a, map, 400, 300, 0);
  const b = createLight(); lightSample(b, map, 620, 300, 0);
  const c = createLight(); lightSample(c, map, 1300, 700, 0);
  check('light-falloff', 'Licht fällt mit dem Abstand ab', a.total > b.total && b.total > c.total,
    `Quelle ${a.total.toFixed(3)} > 220px ${b.total.toFixed(3)} > fern ${c.total.toFixed(3)} (Ambient ${c.total.toFixed(3)})`);
  check('light-ambient', 'Im Dunkeln bleibt nur Ambient', Math.abs(c.total - 0.05) < 1e-9,
    `${c.total.toFixed(3)} — Sehen ist damit konstruktionsbedingt intermittierend`);
}

section('CHECK S5 — GESAMTER EINGANGSVEKTOR: Dimension, Endlichkeit, Normierung');
if (want('input')) {
  const sim = createSim({ seed: 23, creatures: 1, prey: 2, lights: 4 });
  const c = sim.creatures[0];
  const inp = new Float64Array(INPUT_DIM);
  let nf = 0, oor = 0, peak = 0;
  for (let i = 0; i < 400; i++) {
    applyControl(c, {
      tone: 0.5 + 0.5 * Math.sin(i * 0.05),
      emitEcho: i % 20 === 0,
      arms: c.arms.map((_, j) => ({ shoulder: Math.sin(i * 0.13 + j) * 2.5, curl: Math.sin(i * 0.29 + j) * 2.5, adhesion: 0.5 + 0.5 * Math.sin(i * 0.07 + j) })),
    });
    stepSim(sim);
    readProprio(c, inp); readSenses(c, inp, sim.entities);
    for (let k = 0; k < inp.length; k++) {
      if (!Number.isFinite(inp[k])) nf++;
      if (Math.abs(inp[k]) > peak) peak = Math.abs(inp[k]);
      if (Math.abs(inp[k]) > 1.5) oor++;
    }
  }
  check('input-dim', `Eingangsdimension = Proprio ${PROPRIO_DIM} + Sinne ${SENSE_DIM}`, INPUT_DIM === PROPRIO_DIM + SENSE_DIM,
    `INPUT_DIM = ${INPUT_DIM}`);
  check('input-finite', 'Kein NaN/Inf im Eingangsvektor', nf === 0, `${nf} ungültige Werte über 400 Schritte`);
  check('input-range', 'Alle Werte normiert (|x| ≤ 1.5)', oor === 0, `Peak |x| = ${peak.toFixed(3)}, ${oor} außerhalb`);
  const senseOff = PROPRIO_DIM;
  let senseActive = 0;
  for (let k = senseOff; k < INPUT_DIM; k++) if (Math.abs(inp[k]) > 1e-9) senseActive++;
  check('input-senses', 'Sinnes-Eingänge sind belegt', senseActive >= 8,
    `${senseActive}/${SENSE_DIM} Sinne liefern Werte (Echolot/Chem/Licht/Gehör/Tast)`);
}

section('CHECK N1 — NETZ: keine Sättigung, Ausgabe im gültigen Bereich');
if (want('net')) {
  const { inputDim, weightCount, randomGenome, forwardInto, decode, HIDDEN, OUT_DIM } = await import('../src/brain/net.js');
  const nIn = inputDim(42, 43);
  let seed = 99;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const nW = weightCount(nIn, HIDDEN, OUT_DIM);
  let saturated = 0, nonfinite = 0, trials = 0;
  for (let t = 0; t < 40; t++) {
    const w = randomGenome(nIn, HIDDEN, OUT_DIM, rnd);
    const x = new Float64Array(nIn), out = new Float32Array(OUT_DIM), sc = new Float32Array(2 * HIDDEN);
    for (let i = 0; i < nIn; i++) x[i] = Math.sin(i * 0.37 + t) * 0.8;
    forwardInto(w, x, out, sc, nIn, HIDDEN, OUT_DIM);
    for (const v of out) { trials++; if (!Number.isFinite(v)) nonfinite++; if (Math.abs(Math.abs(v) - 1) < 1e-4) saturated++; }
  }
  check('net-finite', 'Netz-Ausgaben endlich und in [-1,1]', nonfinite === 0, `${nonfinite}/${trials} ungültig`);
  // Sättigung würde den Selektionsgradienten töten: alle Genome geben ±1 aus.
  // v2 initialisierte mit (rnd()-0.5)*1.6 über 64 Eingänge — Varianz der
  // Voraktivierung ~25× zu groß, jedes tanh sofort gesättigt.
  check('net-unsaturated', 'Ausgaben nicht gesättigt (1/√fan_in-Skalierung)', saturated / trials < 0.15,
    `${saturated}/${trials} = ${(saturated / trials * 100).toFixed(1)} % exakt ±1`);
  const ctl = { tone: 0, arms: Array.from({ length: 6 }, () => ({ shoulder: 0, curl: 0, adhesion: 0 })), emitEcho: false };
  const out = new Float32Array(OUT_DIM); out[1 + 2 * 3 + 2] = -0.7;
  decode(out, ctl, 6);
  check('net-decode', 'Negative Haftstärke wird zu 0 (nicht-klebrig als Nullpunkt)',
    ctl.arms[2].adhesion === 0, `adhesion = ${ctl.arms[2].adhesion}`);
  check('net-dim', `Genomgröße = ${nW} bei ${nIn} Eingängen`, nW === nIn * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN + HIDDEN * OUT_DIM + OUT_DIM,
    `${nW} Gewichte (Kraken_v2: 5307)`);
}

section('CHECK N2 — ES konvergiert gegen eine bekannte Zielfunktion');
if (want('es')) {
  // Sanity-Test des Optimierers selbst, unabhängig von der Simulation:
  // maximiere f(x) = -||x - ziel||² über 64 Dimensionen.
  const { createES, genomeAt, update } = await import('../src/brain/es.js');
  const N = 64;
  // lr 0.25 war zu aggressiv: der Optimierer schwang um das Optimum
  // (Abstand @60/120/180/240/300 = 1.80 / 2.10 / 2.31 / 1.81 / 2.18 — keine
  // Konvergenz). Mit lr ≤ 0.10 konvergiert er auf ~9 % Restabstand.
  // Die Produktions-Defaults in es.js (sigma 0.06, lr 0.030) liegen im stabilen
  // Bereich; hier wird sigma wegen der größeren Zieldistanz (5.6) hochgesetzt.
  const es = createES(N, { popSize: 32, sigma: 0.5, lr: 0.08, sigmaDecay: 1.0 });
  let seed = 4242;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const target = new Float64Array(N);
  for (let i = 0; i < N; i++) target[i] = Math.sin(i * 0.7);
  const g = new Float64Array(N);
  const fit = new Float64Array(32);
  const d0 = dist(es.theta, target);
  for (let gen = 0; gen < 200; gen++) {
    for (let i = 0; i < 32; i++) {
      genomeAt(es, i, g, rnd);
      fit[i] = -dist(g, target);
    }
    update(es, fit);
  }
  const d1 = dist(es.theta, target);
  function dist(a, b) { let s = 0; for (let i = 0; i < N; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s); }
  check('es-converges', 'ES minimiert den Abstand zum Ziel (64 Dim, 200 Generationen)', d1 < d0 * 0.25,
    `Abstand ${d0.toFixed(2)} → ${d1.toFixed(2)} (${(d1 / d0 * 100).toFixed(0)} %)`);
  // Gespiegelte Paare müssen korrelieren: θ+σε und θ−σε
  const es2 = createES(8, { popSize: 4, sigma: 1 });
  const a = new Float64Array(8), b = new Float64Array(8);
  genomeAt(es2, 0, a, rnd); genomeAt(es2, 1, b, rnd);
  let mirror = true;
  for (let i = 0; i < 8; i++) if (Math.abs((a[i] - es2.theta[i]) + (b[i] - es2.theta[i])) > 1e-9) mirror = false;
  check('es-mirrored', 'Perturbationen sind exakt gespiegelt (Varianzreduktion)', mirror,
    mirror ? 'θ+σε und θ−σε teilen dasselbe ε' : 'Spiegelung defekt');
}

section('CHECK N3 — TRAININGSLAUF ist deterministisch und steigt');
if (want('train')) {
  const { createTrainer, beginGeneration, trainStep, endGeneration } = await import('../src/sim/train.js');
  const runs = [];
  for (let r = 0; r < 2; r++) {
    const tr = createTrainer({ popSize: 16, stepsPerGen: 150, seed: 7, prey: 0 });
    const hist = [];
    for (let g = 0; g < 8; g++) {
      beginGeneration(tr);
      for (let s = 0; s < 150; s++) trainStep(tr);
      hist.push(endGeneration(tr));
    }
    runs.push(hist);
  }
  const same = JSON.stringify(runs[0].map(h => h.best.toFixed(6))) === JSON.stringify(runs[1].map(h => h.best.toFixed(6)));
  check('train-determinism', 'Zwei Läufe mit gleichem Seed sind bit-identisch', same,
    same ? 'reproduzierbar' : 'DIVERGENT — Messreihen wären wertlos');
  const first = runs[0].slice(0, 3).reduce((s, h) => s + h.mean, 0) / 3;
  const last = runs[0].slice(-3).reduce((s, h) => s + h.mean, 0) / 3;
  check('train-improves', 'Mittlerer Score steigt über 8 Generationen', last > first,
    `${first.toFixed(1)} → ${last.toFixed(1)} (${((last - first) / Math.abs(first) * 100).toFixed(0)} %)`);
  const a0 = runs[0][0].anchor, a1 = runs[0][runs[0].length - 1].anchor;
  check('train-grips', 'Anker-Rate steigt (Haften wird entdeckt)', a1 > a0,
    `${(a0 * 100).toFixed(0)} % → ${(a1 * 100).toFixed(0)} %`);
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
section('CHECK 12 — WGSL/Binding-Konsistenz (statisch, ohne GPU)');
if (want('wgsl')) {
  const { execFileSync } = await import('node:child_process');
  try {
    const out = execFileSync('node', [join(ROOT, 'tools/check-wgsl.js')], { encoding: 'utf8' });
    const fails = (out.match(/❌/g) || []).length;
    check('wgsl-static', 'Shader-Structs, Bindings und Entry-Points konsistent zu gpuworld.js', fails === 0,
      fails === 0 ? 'Strides, 6 Bindings, 5 Entry-Points, Workgroup 64, Substep-Reihenfolge, Konstanten-Parität' : fails + ' Inkonsistenz(en)');
  } catch (e) {
    check('wgsl-static', 'Shader-Structs, Bindings und Entry-Points konsistent', false, (e.stdout || e.message).split('\n').filter(l => l.includes('❌')).join(' | '));
  }
  // Shader-Datei muss vom Pages-Workflow ausgeliefert werden
  const wf = readFileSync(join(ROOT, '.github/workflows/deploy-pages.yml'), 'utf8');
  check('wgsl-deployed', 'gpu.html und solver.wgsl werden deployed',
    wf.includes('gpu.html') && wf.includes('solver.wgsl'),
    wf.includes('gpu.html') && wf.includes('solver.wgsl') ? 'Workflow kopiert und verifiziert beide' : 'Workflow kopiert sie nicht');
}

section('ZUSAMMENFASSUNG');
const pass = results.filter(r => r.pass).length;
for (const r of results) if (!r.pass) console.log(`  ❌ ${r.id}: ${r.name} — ${r.detail}`);
console.log(`\n  ${pass}/${results.length} Checks bestanden.`);
console.log(`  Körper: ${BODY.arms} Arme × ${BODY.segsPerArm} Segmente = ${1 + BODY.arms * BODY.segsPerArm} Körper/Krake, Reichweite ${ARM_REACH.toFixed(1)} px`);
process.exit(pass === results.length ? 0 : 2);
