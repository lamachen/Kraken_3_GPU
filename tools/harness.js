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
import { XpbdWorld, SLACK_POS, SLACK_ANG } from '../src/physics/xpbd.js';
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
section('CHECK 1b — KORREKTUR-TOTZONE (f32-Boden der GPU, G1-Drift 1.2789 px)');
if (want('slack')) {
  // Der WGSL-Solver rechnet Float32: bei ~1440 px Koordinaten ist ulp(position)
  // ≈ 1.7e-4 px, das Spawn-Gleichgewicht ist dort kein exakter Fixpunkt.
  // Gemessen an Build d487532 auf einer RTX 3070 Ti: G2 Spawn-Verletzung
  // 2.0e-4 px, G1 Rausch-maxV 0.1606 px/s, G1 Drift-Wachstum 1.2789 px nach
  // 2400 Schritten (CPU-Oracle: 3.4e-9). SLACK_POS/SLACK_ANG überspringen
  // Korrekturen im Rundungsband — in BEIDEN Solvern (Parität prüft
  // check-wgsl.js). Diese beiden Checks pinnen die Totzone von beiden Seiten:
  // (a) Störung innerhalb des Bands darf keinerlei Dynamik erzeugen,
  // (b) eine echte Verletzung (25× slack) muss weiterhin herauskorrigiert
  //     werden — die Totzone darf nicht „zu groß" konfiguriert sein.
  const violPins = (w) => {
    let worst = 0;
    for (const c of w.constraints) {
      if (c.type !== 0 || c.broken) continue;
      const A = c.a, B = c.b;
      const ca = Math.cos(w.ang[A]), sa = Math.sin(w.ang[A]);
      const gax = w.px[A] + c.lax * ca - c.lay * sa, gay = w.py[A] + c.lax * sa + c.lay * ca;
      let gbx, gby;
      if (B >= 0) { const cb = Math.cos(w.ang[B]), sb = Math.sin(w.ang[B]); gbx = w.px[B] + c.lbx * cb - c.lby * sb; gby = w.py[B] + c.lbx * sb + c.lby * cb; }
      else { gbx = c.lbx; gby = c.lby; }
      worst = Math.max(worst, Math.hypot(gax - gbx, gay - gby));
    }
    return worst;
  };
  const map = makeMap(1440, 810, { seed: 5 });
  const sp = findSpawn(map, ARM_REACH + 20, mulberry32(9));

  // (a) Fixpunkt: Störung 0.4× SLACK_POS → kein Körper darf sich bewegen.
  const w1 = new XpbdWorld(); w1.sdf = map.sdf;
  const k1 = spawnKraken(w1, sp.x, sp.y, 0.4);
  const b0 = k1.arms[0].segs[1];
  w1.px[b0] += SLACK_POS * 0.4;
  const px0 = Float64Array.from(w1.px), py0 = Float64Array.from(w1.py), ang0 = Float64Array.from(w1.ang);
  for (let i = 0; i < 240; i++) w1.step();
  let moved = 0;
  for (let i = 0; i < w1.n; i++) {
    moved = Math.max(moved, Math.hypot(w1.px[i] - px0[i], w1.py[i] - py0[i]), Math.abs(w1.ang[i] - ang0[i]));
  }
  check('slack-fixpoint', `Störung < slack (${(SLACK_POS * 0.4).toExponential(1)} px): Null Dynamik, v bleibt 0`,
    moved === 0 && w1.stats.maxSpeed === 0 && w1.stats.maxAngSpeed === 0,
    `max Δ nach 240 Schritten = ${moved.toExponential(2)}  max|v| = ${w1.stats.maxSpeed.toExponential(2)}  (GPU ohne Totzone: 1.28 px Drift aus genau diesem Zittern)`);

  // (b) Echte Verletzung 25× slack (5e-2 px) → muss zurück in die Bandbreite.
  const w2 = new XpbdWorld(); w2.sdf = map.sdf;
  const k2 = spawnKraken(w2, sp.x, sp.y, 0.4);
  const b1 = k2.arms[0].segs[1];
  w2.px[b1] += 0.05;
  const before = violPins(w2);
  for (let i = 0; i < 240; i++) w2.step();
  const after = violPins(w2);
  check('slack-corrects', 'Verletzung 25× slack wird herauskorrigiert (Band < 0.01 px)',
    before > 0.04 && after < 0.01,
    `Pin-Verletzung ${before.toFixed(3)} px → ${after.toExponential(2)} px nach 240 Schritten (slack = ${SLACK_POS} px, ${SLACK_ANG} rad)`);
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

section('CHECK G8 — HYBRID-TRAINER: Spiegel-Feldmapping + Passthrough-Identität (M3c)');
if (want('gputrain')) {
  const { mirrorInto, packControls, GPU_STRIDES } = await import('../src/gpu/gpuworld.js');
  const { createTrainer, beginGeneration, trainStep, endGeneration } = await import('../src/sim/train.js');
  const { gpuTrainStep, beginGenerationGpu, passthroughBackend } = await import('../src/sim/gputrain.js');

  // ── G8a: mirrorInto() muss JEDES Feld an die richtige Stelle legen. ──
  // Der Hybrid-Trainer läuft komplett auf diesem Spiegel: fehlt ein Feld
  // (z. B. av oder grip.force), trainiert die GPU-Variante gegen eine andere
  // Propriozeption als die CPU — stillschweigend und unheilbar.
  {
    const sim = createSim({ seed: 31, creatures: 2 });
    const w = sim.world, nB = w.n, nC = w.constraints.length;
    const bd = new Float32Array(nB * GPU_STRIDES.body);
    const cd = new Float32Array(nC * GPU_STRIDES.constraint);
    const cu = new Uint32Array(cd.buffer);
    for (let i = 0; i < nB; i++) {
      const o = i * GPU_STRIDES.body;
      bd[o] = 100 + i * 0.5; bd[o + 1] = -200 + i * 0.25;
      bd[o + 4] = 1 + i * 0.01; bd[o + 5] = -2 - i * 0.01;
      bd[o + 6] = 0.3 + i * 0.001; bd[o + 8] = 0.05 + i * 0.002;
    }
    for (let k = 0; k < nC; k++) {
      const o = k * GPU_STRIDES.constraint;
      cd[o + 13] = 0.5 + k * 0.001; cd[o + 14] = -0.25 + k * 0.001; cd[o + 15] = k * 0.002;
      cu[o + 17] = k % 2;                     // gebrochen im Wechsel — ÜBER u32
    }
    const stats = mirrorInto(w, bd, cd);
    let bodyOk = true;
    for (let i = 0; i < nB && bodyOk; i++) {
      bodyOk = w.px[i] === bd[i * 12] && w.py[i] === bd[i * 12 + 1]
        && w.vx[i] === bd[i * 12 + 4] && w.vy[i] === bd[i * 12 + 5]
        && w.ang[i] === bd[i * 12 + 6] && w.av[i] === bd[i * 12 + 8];
    }
    let consOk = true, gripOk = true, nGrips = 0;
    const invH = 1 / w.h;
    for (let k = 0; k < nC; k++) {
      const c = w.constraints[k];
      if (c.lambda !== cd[k * 20 + 15]) consOk = false;
      if (c.broken !== (k % 2 === 1)) consOk = false;
      if (c.type === 2) {
        nGrips++;
        const fx = cd[k * 20 + 13], fy = cd[k * 20 + 14];
        if (c.lambdaX !== fx || c.lambdaY !== fy) gripOk = false;
        if (c.force !== Math.hypot(fx, fy) * invH) gripOk = false;   // Parität _postGripBreak
      }
    }
    let expMaxV = 0;
    for (let i = 0; i < nB; i++) { const vx = bd[i * 12 + 4], vy = bd[i * 12 + 5]; expMaxV = Math.max(expMaxV, Math.sqrt(vx * vx + vy * vy)); }
    const statsOk = stats.maxSpeed === expMaxV && stats.nan === 0;
    check('mirror-fields', `mirrorInto: ${nB} Körper, ${nC} Constraints (${nGrips} Grips) exakt gespiegelt`,
      bodyOk && consOk && gripOk && statsOk,
      `Körper=${bodyOk} λ/broken=${consOk} Grip-force=${gripOk} Stats=${statsOk} (broken läuft über die Uint32-Sicht)`);
  }

  // ── G8b: packControls() — Offsets + lbx/lby-Sync des Saugnapf-Reflexes. ──
  // Ohne lbx/lby würde ein mitten in der Episode neu verankerter Grip auf der
  // GPU am ALTEN Anker ziehen (Spawn-Position) — unsichtbar falsch.
  {
    const sim = createSim({ seed: 33, creatures: 1 });
    const w = sim.world, nC = w.constraints.length;
    const cd = new Float32Array(nC * GPU_STRIDES.constraint);
    const cu = new Uint32Array(cd.buffer);
    for (let k = 0; k < nC; k++) {
      const c = w.constraints[k];
      c.compliance = 0.123 + k * 1e-4; c.target = 0.01 * k;
      c.breakForce = k % 3 === 0 ? Infinity : 5 + k;
      c.broken = k % 2 === 0; c.lbx = 11 + k; c.lby = -22 - k;
    }
    packControls(w, cd);
    let ok = true, firstBad = -1;
    for (let k = 0; k < nC; k++) {
      const c = w.constraints[k], o = k * GPU_STRIDES.constraint;
      const good = cd[o + 6] === Math.fround(c.lbx) && cd[o + 7] === Math.fround(c.lby)
        && cd[o + 8] === Math.fround(c.compliance) && cd[o + 10] === Math.fround(c.target)
        && cd[o + 16] === Math.fround(Math.min(c.breakForce, 1e30))
        && cu[o + 17] === (c.broken ? 1 : 0);
      if (!good && firstBad < 0) firstBad = k;
      ok = ok && good;
    }
    check('pack-controls', `packControls: Anker lbx/lby + Steuerfelder auf WGSL-Offsets (${nC} Constraints)`,
      ok, ok ? 'compliance o+8, tgt o+10, lbx/lby o+6/7, breakForce o+16 (∞→1e30), broken o+17 u32' : `Feld falsch ab Constraint ${firstBad}`);
  }

  // ── G8c: Beute-Determinismus — der Math.random()-Bug in makeEntity. ──
  // Gefangen vom Passthrough-Check: zwei Trainer mit gleichem Seed drifteten
  // ab Schritt 85 auseinander, weil heading/seed der Beute aus Math.random()
  // kamen. Mit prey=0 unsichtbar (M3a lief so), mit prey>0 wären alle
  // Messreihen nicht reproduzierbar gewesen.
  {
    const { createSim } = await import('../src/sim/core.js');
    const s1 = createSim({ seed: 5, creatures: 1, prey: 2 });
    const s2 = createSim({ seed: 5, creatures: 1, prey: 2 });
    const same = s1.entities.length === s2.entities.length && s1.entities.every((e, i) => {
      const f = s2.entities[i];
      return e.x === f.x && e.y === f.y && e.heading === f.heading && e.seed === f.seed;
    });
    check('prey-determinism', 'Beute-Entitäten aus gleichem Seed sind identisch (heading/seed)',
      same, same ? `${s1.entities.length} Entitäten bit-identisch (fix: makeEntity nutzt sim.rnd)` : 'Math.random() läuft noch mit');
  }

  // ── G8d: Passthrough-Identität — dieselbe Verdrahtung, bit-identisches Ergebnis. ──
  // gpuTrainStep mit Passthrough-Backend (Physik = CPU, Spiegel-Pfade gezählt)
  // MUSS exakt die Fitness des reinen CPU-Trainers liefern. Jeder Unterschied
  // ist ein Verdrahtungsfehler (Reihenfolge, fehlendes tickSenses, Doppel-
  // Control …) — unabhängig von f32/f64, deshalb EXAKT und nicht Toleranz.
  {
    const OPTS = { popSize: 8, stepsPerGen: 120, seed: 7, prey: 2 };
    const trA = createTrainer(OPTS);                       // Referenz: train.js
    const trB = createTrainer(OPTS);                       // Hybrid: gputrain.js
    const counters = {};
    const backend = passthroughBackend(trB.sim, counters);
    const histA = [], histB = [];
    let fitnessIdentical = true;
    for (let g = 0; g < 2; g++) {
      beginGeneration(trA);
      beginGenerationGpu(trB, backend);
      for (let s = 0; s < OPTS.stepsPerGen; s++) {
        trainStep(trA);
        await gpuTrainStep(trB, backend);
      }
      histA.push(endGeneration(trA));
      histB.push(endGeneration(trB));
      for (let i = 0; i < OPTS.popSize; i++) {
        if (trA.fitness[i] !== trB.fitness[i]) fitnessIdentical = false;
      }
    }
    let stateIdentical = true;
    for (let i = 0; i < trA.sim.world.n; i++) {
      if (trA.sim.world.px[i] !== trB.sim.world.px[i] || trA.sim.world.py[i] !== trB.sim.world.py[i]
        || trA.sim.world.ang[i] !== trB.sim.world.ang[i]) stateIdentical = false;
    }
    const histIdentical = histA.every((h, gi) => h.best === histB[gi].best && h.mean === histB[gi].mean);
    const nSteps = 2 * OPTS.stepsPerGen;
    const callsOk = counters.upload === 2 && counters.uploadControls === nSteps
      && counters.step === nSteps && counters.download === nSteps;
    check('gputrain-identical', 'Hybrid-Trainer (Passthrough) = CPU-Trainer, 2 Gen × 8 Tiere × 120 Schritte',
      fitnessIdentical && stateIdentical && histIdentical && callsOk,
      `Fitness bit-identisch=${fitnessIdentical} Weltzustand=${stateIdentical} Historie=${histIdentical} ` +
      `Aufrufe=${callsOk} (upload ${counters.upload}/2, controls ${counters.uploadControls}/${nSteps}, step ${counters.step}/${nSteps}, download ${counters.download}/${nSteps})`);
  }
}

// ══════════════════════════════════════════════════════════════
section('CHECK G7 — SOLVER-DIVERGENZ: RACE GEFUNDEN, KLASSE NEU GEMESSEN');
if (want('chaos')) {
  // G7 hat zwei Hardware-Runden gebraucht, und die erste Deutung war FALSCH.
  // Chronik:
  //   fb622c8  maxΔ 32.69 px, pathLen -3.99 %  → Tore < 0.5 px FAIL
  //   529fdc7  Tore auf "Solver-Klassen" gelockert, Kontrolle = Ordnung gedreht
  //            → maxΔ 22.02 px OK, aber pathLen -14.8 % gegen Kontrolle -7.9 % FAIL
  // Der Fehler in 529fdc7 war die PRÄMISSE: Die f32-Emulation lief mit der
  // Kollisions-Ordnung der CPU und blieb bei 0.36 px. Also war Numerik nicht
  // die Ursache — und "chaotische Klasse" war nur der Platzhalter dafür, dass
  // die Emulation einen Teil des GPU-Solvers nicht nachgebildet hat.
  // tools/class-spread.mjs hat die fehlende Variable isoliert: solveCollisions
  // dispatchte EINEN THREAD PRO KOLLISIONSPUNKT, aber Mantel (8 Punkte) und
  // Armsegmente (2 Punkte) teilen sich einen Körper → Schreib-Race auf
  // bodies[], letzter Schreiber gewinnt; die CPU korrigiert kumulativ.
  // Messung pop 32, seed 7, 150 Schritte:
  //   Variante                        maxΔ      Flags     pathLen    Fitness
  //   f32, CPU-Kollisionsordnung       0.36 px  100.00 %   +0.01 %    -0.00 %
  //   f32 + Race (letzter gewinnt)    29.77 px   99.44 %  -14.71 %    -3.27 %
  //   f32 + Race (erster gewinnt)     51.10 px   99.35 %  -21.83 %    -3.94 %
  //   f32 + Jacobi (Deltas summiert)  30.54 px   99.27 %  -15.34 %    -3.69 %
  //   ECHTE RTX 3070 Ti (G7)          22.02 px   99.35 %  -14.80 %       —
  // Race trifft die Hardware-Messung, race-frei bleibt 40× darunter — und der
  // Bias stimmt im Vorzeichen (weniger Korrektur → mehr Penetration → mehr
  // Slip/Brüche → kürzerer Pfad). Der Fix: EIN Thread pro Körper, seine Punkte
  // sequentiell (solver.wgsl + collRange-Binding 8, statisch gepinnt von
  // tools/check-wgsl.js, Abschnitt "Kollisions-Race").
  // Was von der Klassen-Erklärung ÜBRIG bleibt, messen die Checks unten:
  // Zustandsrauschen wird absorbiert, die Farb-Ordnung ist frei, und reine
  // f32-Arithmetik kostet < 1 px. Nur die Listenordnung der CPU (Gauss-Seidel
  // über alle Farben gemischt) ist chaotisch — die GPU hat sie nie benutzt.
const { createTrainer } = await import('../src/sim/train.js');
  const { gpuTrainStep, beginGenerationGpu } = await import('../src/sim/gputrain.js');
  const OPTS = { popSize: 8, stepsPerGen: 150, seed: 7, prey: 2 };

  async function lockstep(mutateB) {
    const A = createTrainer(OPTS), B = createTrainer(OPTS);
    beginGenerationGpu(A, null);
    beginGenerationGpu(B, null);
    if (mutateB) mutateB(B);
    let maxD = 0, mism = 0;
    for (let s = 0; s < 150; s++) {
      await gpuTrainStep(A, null);
      await gpuTrainStep(B, null);
      for (let i = 0; i < OPTS.popSize; i++) {
        const ca = A.sim.creatures[i], cb = B.sim.creatures[i];
        const d = Math.hypot(ca.x - cb.x, ca.y - cb.y);
        if (d > maxD) maxD = d;
        for (let a = 0; a < 6; a++) {
          if (ca.arms[a].grip.broken !== cb.arms[a].grip.broken) mism++;
        }
      }
    }
    const pl = (tr) => tr.sim.creatures.reduce((sum, c) => sum + c.pathLen, 0) / OPTS.popSize;
    return { maxD, mism, plA: pl(A), plB: pl(B), nan: A.sim.world.stats.nan + B.sim.world.stats.nan };
  }

  // a) Zustandsrauschen wird absorbiert — Zustandstore (G1/G2) bleiben gültig
  {
    const rnd = mulberry32(123);
    const r = await lockstep((tr) => {
      const w = tr.sim.world, orig = w.step.bind(w);
      w.step = function () {
        for (let i = 0; i < w.n; i++) {
          w.px[i] += (rnd() * 2 - 1) * 1e-3;
          w.py[i] += (rnd() * 2 - 1) * 1e-3;
        }
        orig();
      };
    });
    check('noise-absorbed', 'Zustandsrauschen 1e-3 px/Schritt wird absorbiert (kontraktiv)',
      r.maxD < 0.05 && r.mism <= 10,
      `maxΔ ${r.maxD.toExponential(2)} px, Flips ${r.mism}/7200 (Referenz: 4.2e-3 px, 0 Flips)`);
  }

  // b) Solve-Ordnung gedreht (reines f64!) divergiert chaotisch — die Klasse existiert
  {
    const r = await lockstep((tr) => { tr.sim.world.constraints.reverse(); });
    check('order-chaos', 'CPU-Listenordnung gedreht (f64) divergiert — ordnungs-chaotisch',
      r.maxD > 1,
      `maxΔ ${r.maxD.toFixed(2)} px, Flips ${r.mism}/7200, pathLen ${r.plA.toFixed(2)}→${r.plB.toFixed(2)} ` +
      `(Referenz: 35.35 px, 236 Flips, +25.3 %). Betrifft die CPU: die löst Gauss-Seidel in ` +
      `Listenordnung über ALLE Farben gemischt. Die GPU dispatched farbweise aufsteigend — ` +
      `dieselbe Ordnung wie die CPU nach groupConstraintsByColor, also kein Unterschied.`);
  }

  // c) f32-Solver-Emulation (gleiche Ordnung, per-op fround) bleibt in der Klasse
  {
    const { f32ify } = await import('./f32solver.mjs');
    const pct = (r) => 100 * (r.plB - r.plA) / Math.max(1e-9, r.plA);
    const r = await lockstep((tr) => {
      const emu = f32ify(tr.sim.world, tr.sim.map, { collMode: 'seq' });
      emu.roundState();   // = upload(): Anfangszustand als f32
    });
    check('f32-class', 'f32-Emulation (per-op, race-freie Kollision): NaN 0, maxΔ < 5 px, pathLen ±2 %',
      r.nan === 0 && r.maxD < 5 && Math.abs(pct(r)) < 2,
      `maxΔ ${r.maxD.toFixed(3)} px, pathLen ${r.plA.toFixed(2)}→${r.plB.toFixed(2)} (${pct(r).toFixed(2)} %), NaN ${r.nan} ` +
      `(Referenz seed 7/pop 8: 0.78 px/+0.24 %; pop 32: 0.36 px/+0.01 %) — DAS ist die Erwartung für die GPU nach dem Race-Fix`);

    // d) Der Race selbst — reproduzierbare Ursache statt Vermutung. 'race'
    //    emuliert den Stand 529fdc7 (ein Thread pro Kollisionspunkt, letzter
    //    Schreiber gewinnt). Muss deutlich über der race-freien Zahl liegen,
    //    sonst war die Diagnose Zufall.
    const rRace = await lockstep((tr) => {
      const emu = f32ify(tr.sim.world, tr.sim.map, { collMode: 'race' });
      emu.roundState();
    });
    check('coll-race-repro', 'Kollisions-Race (Stand 529fdc7) reproduziert die GPU-Divergenz',
      rRace.nan === 0 && rRace.maxD > 10 * Math.max(r.maxD, 1e-9),
      `race maxΔ ${rRace.maxD.toFixed(2)} px / pathLen ${pct(rRace).toFixed(2)} %  vs  race-frei ${r.maxD.toFixed(3)} px / ${pct(r).toFixed(2)} % ` +
      `(Hardware RTX 3070 Ti pop 32: 22.02 px / -14.80 %; Emulation pop 32: 29.77 px / -14.71 %)`);

    // d2) Die PHYSIK des Races, nicht nur seine Endzahlen: tools/collprobe.mjs
    //     baut denselben Race auf der CPU nach (Punkte eines Körpers lesen
    //     denselben Zustand, letzter Schreiber gewinnt) und misst, was das im
    //     Untergrund anrichtet. Ohne diesen Check wüsste niemand, WARUM die
    //     Pfade kürzer werden — und die erste Erklärung im Commit-Entwurf war
    //     prompt falsch (mehr Slip/Brüche; gemessen: WENIGER Brüche, 9 → 3).
    {
      const { execFileSync } = await import('node:child_process');
      let out = '';
      try { out = execFileSync('node', [join(ROOT, 'tools/collprobe.mjs')], { encoding: 'utf8' }); }
      catch (e) { out = (e.stdout || '') + (e.stderr || ''); }
      const grab = (label, re2) => {
        const line = out.split('\n').find(l => l.includes(label));
        const m = line && line.match(re2);
        return m ? Number(m[1].replace(',', '.')) : NaN;
      };
      const lostPct = grab('davon verworfen', /=\s*([0-9.]+)\s*%/);
      const resRatio = grab('Rest-Pen. pro Schritt', /\(([0-9.]+)×\)/);
      const plDev = grab('Ø pathLen', /\(([-0-9.]+) %\)/);
      check('coll-race-physics', 'Kollisions-Race: verworfene Korrekturen + Rest-Penetration gemessen',
        Number.isFinite(lostPct) && Number.isFinite(resRatio) && lostPct >= 20 && resRatio >= 3,
        `${lostPct.toFixed(1)} % der Kontaktmeldungen verworfen, ${resRatio.toFixed(2)}× Rest-Penetration pro Schritt, ` +
        `pathLen ${plDev.toFixed(2)} % (Hardware RTX 3070 Ti: -14.80 %) — der CPU-Nachbau ist ein Prädiktor für die GPU`);
    }
  }

  // e) Innerhalb einer Farbklasse ist die Solve-Ordnung BELIEBIG — genau die
  //    Freiheit, die der parallele GPU-Dispatch pro Farbe hat. Bleibt die
  //    Abweichung bei 0, ist die Farbordnung als Ursache ausgeschlossen und
  //    der Race-Fix reicht als Erklärung.
  {
    const r = await lockstep((tr) => {
      const w = tr.sim.world;
      let s0 = 12345;
      const rnd = () => { s0 = (Math.imul(s0, 1103515245) + 12345) & 0x7fffffff; return s0 / 0x7fffffff; };
      for (let c = 0; c < w.colorOff.length; c++) {
        const o = w.colorOff[c], n = w.colorCnt[c];
        for (let i = n - 1; i > 0; i--) {
          const j = (rnd() * (i + 1)) | 0;
          const t = w.constraints[o + i]; w.constraints[o + i] = w.constraints[o + j]; w.constraints[o + j] = t;
        }
      }
    });
    check('color-order-free', 'Ordnung INNERHALB einer Farbklasse ist frei (GPU-Dispatch-Parität)',
      r.maxD < 1e-9 && r.mism === 0,
      `maxΔ ${r.maxD.toExponential(2)} px, Flips ${r.mism}/7200 — Farbgruppen sind körper-disjunkt, ` +
      `der parallele Dispatch pro Farbe ist deshalb exakt, nicht nur näherungsweise`);
  }

  // f) Der Race-Fix selbst, ÄQUIVALENZ statt Quelltext-Vergleich: Der Shader
  //    läuft pro Körper die Punkte in der von groupCollPoints() gelieferten
  //    Reihenfolge ab. Dieser Check geht denselben Weg und muss BIT-identisch
  //    zum kumulativen _solveCollisions() der CPU kommen. Wäre die Gruppierung
  //    lückenhaft oder verdreht, sähe man es hier — ohne GPU-Device.
  {
    const { groupCollPoints } = await import('../src/gpu/gpuworld.js');
    const sim = createSim({ seed: 5, creatures: 3, prey: 0 });
    const w = sim.world, map = sim.map;
    let maxErr = 0, trials = 0, contacts = 0;
    // Posen, die WIRKLICH Kontakt haben: Zufallsstreuen bringt 0 Kontakte
    // (gemessen — der Check war damit wertlos). Stattdessen jede Pose entlang
    // einer Rechteckkante der Karte, 18 Rechtecke × 6 Lagen, plus 20 Posen aus
    // einem laufenden Sim-Schritt (echte Gang-Kontakte).
    const poses = [];
    for (let ri = 0; ri < map.rects.length; ri++) {
      const r = map.rects[ri];
      for (let v = 0; v < 6; v++) {
        const p = { px: new Float64Array(w.n), py: new Float64Array(w.n), ang: new Float64Array(w.n), tag: `rect${ri}/lage${v}` };
        for (let i = 0; i < w.n; i++) {
          const u = ((i + v * 7) % 10) / 10;
          p.px[i] = r.x + r.w * (v % 2 ? u : 0.5) + (v % 3) - 1;
          p.py[i] = r.y + r.h * (v % 2 ? 0.5 : u) + ((v >> 1) % 3) - 1;
          p.ang[i] = (i * 0.7 + v * 1.1) % 6.283;
        }
        poses.push(p);
      }
    }
    {
      const rnd = mulberry32(4242);
      for (let t = 0; t < 20; t++) {
        for (const c of sim.creatures) applyControl(c, { tone: 0.2 + 0.6 * rnd(), emitEcho: false, arms: c.arms.map(() => ({ shoulder: rnd() * 2 - 1, curl: rnd() * 2 - 1, adhesion: rnd() })) });
        stepSim(sim);
        const p = { px: Float64Array.from(w.px), py: Float64Array.from(w.py), ang: Float64Array.from(w.ang), tag: `sim${t}` };
        poses.push(p);
      }
    }
    for (const pose of poses) {
      for (let i = 0; i < w.n; i++) { w.px[i] = pose.px[i]; w.py[i] = pose.py[i]; w.ang[i] = pose.ang[i]; }
      w.ppx.set(w.px); w.ppy.set(w.py); w.angP.set(w.ang);
      trials++;
      const snap = { px: Float64Array.from(w.px), py: Float64Array.from(w.py),
                     ppx: Float64Array.from(w.ppx), ppy: Float64Array.from(w.ppy),
                     angP: Float64Array.from(w.angP) };
      const restore = () => { w.ppx.set(snap.ppx); w.ppy.set(snap.ppy); w.angP.set(snap.angP); };

      restore();
      w._solveCollisions();                      // CPU-Oracle: Punkte sequentiell, kumulativ
      const refX = Float64Array.from(w.ppx), refY = Float64Array.from(w.ppy);

      restore();                                 // GPU-Weg: pro Körper, Punkte in order[]
      const { order, range } = groupCollPoints(w.collPts, w.n);
      const nB = w.n;
      for (let b = 0; b < nB; b++) {
        if (w.im[b] === 0) continue;
        const cnt = range[nB + b];
        let px = w.ppx[b], py = w.ppy[b];
        for (let q = 0; q < cnt; q++) {
          const cp = w.collPts[order[range[b] + q]];
          const ca = Math.cos(w.angP[b]), sa = Math.sin(w.angP[b]);
          const hit = map.sdf(px + cp.lx * ca - cp.ly * sa, py + cp.lx * sa + cp.ly * ca);
          if (!hit) continue;
          const pen = cp.r - hit.d;
          if (pen <= SLACK_POS) continue;
          px += hit.nx * pen; py += hit.ny * pen; contacts++;
        }
        w.ppx[b] = px; w.ppy[b] = py;
      }

      for (let i = 0; i < w.n; i++) {
        maxErr = Math.max(maxErr, Math.abs(w.ppx[i] - refX[i]), Math.abs(w.ppy[i] - refY[i]));
      }
      void snap;
    }
    check('coll-grouping', 'groupCollPoints + Shader-Weg = _solveCollisions, bit-identisch',
      maxErr === 0 && contacts > 500,
      `${trials} Posen × ${w.n} Körper, ${contacts} wirksame Kontakte (Gate > 500, sonst prüft der Check nichts), max|Δ| ${maxErr.toExponential(2)} px ` +
      `(Mantel 8 Punkte, Armsegment 2 — genau die Körper, die den Race ausmachten)`);
  }
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
      fails === 0 ? 'Strides, 9 Bindings, 6 Entry-Points, Workgroup 64, Substep-Reihenfolge, Konstanten-Parität, Kollisions-Race-Fix' : fails + ' Inkonsistenz(en)');
  } catch (e) {
    check('wgsl-static', 'Shader-Structs, Bindings und Entry-Points konsistent', false, (e.stdout || e.message).split('\n').filter(l => l.includes('❌')).join(' | '));
  }
  // Echter Compile-Check (naga als WASM): check-wgsl.js vergleicht nur Texte
  // und kann kein WGSL parsen. Im Race-Fix stand `let cnt = …` und wurde in
  // der Schleife dekrementiert — `invalid left-hand side of assignment`.
  // Ohne Compile-Check hätte das genau einen Browser-Start beim Benutzer
  // gekostet. SKIP (Code 3), wenn naga-wasm nicht installiert ist; die CI
  // installiert es, lokal ist es optional (npm i --no-save naga-wasm@30.2.0).
  {
    const { execFileSync } = await import('node:child_process');
    let code = 0, out = '';
    try { out = execFileSync('node', [join(ROOT, 'tools/compile-wgsl.js'), '--json'], { encoding: 'utf8' }); }
    catch (e) { code = e.status === undefined ? 1 : e.status; out = (e.stdout || '') + (e.stderr || ''); }
    let info = null;
    try { info = JSON.parse(out); } catch { /* SKIP-Ausgabe ist kein JSON */ }
    if (code === 3) {
      console.log('  ⚪ wgsl-compile  SKIP — naga-wasm nicht installiert (npm i --no-save naga-wasm@30.2.0; die CI hat es)');
    } else {
      check('wgsl-compile', 'solver.wgsl compiliert: naga parst, validiert, serialisiert zurück',
        code === 0 && !!info && info.ok === true,
        info && info.ok ? `${info.bytes} B → ${info.roundtripBytes} B IR, Entry-Points ${(info.entryPoints || []).join(', ')}`
                        : (info && info.error ? info.error.split('\n')[0] : out.split('\n')[0]));
    }
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
