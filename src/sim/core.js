/**
 * DOM-freier Simulationskern.
 *
 * Dieselbe Instanz läuft im Browser (mit src/render/) und im Worker/Headless
 * (ohne). Das ist die Voraussetzung dafür, dass der GPU-/Worker-Split kein
 * Retrofit wird: `tools/harness.js` beweist bereits, dass der Kern ohne DOM
 * läuft, und der Worker braucht exakt dieselben Module.
 */
import { XpbdWorld } from '../physics/xpbd.js';
import { makeMap, findSpawn, mulberry32, tickMapLights } from '../world/map.js';
import { spawnKraken, tipPosition, attachGrip, releaseGrip, armCurl, armExtension, toneToCompliance, adhesionToBreakForce, BODY, ARM_REACH, assignColor, MAX_COLORS } from '../body/kraken.js';
import { makeEntity, tickEntity, KIND_PREY } from '../world/entities.js';
import { measureReach } from './fitness.js';
import { createEcho, emit as emitEcho, tick as tickEcho, write as writeEcho, ECHO_DIM } from '../senses/echo.js';
import { createChem, sample as sampleChem, write as writeChem, CHEM_DIM } from '../senses/chem.js';
import { createLight, sample as sampleLight, write as writeLight, LIGHT_DIM } from '../senses/light.js';
import { createHear, sample as sampleHear, write as writeHear, HEAR_DIM, bearing as hearBearing } from '../senses/hear.js';

export const DEFAULTS = {
  width: 1440, height: 810,
  seed: 1,
  creatures: 1,
  substeps: 2,
  iterations: 4,
  h: 1 / 120,
  // Haftung als Reflex: die Spitze haftet, wenn sie relativ zum Untergrund
  // STEHT. Der Untergrund ist statisch, also ist Slip = Spitzengeschwindigkeit.
  // Das ist der Taktgeber des Gangs — ein schwingender Arm gleitet durch, ein
  // gesetzter haftet. Kein Einschalt-Zeitpunkt, den die KI treffen müsste.
  gripSlipMax: 150,      // px/s
  gripCooldown: 0.05,    // s
  prey: 0,               // Anzahl Beute-Entitäten (Duft- und Geräuschquellen)
  preySpeed: 55,
  lights: 4,
};

export function createSim(opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const world = new XpbdWorld({ substeps: o.substeps, iterations: o.iterations, h: o.h });
  const map = makeMap(o.width, o.height, { seed: o.seed });
  world.sdf = map.sdf;
  const rnd = mulberry32(o.seed ^ 0x9e3779b9);

  const creatures = [];
  const colorMask = new Uint32Array(1 << 16);   // Inzidenz-Bitmasken pro Körper
  for (let i = 0; i < o.creatures; i++) {
    // Clearance MUSS größer als die Armreichweite sein (118.5 px), sonst spawnen
    // die Arme in Wänden und die Kollisionsprojektion katapultiert sie heraus.
    const sp = findSpawn(map, ARM_REACH + 20, rnd);
    const k = spawnKraken(world, sp.x, sp.y, rnd() * Math.PI * 2);
    // Constraints dieses Tieres färben (Indices sind global, Maske wächst mit)
    const seen = new Set();
    const all = [...world.constraints, ...k.arms.flatMap(a => a.motors), ...k.arms.map(a => a.grip)];
    for (const c of all) {
      if (seen.has(c)) continue;
      seen.add(c);
      if (c.color === undefined) c.color = assignColor(colorMask, c.a, c.b >= 0 ? c.b : -1);
    }
    const cr = makeCreature(world, k, map, o, i);
    // Verfügbaren Freiraum beim Spawn vermessen — ohne das ist die Fitness
    // reine Lotterie (wer in einer Nische spawnt, kann nie weit kommen).
    cr.spawnReach = measureReach(map, sp.x, sp.y);
    creatures.push(cr);
  }
  // Beute-Entitäten: Duft- und Geräuschquellen. Milestone 4 ergänzt/augmentiert
  // das um einen Bewohner-Agenten mit Policy — dieselbe Struktur.
  const entities = [];
  for (let i = 0; i < o.prey; i++) {
    const sp = findSpawn(map, 40, rnd);
    entities.push(makeEntity(KIND_PREY, sp.x, sp.y, { scent: 1.0, speed: o.preySpeed }));
  }
  groupConstraintsByColor(world);
  return { world, map, creatures, entities, opts: o, rnd, time: 0, stepCount: 0 };
}

function makeCreature(world, k, map, opts, index) {
  return {
    index, mantle: k.mantle, arms: k.arms, spawn: k.spawn,
    map, opts, world,
    // Steuerung (wird vom Gehirn beschrieben)
    tone: 0.5,                       // globaler Muskeltonus 0..1
    perArm: k.arms.map(() => ({ shoulder: 0, curl: 0, adhesion: 0, cd: 0 })),
    // ── Sinne. Alle DOM-frei, alle mit derselben write(out, offset)-Schnittstelle. ──
    echo: createEcho(), chem: createChem(), light: createLight(), hear: createHear(),
    contact: new Float64Array(k.arms.length),
    echoWantsEmit: false, echoLoudness: 0, lastEchoEmit: -1, hearBearing: null, heading: 0,
    // Propriozeption / Messwerte
    prevVx: 0, prevVy: 0, ax: 0, ay: 0,
    pathLen: 0, maxDisplacement: 0, angularAccum: 0, forwardProg: 0,
    gripPullProg: 0, anchoredSteps: 0, anchoredSum: 0, gripCycles: 0,
    freeMove: 0, energySpent: 0, hits: 0, preyMinDist: Infinity, spawnReach: 300,
    wasAnchored: false, prevSpawnDist: 0, steps: 0,
    meanExtension: 1, meanCurl: 0,
  };
}

/** Eigenzustand eines Kraken aus der Welt lesen. */
export function readState(c) {
  const w = c.world, m = c.mantle;
  c.x = w.px[m]; c.y = w.py[m];
  c.vx = w.vx[m]; c.vy = w.vy[m];
  c.angle = w.ang[m]; c.av = w.av[m];
  c.speed = Math.hypot(c.vx, c.vy);
  return c;
}

/**
 * Propriozeption + Steuerwerte auf die Constraints schreiben.
 * Läuft VOR world.step(), einmal pro Sim-Schritt.
 *
 * Eingangs-Layout (pro Krake), identisch zur geplanten NN-Schnittstelle:
 *   [0..5]   pro Arm: Schulterwinkel (normiert)
 *   [6..11]  pro Arm: mittlere Krümmung (normiert)
 *   [12..23] pro Arm: Spitze x/y im RUMPFSYSTEM (normiert auf ARM_REACH)
 *   [24..29] pro Arm: Zuglast am Saugnapf (0 = lose, 1 = kurz vor Abriss)
 *   [30..35] pro Arm: haftet gerade? (0/1)
 *   [36]     Beschleunigung längs   [37] Beschleunigung quer
 *   [38]     Anteil tragender Arme  [39] mittlere Extension
 *   [40]     Tempo                  [41] Winkelgeschwindigkeit
 */
export const PROPRIO_DIM = 42;

/** Gesamte Eingangsdimension = Propriozeption + alle Sinne. */
export const SENSE_DIM = ECHO_DIM + CHEM_DIM + LIGHT_DIM + HEAR_DIM + 6;
export const INPUT_DIM = PROPRIO_DIM + SENSE_DIM;

export function readProprio(c, out) {
  const w = c.world;
  readState(c);
  const ca = Math.cos(-c.angle), sa = Math.sin(-c.angle);
  let anchored = 0, extSum = 0, curlSum = 0;
  let i = 0;
  const shoulder = new Float64Array(c.arms.length);
  for (let a = 0; a < c.arms.length; a++) {
    const arm = c.arms[a];
    shoulder[a] = Math.atan2(Math.sin(w.ang[arm.segs[0]] - c.angle), Math.cos(w.ang[arm.segs[0]] - c.angle));
    extSum += armExtension(w, arm);
    curlSum += armCurl(w, arm);
    if (!arm.grip.broken) anchored++;
  }
  // Normierung hart klemmen: der einseitige Hard-Stop der Gelenkgrenze erlaubt
  // leichtes Überschwingen, gemessen bis 1.61× des Limits. Ohne Klemmung wäre
  // der Eingangsvektor nicht normiert und ein Netz mit tanh-Ausgängen bekäme
  // Inkonsistenzen zwischen Trainings- und Grenzbetrieb.
  const cl = (v, m) => v > m ? m : (v < -m ? -m : v);
  for (let a = 0; a < c.arms.length; a++) out[i++] = cl(shoulder[a] / BODY.shoulderLimit, 1.2);
  for (let a = 0; a < c.arms.length; a++) out[i++] = cl(armCurl(w, c.arms[a]) / BODY.jointLimit, 1.2);
  for (let a = 0; a < c.arms.length; a++) {
    const t = tipPosition(w, c.arms[a]);
    const dx = t.x - c.x, dy = t.y - c.y;
    out[i++] = cl((dx * ca - dy * sa) / ARM_REACH, 1.2);
    out[i++] = cl((dx * sa + dy * ca) / ARM_REACH, 1.2);
  }
  for (let a = 0; a < c.arms.length; a++) {
    const g = c.arms[a].grip;
    out[i++] = g.broken ? 0 : Math.min(1, (g.force || 0) / Math.max(1, g.breakForce));
  }
  for (let a = 0; a < c.arms.length; a++) out[i++] = c.arms[a].grip.broken ? 0 : 1;

  const ax = (c.vx - c.prevVx) / (w.h * w.substeps);
  const ay = (c.vy - c.prevVy) / (w.h * w.substeps);
  out[i++] = Math.max(-1, Math.min(1, (ax * ca - ay * sa) / 600));
  out[i++] = Math.max(-1, Math.min(1, (ax * sa + ay * ca) / 600));
  c.prevVx = c.vx; c.prevVy = c.vy;
  out[i++] = anchored / c.arms.length;
  c.meanExtension = extSum / c.arms.length;
  c.meanCurl = curlSum / c.arms.length;
  out[i++] = c.meanExtension;
  out[i++] = Math.min(1, c.speed / 300);
  out[i++] = Math.max(-1, Math.min(1, c.av / 18));
  c.anchoredCount = anchored;
  return out;
}

/**
 * Steuerwerte anwenden: Muskel-Compliance + Saugnapf-Reflex.
 *
 * Es wird HIER KEINE KRAFT auf den Mantel ausgeübt. Ausschließlich Compliance
 * (Muskelsteifigkeit), Zielwinkel und Adhäsionsgrenze.
 */
export function applyControl(c, control) {
  const w = c.world;
  c.tone = Math.max(0, Math.min(1, control.tone));
  // Echolot-Aussendung ist eine AKTION, kein Dauerzustand: das Netz setzt den
  // Wunsch, tickSenses() feuert, sobald der Cooldown frei ist.
  c.echoWantsEmit = !!control.emitEcho;
  const comp = toneToCompliance(c.tone);
  for (let a = 0; a < c.arms.length; a++) {
    const arm = c.arms[a], st = c.perArm[a], ctl = control.arms[a];
    for (const m of arm.motors) m.compliance = comp;
    // target ist die kommandierte ABWEICHUNG von der Ruhe-Richtung; der Solver
    // legt `rest` selbst an. Also einfach klemmen und zuweisen.
    arm.motors[0].target = Math.max(-BODY.shoulderLimit, Math.min(BODY.shoulderLimit, ctl.shoulder));
    for (let j = 1; j < arm.motors.length; j++) {
      arm.motors[j].target = Math.max(-BODY.jointLimit, Math.min(BODY.jointLimit, ctl.curl));
    }
    st.adhesion = Math.max(0, Math.min(1, ctl.adhesion));

    // ── Saugnapf-Reflex ──
    if (st.cd > 0) st.cd -= w.h * w.substeps;
    const g = arm.grip;
    if (g.broken) {
      if (st.adhesion > 0.08 && st.cd <= 0) {
        const t = tipPosition(w, arm);
        const slip = Math.hypot(w.vx[arm.tip], w.vy[arm.tip]);
        if (slip < c.opts.gripSlipMax) {
          attachGrip(w, arm, adhesionToBreakForce(st.adhesion));
        } else {
          st.cd = c.opts.gripCooldown;
        }
        void t;
      }
    } else {
      g.breakForce = adhesionToBreakForce(st.adhesion);
      if (st.adhesion < 0.04) releaseGrip(arm);
    }
  }
}

/**
 * Alle Sinne abtasten und in `out` ab PROPRIO_DIM schreiben.
 *
 * Die Reihenfolge ist Teil der Schnittstelle und darf nicht mehr umgestellt
 * werden, ohne bestehende Genome zu verwerfen:
 *   [42..65]  Echolot: 12 Strahlen × (normierte Distanz, Intensität)
 *   [66..71]  Chemorezeptorbogen, 6 Werte
 *   [72..74]  Licht: links, rechts, gesamt
 *   [75..78]  passives Gehör: 4 Quadranten
 *   [79..84]  Tastkontakt pro Arm
 */
export function readSenses(c, out, entities) {
  readState(c);
  let o = PROPRIO_DIM;
  o = writeEcho(c.echo, out, o);
  o = writeChem(c.chem, out, o);
  o = writeLight(c.light, out, o);
  o = writeHear(c.hear, out, o);
  for (let a = 0; a < c.contact.length; a++) out[o++] = c.contact[a];
  return out;
}

/** Sinne fortschreiben — nach der Physik, vor dem nächsten Netz-Durchlauf. */
export function tickSenses(c, sim) {
  const w = c.world;
  readState(c);
  const dt = w.h * w.substeps;
  const heading = c.angle - Math.PI * 0.5;   // Augen liegen bei lokalem -y
  c.heading = heading;
  tickEcho(c.echo, dt, sim.time);
  if (c.echoWantsEmit) {
    const loud = emitEcho(c.echo, sim.map, c.x, c.y, heading, sim.time);
    if (loud > 0) { c.echoWantsEmit = false; c.lastEchoEmit = sim.time; c.echoLoudness = loud; }
  }
  sampleChem(c.chem, sim.entities, c.x, c.y, heading);
  sampleLight(c.light, sim.map, c.x, c.y, heading);
  sampleHear(c.hear, sim.entities, c.x, c.y);
  c.hearBearing = hearBearing(c.hear, heading);
  // Tastkontakt: Spitze nahe an der Karte?
  for (let a = 0; a < c.arms.length; a++) {
    const t = tipPosition(w, c.arms[a]);
    const hit = sim.map.sdf(t.x, t.y);
    const near = hit ? Math.max(0, 1 - hit.d / 9) : 0;
    c.contact[a] = c.contact[a] * 0.7 + near * 0.3;
  }
}


/**
 * Constraints stabil nach Farbe umsortieren, Offsets merken.
 * Objektidentität bleibt (Arme halten Referenzen); nur die Array-Reihenfolge
 * ändert sich. CPU und GPU sehen dadurch dieselbe Reihenfolge, und die GPU
 * fährt pro Farbe einen race-freien Dispatch.
 */
export function groupConstraintsByColor(world) {
  const buckets = Array.from({ length: MAX_COLORS }, () => []);
  for (const c of world.constraints) buckets[c.color | 0].push(c);
  const off = new Uint32Array(MAX_COLORS), cnt = new Uint32Array(MAX_COLORS);
  let n = 0, maxCnt = 0;
  for (let c = 0; c < MAX_COLORS; c++) {
    off[c] = n; cnt[c] = buckets[c].length;
    if (cnt[c] > maxCnt) maxCnt = cnt[c];
    n += cnt[c];
  }
  world.constraints = buckets.flat();
  world.colorOff = off; world.colorCnt = cnt; world.colorMax = maxCnt;
}

/** Messwerte VOR dem Physik-Schritt sichern. */
export function preStep(c) {
  readState(c);
  c._sx = c.x; c._sy = c.y;
}

/** Messwerte NACH dem Physik-Schritt aufsummieren. */
export function postStep(c) {
  const w = c.world;
  readState(c);
  c.steps++;
  const sx = c._sx, sy = c._sy;

  const dx = c.x - sx, dy = c.y - sy;
  const stepDist = Math.hypot(dx, dy);
  c.pathLen += stepDist;
  c.angularAccum += Math.abs(c.av) * w.h * w.substeps;
  const fa = c.angle - Math.PI * 0.5;
  const fwd = dx * Math.cos(fa) + dy * Math.sin(fa);
  if (fwd > 0) c.forwardProg += fwd;

  const anchored = c.arms.reduce((n, a) => n + (a.grip.broken ? 0 : 1), 0);
  c.anchoredSum += anchored;
  if (anchored > 0) {
    c.anchoredSteps++;
    const dSpawn = Math.hypot(c.x - c.spawn.x, c.y - c.spawn.y);
    if (dSpawn > c.prevSpawnDist) c.gripPullProg += dSpawn - c.prevSpawnDist;
    c.prevSpawnDist = dSpawn;
  } else {
    c.prevSpawnDist = Math.hypot(c.x - c.spawn.x, c.y - c.spawn.y);
  }
  if (anchored === 0 && stepDist > 0.3) c.freeMove += stepDist;
  // Energie: Muskeltonus und Haftstärke kosten, Bewegung kostet.
  // Bewusst KEIN Bonus dafür, überhaupt Energie auszugeben — v2 hatte
  // `idleP − energySpent*1.5` und belohnte damit Verschwendung.
  c.energySpent += (c.tone * 0.0006 + stepDist * 0.0004) * c.arms.length;
  if (anchored > 0 && !c.wasAnchored) c.gripCycles++;
  c.wasAnchored = anchored > 0;
  c.maxDisplacement = Math.max(c.maxDisplacement, Math.hypot(c.x - c.spawn.x, c.y - c.spawn.y));
  return stepDist;
}

export function stepSim(sim) {
  // Die Physik wird EINMAL pro Sim-Schritt für die ganze Welt integriert —
  // nicht pro Krake. Der erste Entwurf rief w.step() inside stepCreature auf,
  // was bei einem geteilten World n komplette Welt-Schritte pro Sim-Schritt
  // bedeutete: O(n²). Gemessen 1083 ms/Schritt bei 64 Kraken statt ~18 ms.
  //
  // Das ist genau die Semantik, die das GPU-Batching braucht: N Tiere sind N
  // unabhängige Lanes, die im selben Dispatch integriert werden. Wer hier pro
  // Tier einen eigenen Solverlauf macht, macht Batching unmöglich.
  for (const c of sim.creatures) preStep(c);
  sim.world.step();
  for (const c of sim.creatures) postStep(c);
  sim.time += sim.opts.h * sim.opts.substeps;
  sim.stepCount++;
  // Welt-Update: Licht wandert, Entitäten bewegen sich und emittieren dabei,
  // Sinne werden nachgeführt. Reihenfolge: erst Physik, dann Wahrnehmung —
  // das Tier sieht den Zustand NACH seinem eigenen Schritt.
  const dtS = sim.opts.h * sim.opts.substeps;
  tickMapLights(sim.map, dtS);
  if (sim.entities.length) {
    const threats = sim.creatures.map(c => ({ x: c.x, y: c.y }));
    for (const e of sim.entities) tickEntity(e, dtS, sim.map.width, sim.map.height, threats, sim.rnd);
  }
  for (const c of sim.creatures) tickSenses(c, sim);
}

/** Nur für Einzel-Tier-Tests: ein Tier in eigener Welt einen Schritt fahren. */
export function stepCreature(c) {
  preStep(c);
  c.world.step();
  postStep(c);
}
