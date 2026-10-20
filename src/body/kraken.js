/**
 * Körpermodell: Mantel + 6 Arme × 7 Segmente, EIN Saugnapf pro Arm an der Spitze.
 *
 * Gelenke sind Pin-Constraints (positionsgekoppelt, rotationsfrei = Scharnier).
 * Muskeln sind WINKEL-Constraints, deren Compliance die KI steuert — kein
 * Drehmoment, sondern Steifigkeit. Das ist das Muskel-Hydrostat-Prinzip in
 * seiner natürlichen Formulierung.
 *
 * Es wirkt zu keinem Zeitpunkt eine Antriebskraft auf den Mantel. Fortbewegung
 * entsteht ausschließlich so: Spitze haftet am Untergrund → Muskeln knicken den
 * Arm ein → der Arm wird kürzer → der Mantel wird herangezogen. Hält die Spitze
 * stattdessen etwas Freies und Leichtes, zieht dieselbe Mechanik das Objekt heran.
 *
 * Massenverhältnis Rumpf : proximales Segment ≈ 6:1. XPBD- und
 * Sequential-Impulse-Solver werden ab etwa 10:1 instabil — in Kraken_v2 war das
 * Verhältnis durch degenerierte Kapselgeometrie 250:1, mitursächlich für die
 * Spawn-Explosion (max|ω| = 3918 rad/step im ersten Frame).
 */
import { C_ANGLE, C_GRIP, C_PIN, makeConstraint } from '../physics/xpbd.js';

export const BODY = {
  arms: 6,
  segsPerArm: 7,
  segLen: 13.5,
  shoulderR: 24,
  mantleRadius: 18,
  mantleMass: 20,
  mantleInertiaFactor: 2.5,   // Mantel ist träge: Richtungs-Anker
  segMassProx: 3.2,
  segMassDist: 1.8,
  segThickProx: 11.9,
  segThickDist: 5.9,
  jointLimit: Math.PI * 0.42,    // ±75.6° pro Gelenk
  shoulderLimit: Math.PI * 0.62,
  spawnCurl: 0.28,
  // XPBD: α̃ = α/h², und α̃ muss in der Größenordnung der effektiven Winkelmasse
  // wSum = invI_A + invI_B ≈ 0.33 liegen, sonst ist der Muskel entweder wirkungs-
  // los (α̃ >> wSum) oder ein Starrkörper (α̃ << wSum). Mit h = 1/120 ist
  // α̃ = 14400/k, also k ≈ 43.600 für α̃ = wSum. Erster Entwurf hatte
  // k ∈ [8, 400] → α̃ = 36…1800 → Muskeln praktisch inert (gemessen: Gelenk-
  // korrektur 1e-7 rad pro Iteration).
  muscleStiffnessMin: 2000,      // schlaff  (α̃ ≈ 7.2,  ~22× wSum)
  muscleStiffnessMax: 400000,    // steif    (α̃ ≈ 0.036, ~0.1× wSum)
  gripBreakMin: 40,              // N — Adhäsionsgrenze bei Haftstärke 0
  gripBreakMax: 900,             // N — bei Haftstärke 1
};

export const ARM_REACH = BODY.shoulderR + BODY.segsPerArm * BODY.segLen;   // 118.5 px

const rectInertia = (m, w, h) => m * (w * w + h * h) / 12;

/**
 * Farben für race-freies paralleles Constraint-Solving.
 *
 * Zwei Constraints dürfen im SELBEN Dispatch laufen, solange sie keinen Körper
 * teilen. Greedy über Inzidenz-Bitmasken: kleinste freie Farbe wählen.
 * Notwendig, weil paralleles Schreiben auf ppx/angP ohne Atomics ein Data-Race
 * ist — gemessen auf RTX 3070 Ti: Determinismus-Check kippte, passives
 * Gleichgewicht driftete 0.05 px. Maximaler Grad: Mantel mit 12 Constraints
 * (6 Pins + 6 Motoren) → 16 Farben reichen sicher (Vizing: Δ+1).
 */
export const MAX_COLORS = 16;

export function assignColor(colorMask, a, b) {
  const used = colorMask[a] | (b >= 0 ? colorMask[b] : 0);
  let c = 0;
  while (used & (1 << c)) c++;
  colorMask[a] |= 1 << c;
  if (b >= 0) colorMask[b] |= 1 << c;
  return c;
}

/**
 * Legt einen Kraken constraint-frei in die Welt (Verletzung exakt 0.0000).
 *
 * Verkettung: halber Schritt zum Gelenk → drehen → halber Schritt zum Zentrum.
 * Constraints werden NACH der Pose erzeugt. In Kraken_v2 wurde die Pose nach
 * der Constraint-Erzeugung gesetzt, und weil Matter `pointA` nur inkrementell ab
 * dem bei create() eingefrorenen `angleA` rotiert, war der Schulteranker um bis
 * zu 57 px versetzt.
 */
export function spawnKraken(world, x, y, angle, opts = {}) {
  const B = BODY;
  const curl = opts.curl !== undefined ? opts.curl : B.spawnCurl;

  const mantleI = (B.mantleMass * B.mantleRadius * B.mantleRadius / 2) * B.mantleInertiaFactor;
  const mantle = world.addBody(x, y, angle, B.mantleMass, mantleI);
  // Mantel-Kollisionspunkte (Kreis-Approximation, 8 Abtastpunkte)
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    world.addCollisionPoint(mantle, Math.cos(a) * B.mantleRadius * 0.7, Math.sin(a) * B.mantleRadius * 0.7, B.mantleRadius * 0.45);
  }

  const arms = [];
  for (let a = 0; a < B.arms; a++) {
    const relAng = (a / B.arms) * Math.PI * 2;
    const th0 = angle + relAng;
    const segs = [];
    let jx = x + Math.cos(th0) * B.shoulderR;
    let jy = y + Math.sin(th0) * B.shoulderR;
    let th = th0;

    for (let i = 0; i < B.segsPerArm; i++) {
      const t = B.segsPerArm > 1 ? i / (B.segsPerArm - 1) : 0;
      const mass = B.segMassProx + (B.segMassDist - B.segMassProx) * t;
      const thick = B.segThickProx + (B.segThickDist - B.segThickProx) * t;
      const ux = Math.cos(th), uy = Math.sin(th);
      const cx = jx + ux * B.segLen * 0.5;
      const cy = jy + uy * B.segLen * 0.5;
      const b = world.addBody(cx, cy, th, mass, rectInertia(mass, B.segLen, thick));
      const r = thick * 0.5;
      // Kapsel als sphere-swept: zwei Endpunkte. Einfach, robust, GPU-trivial.
      world.addCollisionPoint(b, -B.segLen * 0.5 + r * 0.4, 0, r);
      world.addCollisionPoint(b, B.segLen * 0.5 - r * 0.4, 0, r);
      segs.push(b);
      jx = cx + ux * B.segLen * 0.5;
      jy = cy + uy * B.segLen * 0.5;
      th += curl;
    }

    // Schulter-Pin: Mantel → Segment 0
    world.addConstraint(makeConstraint(C_PIN, mantle, segs[0],
      Math.cos(relAng) * B.shoulderR, Math.sin(relAng) * B.shoulderR,
      -B.segLen * 0.5, 0, 0));

    const motors = [];
    // A = parent (Mantel), B = child (Segment 0), damit die Nebenbedingung
    // C = θ_B − θ_A − target ist. Umgekehrt wäre das Vorzeichen invertiert.
    //
    // Ziel und Grenzen liegen RELATIV ZUR RUHE-RICHTUNG relAng, nicht bei 0.
    // Der erste Entwurf hatte target = 0, womit der Motor alle sechs Arme auf
    // die Mantelachse falten wollte — ein frisch gespawnter Krake war damit
    // nicht im Gleichgewicht und driftete (gemessen 23 px in 4 s ohne jeden
    // Antrieb). target = relAng macht den Spawn-Zustand zum echten Fixpunkt.
    const mShoulder = makeConstraint(C_ANGLE, mantle, segs[0], 0, 0, 0, 0, 1 / B.muscleStiffnessMin);
    mShoulder.rest = relAng;   // Ruhe-Richtung des Arms
    mShoulder.target = 0;      // kommandierte Abweichung = 0 im Ruhezustand
    mShoulder.minAngle = -B.shoulderLimit; mShoulder.maxAngle = B.shoulderLimit;
    mShoulder.owner = a;
    world.addConstraint(mShoulder);
    motors.push(mShoulder);

    for (let i = 0; i < segs.length - 1; i++) {
      world.addConstraint(makeConstraint(C_PIN, segs[i], segs[i + 1],
        B.segLen * 0.5, 0, -B.segLen * 0.5, 0, 0));
      const motor = makeConstraint(C_ANGLE, segs[i], segs[i + 1], 0, 0, 0, 0, 1 / B.muscleStiffnessMin);
      motor.rest = curl;     // Spawn-Pose ist der Fixpunkt
      motor.target = 0;
      motor.minAngle = -B.jointLimit; motor.maxAngle = B.jointLimit;
      motor.owner = a;
      world.addConstraint(motor);
      motors.push(motor);
    }

    // EIN Saugnapf, an der Spitze. Verankert an einem WELT-Punkt (Untergrund).
    const tip = segs[segs.length - 1];
    const grip = makeConstraint(C_GRIP, tip, -1, B.segLen * 0.5, 0, x, y, 0);
    grip.broken = true;
    grip.breakForce = 0;
    grip.owner = a;
    world.addConstraint(grip);

    arms.push({ relAng, segs, motors, grip, tip, index: a });
  }
  return { mantle, arms, spawn: { x, y, angle } };
}

export function tipPosition(world, arm) {
  const b = arm.tip, L = BODY.segLen * 0.5;
  const c = Math.cos(world.ang[b]), s = Math.sin(world.ang[b]);
  return { x: world.px[b] + L * c, y: world.py[b] + L * s };
}

/** Spitze am Untergrund verankern. breakForce = Adhäsionsgrenze in Newton. */
export function attachGrip(world, arm, breakForce) {
  const t = tipPosition(world, arm), g = arm.grip;
  g.lbx = t.x; g.lby = t.y;
  g.broken = false;
  g.breakForce = breakForce;
  g.lambdaX = 0; g.lambdaY = 0; g.force = 0;
  return g;
}

export function releaseGrip(arm) { arm.grip.broken = true; arm.grip.force = 0; }

/** Mittlere Krümmung des Arms — Propriozeption [1]. */
export function armCurl(world, arm) {
  let sum = 0, n = 0;
  for (let i = 1; i < arm.segs.length; i++) {
    const d = world.ang[arm.segs[i]] - world.ang[arm.segs[i - 1]];
    sum += Math.atan2(Math.sin(d), Math.cos(d)); n++;
  }
  return n ? sum / n : 0;
}

/** 0 = ganz eingerollt, 1 = ganz gestreckt — Propriozeption [3]. */
export function armExtension(world, arm) {
  const t = tipPosition(world, arm);
  const s0 = arm.segs[0];
  return Math.min(1, Math.hypot(t.x - world.px[s0], t.y - world.py[s0]) / Math.max(1, ARM_REACH));
}

/** Muskel-Compliance aus einem Tonus 0..1 (0 = schlaff, 1 = steif). */
export function toneToCompliance(tone) {
  const B = BODY;
  const t = Math.max(0, Math.min(1, tone));
  const stiff = B.muscleStiffnessMin + t * (B.muscleStiffnessMax - B.muscleStiffnessMin);
  return 1 / stiff;
}

/** Adhäsionsgrenze aus einer Haftstärke 0..1. */
export function adhesionToBreakForce(a) {
  const B = BODY;
  const c = Math.max(0, Math.min(1, a));
  return B.gripBreakMin + (B.gripBreakMax - B.gripBreakMin) * c;
}
