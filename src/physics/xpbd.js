/**
 * XPBD-Solver (CPU-Referenzimplementierung).
 *
 * Diese Datei ist das ORAKEL für die WebGPU-Portierung: derselbe Algorithmus,
 * dieselben Datenlayouts, dieselbe Reihenfolge. `tools/harness.js` vergleicht
 * GPU gegen CPU — ohne diese Referenz ist ein Shader-Fehler nicht von einem
 * Designproblem zu unterscheiden.
 *
 * ── Warum XPBD und kein Rigid-Body-Engine ──────────────────────────
 * Kraken_v2 benutzte matter-js. Vier voneinander unabhängige Engine-Eigenheiten
 * haben dort jeweils Stunden gekostet und sind hier strukturell ausgeschlossen:
 *
 *   1. `angularStiffness` (Default 0) führte in Constraint.solve() zu
 *      `bodyA.angle -= p` OHNE `anglePrev` mitzunehmen — der Solver erzeugte
 *      Winkelgeschwindigkeit aus dem Nichts. Isoliert nachgestellt: eine
 *      gekrümmte Kette mit Kräften=0, Momenten=0 und Constraint-Verletzung
 *      0.00000 explodierte auf max|v| = 69 px/step und max|ω| = 95 rad/step
 *      im allerersten Update.
 *   2. `chamfer radius > min(w,h)/2` erzeugte degenerierte 1.6-px-Splitter mit
 *      inertia 1.20 statt 10.52 und einem Trägheitsverhältnis von 250:1.
 *   3. `constraint.pointA` wird nur INKREMENTELL um (angle - angleA) rotiert —
 *      wer die Pose nach Constraint.create() setzt, hat einen versetzten Anker.
 *   4. `setVelocity` normalisiert auf `_baseDelta`, nicht auf das echte Delta.
 *
 * XPBD umgeht die ganze Klasse: Positionen werden projiziert, und die
 * Geschwindigkeiten werden AM ENDE aus der tatsächlichen Positionsänderung
 * zurückgerechnet (`v = (pPred - p)/h`). Projektion und Impuls können damit
 * nicht auseinanderlaufen. Zusätzlich ist XPBD von Natur aus dissipativ.
 *
 * ── Muskelmodell ───────────────────────────────────────────────────
 * Ein Muskel ist hier KEIN Drehmoment, sondern die COMPLIANCE eines
 * Winkel-Constraints. Compliance α = 1/Steifigkeit, und die KI steuert α
 * direkt. Das ist exakt das Muskel-Hydrostat-Prinzip: ein schlaffer Arm hat
 * hohe Compliance (gibt nach), ein gespannter niedrige (hält Form und
 * überträgt Zug). In v2 musste dieser Zusammenhang über einen skalierten
 * Drehmoment-Term nachgebildet werden — hier ist er die natürliche Formulierung.
 *
 * ── Saugnapf ───────────────────────────────────────────────────────
 * Ein Grip ist ein Pin-Constraint gegen einen Welt-Anker. Der akkumulierte
 * Lagrange-Multiplikator λ IST die Constraint-Kraft (F = λ/h). Die
 * Haltekraft ist also keine heuristic Distanzschwelle mehr (v2: GRIP_BREAK
 * = 22 px Dehnung), sondern die echte Kraft: reißt ab, wenn |λ|/h die vom
 * Netz gewählte Adhäsionsstärke übersteigt.
 */

export const SOLVER_DEFAULTS = {
  substeps: 2,
  iterations: 4,
  h: 1 / 120,          // s pro Substep → 2 Substeps = 1/60 s Sim-Schritt
  linDrag: 3.0,        // 1/s  (exponentielle Dämpfung)
  angDrag: 3.0,        // 1/s
  maxSpeed: 360,       // px/s   hartes Limit
  maxAngSpeed: 18,     // rad/s  hartes Limit
};

/**
 * Korrektur-Totzone (slack) — der f32-Boden des GPU-Ports.
 *
 * Die CPU rechnet Float64, WGSL Float32. Bei Koordinaten um 1440 px ist
 * ulp(position) ≈ 1.7e-4 px; das in Float64 berechnete Spawn-Gleichgewicht ist
 * nach der Rundung in die f32-Buffer deshalb KEIN exakter Fixpunkt mehr: der
 * Solver sieht Mikro-Verletzungen (G2 gemessen, Build d487532: schlechteste
 * Pin-Verletzung nach Upload 2.0e-4 px — exakt diese Größenordnung) und
 * „korrigiert" sie in jedem Substep. `v = (pPred - p)/h` hebt das 1e-4-px-
 * Zittern auf ~0.16 px/s Rauschgeschwindigkeit, die sich linear integriert:
 * G1-Wachstum gemessen 1.2789 px nach 2400 Schritten, CPU-Oracle 3.4e-9.
 * Kein Race und kein Formelfehler — die numerische Bodenhöhe von f32.
 *
 * Heilung: Korrekturen unterhalb slack werden übersprungen, in BEIDEN Solvern
 * (Oracle-Vergleich bleibt 1:1, check-wgsl.js prüft die Parität). Ein Zustand
 * mit allen |C| < slack und allen v = 0 ist damit ein BITEXAKTER Fixpunkt:
 * pPred = p + h·0 = p, keine Projektion, v' = (p-p)/h = 0 — für immer.
 * Echte Physik spielt 1–2 Zehnerpotenzen darüber: Muskelzug verformt Gelenke
 * um ~0.01–1 px pro Substep, Grip-Lasten erzeugen |λ| weit über dem Band.
 * 2e-3 px = 1/59 000 der Armreichweite (118.5 px) — physikalisch unsichtbar.
 */
export const SLACK_POS = 2e-3;   // px  — Pin-/Grip-Verletzung, Kollisions-Penetration
export const SLACK_ANG = 2e-4;   // rad — Muskel-Abweichung, Gelenkgrenz-Überschuss (0.011°)

/** Winkel auf (-π, π] wickeln — O(1), keine while-Schleife. */
/**
 * Farben für race-freies paralleles Constraint-Solving.
 *
 * Zwei Constraints dürfen im SELBEN Dispatch laufen, solange sie keinen Körper
 * teilen. Greedy über Inzidenz-Bitmasken: kleinste freie Farbe. Notwendig, weil
 * paralleles Schreiben auf ppx/angP ohne Atomics ein Data-Race ist (gemessen:
 * Determinismus-Check kippte, passives Gleichgewicht driftete 0.05 px).
 * Maximaler Grad: Mantel mit 12 Constraints (6 Pins + 6 Motoren) → 16 Farben
 * reichen sicher (Vizing: Δ+1).
 *
 * Liegt bewusst in der PHYSIK-Schicht, nicht in core.js: gpu.html baut Welten
 * direkt aus XpbdWorld + spawnKraken (ohne createSim), und ohne Färbung wären
 * dort alle Constraints in Farbe 0 gelandet — der Race wäre unbemerkt zurück.
 * ensureColored() wird deshalb auch in gpuworld.create() defensiv aufgerufen.
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

/** Färbt fehlende Constraints und gruppiert stabil nach Farbe. Idempotent. */
export function ensureColored(world) {
  if (world.colorOff) return world.colorOff;
  const mask = new Uint32Array(Math.max(1024, world.n * 2));
  for (const c of world.constraints) {
    if (c.color === undefined) c.color = assignColor(mask, c.a, c.b >= 0 ? c.b : -1);
  }
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
  return off;
}

export function wrapAngle(a) {
  const TAU = Math.PI * 2;
  let d = a % TAU;
  if (d > Math.PI) d -= TAU;
  else if (d < -Math.PI) d += TAU;
  return d;
}

/** Winkel-Differenz a-b auf (-π, π]. O(1) — v2 hatte hier eine while-Schleife. */
export function angleDelta(a, b) {
  let d = (a - b) % (Math.PI * 2);
  if (!Number.isFinite(d)) return 0;
  if (d > Math.PI) d -= Math.PI * 2;
  else if (d < -Math.PI) d += Math.PI * 2;
  return d;
}

// ─────────────────────────────────────────────────────────────
//  Constraint-Typen
// ─────────────────────────────────────────────────────────────
export const C_PIN = 0;    // Punkt-zu-Punkt-Gelenk (Scharnier)
export const C_ANGLE = 1;  // relativer Winkel mit Ziel — der Muskel
export const C_GRIP = 2;   // Pin gegen Welt-Anker, reißt bei Kraftüberschreitung

/**
 * @param {number} type     C_PIN | C_ANGLE | C_GRIP
 * @param {number} a        Index Body A
 * @param {number} b        Index Body B, oder -1 für Welt-Anker
 * @param {number} lax,lay  Anker im Lokalsystem von A
 * @param {number} lbx,lby  Anker im Lokalsystem von B (bei b<0: Welt-Koordinaten)
 * @param {number} compliance  α (0 = starr). Für C_ANGLE: 1/Muskelsteifigkeit.
 */
export function makeConstraint(type, a, b, lax, lay, lbx, lby, compliance = 0) {
  return {
    type, a, b, lax, lay, lbx, lby, compliance,
    lambda: 0, lambdaX: 0, lambdaY: 0,   // akkumuliert pro Substep
    broken: false,
    breakForce: Infinity,                 // für C_GRIP: Adhäsionsgrenze in N
    target: 0,                            // für C_ANGLE: kommandierte ABWEICHUNG von rest
    rest: 0,                              // für C_ANGLE: Ruhe-Richtung des Gelenks
    minAngle: -Infinity, maxAngle: Infinity,
    owner: -1,                            // Index des Arms, für Sensorik
  };
}

/**
 * XPBD-Solver über flachen Arrays.
 *
 * Flache Typisierte Arrays statt Objekten — bewusst: dasselbe Layout wird 1:1
 * als GPU-Buffer gespiegelt, die Portierung ist dann ein memcpy pro Puffer
 * statt einer Neuformulierung.
 */
export class XpbdWorld {
  constructor(opts = {}) {
    const o = { ...SOLVER_DEFAULTS, ...opts };
    this.substeps = o.substeps;
    this.iterations = o.iterations;
    this.h = o.h;
    this.linDrag = o.linDrag;
    this.angDrag = o.angDrag;
    this.maxSpeed = o.maxSpeed;
    this.maxAngSpeed = o.maxAngSpeed;

    this.n = 0;
    this.cap = 0;
    this.px = null; this.py = null;         // Position (aktuell)
    this.ppx = null; this.ppy = null;       // prädizierte Position (während Solve)
    this.vx = null; this.vy = null;
    this.ang = null; this.angP = null;
    this.av = null;                         // Winkelgeschwindigkeit
    this.im = null;                         // inverse Masse
    this.ii = null;                         // inverse Trägheit
    this.fx = null; this.fy = null;         // externe Kraft (pro Substep konsumiert)
    this.tq = null;                         // externes Drehmoment
    this.collPts = [];                      // [{body, lx, ly, r}] Kollisionsabtastpunkte
    this.constraints = [];
    this.sdf = null;                        // Signierte-Distanz-Funktion der Karte
    this.stats = { maxSpeed: 0, maxAngSpeed: 0, nan: 0, collisions: 0, broken: 0 };
  }

  /** Einen Körper anlegen. Gibt den Index zurück. */
  addBody(x, y, angle, mass, inertia) {
    const i = this.n++;
    if (i >= this.cap) this._grow();
    this.px[i] = x; this.py[i] = y;
    this.ppx[i] = x; this.ppy[i] = y;
    this.vx[i] = 0; this.vy[i] = 0;
    this.ang[i] = angle; this.angP[i] = angle; this.av[i] = 0;
    this.im[i] = mass > 0 ? 1 / mass : 0;
    this.ii[i] = inertia > 0 ? 1 / inertia : 0;
    this.fx[i] = 0; this.fy[i] = 0; this.tq[i] = 0;
    return i;
  }

  _grow() {
    const cap = Math.max(64, this.cap * 2);
    const g = (src) => { const d = new Float64Array(cap); if (src) d.set(src); return d; };
    this.px = g(this.px); this.py = g(this.py);
    this.ppx = g(this.ppx); this.ppy = g(this.ppy);
    this.vx = g(this.vx); this.vy = g(this.vy);
    this.ang = g(this.ang); this.angP = g(this.angP); this.av = g(this.av);
    this.im = g(this.im); this.ii = g(this.ii);
    this.fx = g(this.fx); this.fy = g(this.fy); this.tq = g(this.tq);
    this.cap = cap;
  }

  /** Kollisionsabtastpunkt an einem Körper (lokaler Offset, Radius). */
  addCollisionPoint(body, lx, ly, r) {
    this.collPts.push({ body, lx, ly, r });
  }

  addConstraint(c) { this.constraints.push(c); return c; }

  /**
   * Ein Substep: Kräfte → Prädiktion → Projektion → Geschwindigkeiten zurück.
   *
   * Die Reihenfolge ist wesentlich. Insbesondere wird `v` AM ENDE aus
   * `(pPred - p)/h` neu gebildet — nicht vorher. Damit ist garantiert, dass
   * Positionsprojektion und Impuls konsistent sind (genau das war in matter-js
   * die Quelle der Energieinjektion).
   */
  substep() {
    const h = this.h;
    const n = this.n;
    const { px, py, ppx, ppy, vx, vy, ang, angP, av, im, ii, fx, fy, tq } = this;

    // ── 1. externe Kräfte, Dämpfung, Limits, Prädiktion ──
    const linDamp = Math.exp(-this.linDrag * h);
    const angDamp = Math.exp(-this.angDrag * h);
    for (let i = 0; i < n; i++) {
      if (im[i] === 0 && ii[i] === 0) { // statisch
        ppx[i] = px[i]; ppy[i] = py[i]; angP[i] = ang[i];
        vx[i] = 0; vy[i] = 0; av[i] = 0;
        continue;
      }
      vx[i] = (vx[i] + h * im[i] * fx[i]) * linDamp;
      vy[i] = (vy[i] + h * im[i] * fy[i]) * linDamp;
      av[i] = (av[i] + h * ii[i] * tq[i]) * angDamp;

      const sp = Math.hypot(vx[i], vy[i]);
      if (sp > this.maxSpeed) { const k = this.maxSpeed / sp; vx[i] *= k; vy[i] *= k; }
      if (av[i] > this.maxAngSpeed) av[i] = this.maxAngSpeed;
      else if (av[i] < -this.maxAngSpeed) av[i] = -this.maxAngSpeed;

      if (!Number.isFinite(vx[i]) || !Number.isFinite(vy[i])) { vx[i] = 0; vy[i] = 0; this.stats.nan++; }
      if (!Number.isFinite(av[i])) { av[i] = 0; this.stats.nan++; }

      ppx[i] = px[i] + h * vx[i];
      ppy[i] = py[i] + h * vy[i];
      angP[i] = ang[i] + h * av[i];
      fx[i] = 0; fy[i] = 0; tq[i] = 0;
    }

    // ── 2. Projektions-Iterationen ──
    for (let it = 0; it < this.iterations; it++) {
      for (let k = 0; k < this.constraints.length; k++) {
        const c = this.constraints[k];
        if (c.broken) continue;
        if (c.type === C_ANGLE) this._solveAngle(c, it === 0);
        else this._solvePin(c, it === 0);
      }
      this._solveCollisions();
    }

    // ── 3. Geschwindigkeiten aus der Positionsänderung zurückrechnen ──
    // Die Limits werden HIER nocheinmal angewendet, nicht nur in Phase 1:
    // die Constraint-Projektion verändert angP/ppx selbst und kann damit über
    // das Vorhersage-Limit hinausgehen. Ein Muskel, der ein Gelenk in einem
    // Substep um 1 rad dreht, ergibt sonst av = 120 rad/s (gemessen: 118.9).
    // Die Klemmung ist unelastisch — sie entzieht Energie, erzeugt keine.
    for (let i = 0; i < n; i++) {
      vx[i] = (ppx[i] - px[i]) / h;
      vy[i] = (ppy[i] - py[i]) / h;
      av[i] = angleDelta(angP[i], ang[i]) / h;
      const sp2 = Math.hypot(vx[i], vy[i]);
      if (sp2 > this.maxSpeed) { const k = this.maxSpeed / sp2; vx[i] *= k; vy[i] *= k; }
      if (av[i] > this.maxAngSpeed) av[i] = this.maxAngSpeed;
      else if (av[i] < -this.maxAngSpeed) av[i] = -this.maxAngSpeed;
      px[i] = ppx[i]; py[i] = ppy[i];
      // WICHTIG: den Winkel aus dem Prädiktor in den Zustand übernehmen.
      // Ohne diese Zeile bleibt `ang` auf dem Vor-Substep-Wert stehen, jeder
      // folgende Substep berechnet av = (angP - ang)/h aus einem veralteten
      // Winkel, und die Winkelgeschwindigkeit hängt dauerhaft am Limit
      // (gemessen: max|ω| = 18.000 rad/s bei einem körperlich ruhenden Tier).
      ang[i] = angP[i];
      // Winkel einwickeln: cos/sin bleiben identisch, Zahlen klein.
      // Kraken_v2 ließ body.angle unbegrenzt wachsen (gemessen 1.784.349 rad) —
      // die while-Schleife in angDiff() lief dann 284.000 Durchgänge pro Aufruf.
      if (ang[i] > 50 || ang[i] < -50) { const w = wrapAngle(ang[i]); ang[i] = w; angP[i] = w; }
    }

    this._measure();
  }

  /** Pin-/Grip-Constraint: Vektor-Nebenbedingung C = gA - gB = 0, exakte 2×2-Masse. */
  _solvePin(c, firstIteration) {
    if (firstIteration) { c.lambdaX = 0; c.lambdaY = 0; }
    const A = c.a, B = c.b;
    const { ppx, ppy, angP, im, ii } = this;

    const ca = Math.cos(angP[A]), sa = Math.sin(angP[A]);
    const rax = c.lax * ca - c.lay * sa, ray = c.lax * sa + c.lay * ca;
    const gax = ppx[A] + rax, gay = ppy[A] + ray;

    let gbx, gby, rbx = 0, rby = 0, wB = 0, wIB = 0;
    if (B >= 0) {
      const cb = Math.cos(angP[B]), sb = Math.sin(angP[B]);
      rbx = c.lbx * cb - c.lby * sb; rby = c.lbx * sb + c.lby * cb;
      gbx = ppx[B] + rbx; gby = ppy[B] + rby;
      wB = im[B]; wIB = ii[B];
    } else {
      gbx = c.lbx; gby = c.lby;   // Welt-Anker
    }

    let Cx = gax - gbx, Cy = gay - gby;
    if (!Number.isFinite(Cx) || !Number.isFinite(Cy)) return;
    // Totzone (SLACK_POS): unterhalb des f32-Bodens gilt das Constraint als
    // erfüllt. Ohne das integriert sich Rundungszittern über v = Δp/h zu
    // dauerhafter Rauschgeschwindigkeit auf (G1 gemessen: 1.2789 px Drift).
    if (Cx * Cx + Cy * Cy < SLACK_POS * SLACK_POS) return;

    // perp(r) = (-ry, rx) — Jacobian-Spalte zur Winkel-Freiheitsgrade
    const pax = -ray, pay = rax;
    const pbx = -rby, pby = rbx;
    const wA = im[A], wIA = ii[A];

    // Effektive 2×2-Masse:  K = (wA+wB)·I + wIA·uA uAᵀ + wIB·uB uBᵀ
    // mit u = perp(r) = (-ry, rx), also u uᵀ = [[ux², ux·uy], [ux·uy, uy²]].
    const wSum = wA + wB;
    const k00 = wSum + wIA * pax * pax + wIB * pbx * pbx;
    const k01 =          wIA * pax * pay + wIB * pbx * pby;
    const k11 = wSum + wIA * pay * pay + wIB * pby * pby;

    const det = k00 * k11 - k01 * k01;
    if (Math.abs(det) < 1e-12) return;

    const alphaTilde = c.compliance / (this.h * this.h);
    // Δλ = -K⁻¹ (C + α̃ λ)
    let bx = Cx + alphaTilde * c.lambdaX;
    let by = Cy + alphaTilde * c.lambdaY;
    const i00 = k11 / det, i11 = k00 / det, i01 = -k01 / det;
    let dlx = -(i00 * bx + i01 * by);
    let dly = -(i01 * bx + i11 * by);
    if (!Number.isFinite(dlx) || !Number.isFinite(dly)) return;

    c.lambdaX += dlx; c.lambdaY += dly;

    ppx[A] += wA * dlx; ppy[A] += wA * dly;
    angP[A] += wIA * (pax * dlx + pay * dly);
    if (B >= 0) {
      ppx[B] -= wB * dlx; ppy[B] -= wB * dly;
      angP[B] -= wIB * (pbx * dlx + pby * dly);
    }
  }

  /** Winkel-Constraint (Muskel): skalare Nebenbedingung C = wrap(θB - θA - target). */
  _solveAngle(c, firstIteration) {
    if (firstIteration) c.lambda = 0;
    const A = c.a, B = c.b;
    const { angP, ii } = this;
    const wA = ii[A], wB = B >= 0 ? ii[B] : 0;
    const wSum = wA + wB;
    if (wSum <= 0) return;

    // Alle Winkelgrößen werden als ABWEICHUNG VON DER RUHE-RICHTUNG geführt.
    //
    // Warum nicht absolut: `angleDelta` wickelt auf (-π, π], aber die Ruhe-
    // Richtung eines Arms ist relAng = k·2π/6, also bis 5.24 rad. Vergleicht man
    // einen gewickelten Ist-Winkel mit einer ungewickelten Grenze, entsteht eine
    // Phantom-Überschreitung von mehreren rad — gemessen: max|v| am 360-px/s-
    // Limit und 3840 px Pfad bei einem eigentlich passiven Tier.
    //
    // In Abweichungs-Koordinaten sind Ziel und Grenzen immer klein und das
    // Wickeln ist harmlos. `c.target` ist damit die KOMMANDIERTE Abweichung —
    // genau das, was ein Netz ausgeben sollte.
    const rel = B >= 0 ? angleDelta(angP[B], angP[A]) : angP[A];
    const dev = angleDelta(rel, c.rest);
    const C = angleDelta(dev, c.target);
    if (!Number.isFinite(C)) return;

    // Totzone (SLACK_ANG): Muskelkorrektur nur über dem f32-Boden. Ruhe-
    // zustände (C ≈ 1e-6 rad aus Rundung) werden damit zu exakten Fixpunkten.
    if (Math.abs(C) >= SLACK_ANG) {
      const alphaTilde = c.compliance / (this.h * this.h);
      const dl = -(C + alphaTilde * c.lambda) / (wSum + alphaTilde);
      if (!Number.isFinite(dl)) return;
      c.lambda += dl;

      if (B >= 0) { angP[B] += wB * dl; angP[A] -= wA * dl; }
      else        { angP[A] += wA * dl; }
    }

    // Gelenkgrenze als EINSEITIGE harte Nebenbedingung auf die Abweichung.
    const rel2 = B >= 0 ? angleDelta(angP[B], angP[A]) : angP[A];
    const dev2 = angleDelta(rel2, c.rest);
    let over = 0;
    if (dev2 > c.maxAngle) over = dev2 - c.maxAngle;
    else if (dev2 < c.minAngle) over = dev2 - c.minAngle;
    if (Math.abs(over) >= SLACK_ANG && wSum > 0) {
      const dl2 = -over / wSum;
      if (B >= 0) { angP[B] += wB * dl2; angP[A] -= wA * dl2; }
      else        { angP[A] += wA * dl2; }
    }
  }

  /**
   * Kollision gegen das Signed-Distance-Field der Karte.
   *
   * Analytisch über die Rechteckliste statt gebackener Textur: exakte Distanzen,
   * kein Baking-Schritt, und portiert als reine Arithmetik 1:1 nach WGSL.
   */
  _solveCollisions() {
    if (!this.sdf) return;
    const { ppx, ppy, angP, im } = this;
    for (let k = 0; k < this.collPts.length; k++) {
      const cp = this.collPts[k];
      const B = cp.body;
      const ca = Math.cos(angP[B]), sa = Math.sin(angP[B]);
      const wx = ppx[B] + cp.lx * ca - cp.ly * sa;
      const wy = ppy[B] + cp.lx * sa + cp.ly * ca;
      const hit = this.sdf(wx, wy);
      if (!hit) continue;
      const pen = cp.r - hit.d;
      // Totzone: auch Penetration unter dem f32-Boden überspringen, sonst
      // zittert ein aufliegender Körper dauerhaft (Kontakt-Rauschen).
      if (pen <= SLACK_POS) continue;
      const w = im[B];
      if (w === 0) continue;
      // Vollständige Positionsprojektion entlang der Normalen (statisch, keine Karte)
      ppx[B] += hit.nx * pen;
      ppy[B] += hit.ny * pen;
      this.stats.collisions++;
    }
  }

  _measure() {
    let mv = 0, mw = 0, nan = 0;
    for (let i = 0; i < this.n; i++) {
      const s = Math.hypot(this.vx[i], this.vy[i]);
      if (s > mv) mv = s;
      const a = Math.abs(this.av[i]);
      if (a > mw) mw = a;
      if (!Number.isFinite(this.px[i]) || !Number.isFinite(this.py[i]) ||
          !Number.isFinite(this.ang[i])) nan++;
    }
    this.stats.maxSpeed = mv;
    this.stats.maxAngSpeed = mw;
    this.stats.nan = nan;
  }

  /** Ein voller Sim-Schritt (= `substeps` Substeps). */
  step() {
    for (let s = 0; s < this.substeps; s++) this.substep();
    this._postGripBreak();
  }

  /**
   * Grip-Bruch nach echter Kraft, nicht nach Dehnung.
   * λ ist der akkumulierte Lagrange-Multiplikator, F = |λ|/h.
   */
  _postGripBreak() {
    const inv = 1 / this.h;
    for (const c of this.constraints) {
      if (c.type !== C_GRIP || c.broken) continue;
      const f = Math.hypot(c.lambdaX, c.lambdaY) * inv;
      c.force = f;
      if (f > c.breakForce) { c.broken = true; this.stats.broken++; }
    }
  }
}
