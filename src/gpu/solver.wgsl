// XPBD-Solver als Compute-Shader.
//
// 1:1-Port von src/physics/xpbd.js — dieselbe Reihenfolge, dieselben Formeln,
// dieselben Konstanten. Die CPU-Fassung ist das ORAKEL: gpu.html vergleicht
// beide auf identischen Eingaben.
//
// ── Warum nicht bit-identisch ───────────────────────────────────────
// Die CPU löst Constraints sequentiell (Gauss-Seidel), die GPU parallel ohne
// Atomics (Jacobi). Bei mehreren Constraints am selben Körper ist das Ergebnis
// reihenfolgeabhängig und weicht deshalb numerisch ab. Bit-Identität zu fordern
// wäre ein falsches Kriterium — validiert wird stattdessen:
//   EXAKT   passives Gleichgewicht (alles null), Spawn-Verletzung, NaN, Determinismus
//   TOLERANZ grip-pull-Weg, Energiebegrenzung, Spitzenwerte
//
// ── Lanes ───────────────────────────────────────────────────────────
// Jedes Tier ist eine unabhängige Lane. Körper und Constraints tragen keinen
// Lane-Index, weil Tiere nicht miteinander interagieren — die Welt ist einfach
// die Konkatenation aller Tiere. Die Karte (rects) ist für alle Lanes gleich.

struct Params {
  h: f32,
  iterations: i32,
  linDamp: f32,
  angDamp: f32,
  maxSpeed: f32,
  maxAngSpeed: f32,
  nBodies: u32,
  nConstraints: u32,
  nCollPts: u32,
  nRects: u32,
  invH: f32,
  invH2: f32,
  pad0: f32,
  pad1: f32,
};

// 12 f32 = 48 Bytes, Stride 48
struct Body {
  px: f32, py: f32, ppx: f32, ppy: f32,
  vx: f32, vy: f32, ang: f32, angP: f32,
  av: f32, invMass: f32, invInertia: f32, pad: f32,
};

// 20 × 4 = 80 Bytes
struct Constraint {
  aIdx: u32, bIdx: u32, cType: u32, owner: u32,
  lax: f32, lay: f32, lbx: f32, lby: f32,
  compliance: f32, rest: f32, tgt: f32, minAng: f32,
  maxAng: f32, lamX: f32, lamY: f32, lam: f32,
  breakForce: f32, broken: u32, pad0: u32, pad1: u32,
};

struct CollPt { body: u32, lx: f32, ly: f32, r: f32 };
struct Rect  { x: f32, y: f32, w: f32, h: f32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(2) var<storage, read_write> cons: array<Constraint>;
@group(0) @binding(3) var<storage, read_write> collPts: array<CollPt>;
@group(0) @binding(4) var<storage, read> rects: array<Rect>;
@group(0) @binding(5) var<storage, read_write> forces: array<vec4f>;   // fx, fy, torque, adhesion
// [0..15]  Start-Offset jeder Farbgruppe im Constraint-Array
// [16..31] Anzahl Constraints pro Farbe
// [32]     Workgroups pro Farbe = ceil(maxCnt/64)
// EIN Dispatch deckt alle 16 Farben ab: Thread-Id -> Farbe -> Offset.
// Innerhalb einer Farbe teilt kein Körper zwei Constraints, also kein
// Schreib-Race ohne Atomics. Kein pro-Dispatch-Uniform-Update nötig.
@group(0) @binding(6) var<storage, read> colorInfo: array<u32>;
// Aktuelle Farbe dieses Dispatches. Es gibt 16 Bind-Groups (eine pro Farbe),
// die sich ALLE Buffer teilen und sich nur in diesem 4-Byte-Uniform
// unterscheiden. Dadurch kostet der sequentielle Farb-Dispatch nichts pro
// Schritt: keine writeBuffer-Aufrufe, keine Synchronisation.
struct ColorU { c: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(7) var<uniform> UC: ColorU;

const TAU: f32 = 6.28318530718;

// ── Korrektur-Totzone — PARITÄT zu SLACK_POS/SLACK_ANG in xpbd.js ──────────
// (statisch geprüft von tools/check-wgsl.js). Grund: f32-Boden. Bei ~1440 px
// Koordinaten ist ulp(position) ≈ 1.7e-4 px, das Spawn-Gleichgewicht ist in
// f32 kein exakter Fixpunkt (G2 gemessen, Build d487532: 2.0e-4 px Verletzung
// nach Upload). Ohne Totzone integrierte sich das Rundungszittern über
// v = Δp/h zu Rauschgeschwindigkeit 0.1606 px/s und linearem Drift: G1
// Wachstum 1.2789 px nach 2400 Schritten, CPU-Oracle 3.4e-9. Mit Totzone ist
// jeder Ruhezustand bitexakt stationär: pPred = p + h·0 = p, alle |C| < slack
// → keine Projektion → v bleibt 0. Echte Lasten liegen 1–2 Zehnerpotenzen
// über dem Band (Muskelzug: ~0.01–1 px Verletzung pro Substep).
const SLACK_POS: f32 = 0.002;    // px  — identisch zu SLACK_POS = 2e-3 (JS)
const SLACK_ANG: f32 = 0.0002;   // rad — identisch zu SLACK_ANG = 2e-4 (JS)

fn wrapAngle(a: f32) -> f32 {
  var d = a % TAU;
  if (d > 3.14159265359) { return d - TAU; }
  if (d < -3.14159265359) { return d + TAU; }
  return d;
}

fn angleDelta(a: f32, b: f32) -> f32 {
  var d = (a - b) % TAU;
  if (d > 3.14159265359) { d = d - TAU; }
  if (d < -3.14159265359) { d = d + TAU; }
  return d;
}

// ── Signierte Distanz zur Karte (min über alle Rechtecke) ──────────
// Identisch zu makeSdf() in src/world/map.js, nur ohne Objekt-Allokation.
struct SdfHit { d: f32, nx: f32, ny: f32, valid: bool };

fn sdf(x: f32, y: f32) -> SdfHit {
  var best: f32 = 1e30;
  var bnx: f32 = 0.0;
  var bny: f32 = 0.0;
  for (var i: u32 = 0u; i < P.nRects; i = i + 1u) {
    let r = rects[i];
    let cx = clamp(x, r.x, r.x + r.w);
    let cy = clamp(y, r.y, r.y + r.h);
    let dx = x - cx;
    let dy = y - cy;
    let d2 = dx * dx + dy * dy;
    if (d2 > 1e-12) {
      if (d2 < best * best) {
        best = sqrt(d2);
        bnx = dx / best;
        bny = dy / best;
      }
    } else {
      let dl = x - r.x;
      let dr = r.x + r.w - x;
      let dt = y - r.y;
      let db = r.y + r.h - y;
      var m = dl;
      var nx: f32 = -1.0; var ny: f32 = 0.0;
      if (dr < m) { m = dr; nx = 1.0; ny = 0.0; }
      if (dt < m) { m = dt; nx = 0.0; ny = -1.0; }
      if (db < m) { m = db; nx = 0.0; ny = 1.0; }
      let d = -m;
      if (d < best) { best = d; bnx = nx; bny = ny; }
    }
  }
  if (best > 1e29) { return SdfHit(0.0, 0.0, 0.0, false); }
  return SdfHit(best, bnx, bny, true);
}

// ══ 0. Lambda-Reset pro Substep ════════════════════════════════════
// XPBD akkumuliert den Lagrange-Multiplikator ÜBER DIE ITERATIONEN EINES
// SUBSTEPS; der Compliance-Term α̃·λ darf nicht über Substep-Grenzen wachsen.
// Die CPU-Fassung resettet in _solvePin/_solveAngle beim ersten Iterations-
// durchlauf (firstIteration-Flag). Ohne dieses Kernel wuchs λ auf der GPU
// unbegrenzt: gemessen passiv 13.85 px Drift und max|v| exakt am 360-px/s-Limit,
// während die CPU bei 1.36e-10 stand.
@compute @workgroup_size(64)
fn resetLambda(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= P.nConstraints) { return; }
  var c = cons[k];
  c.lamX = 0.0; c.lamY = 0.0; c.lam = 0.0;
  cons[k] = c;
}

// ══ 1. Integration: Kräfte, Dämpfung, Limits, Prädiktion ══════════
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.nBodies) { return; }
  var b = bodies[i];
  let f = forces[i];

  if (b.invMass == 0.0 && b.invInertia == 0.0) {
    b.ppx = b.px; b.ppy = b.py; b.angP = b.ang;
    b.vx = 0.0; b.vy = 0.0; b.av = 0.0;
    bodies[i] = b;
    return;
  }

  let linDamp = exp(-P.linDamp * P.h);
  let angDamp = exp(-P.angDamp * P.h);

  var vx = (b.vx + P.h * b.invMass * f.x) * linDamp;
  var vy = (b.vy + P.h * b.invMass * f.y) * linDamp;
  var av = (b.av + P.h * b.invInertia * f.z) * angDamp;

  let sp = length(vec2f(vx, vy));
  if (sp > P.maxSpeed) {
    let k = P.maxSpeed / sp;
    vx = vx * k; vy = vy * k;
  }
  av = clamp(av, -P.maxAngSpeed, P.maxAngSpeed);

  // NaN-Guard: einmal pro Körper, billig, und ein NaN würde sich sonst
  // über die Constraints auf die ganze Lane ausbreiten.
  if (!(vx == vx) || !(vy == vy)) { vx = 0.0; vy = 0.0; }
  if (!(av == av)) { av = 0.0; }

  b.vx = vx; b.vy = vy; b.av = av;
  b.ppx = b.px + P.h * vx;
  b.ppy = b.py + P.h * vy;
  b.angP = b.ang + P.h * av;
  bodies[i] = b;
  forces[i] = vec4f(0.0, 0.0, 0.0, f.w);   // Kräfte konsumieren, Adhäsion behalten
}

// ══ 2a. Constraints (Pin / Winkel / Grip) ═════════════════════════
@compute @workgroup_size(64)
fn solveConstraints(@builtin(global_invocation_id) gid: vec3u) {
  // EIN Dispatch pro Farbe, sequentiell. Farben MÜSSEN sequentiell laufen:
  // innerhalb einer Farbe teilt kein Körper zwei Constraints (race-frei),
  // aber Constraints VERSCHIEDENER Farben teilen Körper sehr wohl. Ein
  // Sammel-Dispatch über alle Farben (früherer Stand) lief deshalb trotz
  // Färbung parallel auf gemeinsame Körper — gemessen: G5 nondeterministisch,
  // G3-Grip bricht grundlos, G1 explodiert nach ~240 Schritten ans Limit.
  let color = UC.c;
  if (gid.x >= colorInfo[16 + color]) { return; }
  let k = colorInfo[color] + gid.x;
  var c = cons[k];
  if (c.broken != 0u) { return; }
  if (c.cType == 1u) { solveAngle(k); return; }
  solvePin(k);
}

fn solvePin(k: u32) {
  var c = cons[k];
  let A = c.aIdx;
  var a = bodies[A];
  let ca = cos(a.angP); let sa = sin(a.angP);
  let rax = c.lax * ca - c.lay * sa;
  let ray = c.lax * sa + c.lay * ca;
  let gax = a.ppx + rax;
  let gay = a.ppy + ray;

  var gbx = c.lbx; var gby = c.lby;
  var rbx = 0.0; var rby = 0.0;
  var wB = 0.0; var wIB = 0.0;
  var bIdx = 0xffffffffu;
  if (c.bIdx != 0xffffffffu) {
    bIdx = c.bIdx;
    let bb = bodies[bIdx];
    let cb = cos(bb.angP); let sb = sin(bb.angP);
    rbx = c.lbx * cb - c.lby * sb;
    rby = c.lbx * sb + c.lby * cb;
    gbx = bb.ppx + rbx; gby = bb.ppy + rby;
    wB = bb.invMass; wIB = bb.invInertia;
  }

  let Cx = gax - gbx;
  let Cy = gay - gby;
  if (!(Cx == Cx) || !(Cy == Cy)) { return; }
  // Totzone (SLACK_POS) — Parität zu _solvePin in xpbd.js.
  if (Cx * Cx + Cy * Cy < SLACK_POS * SLACK_POS) { return; }

  let pax = -ray; let pay = rax;      // perp(r)
  let pbx = -rby; let pby = rbx;
  let wA = a.invMass; let wIA = a.invInertia;
  let wSum = wA + wB;

  let k00 = wSum + wIA * pax * pax + wIB * pbx * pbx;
  let k01 = wIA * pax * pay + wIB * pbx * pby;
  let k11 = wSum + wIA * pay * pay + wIB * pby * pby;
  let det = k00 * k11 - k01 * k01;
  if (abs(det) < 1e-12) { return; }

  let alphaTilde = c.compliance * P.invH2;
  let bx = Cx + alphaTilde * c.lamX;
  let by = Cy + alphaTilde * c.lamY;
  let i00 = k11 / det; let i11 = k00 / det; let i01 = -k01 / det;
  var dlx = -(i00 * bx + i01 * by);
  var dly = -(i01 * bx + i11 * by);
  if (!(dlx == dlx) || !(dly == dly)) { return; }

  c.lamX = c.lamX + dlx;
  c.lamY = c.lamY + dly;
  cons[k] = c;

  a.ppx = a.ppx + wA * dlx;
  a.ppy = a.ppy + wA * dly;
  a.angP = a.angP + wIA * (pax * dlx + pay * dly);
  bodies[A] = a;

  if (bIdx != 0xffffffffu) {
    var bb = bodies[bIdx];
    bb.ppx = bb.ppx - wB * dlx;
    bb.ppy = bb.ppy - wB * dly;
    bb.angP = bb.angP - wIB * (pbx * dlx + pby * dly);
    bodies[bIdx] = bb;
  }
}

fn solveAngle(k: u32) {
  var c = cons[k];
  let A = c.aIdx;
  let hasB = c.bIdx != 0xffffffffu;
  var a = bodies[A];
  let wA = a.invInertia;
  var wB = 0.0;
  var rel = a.angP;
  if (hasB) {
    let bb = bodies[c.bIdx];
    wB = bb.invInertia;
    rel = angleDelta(bb.angP, a.angP);
  }
  let wSum = wA + wB;
  if (wSum <= 0.0) { return; }

  // Abweichungs-Koordinaten — identisch zur CPU-Fassung. Absolutwinkel würden
  // gegen gewickelte Ist-Werte verglichen und Phantom-Überschreitungen erzeugen.
  let dev = angleDelta(rel, c.rest);
  let C = angleDelta(dev, c.tgt);
  if (!(C == C)) { return; }

  // Totzone (SLACK_ANG) — Parität zu _solveAngle in xpbd.js.
  if (abs(C) >= SLACK_ANG) {
    let alphaTilde = c.compliance * P.invH2;
    let dl = -(C + alphaTilde * c.lam) / (wSum + alphaTilde);
    if (!(dl == dl)) { return; }
    c.lam = c.lam + dl;
    if (hasB) {
      var bb = bodies[c.bIdx];
      bb.angP = bb.angP + wB * dl;
      bodies[c.bIdx] = bb;
      a.angP = a.angP - wA * dl;
    } else {
      a.angP = a.angP + wA * dl;
    }
  }

  // Gelenkgrenze als einseitige harte Nebenbedingung
  var rel2 = a.angP;
  if (hasB) { rel2 = angleDelta(bodies[c.bIdx].angP, a.angP); }
  let dev2 = angleDelta(rel2, c.rest);
  var over = 0.0;
  if (dev2 > c.maxAng) { over = dev2 - c.maxAng; }
  if (dev2 < c.minAng) { over = dev2 - c.minAng; }
  if (abs(over) >= SLACK_ANG) {
    let dl2 = -over / wSum;
    if (hasB) {
      var bb = bodies[c.bIdx];
      bb.angP = bb.angP + wB * dl2;
      bodies[c.bIdx] = bb;
      a.angP = a.angP - wA * dl2;
    } else {
      a.angP = a.angP + wA * dl2;
    }
  }
  cons[k] = c;
  bodies[A] = a;
}

// ══ 2b. Kollision gegen das SDF ═══════════════════════════════════
@compute @workgroup_size(64)
fn solveCollisions(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= P.nCollPts) { return; }
  let cp = collPts[k];
  var b = bodies[cp.body];
  if (b.invMass == 0.0) { return; }
  let ca = cos(b.angP); let sa = sin(b.angP);
  let wx = b.ppx + cp.lx * ca - cp.ly * sa;
  let wy = b.ppy + cp.lx * sa + cp.ly * ca;
  let hit = sdf(wx, wy);
  if (!hit.valid) { return; }
  let pen = cp.r - hit.d;
  // Totzone — Parität zu _solveCollisions in xpbd.js: Kontakt-Rauschen unter
  // dem f32-Boden überspringen, sonst zittert ein aufliegender Körper.
  if (pen <= SLACK_POS) { return; }
  b.ppx = b.ppx + hit.nx * pen;
  b.ppy = b.ppy + hit.ny * pen;
  bodies[cp.body] = b;
}

// ══ 3. Abschluss: Geschwindigkeiten aus der Positionsänderung ══════
@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.nBodies) { return; }
  var b = bodies[i];
  var vx = (b.ppx - b.px) * P.invH;
  var vy = (b.ppy - b.py) * P.invH;
  var av = angleDelta(b.angP, b.ang) * P.invH;

  // Nachklemmung: die Projektion kann über die Vorhersage-Limits hinausgehen.
  let sp = length(vec2f(vx, vy));
  if (sp > P.maxSpeed) { let k = P.maxSpeed / sp; vx = vx * k; vy = vy * k; }
  av = clamp(av, -P.maxAngSpeed, P.maxAngSpeed);

  b.vx = vx; b.vy = vy; b.av = av;
  b.px = b.ppx; b.py = b.ppy;
  // WICHTIG: Winkel aus dem Prädiktor in den Zustand übernehmen. Ohne das
  // bleibt `ang` auf dem Vor-Substep-Wert, jeder folgende Substep rechnet av
  // aus einem veralteten Winkel, und ω hängt dauerhaft am Limit.
  b.ang = b.angP;
  if (b.ang > 50.0 || b.ang < -50.0) {
    let w = wrapAngle(b.ang);
    b.ang = w; b.angP = w;
  }
  bodies[i] = b;
}

// ══ 4. Grip-Bruch über echte Kraft ════════════════════════════════
@compute @workgroup_size(64)
fn gripBreak(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= P.nConstraints) { return; }
  var c = cons[k];
  if (c.cType != 2u || c.broken != 0u) { return; }
  let f = length(vec2f(c.lamX, c.lamY)) * P.invH;
  if (f > c.breakForce) { c.broken = 1u; c.lamX = 0.0; c.lamY = 0.0; }
  cons[k] = c;
}
