/**
 * f32-Solver-Emulation — die headlose Entsprechung des GPU-Solvers.
 *
 * Zweck: Die G7-Frage „Numerik oder Verdrahtungs-Bug?" ohne WebGPU-Device
 * beantworten. f32ify(world, map) ersetzt step/substep/_postGripBreak einer
 * XpbdWorld durch eine Fassung, die solver.wgsl Operation für Operation
 * spiegelt: dieselbe Reihenfolge (resetLambda → integrate → iterations ×
 * (constraints in Listen-==Farbordnung → collisions) → finalize → gripBreak),
 * dieselben Formeln, jede Arithmetik mit Math.fround auf f32 gerundet,
 * statische Felder als f32-Snapshot (die GPU hält sie in f32-Puffern).
 *
 * KEIN Bit-Anspruch an die echte GPU: WGSL erlaubt fma-Kontraktion und
 * implementation-defined Genauigkeit für cos/sin/exp (±2–3 ulp). Diese
 * Fassung emittiert einen REPRÄSENTANTEN der Klasse „gültige f32-
 * Implementierung desselben Algorithmus" — genau die Klasse, deren
 * Divergenz-Verhalten G7 messen soll.
 *
 * Kontrolliert durch tools/f32parity-probe.mjs:
 *   A = f64-Oracle (unverändert), B = f32ify — gleiche Ordnung, gleiche
 *   Steuerlogik, nur die Solver-Präzision differs.
 */

const f = Math.fround;

// ── Konstanten — identisch zu solver.wgsl ──────────────────────────
const SLACK_POS = f(0.002);
const SLACK_ANG = f(0.0002);
const TAU = f(6.28318530718);
const PI = f(3.14159265359);
const EPS_DET = f(1e-12);
const EPS_SDF = f(1e-12);
const SDF_INF = f(1e30);
const SDF_VALID = f(1e29);

/** WGSL: d = (a-b) % TAU mit FRem-Semantik (x - y·trunc(x/y)), dann falten. */
function angleDeltaF(a, b) {
  const x = f(a - b);
  const q = Math.trunc(x / TAU);
  let d = f(x - f(TAU * q));
  if (d > PI) return f(d - TAU);
  if (d < -PI) return f(d + TAU);
  return d;
}

function wrapAngleF(a) {
  const q = Math.trunc(a / TAU);
  let d = f(a - f(TAU * q));
  if (d > PI) return f(d - TAU);
  if (d < -PI) return f(d + TAU);
  return d;
}

/** WGSL length(vec2f) — sqrt(dot) mit f32-Zwischenwerten. */
function lengthF(x, y) {
  return f(Math.sqrt(f(f(x * x) + f(y * y))));
}

/**
 * Welt auf f32-Solver-Emulation umschalten. Gibt einen Snapshot-Helper
 * zurück; roundState() einmal nach jedem Respawn/Upload aufrufen (das
 * entspricht upload() → f32-Puffer).
 */
export function f32ify(w, map) {
  const n = w.n;
  const nC = w.constraints.length;

  // ── statische f32-Snapshots (GPU: in Puffern, einmal geschrieben) ──
  const h = f(w.h);
  const invH = f(1 / w.h);            // writeStatic: F[10] = 1/cpu.h
  const invH2 = f(1 / (w.h * w.h));   // F[11]
  const linDamp = f(Math.exp(f(-f(w.linDrag) * h)));   // integrate: exp(-P.linDamp*P.h)
  const angDamp = f(Math.exp(f(-f(w.angDrag) * h)));
  const maxSpeed = f(w.maxSpeed);
  const maxAngSpeed = f(w.maxAngSpeed);

  const imF = new Float32Array(n), iiF = new Float32Array(n);
  for (let i = 0; i < n; i++) { imF[i] = w.im[i]; iiF[i] = w.ii[i]; }

  const laxF = new Float32Array(nC), layF = new Float32Array(nC);
  const restF = new Float32Array(nC), minF = new Float32Array(nC), maxF = new Float32Array(nC);
  const typeArr = new Uint8Array(nC), aArr = new Int32Array(nC), bArr = new Int32Array(nC);
  for (let k = 0; k < nC; k++) {
    const c = w.constraints[k];
    laxF[k] = c.lax; layF[k] = c.lay;
    restF[k] = c.rest; minF[k] = c.minAngle; maxF[k] = c.maxAngle;
    typeArr[k] = c.type; aArr[k] = c.a; bArr[k] = c.b;
  }

  const rectsF = [];
  for (const r of map.rects) {
    rectsF.push({ x: f(r.x), y: f(r.y), w: f(r.w), h: f(r.h),
      xw: f(f(r.x) + f(r.w)), yh: f(f(r.y) + f(r.h)) });
  }

  const nP = w.collPts.length;
  const cpBody = new Int32Array(nP);
  const cpLx = new Float32Array(nP), cpLy = new Float32Array(nP), cpR = new Float32Array(nP);
  for (let i = 0; i < nP; i++) {
    const p = w.collPts[i];
    cpBody[i] = p.body; cpLx[i] = p.lx; cpLy[i] = p.ly; cpR[i] = p.r;
  }

  // ── dynamische f32-Snapshots (GPU: packControls pro Schritt) ───────
  const compF = new Float32Array(nC), tgtF = new Float32Array(nC);
  const lbxF = new Float32Array(nC), lbyF = new Float32Array(nC);
  const brkF = new Float32Array(nC);
  function snapshotControls() {
    for (let k = 0; k < nC; k++) {
      const c = w.constraints[k];
      compF[k] = c.compliance; tgtF[k] = c.target;
      lbxF[k] = c.lbx; lbyF[k] = c.lby;
      brkF[k] = Number.isFinite(c.breakForce) ? Math.min(c.breakForce, 1e30) : 1e30;
    }
  }

  // Zustand nach Respawn auf f32 runden (= upload() in die GPU-Puffer)
  function roundState() {
    for (let i = 0; i < n; i++) {
      w.px[i] = f(w.px[i]); w.py[i] = f(w.py[i]);
      w.ppx[i] = f(w.ppx[i]); w.ppy[i] = f(w.ppy[i]);
      w.vx[i] = f(w.vx[i]); w.vy[i] = f(w.vy[i]);
      w.ang[i] = f(w.ang[i]); w.angP[i] = f(w.angP[i]); w.av[i] = f(w.av[i]);
    }
  }

  // ── SDF — makeSdf() als reine f32-Arithmetik (wie solver.wgsl) ─────
  let sdfHit = null;
  function sdfF(x, y) {
    let best = SDF_INF, bnx = 0, bny = 0;
    for (let i = 0; i < rectsF.length; i++) {
      const r = rectsF[i];
      let cx = x < r.x ? r.x : (x > r.xw ? r.xw : x);
      let cy = y < r.y ? r.y : (y > r.yh ? r.yh : y);
      const dx = f(x - cx), dy = f(y - cy);
      const d2 = f(f(dx * dx) + f(dy * dy));
      if (d2 > EPS_SDF) {
        if (d2 < f(best * best)) {
          best = f(Math.sqrt(d2));
          bnx = f(dx / best); bny = f(dy / best);
        }
      } else {
        const dl = f(x - r.x), dr = f(r.xw - x), dt = f(y - r.y), db = f(r.yh - y);
        let m = dl, nx = -1, ny = 0;
        if (dr < m) { m = dr; nx = 1; ny = 0; }
        if (dt < m) { m = dt; nx = 0; ny = -1; }
        if (db < m) { m = db; nx = 0; ny = 1; }
        const d = f(-m);
        if (d < best) { best = d; bnx = nx; bny = ny; }
      }
    }
    if (best > SDF_VALID) return null;
    if (!sdfHit) sdfHit = { d: 0, nx: 0, ny: 0 };
    sdfHit.d = best; sdfHit.nx = bnx; sdfHit.ny = bny;
    return sdfHit;
  }

  // ── Kernel: resetLambda ────────────────────────────────────────────
  function resetLambdaF() {
    for (let k = 0; k < nC; k++) {
      const c = w.constraints[k];
      c.lambdaX = 0; c.lambdaY = 0; c.lambda = 0;
    }
  }

  // ── Kernel: integrate ─────────────────────────────────────────────
  function integrateF() {
    for (let i = 0; i < n; i++) {
      if (imF[i] === 0 && iiF[i] === 0) {
        w.ppx[i] = w.px[i]; w.ppy[i] = w.py[i]; w.angP[i] = w.ang[i];
        w.vx[i] = 0; w.vy[i] = 0; w.av[i] = 0;
        continue;
      }
      // externe Kräfte sind in dieser Welt immer 0 — Formel trotzdem 1:1
      let vx = f(f(w.vx[i] + f(f(h * imF[i]) * f(w.fx[i]))) * linDamp);
      let vy = f(f(w.vy[i] + f(f(h * imF[i]) * f(w.fy[i]))) * linDamp);
      let av = f(f(w.av[i] + f(f(h * iiF[i]) * f(w.tq[i]))) * angDamp);
      const sp = lengthF(vx, vy);
      if (sp > maxSpeed) { const k = f(maxSpeed / sp); vx = f(vx * k); vy = f(vy * k); }
      if (av > maxAngSpeed) av = maxAngSpeed;
      else if (av < -maxAngSpeed) av = -maxAngSpeed;
      if (!(vx === vx) || !(vy === vy)) { vx = 0; vy = 0; }
      if (!(av === av)) av = 0;
      w.vx[i] = vx; w.vy[i] = vy; w.av[i] = av;
      w.ppx[i] = f(w.px[i] + f(h * vx));
      w.ppy[i] = f(w.py[i] + f(h * vy));
      w.angP[i] = f(w.ang[i] + f(h * av));
      w.fx[i] = 0; w.fy[i] = 0; w.tq[i] = 0;
    }
  }

  // ── Kernel: solveConstraints (Listen- == Farbordnung) ─────────────
  function solvePinF(k) {
    const c = w.constraints[k];
    const A = aArr[k];
    const ca = f(Math.cos(w.angP[A])), sa = f(Math.sin(w.angP[A]));
    const rax = f(f(laxF[k] * ca) - f(layF[k] * sa));
    const ray = f(f(laxF[k] * sa) + f(layF[k] * ca));
    const gax = f(w.ppx[A] + rax), gay = f(w.ppy[A] + ray);

    let gbx = lbxF[k], gby = lbyF[k];
    let rbx = 0, rby = 0, wB = 0, wIB = 0;
    const B = bArr[k];
    if (B >= 0) {
      const cb = f(Math.cos(w.angP[B])), sb = f(Math.sin(w.angP[B]));
      rbx = f(f(lbxF[k] * cb) - f(lbyF[k] * sb));
      rby = f(f(lbxF[k] * sb) + f(lbyF[k] * cb));
      gbx = f(w.ppx[B] + rbx); gby = f(w.ppy[B] + rby);
      wB = imF[B]; wIB = iiF[B];
    }

    const Cx = f(gax - gbx), Cy = f(gay - gby);
    if (!(Cx === Cx) || !(Cy === Cy)) return;
    if (f(f(Cx * Cx) + f(Cy * Cy)) < f(SLACK_POS * SLACK_POS)) return;

    const pax = f(-ray), pay = rax;
    const pbx = f(-rby), pby = rbx;
    const wA = imF[A], wIA = iiF[A];
    const wSum = f(wA + wB);

    const k00 = f(f(wSum + f(f(wIA * pax) * pax)) + f(f(wIB * pbx) * pbx));
    const k01 = f(f(f(wIA * pax) * pay) + f(f(wIB * pbx) * pby));
    const k11 = f(f(wSum + f(f(wIA * pay) * pay)) + f(f(wIB * pby) * pby));
    const det = f(f(k00 * k11) - f(k01 * k01));
    if (Math.abs(det) < EPS_DET) return;

    const alphaTilde = f(compF[k] * invH2);
    const bx = f(Cx + f(alphaTilde * c.lambdaX));
    const by = f(Cy + f(alphaTilde * c.lambdaY));
    const i00 = f(k11 / det), i11 = f(k00 / det), i01 = f(-f(k01 / det));
    const dlx = f(-f(f(i00 * bx) + f(i01 * by)));
    const dly = f(-f(f(i01 * bx) + f(i11 * by)));
    if (!(dlx === dlx) || !(dly === dly)) return;

    c.lambdaX = f(c.lambdaX + dlx);
    c.lambdaY = f(c.lambdaY + dly);

    w.ppx[A] = f(w.ppx[A] + f(wA * dlx));
    w.ppy[A] = f(w.ppy[A] + f(wA * dly));
    w.angP[A] = f(w.angP[A] + f(wIA * f(f(pax * dlx) + f(pay * dly))));
    if (B >= 0) {
      w.ppx[B] = f(w.ppx[B] - f(wB * dlx));
      w.ppy[B] = f(w.ppy[B] - f(wB * dly));
      w.angP[B] = f(w.angP[B] - f(wIB * f(f(pbx * dlx) + f(pby * dly))));
    }
  }

  function solveAngleF(k) {
    const c = w.constraints[k];
    const A = aArr[k], B = bArr[k];
    const hasB = B >= 0;
    const wA = iiF[A];
    let wB = 0;
    let rel = w.angP[A];
    if (hasB) { wB = iiF[B]; rel = angleDeltaF(w.angP[B], w.angP[A]); }
    const wSum = f(wA + wB);
    if (wSum <= 0) return;

    const dev = angleDeltaF(rel, restF[k]);
    const C = angleDeltaF(dev, tgtF[k]);
    if (!(C === C)) return;

    if (Math.abs(C) >= SLACK_ANG) {
      const alphaTilde = f(compF[k] * invH2);
      const dl = f(-f(f(C + f(alphaTilde * c.lambda)) / f(wSum + alphaTilde)));
      if (!(dl === dl)) return;
      c.lambda = f(c.lambda + dl);
      if (hasB) {
        w.angP[B] = f(w.angP[B] + f(wB * dl));
        w.angP[A] = f(w.angP[A] - f(wA * dl));
      } else {
        w.angP[A] = f(w.angP[A] + f(wA * dl));
      }
    }

    let rel2 = w.angP[A];
    if (hasB) rel2 = angleDeltaF(w.angP[B], w.angP[A]);
    const dev2 = angleDeltaF(rel2, restF[k]);
    let over = 0;
    if (dev2 > maxF[k]) over = f(dev2 - maxF[k]);
    else if (dev2 < minF[k]) over = f(dev2 - minF[k]);
    if (Math.abs(over) >= SLACK_ANG) {
      const dl2 = f(-f(over / wSum));
      if (hasB) {
        w.angP[B] = f(w.angP[B] + f(wB * dl2));
        w.angP[A] = f(w.angP[A] - f(wA * dl2));
      } else {
        w.angP[A] = f(w.angP[A] + f(wA * dl2));
      }
    }
  }

  function solveConstraintsF() {
    for (let k = 0; k < nC; k++) {
      const c = w.constraints[k];
      if (c.broken) continue;
      if (typeArr[k] === 1) solveAngleF(k);
      else solvePinF(k);
    }
  }

  // ── Kernel: solveCollisions ───────────────────────────────────────
  function solveCollisionsF() {
    for (let i = 0; i < nP; i++) {
      const b = cpBody[i];
      if (imF[b] === 0) continue;
      const ca = f(Math.cos(w.angP[b])), sa = f(Math.sin(w.angP[b]));
      const wx = f(w.ppx[b] + f(f(cpLx[i] * ca) - f(cpLy[i] * sa)));
      const wy = f(w.ppy[b] + f(f(cpLx[i] * sa) + f(cpLy[i] * ca)));
      const hit = sdfF(wx, wy);
      if (!hit) continue;
      const pen = f(cpR[i] - hit.d);
      if (pen <= SLACK_POS) continue;
      w.ppx[b] = f(w.ppx[b] + f(hit.nx * pen));
      w.ppy[b] = f(w.ppy[b] + f(hit.ny * pen));
      w.stats.collisions++;
    }
  }

  // ── Kernel: finalize ──────────────────────────────────────────────
  function finalizeF() {
    for (let i = 0; i < n; i++) {
      if (imF[i] === 0 && iiF[i] === 0) {
        // statische Körper: ppx==px bereits in integrate gesetzt; der
        // GPU-finalize läuft über ALLE Körper — vx=(ppx-px)*invH=0 etc.
        w.vx[i] = 0; w.vy[i] = 0; w.av[i] = 0;
        continue;
      }
      let vx = f(f(w.ppx[i] - w.px[i]) * invH);
      let vy = f(f(w.ppy[i] - w.py[i]) * invH);
      let av = f(angleDeltaF(w.angP[i], w.ang[i]) * invH);
      const sp = lengthF(vx, vy);
      if (sp > maxSpeed) { const k = f(maxSpeed / sp); vx = f(vx * k); vy = f(vy * k); }
      if (av > maxAngSpeed) av = maxAngSpeed;
      else if (av < -maxAngSpeed) av = -maxAngSpeed;
      w.vx[i] = vx; w.vy[i] = vy; w.av[i] = av;
      w.px[i] = w.ppx[i]; w.py[i] = w.ppy[i];
      w.ang[i] = w.angP[i];
      if (w.ang[i] > 50 || w.ang[i] < -50) {
        const wr = wrapAngleF(w.ang[i]);
        w.ang[i] = wr; w.angP[i] = wr;
      }
    }
  }

  // ── Kernel: gripBreak (pro Schritt, nach den Substeps) ────────────
  function gripBreakF() {
    const invH64 = 1 / w.h;   // mirrorInto rechnet force in f64 auf f32-λ
    for (let k = 0; k < nC; k++) {
      const c = w.constraints[k];
      if (typeArr[k] !== 2 || c.broken) continue;
      const fm = f(lengthF(c.lambdaX, c.lambdaY) * invH);
      if (fm > brkF[k]) {
        c.broken = true; c.lambdaX = 0; c.lambdaY = 0;
        c.force = 0;                       // Spiegel: λ=0 → force=0
        w.stats.broken++;
      } else {
        c.force = Math.hypot(c.lambdaX, c.lambdaY) * invH64;   // mirrorInto-Parität
      }
    }
  }

  // ── Patch: step/substep ersetzen ──────────────────────────────────
  w.substep = function () {
    resetLambdaF();
    integrateF();
    for (let it = 0; it < w.iterations; it++) {
      solveConstraintsF();
      solveCollisionsF();
    }
    finalizeF();
    w._measure();
  };
  w._postGripBreak = gripBreakF;
  w.step = function () {
    snapshotControls();
    for (let s = 0; s < w.substeps; s++) w.substep();
    w._postGripBreak();
  };

  return { roundState, snapshotControls };
}
