/**
 * WebGPU-Bindung für den XPBD-Solver.
 *
 * Spiegelt einen CPU-`XpbdWorld` 1:1 in GPU-Buffer und fährt dieselben Schritte
 * als Compute-Dispatches. Die CPU-Fassung (src/physics/xpbd.js) bleibt das
 * ORAKEL: gpu.html fährt beide auf identischen Eingaben und vergleicht.
 *
 * `navigator.gpu` wird erst INNERHALB von create() berührt, damit dieses Modul
 * auch in Node importierbar bleibt (tools/harness.js prüft es als Teil von
 * src/). Ohne WebGPU wirft create() eine klare Meldung statt eines TypeError.
 *
 * ── Layout (muss exakt zu solver.wgsl passen) ─────────────────────
 *   Body        12 f32 = 48 B     px py ppx ppy vx vy ang angP av invMass invInertia pad
 *   Constraint  20 × 4 = 80 B     aIdx bIdx cType owner | lax lay lbx lby |
 *                                 compliance rest target minAng | maxAng lamX lamY lam |
 *                                 breakForce broken pad pad
 *   CollPt       4 × 4 = 16 B     body lx ly r
 *   Rect         4 × 4 = 16 B     x y w h
 *   Force        4 × 4 = 16 B     fx fy torque adhesion
 */

import { ensureColored } from '../physics/xpbd.js';

const BODY_STRIDE = 12;
const CONS_STRIDE = 20;
const WORKGROUP = 64;

export const GPU_STRIDES = { body: BODY_STRIDE, constraint: CONS_STRIDE, collPt: 4, rect: 4, force: 4 };

export function isAvailable() {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

export async function create(cpu, map, wgslSource, device) {
  // `device` MUSS von außen kommen, wenn mehrere Welten gebaut werden.
  //
  // Gemessen: gpu.html baut ~10 Welten (G1..G6), und jede rief hier
  // requestAdapter() + requestDevice() auf. Irgendwann lieferte
  // requestAdapter() null auf einer Maschine, die Minuten zuvor noch
  // "nvidia ampere RTX 3070 Ti" gemeldet hatte — der GPU-Prozess war
  // überlastet bzw. gecrasht. Ein geteiltes Device behebt das und ist
  // nebenbei schneller: kein wiederholter Pipeline-/Device-Aufbau.
  if (!device) {
    if (!isAvailable()) {
      throw new Error('WebGPU ist in diesem Browser nicht verfügbar (navigator.gpu fehlt). ' +
        'Chrome/Edge ab 113, Safari ab 26, Firefox ab 141/145.');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('requestAdapter() lieferte null. Häufigste Ursache: der GPU-Prozess ' +
        'ist überlastet oder gecrasht (viele Devices, Treiber-Reset, Tab im Hintergrund). ' +
        'Seite mit Strg+Shift+R neu laden; bleibt es, chrome://gpu prüfen.');
    }
    device = await adapter.requestDevice();
  }

  // Defensiv: Welten, die ohne createSim() gebaut wurden (gpu.html, spätere
  // Worker), haben möglicherweise keine Färbung. Ohne sie lägen ALLE
  // Constraints in Farbe 0 (Schreib-Race zurück) und upload() crasht an
  // colorOff === undefined — der von dir gemessene TypeError gpuworld.js:225.
  ensureColored(cpu);
  const nB = cpu.n;
  const nC = cpu.constraints.length;
  const nP = cpu.collPts.length;
  const nR = map.rects.length;

  const buf = (bytes, usage) => device.createBuffer({
    size: Math.max(16, Math.ceil(bytes / 16) * 16), usage,
  });
  const SS = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const buffers = {
    params: buf(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
    bodies: buf(nB * BODY_STRIDE * 4, SS),
    cons: buf(nC * CONS_STRIDE * 4, SS),
    collPts: buf(nP * 4 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    rects: buf(nR * 4 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    forces: buf(nB * 4 * 4, SS),
    colorInfo: buf(33 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    readBodies: buf(nB * BODY_STRIDE * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
    readCons: buf(nC * CONS_STRIDE * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
  };

  if (!wgslSource) throw new Error('create() braucht den WGSL-Quelltext als dritten Parameter.');
  const module = device.createShaderModule({ code: wgslSource });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(m => m.type === 'error');
  if (errors.length) {
    throw new Error('WGSL-Compilierfehler:\n' + errors.slice(0, 8)
      .map(e => `  Zeile ${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
  }
  const warnings = info.messages.filter(m => m.type === 'warning');

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const pipelineLayout = device.createBindGroupLayout ? device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }) : null;
  const mk = (entry) => device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: entry } });
  const pipelines = {
    resetLambda: mk('resetLambda'),
    integrate: mk('integrate'),
    solveConstraints: mk('solveConstraints'),
    solveCollisions: mk('solveCollisions'),
    finalize: mk('finalize'),
    gripBreak: mk('gripBreak'),
  };
  // 16 Color-Uniforms (einmalig) und 16 Bind-Groups, die sich alles teilen
  // und nur in Binding 7 differieren. Sequentieller Farb-Dispatch ohne
  // jegliche CPU-Kosten pro Schritt.
  const colorUniforms = [];
  const bindGroups = [];
  for (let c = 0; c < 16; c++) {
    const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ub, 0, new Uint32Array([c, 0, 0, 0]));
    colorUniforms.push(ub);
    bindGroups.push(device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.params } },
        { binding: 1, resource: { buffer: buffers.bodies } },
        { binding: 2, resource: { buffer: buffers.cons } },
        { binding: 3, resource: { buffer: buffers.collPts } },
        { binding: 4, resource: { buffer: buffers.rects } },
        { binding: 5, resource: { buffer: buffers.forces } },
        { binding: 6, resource: { buffer: buffers.colorInfo } },
        { binding: 7, resource: { buffer: ub } },
      ],
    }));
  }
  const bindGroup = bindGroups[0];

  const g = {
    device, buffers, pipelines, bindGroup, bindGroups, colorUniforms, warnings,
    cpu, map, nB, nC, nP, nR,
    // ALLE Felder sind 4 Byte, daher ist der Element-Index für Float32-,
    // Int32- und Uint32-Sicht derselbe. Ganzzahlen MÜSSEN über die Int-Sicht
    // geschrieben werden: ein Float32Array speichert das Bitmuster von 4.0
    // (0x40800000), und der Shader liest daraus als i32 = 1.082.130.432.
    // Genau das hat die GPU hängen lassen (DXGI_ERROR_DEVICE_HUNG, TDR-Reset),
    // weil die SDF-Schleife dann milliardenfach lief.
    paramsF: new Float32Array(16),
    paramsI: null, paramsU: null,
    bodyData: new Float32Array(nB * BODY_STRIDE),
    consData: new Float32Array(nC * CONS_STRIDE),
    collData: new Float32Array(nP * 4),
    rectData: new Float32Array(nR * 4),
    forceData: new Float32Array(nB * 4),
    dispatches: 0,
  };
  g.paramsI = new Int32Array(g.paramsF.buffer);
  g.paramsU = new Uint32Array(g.paramsF.buffer);
  writeStatic(g);
  return g;
}

function writeStatic(g) {
  const { cpu, map } = g;
  const F = g.paramsF, I = g.paramsI, U = g.paramsU;
  F[0] = cpu.h;
  I[1] = cpu.iterations;                 // i32 im Shader
  F[2] = cpu.linDrag;
  F[3] = cpu.angDrag;
  F[4] = cpu.maxSpeed;
  F[5] = cpu.maxAngSpeed;
  U[6] = cpu.n;                          // u32
  U[7] = cpu.constraints.length;         // u32
  U[8] = cpu.collPts.length;             // u32
  U[9] = map.rects.length;               // u32
  F[10] = 1 / cpu.h;
  F[11] = 1 / (cpu.h * cpu.h);
  // SANITY-GUARD: ein falscher Wert hier kostet sonst einen GPU-Hang samt
  // TDR-Reset und einem danach blockierten Adapter — also laut und früh werfen.
  if (!(I[1] >= 1 && I[1] <= 64)) throw new Error(`iterations unplausibel: ${I[1]}`);
  if (!(U[9] >= 1 && U[9] <= 100000)) throw new Error(`nRects unplausibel: ${U[9]}`);
  if (!(U[6] >= 1 && U[6] <= 4000000)) throw new Error(`nBodies unplausibel: ${U[6]}`);
  if (!(F[0] > 0 && F[0] < 1)) throw new Error(`h unplausibel: ${F[0]}`);
  g.device.queue.writeBuffer(g.buffers.params, 0, F);

  for (let i = 0; i < map.rects.length; i++) {
    const r = map.rects[i], o = i * 4;
    g.rectData[o] = r.x; g.rectData[o + 1] = r.y; g.rectData[o + 2] = r.w; g.rectData[o + 3] = r.h;
  }
  g.device.queue.writeBuffer(g.buffers.rects, 0, g.rectData);

  const collU = new Uint32Array(g.collData.buffer);
  for (let i = 0; i < cpu.collPts.length; i++) {
    const p = cpu.collPts[i], o = i * 4;
    collU[o] = p.body;                              // u32 im Shader
    g.collData[o + 1] = p.lx; g.collData[o + 2] = p.ly; g.collData[o + 3] = p.r;
    if (p.body >= cpu.n) throw new Error(`collPt ${i} verweist auf Body ${p.body} von ${cpu.n}`);
  }
  g.device.queue.writeBuffer(g.buffers.collPts, 0, g.collData);
}

/** CPU-Zustand → GPU (vor jedem Vergleichslauf). */
export function upload(g) {
  const { cpu } = g;
  for (let i = 0; i < cpu.n; i++) {
    const o = i * BODY_STRIDE;
    g.bodyData[o] = cpu.px[i]; g.bodyData[o + 1] = cpu.py[i];
    g.bodyData[o + 2] = cpu.ppx[i]; g.bodyData[o + 3] = cpu.ppy[i];
    g.bodyData[o + 4] = cpu.vx[i]; g.bodyData[o + 5] = cpu.vy[i];
    g.bodyData[o + 6] = cpu.ang[i]; g.bodyData[o + 7] = cpu.angP[i];
    g.bodyData[o + 8] = cpu.av[i]; g.bodyData[o + 9] = cpu.im[i];
    g.bodyData[o + 10] = cpu.ii[i]; g.bodyData[o + 11] = 0;
    const fo = i * 4;
    g.forceData[fo] = cpu.fx[i]; g.forceData[fo + 1] = cpu.fy[i];
    g.forceData[fo + 2] = cpu.tq[i]; g.forceData[fo + 3] = 0;
  }
  const consU = new Uint32Array(g.consData.buffer);
  for (let k = 0; k < cpu.constraints.length; k++) {
    const c = cpu.constraints[k], o = k * CONS_STRIDE;
    consU[o] = c.a; consU[o + 1] = c.b >= 0 ? c.b : 0xffffffff;   // u32 im Shader
    consU[o + 2] = c.type; consU[o + 3] = c.owner;                // u32
    if (c.a >= cpu.n || (c.b >= 0 && c.b >= cpu.n)) {
      throw new Error(`Constraint ${k}: Body-Index ${c.a}/${c.b} außerhalb von ${cpu.n} — das wäre ein Out-of-Bounds-Read im Shader`);
    }
    g.consData[o + 4] = c.lax; g.consData[o + 5] = c.lay;
    g.consData[o + 6] = c.lbx; g.consData[o + 7] = c.lby;
    g.consData[o + 8] = c.compliance; g.consData[o + 9] = c.rest;
    g.consData[o + 10] = c.target; g.consData[o + 11] = c.minAngle;
    g.consData[o + 12] = c.maxAngle; g.consData[o + 13] = c.lambdaX || 0;
    g.consData[o + 14] = c.lambdaY || 0; g.consData[o + 15] = c.lambda;
    g.consData[o + 16] = Number.isFinite(c.breakForce) ? Math.min(c.breakForce, 1e30) : 1e30;
    consU[o + 17] = c.broken ? 1 : 0;                              // u32
    consU[o + 18] = 0; consU[o + 19] = 0;                          // pads
  }
  g.device.queue.writeBuffer(g.buffers.bodies, 0, g.bodyData);
  g.device.queue.writeBuffer(g.buffers.cons, 0, g.consData);
  g.device.queue.writeBuffer(g.buffers.forces, 0, g.forceData);

  // Farbgruppen: Offsets, Anzahlen, Workgroups pro Farbe.
  // Ganzzahlen MÜSSEN über die Uint32-Sicht — derselbe Fehlerklassen-Fix wie
  // d24df09 (Float-Bitmuster in u32-Feldern ließ das Device hängen).
  const ci = new Uint32Array(33);
  let maxCnt = 0;
  for (let c = 0; c < 16; c++) {
    ci[c] = cpu.colorOff[c]; ci[16 + c] = cpu.colorCnt[c];
    if (cpu.colorCnt[c] > maxCnt) maxCnt = cpu.colorCnt[c];
  }
  ci[32] = Math.max(1, Math.ceil(maxCnt / 64));
  g.colorWG = ci[32];
  g.device.queue.writeBuffer(g.buffers.colorInfo, 0, ci);
}

const disp = (n) => Math.ceil(n / WORKGROUP);

/**
 * Nur die veränderlichen Constraint-Felder neu schreiben (compliance, target,
 * breakForce, broken). upload() würde die Körperzustände aus den CPU-Arrays
 * zurückschreiben und damit den GPU-Fortschritt vernichten.
 *
 * Ohne das sahen G3/G4 auf der GPU ewig die Spawn-Ziele: die Tests änderten
 * Muskelziele pro Schritt auf der CPU, luden sie aber nie hoch — die GPU
 * simulierte folgerichtig ein Tier im Gleichgewicht (G4 maxV 0.132 statt 360).
 * Ganzzahlen (broken) über die Uint32-Sicht, siehe d24df09.
 */
export function uploadControls(g) {
  packControls(g.cpu, g.consData);
  g.device.queue.writeBuffer(g.buffers.cons, 0, g.consData);
}

/**
 * Steuerrelevante Constraint-Felder aus der CPU-Welt in den flachen Block
 * packen — reine Funktion, damit tools/harness.js das Packing ohne Device
 * prüfen kann (mirror-into-/pack-controls-Checks).
 *
 * Felder: compliance (o+8), tgt (o+10), lbx/lby (o+6/7), breakForce (o+16),
 * broken (o+17, ÜBER DIE UINT32-SICHT — d24df09: Ganzzahlen nie durch die
 * Float-Sicht schreiben).
 *
 * lbx/lby SIND PFLICHT: Der Saugnapf-Reflex (applyControl) verankert die
 * Spitze mitten in der Episode NEU — attachGrip schreibt lbx/lby auf die
 * aktuelle Spitzenposition. Ohne Sync würde die GPU mit dem Anker aus dem
 * letzten vollen upload() ziehen (Spawn-Position oder 0/0).
 */
export function packControls(cpu, consData) {
  const consU = new Uint32Array(consData.buffer, consData.byteOffset, consData.length);
  for (let k = 0; k < cpu.constraints.length; k++) {
    const c = cpu.constraints[k], o = k * CONS_STRIDE;
    consData[o + 6] = c.lbx; consData[o + 7] = c.lby;
    consData[o + 8] = c.compliance;
    consData[o + 10] = c.target;
    consData[o + 16] = Number.isFinite(c.breakForce) ? Math.min(c.breakForce, 1e30) : 1e30;
    consU[o + 17] = c.broken ? 1 : 0;
  }
}

/** Ein Substep — dieselbe Reihenfolge wie XpbdWorld.substep(). */
export function substep(g) {
  const d = g.device;
  const enc = d.createCommandEncoder();
  const pass = () => { const p = enc.beginComputePass(); return p; };
  let p = pass();
  p.setPipeline(g.pipelines.resetLambda); p.setBindGroup(0, g.bindGroup);
  p.dispatchWorkgroups(disp(g.nC));
  p.setPipeline(g.pipelines.integrate); p.setBindGroup(0, g.bindGroup);
  p.dispatchWorkgroups(disp(g.nB)); p.end();
  for (let it = 0; it < g.cpu.iterations; it++) {
    p = pass();
    p.setPipeline(g.pipelines.solveConstraints);
    // Sequentiell pro Farbe: nur so ist die Färbung race-frei.
    for (let c = 0; c < 16; c++) {
      const cnt = g.cpu.colorCnt[c];
      if (!cnt) continue;
      p.setBindGroup(0, g.bindGroups[c]);
      p.dispatchWorkgroups(Math.ceil(cnt / WORKGROUP));
    }
    p.setPipeline(g.pipelines.solveCollisions); p.setBindGroup(0, g.bindGroup);
    p.dispatchWorkgroups(disp(g.nP));
    p.end();
  }
  p = pass();
  p.setPipeline(g.pipelines.finalize); p.setBindGroup(0, g.bindGroup);
  p.dispatchWorkgroups(disp(g.nB)); p.end();
  d.queue.submit([enc.finish()]);
  g.dispatches += 3 + g.cpu.iterations * 2 + 1;
}

export async function step(g) {
  for (let s = 0; s < g.cpu.substeps; s++) substep(g);
  const enc = g.device.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(g.pipelines.gripBreak); p.setBindGroup(0, g.bindGroup);
  p.dispatchWorkgroups(disp(g.nC)); p.end();
  g.device.queue.submit([enc.finish()]);
  g.dispatches += 1;
}

/** GPU → CPU, zurück in dieselben flachen Arrays (für den Vergleich). */
export async function download(g) {
  const enc = g.device.createCommandEncoder();
  enc.copyBufferToBuffer(g.buffers.bodies, 0, g.buffers.readBodies, 0, g.nB * BODY_STRIDE * 4);
  enc.copyBufferToBuffer(g.buffers.cons, 0, g.buffers.readCons, 0, g.nC * CONS_STRIDE * 4);
  g.device.queue.submit([enc.finish()]);
  await g.buffers.readBodies.mapAsync(GPUMapMode.READ);
  await g.buffers.readCons.mapAsync(GPUMapMode.READ);
  const bd = new Float32Array(g.buffers.readBodies.getMappedRange().slice(0));
  const cd = new Float32Array(g.buffers.readCons.getMappedRange().slice(0));
  g.buffers.readBodies.unmap(); g.buffers.readCons.unmap();
  return mirrorInto(g.cpu, bd, cd);
}

/**
 * Spiegelung des GPU-Zustands in die CPU-Welt — reine Funktion (ohne Device),
 * damit tools/harness.js das Feld-Mapping exakt pinnen kann.
 *
 * Das ist das Fundament des Hybrid-Trainings (Milestone 3c): Nach der
 * Spiegelung laufen readProprio/tickSenses/applyControl (inkl. Saugnapf-
 * Reflex)/postStep UNVERÄNDERT auf der CPU — dieselben Code-Pfade wie im
 * CPU-Trainer, nur dass die Physik von der GPU kam. Keine Sinn-Logik wird
 * dupliziert, die CPU bleibt das Oracle.
 *
 * Gespiegelt werden: px, py, vx, vy, ang, av (alle Körper), lambda (Motoren),
 * lambdaX/lambdaY + broken + force (Grips). NICHT gespiegelt: ppx/ppy/angP
 * (liest kein CPU-Verbraucher; respawnCreature setzt sie vor dem nächsten
 * upload() selbst) und stats.broken (Zähler, kein Zustand).
 *
 * Grip-force: Parität zu _postGripBreak() in xpbd.js — F = |λ|/h. Die GPU
 * nullt λ beim Bruch (gripBreak), CPU behält den Wert; beobachtbar ist das
 * nicht, denn readProprio gibt für gebrochene Grips ohnehin 0 aus.
 */
export function mirrorInto(cpu, bodyData, consData) {
  const consU = new Uint32Array(consData.buffer, consData.byteOffset, consData.length);
  // Enger Loop ohne Allokationen: die frühere Fassung baute pro Körper ein
  // 5er-Array für .every(Number.isFinite) und nutzte Math.hypot für die
  // Statistik — 22 016 Körper × (Allokation + hypot) kostete ~5.8 ms/Schritt
  // bei B=512. Stats sind Diagnosewerte (G1-Tor: < 0.05 px/s), deshalb ist
  // sqrt(vx²+vy²) statt hypot hier statthaft; die Grip-KRAFT bleibt exakt
  // Math.hypot — sie ist Parität zu _postGripBreak() und NN-Eingang.
  let maxS2 = 0, maxAngSpeed = 0, nan = 0;
  for (let i = 0; i < cpu.n; i++) {
    const o = i * BODY_STRIDE;
    const px = bodyData[o], py = bodyData[o + 1];
    const vx = bodyData[o + 4], vy = bodyData[o + 5];
    const ang = bodyData[o + 6], av = bodyData[o + 8];
    cpu.px[i] = px; cpu.py[i] = py;
    cpu.vx[i] = vx; cpu.vy[i] = vy;
    cpu.ang[i] = ang; cpu.av[i] = av;
    const s2 = vx * vx + vy * vy;
    if (s2 > maxS2) maxS2 = s2;
    const aa = av < 0 ? -av : av;
    if (aa > maxAngSpeed) maxAngSpeed = aa;
    if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(vx)
      || !Number.isFinite(vy) || !Number.isFinite(ang)) nan++;
  }
  const maxSpeed = Math.sqrt(maxS2);
  const invH = 1 / cpu.h;
  for (let k = 0; k < cpu.constraints.length; k++) {
    const c = cpu.constraints[k], o = k * CONS_STRIDE;
    c.lambda = consData[o + 15];
    c.broken = consU[o + 17] !== 0;
    if (c.type === 2) {                    // C_GRIP — Parität zu xpbd.js
      c.lambdaX = consData[o + 13]; c.lambdaY = consData[o + 14];
      c.force = Math.hypot(c.lambdaX, c.lambdaY) * invH;
    }
  }
  cpu.stats.maxSpeed = maxSpeed;
  cpu.stats.maxAngSpeed = maxAngSpeed;
  cpu.stats.nan = nan;
  return { maxSpeed, maxAngSpeed, nan };
}

/** Relativer Pfad zum Shader — gpu.html fetcht ihn und übergibt ihn an create(). */
export const WGSL_PATH = 'src/gpu/solver.wgsl';

/**
 * Gibt alle Buffer der Welt frei. Wichtig, wenn Szenarien nacheinander laufen:
 * ungefreed storage buffer halten Device-Speicher und tragen zur Überlastung bei,
 * die requestAdapter() später mit null beantwortet.
 */
export function destroy(g) {
  if (!g) return;
  for (const k of Object.keys(g.buffers)) {
    try { g.buffers[k].destroy(); } catch (e) { /* bereits zerstört */ }
  }
  g.buffers = {};
}

export async function loadWgsl(base = '') {
  const url = base ? base + '/' + WGSL_PATH : WGSL_PATH;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Shader nicht ladbar: ${url} → HTTP ${r.status}`);
  return r.text();
}
