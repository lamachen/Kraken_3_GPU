#!/usr/bin/env node
/**
 * Statische Konsistenzprüfung Shader ↔ Bindings.
 *
 * WGSL lässt sich ohne GPU/Browser nicht kompilieren. Was sich aber prüfen lässt
 * — und was die häufigste Fehlerquelle ist — sind die Vereinbarungen zwischen
 * solver.wgsl und gpuworld.js: Struct-Feldanzahl gegen Byte-Stride, Entry-Points,
 * Binding-Nummern, read-only gegen read_write, Workgroup-Größe.
 *
 *   node tools/check-wgsl.js
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const wg = readFileSync(join(ROOT, 'src/gpu/solver.wgsl'), 'utf8');
const js = readFileSync(join(ROOT, 'src/gpu/gpuworld.js'), 'utf8');

let fail = 0;
const good = (m) => console.log('  ✅ ' + m);
const bad = (m) => { fail++; console.log('  ❌ ' + m); };

// Kommentare entfernen, damit sie die Feldzählung nicht stören
// Kommentare entfernen UND Whitespace normalisieren: `struct Rect  { x: f32, … }`
// steht einzeilig mit Doppel-Leerzeichen, ein naiver indexOf('struct Rect {')
// findet das nicht.
const clean = wg.replace(/\/\/[^\n]*/g, '').replace(/\s+/g, ' ');

function fieldCount(name) {
  const i = clean.indexOf('struct ' + name + ' {');
  if (i < 0) return null;
  let depth = 0, j = i;
  for (; j < clean.length; j++) { if (clean[j] === '{') depth++; else if (clean[j] === '}') { depth--; if (!depth) break; } }
  const body = clean.slice(clean.indexOf('{', i) + 1, j);
  return body.split(',').map(s => s.trim()).filter(s => s.includes(':')).length;
}

console.log('── Struct-Felder gegen JS-Stride ──');
const strides = { body: 'BODY_STRIDE = (\\d+)', constraint: 'CONS_STRIDE = (\\d+)' };
const spec = [
  ['Body', 'BODY_STRIDE', 48],
  ['Constraint', 'CONS_STRIDE', 80],
  ['CollPt', null, 16],
  ['Rect', null, 16],
];
for (const [name, jsConst, bytes] of spec) {
  const f = fieldCount(name);
  if (f === null) { bad(`struct ${name} nicht gefunden`); continue; }
  if (f * 4 !== bytes) { bad(`${name}: ${f} Felder = ${f * 4} B, erwartet ${bytes} B`); continue; }
  if (jsConst) {
    const m = js.match(new RegExp(jsConst + ' = (\\d+)'));
    if (!m) { bad(`${jsConst} in gpuworld.js nicht gefunden`); continue; }
    if (Number(m[1]) !== f) { bad(`${jsConst} = ${m[1]}, aber WGSL ${name} hat ${f} Felder`); continue; }
  }
  good(`${name}: ${f} Felder × 4 B = ${bytes} B (Stride stimmt)`);
}

console.log('\n── Params-Uniform ──');
{
  const f = fieldCount('Params');
  const size = f * 4;
  const m = js.match(/params: buf\((\d+)/);
  const alloc = m ? Number(m[1]) : 0;
  if (size <= alloc) good(`Params: ${f} Felder = ${size} B, Buffer ${alloc} B`);
  else bad(`Params braucht ${size} B, Buffer ist nur ${alloc} B`);
}

console.log('\n── Entry-Points ──');
{
  const eps = [...clean.matchAll(/@compute\s*@workgroup_size\(\d+\)\s*fn\s+(\w+)/g)].map(m => m[1]);
  const need = ['resetLambda', 'integrate', 'solveConstraints', 'solveCollisions', 'finalize', 'gripBreak'];
  for (const e of need) {
    if (!eps.includes(e)) bad(`Entry-Point ${e} fehlt im Shader`);
    else if (!js.includes(`mk('${e}')`)) bad(`${e} wird in gpuworld.js nicht als Pipeline angelegt`);
    else good(`${e} vorhanden und verdrahtet`);
  }
  const extra = eps.filter(e => !need.includes(e));
  if (extra.length) bad(` ungenutzte Entry-Points: ${extra.join(', ')}`);
}

console.log('\n── Bindings ──');
{
  const wgB = [...clean.matchAll(/@group\(0\)\s*@binding\((\d+)\)\s*var<([^>]+)>\s*(\w+)/g)]
    .map(m => ({ n: +m[1], kind: m[2], name: m[3] }));
  const seen = new Set();
  for (const b of wgB) {
    if (seen.has(b.n)) { bad(`Binding ${b.n} doppelt belegt`); continue; }
    seen.add(b.n);
    const layout = js.match(new RegExp(`binding: ${b.n},[^}]*type: '([a-z-]+)'`));
    // WGSL <storage, read> VERLANGT im Layout 'read-only-storage' — das ist
    // keine Abweichung, sondern die vorgeschriebene Zuordnung.
    const want = b.kind.includes('uniform') ? 'uniform'
      : b.kind === 'storage, read' ? 'read-only-storage' : 'storage';
    if (!layout) { bad(`Binding ${b.n} (${b.name}) hat kein Layout in gpuworld.js`); continue; }
    if (layout[1] !== want) { bad(`Binding ${b.n} (${b.name}): WGSL <${b.kind}> braucht '${want}', Layout sagt '${layout[1]}'`); continue; }
    good(`Binding ${b.n} ${b.name.padEnd(8)} <${b.kind}> → '${layout[1]}'`);
  }
  const jsB = new Set([...js.matchAll(/binding: (\d), visibility/g)].map(m => +m[1]));
  for (const n of jsB) if (!seen.has(n)) bad(`Layout definiert Binding ${n}, der Shader nutzt sie nicht`);
}

console.log('\n── Kollisions-Race (Regression gegen Build 529fdc7) ──');
{
  // URSACHE der G7-Divergenz, headless reproduziert (tools/class-spread.mjs):
  // solveCollisions lief mit EINEM THREAD PRO KOLLISIONSPUNKT, aber Mantel (8
  // Punkte) und Armsegmente (2 Punkte) teilen sich einen Körper — mehrere
  // Threads lasen/korrigierten/schrieben dasselbe bodies[i]. Letzter gewinnt,
  // der Rest der Korrektur geht verloren. Messung pop 32, seed 7, 150 Schritte:
  //   f32 mit CPU-Kollisionsordnung  maxΔ  0.36 px · pathLen +0.01 %
  //   f32 mit Race (letzter gewinnt) maxΔ 29.77 px · pathLen -14.71 %
  //   echte RTX 3070 Ti              maxΔ 22.02 px · pathLen -14.80 %
  // Fix: ein Thread pro Körper, seine Punkte sequentiell in CPU-Ordnung.
  const perBody = /fn solveCollisions[\s\S]*?gid\.x;\s*if \(i >= P\.nBodies\)/.test(clean);
  if (perBody) good('solveCollisions begrenzt auf P.nBodies — ein Thread pro Körper');
  else bad('solveCollisions läuft nicht pro Körper (Schreib-Race über bodies[] möglich)');

  const noPointDispatch = /solveCollisions\);\s*p\.setBindGroup\(0, g\.bindGroup\);\s*p\.dispatchWorkgroups\(disp\(g\.nB\)\)/.test(js.replace(/\s+/g, ' ').replace(/setPipeline\(g\.pipelines\./g, ''));
  if (noPointDispatch) good('Dispatch solveCollisions = disp(nB), nicht disp(nP)');
  else bad('solveCollisions wird nicht mit disp(g.nB) dispatched');

  if (/gid\.x;\s*if \(k >= P\.nCollPts\)/.test(clean.replace(/fn solveCollisions[^]*?$/, '')) === false &&
      !/fn solveCollisions[\s\S]*?P\.nCollPts/.test(clean)) good('kein Thread-Index mehr über P.nCollPts in solveCollisions');
  else bad('solveCollisions indexiert weiterhin über P.nCollPts (ein Thread pro Punkt)');

  const hasRange = wg.includes('@binding(8)') && wg.includes('collRange') && js.includes('collRange:');
  if (hasRange) good('collRange (Binding 8) in Shader und gpuworld.js vorhanden');
  else bad('collRange fehlt (Shader oder gpuworld.js)');

  // Die Gruppierung ist eine reine exportierte Funktion (groupCollPoints),
  // damit der Harness sie headless gegen _solveCollisions prüfen kann —
  // gpuworld.js selbst ist ohne GPUBufferUsage nicht importierbar.
  const grouped = /export function groupCollPoints/.test(js)
    && js.includes('groupCollPoints(cpu.collPts, cpu.n)')
    && js.includes('buffers.collRange');
  if (grouped) good('Kollisionspunkte werden per groupCollPoints() nach Körper gruppiert (stabil, CPU-Ordnung)');
  else bad('Kollisionspunkte werden nicht nach Körper gruppiert — Shader-Loop wäre lückenhaft');

  const loopOk = /var cnt = collRange\[nB \+ i\];[\s\S]*?let cp = collPts\[o\];[\s\S]*?bodies\[i\] = b;/.test(clean);
  // `var`, nicht `let`: cnt wird pro Punkt dekrementiert. `let cnt` ist ein
  // Compile-Fehler (invalid left-hand side) — von tools/compile-wgsl.js gefangen.
  if (loopOk) good('Punkte des Körpers werden sequentiell abgearbeitet, EIN Rückschrieb am Ende');
  else bad('Kollisions-Schleife im Shader nicht in der erwarteten Form');
}

console.log('\n── Workgroup-Größe ──');
{
  const ws = [...new Set([...clean.matchAll(/@workgroup_size\((\d+)\)/g)].map(m => +m[1]))];
  const m = js.match(/WORKGROUP = (\d+)/);
  const jsWs = m ? Number(m[1]) : NaN;
  if (ws.length === 1 && ws[0] === jsWs) good(`Shader ${ws[0]}, JS ${jsWs} — identisch`);
  else bad(`Shader nutzt ${ws.join('/')}, JS dispatcht mit ${jsWs}`);
}

console.log('\n── Reservierte WGSL-Schlüsselwörter als Bezeichner ──');
{
  // Gemessen am 19.10.2026 in Chrome auf einer RTX 3070 Ti: das Feld `target`
  // im Constraint-Struct schlug mit "'target' is a reserved keyword" fehl,
  // Zeile 40:31 und 255:29. Solche Fehler kosten hier genau einen Browser-Start,
  // deshalb statisch abfangen, bevor der Shader den Compiler sieht.
  const RESERVED = new Set([
    'asm','bf16','const_cast','dynamic_cast','enum','handle','inline','nointerpolation',
    'namespace','precise','reinterpret_cast','row_major','sample','shared','short',
    'sizeof','static','static_assert','static_cast','template','this','thread_local',
    'typedef','union','using','varying','volatile','target','co_await','co_return',
    'co_yield','goto','class','delete','new','try','catch','do','auto','type','vec',
    'let','out','override','requires','uniform','input','output','resource','filter',
    'active','passive','mut','nonmut','public','private','protected','friend',
  ]);
  const clean2 = wg.replace(/\/\/[^\n]*/g, '');
  const offenders = [];
  for (const m of clean2.matchAll(/struct\s+(\w+)\s*\{([^}]*)\}/g)) {
    for (const f of m[2].split(',').map(s => s.trim()).filter(s => s.includes(':'))) {
      const name = f.split(':')[0].trim();
      if (RESERVED.has(name)) offenders.push(m[1] + '.' + name);
    }
  }
  for (const m of clean2.matchAll(/(?:let|var<[^>]*>|var|const)\s+(\w+)\s*(?::|=)/g)) {
    if (RESERVED.has(m[1])) offenders.push('Deklaration ' + m[1]);
  }
  if (offenders.length === 0) good('keine reservierten Schlüsselwörter als Feld- oder Variablennamen');
  else bad('reservierte WGSL-Schlüsselwörter: ' + offenders.join(', '));
}

console.log('\n── Typkorrektheit der Buffer-Schreibweise (DEVICE_HUNG-Regression) ──');
{
  // Gemessen am 19.10.2026 auf einer RTX 3070 Ti: Ganzzahlen, die über ein
  // Float32Array in u32/i32-Shaderfelder geschrieben werden, landen dort als
  // Float-BITMUSTER. iterations=4 wurde zu i32 1.082.130.432, nRects zu einer
  // Milliarden-Zahl — die SDF-Schleife lief praktisch endlos, die GPU hing,
  // Windows setzte sie per TDR zurück (DXGI_ERROR_DEVICE_HUNG 0x887A0006),
  // und danach lieferte requestAdapter() bis zum Browser-Neustart null.
  //
  // Dieser Check pinnt die korrekte Schreibweise fest.
  const BAD = [
    /g\.params\[\s*\d+\s*\]\s*=\s*cpu\.(iterations|n)\b/,
    /g\.params\[\s*\d+\s*\]\s*=\s*(cpu\.)?(constraints|collPts)\.length/,
    /consData\[o\s*\+?\s*0?\]\s*=\s*c\.a\b/,
    /consData\[o\s*\+\s*1\]\s*=\s*c\.b\b/,
    /consData\[o\s*\+\s*2\]\s*=\s*c\.type/,
    /consData\[o\s*\+\s*17\]\s*=\s*c\.broken/,
    /collData\[o\]\s*=\s*p\.body/,
  ];
  let bad = 0;
  for (const re of BAD) if (re.test(js)) { bad++; console.log('    ❌ altes Muster vorhanden: ' + re); }
  const GOOD = [
    ['I[1] = cpu.iterations', 'iterations über Int32-Sicht'],
    ['U[6] = cpu.n', 'nBodies über Uint32-Sicht'],
    ['U[9] = map.rects.length', 'nRects über Uint32-Sicht'],
    ['consU[o] = c.a', 'Constraint-Body-Indizes über Uint32-Sicht'],
    ['consU[o + 17] = c.broken', 'broken-Flag über Uint32-Sicht'],
    ['collU[o] = p.body', 'CollPt-Body-Index über Uint32-Sicht'],
  ];
  for (const [needle, label] of GOOD) {
    if (js.includes(needle)) good(label);
    else { bad++; console.log('    ❌ fehlt: ' + label + '  (' + needle + ')'); }
  }
  if (bad === 0) good('keine Ganzzahl wird über eine Float-Sicht in ein u32/i32-Feld geschrieben');
  else { fail += bad; console.log('    ❌ ' + bad + ' typinkorrekte Schreibweise(n)'); }
  for (const needle of ['iterations unplausibel', 'nRects unplausibel', 'nBodies unplausibel']) {
    if (js.includes(needle)) good('Sanity-Guard: ' + needle);
    else { fail++; console.log('    ❌ Guard fehlt: ' + needle); }
  }
}

console.log('\n── Reihenfolge der Substeps (Shader-Aufrufe in gpuworld.js) ──');
{
  const order = [...js.matchAll(/pipelines\.(\w+)/g)].map(m => m[1]);
  const i = order.indexOf('integrate');
  const f = order.indexOf('finalize');
  const sc = order.indexOf('solveConstraints');
  if (i >= 0 && sc > i && f > sc) good(`integrate → solveConstraints/solveCollisions → finalize (wie XpbdWorld.substep)`);
  else bad(`falsche Reihenfolge: ${order.join(' → ')}`);
}

console.log('\n── CPU-Oracle-Parität: dieselben Konstanten ──');
{
  const cpu = readFileSync(join(ROOT, 'src/physics/xpbd.js'), 'utf8');
  for (const k of ['maxSpeed', 'maxAngSpeed', 'linDrag', 'angDrag', 'iterations', 'substeps']) {
    if (js.includes('cpu.' + k)) good(`${k} wird aus dem CPU-World gelesen, nicht dupliziert`);
    else bad(`${k} wird nicht vom CPU-World übernommen — Gefahr von Drift`);
  }

  // Totzone (SLACK_POS/SLACK_ANG): Werte UND Verwendung müssen auf beiden
  // Seiten identisch sein. Driften sie auseinander, „korrigiert" nur eine
  // Seite den f32-Boden (G1-Drift 1.2789 px, gemessen an Build d487532) oder
  // schluckt echte Verletzungen — beides wäre im Oracle-Vergleich unsichtbar,
  // wenn die Konstanten nicht statisch gepinnt sind.
  for (const name of ['SLACK_POS', 'SLACK_ANG']) {
    const mJs = cpu.match(new RegExp('export const ' + name + ' = ([0-9.eE+-]+);'));
    const mWg = wg.match(new RegExp('const ' + name + ': f32 = ([0-9.eE+-]+);'));
    if (!mJs || !mWg) { bad(`${name} fehlt (xpbd.js: ${!!mJs}, solver.wgsl: ${!!mWg})`); continue; }
    if (Number(mJs[1]) !== Number(mWg[1])) bad(`${name}: CPU ${mJs[1]} ≠ GPU ${mWg[1]}`);
    else good(`${name} = ${Number(mJs[1])} — Parität xpbd.js ↔ solver.wgsl`);
  }
  const deadzoneUses = [
    ['xpbd.js', cpu, ['SLACK_POS * SLACK_POS', 'Math.abs(C) >= SLACK_ANG', 'Math.abs(over) >= SLACK_ANG', 'pen <= SLACK_POS']],
    // Im Shader steht die Totzone als positiver Guard (`if (pen > SLACK_POS)`),
    // weil die Kollision dort in einer Körper-Schleife statt per early-return
    // läuft — dieselbe Semantik, andere Schreibweise.
    ['solver.wgsl', wg, ['SLACK_POS * SLACK_POS', 'abs(C) >= SLACK_ANG', 'abs(over) >= SLACK_ANG', 'pen > SLACK_POS']],
  ];
  for (const [label, src, needles] of deadzoneUses) {
    const miss = needles.filter(nd => !src.includes(nd));
    if (miss.length) bad(`Totzone in ${label} unvollständig: fehlt ${miss.join(', ')}`);
    else good(`${label}: Totzone in Pin, Muskel, Gelenkgrenze und Kollision aktiv`);
  }
}

console.log(`\n${fail === 0 ? '✅' : '❌'} ${fail} Problem(e)`);
process.exit(fail === 0 ? 0 : 2);
