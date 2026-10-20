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
  const need = ['integrate', 'solveConstraints', 'solveCollisions', 'finalize', 'gripBreak'];
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

console.log('\n── Workgroup-Größe ──');
{
  const ws = [...new Set([...clean.matchAll(/@workgroup_size\((\d+)\)/g)].map(m => +m[1]))];
  const m = js.match(/WORKGROUP = (\d+)/);
  const jsWs = m ? Number(m[1]) : NaN;
  if (ws.length === 1 && ws[0] === jsWs) good(`Shader ${ws[0]}, JS ${jsWs} — identisch`);
  else bad(`Shader nutzt ${ws.join('/')}, JS dispatcht mit ${jsWs}`);
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
}

console.log(`\n${fail === 0 ? '✅' : '❌'} ${fail} Problem(e)`);
process.exit(fail === 0 ? 0 : 2);
