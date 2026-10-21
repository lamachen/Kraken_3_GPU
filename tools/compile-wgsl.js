#!/usr/bin/env node
/**
 * Echter WGSL-Compile-Check — naga (der Shader-Übersetzer aus wgpu) als WASM.
 *
 * Warum: `tools/check-wgsl.js` vergleicht nur Texte (Strides, Bindings,
 * Entry-Points). Es kann KEIN WGSL parsen. Genau das hat fast einen
 * ausgelieferten Compile-Fehler gekostet: im Race-Fix stand `let cnt = …`
 * und wurde in der Schleife dekrementiert — `invalid left-hand side of
 * assignment`. Ein Browser-Start beim Benutzer, um das herauszufinden.
 *
 *   node tools/compile-wgsl.js [--json]
 *
 * Optional: ohne installiertes `naga-wasm` endet das Skript mit Code 3
 * (SKIP, nicht FAIL) — das Repo bleibt buildless und abhängigkeitsfrei.
 *   npm i --no-save naga-wasm@30.2.0
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const SHADER = join(ROOT, 'src/gpu/solver.wgsl');
const WANT_JSON = process.argv.includes('--json');

let naga;
try {
  naga = await import('naga-wasm');
} catch {
  console.log('SKIP naga-wasm ist nicht installiert — kein Compile-Check möglich.');
  console.log('     npm i --no-save naga-wasm@30.2.0   (2 MB WASM, keine Repo-Abhängigkeit)');
  process.exit(3);
}

const src = readFileSync(SHADER, 'utf8');
const out = { file: 'src/gpu/solver.wgsl', bytes: src.length, ok: false, error: null, entryPoints: [] };
try {
  const module = naga.parseWgsl(src);
  const info = naga.validate(module);
  out.ok = true;
  // Einmal WGSL zurückgeschrieben = Beweis, dass der IR vollständig aufgebaut
  // wurde (validate() allein prüft nur; writeWgsl serialisiert jeden Eintrag).
  const back = naga.writeWgsl(module, info, {});
  out.roundtripBytes = back.length;
  for (const ep of ['resetLambda', 'integrate', 'solveConstraints', 'solveCollisions', 'finalize', 'gripBreak']) {
    out.entryPoints.push(ep);
    if (!new RegExp('fn\\s+' + ep + '\\b').test(back)) throw new Error(`Entry-Point ${ep} fehlt im zurückserialisierten Modul`);
  }
  module.free?.(); info.free?.();
} catch (e) {
  out.error = String(e && (e.message || e)).split('\n').slice(0, 12).join('\n');
}

if (WANT_JSON) { console.log(JSON.stringify(out, null, 1)); }
else if (out.ok) {
  console.log(`✅ ${out.file} parst, validiert und serialisiert zurück (${out.bytes} B → ${out.roundtripBytes} B, naga 30.x)`);
  console.log(`   Entry-Points im IR: ${out.entryPoints.join(', ')}`);
} else {
  console.log(`❌ ${out.file} — WGSL-Compile-Fehler:\n${out.error}`);
}
process.exit(out.ok ? 0 : 1);
