#!/usr/bin/env node
/**
 * Hängt an JEDE relative Modul- und Shader-URL im Site-Artefakt einen
 * Build-Stempel:  ./src/x.js  →  ./src/x.js?v=<sha>
 *
 * Warum das nötig ist (gemessen, nicht vermutet): GitHub Pages liefert über
 * ein CDN mit Cache-Control aus. Nachdem ein Fix deployed war, bekam ein
 * Browser weiterhin die VORHERIGE gpuworld.js serviert und crashte an einer
 * Zeile, die es im neuen Build nicht mehr gab. Drei Diagnose-Durchläufe gingen
 * dafür drauf, und der Benutzer wurde zu Unrecht auf seinen Browser verwiesen.
 *
 * Mit gestempelten URLs ist jede Deploy-Version global eindeutig: kein
 * Browser-Cache, keine CDN-Edge kann jemals eine alte Datei zur neuen HTML
 * liefern. Der Stempel kommt aus GITHUB_SHA und wird vom Workflow eingesetzt.
 *
 *   node tools/bust-cache.js <site-dir> <sha>
 */
import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const dir = process.argv[2];
const sha = (process.argv[3] || 'dev').slice(0, 7);
if (!dir) { console.error('usage: node tools/bust-cache.js <site-dir> <sha>'); process.exit(2); }

const RE = /(['"])(\.{0,2}\/[^'"]+?\.(?:js|wgsl))\1/g;
let files = 0, hits = 0;

function walk(d) {
  for (const name of readdirSync(d)) {
    const p = join(d, name);
    const st = statSync(p);
    if (st.isDirectory()) { walk(p); continue; }
    if (!/\.(html|js)$/.test(name)) continue;
    let s = readFileSync(p, 'utf8');
    const before = s;
    s = s.replace(RE, (m, q, path) => {
      if (path.includes('?')) return m;          // bereits gestempelt
      hits++;
      return `${q}${path}?v=${sha}${q}`;
    });
    if (s !== before) { writeFileSync(p, s); files++; }
  }
}
walk(dir);
console.log(`bust-cache: ${files} Dateien umgeschrieben, ${hits} URLs gestempelt mit v=${sha}`);
