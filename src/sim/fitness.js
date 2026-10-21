/**
 * Fitness — die Lehren aus Kraken_v2, in Code gegossen.
 *
 * v2 hatte: `gripLoco = gripMoveDist * 3.5` mit `gripMoveDist = Σ|v|` solange
 * IRGENDEIN Arm hielt. Drehen auf der Stelle liefert volles |v|, keinen
 * Drift-Penalty und fast keinen Energieaufwand — Kreiseln war damit die
 * fitness-optimale Strategie. Gemessen: Score 3244 bei 109 px echter Distanz
 * und 198.8 Umdrehungen. `angularAccum` war als "spinning penalty" deklariert,
 * wurde aber nie erhöht und nie verwendet.
 *
 * Drei Regeln, die hier durchgehalten werden:
 *   1. Nur NETTO-Fortschritt zählt, nie Pfadlänge. Zappeln liefert 0.
 *   2. Lageunabhängig: `reach` normiert auf den beim Spawn vermessenen Freiraum,
 *      sonst selektiert man Spawn-Lotterie (v2: Score-Streuung 150…571 OHNE
 *      Trend über 22 Generationen).
 *   3. Jeder Term einzeln nicht farmbar — insbesondere kein Bonus dafür,
 *      Energie auszugeben (v2: `idleP − energySpent*1.5` belohnte Verschwendung).
 *
 * ── Revision nach Messung (tools/gaitprobe.mjs, tools/strideprobe.mjs) ──────
 * Regeln 1 und 3 waren an drei Stellen verletzt — so, dass die Fitness Verhalten
 * belohnte, das die Messstände danach als Sackgasse auswiesen:
 *
 *   • `gripPullProg` summierte jeden Anstieg des Spawn-Abstands, also auch den
 *     zweiten Vorstoß nach einem Zurückrutschen. Rattern war farmbar: der
 *     Skript-Gang `wave T=16` holte 126.9 px „Zug" aus 28.3 px Netto (4.5×),
 *     und dieser eine Term trug 330 von 454 Punkten — der höchste Score der
 *     ganzen Sweep-Tabelle gehörte einem Tier, das kaum vorankam. Jetzt zählt
 *     nur das Ratchet auf neue Maxima: `gripPullProg ≤ maxDisplacement`
 *     (gepinnt von tools/strideprobe.mjs über alle Konfigurationen).
 *   • `directed` und `headingHold` sind Verhältnisse ohne Fortschritts-Schwelle
 *     und damit durch STILLSTEHEN maximierbar. Gemessen: „halten (Adhäsion 1,
 *     keine Bewegung)" scored 104.4 und „zappeln" 255.8 — das trainierte Netz
 *     nach 6 Generationen ES: 149.8. Eine Policy schlechter als ein toter
 *     Krake war fitness-optimal. Jetzt: `progGate` (1.0 ab 25 % des Freiraums)
 *     auf alle Qualitätsterme; „halten" 104.4 → 44.4, „zappeln" 255.8 → 61.0.
 *   • Der gait-Term war blind: `gripCycles` verlangt einen Schritt mit NULL
 *     haftenden Armen. Gemessen haften im Schnitt 2.33 von 6 Armen, nur
 *     0.46 % der Schritte haben null Anker, 79 % der Greif-Ereignisse sind
 *     Wechselgriffe. Der Term trug 4.2 von 360 möglichen Punkten (1.4 % des
 *     Scores): toter Gradient, nicht statisches Verhalten — die alte
 *     README-Behauptung „einmal festhalten und nie mehr loslassen" war falsch.
 *     Ersetzt durch `stride` = Wechselgriffe pro Arm (postStep zählt sie,
 *     `c.strides`), gedeckelt auf `strideCap` und multiplikativ auf den
 *     Fortschritt.
 *
 * Der physikalische Kopfraum war nie das Problem: getaktete Skript-Controller
 * erreichen 64.4 px netto gegen 50.0 px des trainierten Netzes nach 20
 * Generationen — dieselbe Welt, dieselben 32 Spawns, dieselbe Physik.
 */
import { ARM_REACH } from '../body/kraken.js';

export const FIT_WEIGHTS = {
  gripPull: 2.6,      // Netto-Fortschritt vom Spawn weg, NUR während ein Arm hält (Ratchet)
  strideBonus: 0.5,   // Wechselgriffe VERVIELFACHEN den Fortschritt um bis zu diesen Anteil
  strideCap: 60,      // gedeckelt: 60 ≈ ein Griff pro 8 Schritte bei 500-Schritt-Episoden
  reach: 320,         // Anteil des verfügbaren Freiraums
  directed: 70,       // Geradlinigkeit (Netto / Pfad)                     — × progGate
  headingHold: 28,    // ruhiger Kurs                                      — × progGate
  meanSpeed: 14,      // überhaupt mobil sein (klein, sonst Zappel-Anreiz) — × progGate
  efficiency: 3,      // Strecke pro aufgewendete Energie                  — × progGate
  progGateKnee: 0.25, // Anteil von spawnReach, ab dem Qualitätsterme voll zählen
  spin: 0.55,         // Strafe für kumulierte Rotation
  drift: 0.05,        // Bewegung ohne Halt ist verschwendete Energie
  preyHit: 25,        // Beute getroffen
  preyNear: 0.06,     // Annäherung an Beute
  energy: 0.30,       // Energieverbrauch
};

/**
 * Beim Spawn den verfügbaren Freiraum vermessen (16 Strahlen).
 * Ohne das ist die Fitness reine Lotterie: wer in einer Nische mit der Wand im
 * Rücken spawnt, kann physikalisch nie weit kommen.
 */
export function measureReach(map, x, y) {
  let sum = 0, best = 0;
  const N = 16;
  for (let k = 0; k < N; k++) {
    const a = (k / N) * Math.PI * 2;
    let d = 0;
    while (d < ARM_REACH * 3.5) {
      d += 12;
      const h = map.sdf(x + Math.cos(a) * d, y + Math.sin(a) * d);
      if (!h || h.d <= 0) break;
    }
    sum += d; if (d > best) best = d;
  }
  return Math.max(70, (sum / N) * 0.55 + best * 0.45);
}

/**
 * Reihenfolge der Terme. Sie ist NICHT beliebig: `score()` summiert in
 * Float64, und Addition ist nicht assoziativ. Jede Umstellung ändert das
 * Ergebnis in den letzten Bits — und G7 vergleicht Fitness zwischen CPU-Oracle
 * und GPU auf ±5 %, während `train-determinism` Bit-Identität verlangt.
 * `scoreTerms()` liefert die Beiträge, `score()` summiert sie in genau dieser
 * Reihenfolge; negative Beiträge stehen als negative Zahlen darin
 * (`s += -x` ist bitgleich zu `s -= x`).
 */
export const TERM_ORDER = [
  'gripPull', 'stride', 'reach', 'directed', 'headingHold', 'meanSpeed',
  'efficiency', 'spin', 'drift', 'energy', 'preyHit', 'preyNear',
];

/**
 * Fitness in ihre Terme zerlegt — derselbe Code, den `score()` summiert.
 *
 * Warum das exportiert ist: „Score 418" ist keine Aussage. Erst die Zerlegung
 * zeigt, wonach die Selektion tatsächlich greift. Gemessen mit
 * `tools/gaitprobe.mjs` am Stand vor dieser Revision: der gait-Term trug 4 von
 * 418 Punkten (1.4 %), weil `gripCycles` einen Vollverlust aller sechs Anker
 * verlangt, den ein Wechselgang nie produziert — ein toter Gradient. Und
 * `gripPull` trug 330 von 454 Punkten eines Ganges, der 28 px netto lief und
 * 342 px Weg machte: die Fitness hatte das Rattern belohnt, nicht das Reisen.
 */
export function scoreTerms(c, opts = {}) {
  const W = FIT_WEIGHTS;
  const steps = Math.max(1, c.steps);
  const path = c.pathLen;
  const disp = c.maxDisplacement;
  const reach = c.spawnReach > 0 ? c.spawnReach : 300;

  const reachFrac = Math.min(1.2, disp / reach);
  const directed = path > 1 ? Math.min(1, disp / path) : 0;
  // Kopfgesteuerte Rotation: ein voll ausgelastetes Winkel-Limit über die ganze
  // Episode ist die Obergrenze; davon der verbrauchte Anteil ist die Strafe.
  const spinBudget = steps * (opts.maxAngSpeed || 18) * (opts.dt || 1 / 60);
  const headingHold = 1 - Math.min(1, c.angularAccum / Math.max(1e-6, spinBudget));
  const meanSpeed = path / steps / 60;             // px/step, grob normiert
  const efficiency = (c.forwardProg || 0) / Math.max(0.5, c.energySpent || 0.5);
  // Fortschritts-Basis: die Terme, die echten Raumgewinn messen. Der Gang
  // multipliziert SIE, statt sich selbst zu belohnen — Takt ist damit nicht
  // ohne Reise zu kassieren. (Additiv mit progGate war der erste Versuch und
  // ist gemessen durchgefallen: nach 20 Generationen 75 Wechselgriffe pro
  // Episode bei nur 22.7 px netto, gegen 50.0 px beim alten Stand — die ES
  // hatte den leichteren der beiden Wege genommen. Multiplikativ zahlt
  // derselbe Takt einem Tier, das 50 px reist, ein Vielfaches dessen, was er
  // einem Tier zahlt, das auf der Stelle tritt.)
  const progressBase = (c.gripPullProg || 0) * W.gripPull + reachFrac * W.reach;
  const strideFrac = Math.min(c.strides || 0, W.strideCap) / W.strideCap;   // 0..1

  // Fortschritts-Tor für alle QUALITÄTSTERME. Verhältnisse wie Netto/Pfad oder
  // „ruhiger Kurs" sind ohne Schwelle durch Stillstand maximierbar (gemessen:
  // 104.4 Punkte für ein Tier, das sich praktisch nicht bewegt, 255.8 für eines,
  // das nur zappelt — das trainierte Netz lag nach 6 Generationen bei 149.8).
  // Die Fortschrittsterme selbst (gripPull, reach, Beute) bleiben ungetorrt —
  // sie SIND der Fortschritt. Strafterme bleiben Strafterme.
  const progGate = Math.min(1, disp / Math.max(1e-6, W.progGateKnee * reach));

  const t = {
    gripPull: (c.gripPullProg || 0) * W.gripPull,
    stride: progressBase * W.strideBonus * strideFrac,
    reach: reachFrac * W.reach,
    directed: directed * W.directed * progGate,
    headingHold: headingHold * W.headingHold * progGate,
    meanSpeed: Math.min(1, meanSpeed) * W.meanSpeed * progGate,
    efficiency: Math.min(20, efficiency) * W.efficiency * progGate,
    spin: -(c.angularAccum * W.spin * 0.02),
    drift: -((c.freeMove || 0) * W.drift),
    energy: -((c.energySpent || 0) * W.energy),
    preyHit: (c.hits || 0) * W.preyHit,
    preyNear: (c.preyMinDist !== undefined && c.preyMinDist < 500)
      ? Math.max(0, 500 - c.preyMinDist) * W.preyNear : 0,
  };
  return t;
}

export function score(c, opts = {}) {
  const t = scoreTerms(c, opts);
  let s = 0;
  for (let i = 0; i < TERM_ORDER.length; i++) s += t[TERM_ORDER[i]];
  return Number.isFinite(s) ? s : 0;
}
