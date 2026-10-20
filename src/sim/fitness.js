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
 */
import { ARM_REACH } from '../body/kraken.js';

export const FIT_WEIGHTS = {
  gripPull: 2.6,     // Netto-Fortschritt vom Spawn weg, NUR während ein Arm hält
  gait: 3.0,         // abgeschlossene Zugzyklen — den Gang überhaupt entdecken
  gaitCap: 120,      // gedeckelt: schnelles Greifen/Lösen soll nicht farmen
  reach: 320,        // Anteil des verfügbaren Freiraums
  directed: 70,      // Geradlinigkeit (Netto / Pfad)
  headingHold: 28,   // ruhiger Kurs
  meanSpeed: 14,     // überhaupt mobil sein (klein, sonst Zappel-Anreiz)
  efficiency: 3,     // Strecke pro aufgewendeter Energie
  spin: 0.55,        // Strafe für kumulierte Rotation
  drift: 0.05,       // Bewegung ohne Halt ist verschwendete Energie
  preyHit: 25,       // Beute getroffen
  preyNear: 0.06,    // Annäherung an Beute
  energy: 0.30,      // Energieverbrauch
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

export function score(c, opts = {}) {
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

  let s = 0;
  s += (c.gripPullProg || 0) * W.gripPull;
  s += Math.min(c.gripCycles || 0, W.gaitCap) * W.gait;
  s += reachFrac * W.reach;
  s += directed * W.directed;
  s += headingHold * W.headingHold;
  s += Math.min(1, meanSpeed) * W.meanSpeed;
  s += Math.min(20, efficiency) * W.efficiency;
  s -= c.angularAccum * W.spin * 0.02;
  s -= (c.freeMove || 0) * W.drift;
  s -= (c.energySpent || 0) * W.energy;
  s += (c.hits || 0) * W.preyHit;
  if (c.preyMinDist !== undefined && c.preyMinDist < 500) {
    s += Math.max(0, 500 - c.preyMinDist) * W.preyNear;
  }
  return Number.isFinite(s) ? s : 0;
}
