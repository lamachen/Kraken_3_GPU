/**
 * Schritt-Taktgeber für Viewer und Training-Konsole.
 *
 * DOM-frei (wie alles außer view.js) — die Logik ist damit headless prüfbar
 * und die Browserseiten fahren exakt denselben Code (Harness-Checks
 * `pace-realtime`, `pace-multiplier`, `pace-turbo-pause`).
 *
 * Warum es ihn gibt (gemessen, Build 57da2d4): index.html lief de facto immer
 * im Turbo — bis zu 40 Schritte pro Frame (~2400 Schritte/s ≈ 40× Echtzeit).
 * Der Platzhalter-Gang hat eine Periode von 150 Schritten = 2.5 s Sim-Zeit;
 * bei 40× wedeln die Arme mit ~16 Hz, die Netto-Drift von 4–6 px/s (Probe:
 * tools/demogait-probe.mjs) bleibt unsichtbar. Das Tier wirkte wie ein
 * Kreisel. Der Pacer macht das Tempo explizit: 1× = 60 Schritte/s = Echtzeit.
 *
 * Ein Sim-Schritt dauert h × substeps Sim-Zeit (DEFAULTS aus core.js:
 * 1/120 × 2 = 1/60 s). Tempo m heißt deshalb: 60·m Schritte pro Echtsekunde.
 */
import { DEFAULTS } from '../sim/core.js';

export const SIM_DT = DEFAULTS.h * DEFAULTS.substeps;   // 1/60 s pro Schritt

/**
 * @param {object} o
 * @param {number} o.speed        Tempo-Faktor (1 = Echtzeit, Infinity = Turbo)
 * @param {number} o.maxPerFrame  harte Obergrenze Schritte/Frame (Ruckel-Schutz)
 * @param {number} o.turboBudgetMs  Zeitbudget pro Frame im Turbo-Modus
 */
export function createPacer({ speed = 1, maxPerFrame = 40, turboBudgetMs = 12 } = {}) {
  let sp = speed;
  let turbo = !isFinite(sp);
  let paused = false;
  let acc = 0;   // aufgelaufene Schritt-Schuld in Echtzeit

  return {
    /** Tempo setzen; Infinity = Turbo (so schnell das Budget erlaubt). */
    setSpeed(s) {
      sp = s; turbo = !isFinite(s);
      acc = 0;                       // kein Schulden-Burst beim Umschalten
    },
    getSpeed() { return sp; },
    isTurbo() { return turbo; },
    setPaused(p) { paused = !!p; if (paused) acc = 0; },
    isPaused() { return paused; },

    /**
     * Einen Frame anmelden.
     * @param {number} ms vergangene Echtzeit in ms seit dem letzten update()
     * @returns {number} fällige Schritte: ≥ 0 = genau so viele laufen lassen,
     *                   −1 = Turbo-Modus (Aufrufer läuft gegen turboBudgetMs)
     */
    update(ms) {
      if (paused) return 0;
      if (turbo) return -1;
      acc += (ms / 1000) * sp / SIM_DT;
      let n = Math.floor(acc);
      if (n > maxPerFrame) n = maxPerFrame;
      acc -= n;
      // Deckel gegen Aufhol-Bursts nach Tab-Wechseln/Standbild:
      if (acc > maxPerFrame) acc = maxPerFrame;
      return n;
    },

    /** Turbo-Modus: Schritte gegen das Zeitbudget laufen lassen. */
    runTurbo(stepFn, nowFn = () => performance.now()) {
      const t0 = nowFn();
      let n = 0;
      while (nowFn() - t0 < turboBudgetMs) { stepFn(); n++; }
      return n;
    },
  };
}
