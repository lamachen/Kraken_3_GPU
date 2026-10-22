/**
 * θ-Controller: ein evolviertes Genom fährt ein Tier — exakt der Verdrahtungs-
 * pfad aus trainStep() (src/sim/train.js), nur ohne Trainer/ES-Überbau, damit
 * der Viewer dasselbe Verhalten zeigt, das im Training gemessen wurde.
 *
 * DOM-frei wie der ganze Kern (view.js ist die einzige DOM-Schicht). Der
 * Harness-Check `theta-viewer-parity` pinnt die Bit-Identität: dieselbe Pose,
 * dasselbe Genom → nach 300 Schritten maxΔ = 0 px zwischen theta.js-Pfad und
 * trainStep-Pfad. Jede Abweichung in Reihenfolge oder Puffer-Layout (z. B.
 * rekurrente Eingänge an der falschen Offset-Position) würde hier auffliegen.
 *
 * Der rekurrente Zustand (die 20 Ausgaben von t−1) lebt pro Tier in einer
 * WeakMap und überlebt Episoden-Respawns — genau wie tr.outputs im Trainer,
 * den resetCreatures() ebenfalls nicht löscht.
 */
import { readProprio, readSenses, PROPRIO_DIM, SENSE_DIM } from '../sim/core.js';
import { forwardInto, decode, HIDDEN, OUT_DIM } from '../brain/net.js';
import { N_IN, N_W } from '../sim/train.js';

/**
 * @param {Float32Array|number[]} theta  flaches Genom (N_W Gewichte)
 * @returns {{ theta: Float32Array, control(c: object, sim: object): object }}
 *   control() hat dieselbe Rolle wie demoControl(): liefert die Steuerstruktur
 *   für DEN Schritt; applyControl() ruft der Aufrufer (Viewer-Loop).
 */
export function createThetaControl(theta) {
  const w = theta instanceof Float32Array ? theta : Float32Array.from(theta);
  if (w.length !== N_W) throw new Error(`θ hat ${w.length} Gewichte, erwartet ${N_W}`);
  const states = new WeakMap();

  function stateFor(c) {
    let s = states.get(c);
    if (!s) {
      s = {
        x: new Float32Array(N_IN),                    // Eingangs-Block (Proprio|Sinne|Out t−1)
        out: new Float32Array(OUT_DIM),               // Ausgaben = rekurrenter Zustand
        scratch: new Float32Array(2 * HIDDEN),        // h1|h2
        control: {
          tone: 0.5, emitEcho: false,
          arms: c.arms.map(() => ({ shoulder: 0, curl: 0, adhesion: 0 })),
        },
      };
      states.set(c, s);
    }
    return s;
  }

  return {
    theta: w,
    control(c, sim) {
      const s = stateFor(c);
      const x = s.x;
      readProprio(c, x);
      readSenses(c, x, sim ? sim.entities : []);
      // rekurrente Eingänge: die Ausgaben des vorigen Schritts (Offset wie im Trainer)
      for (let k = 0; k < OUT_DIM; k++) x[PROPRIO_DIM + SENSE_DIM + k] = s.out[k];
      forwardInto(w, x, s.out, s.scratch, N_IN, HIDDEN, OUT_DIM);
      return decode(s.out, s.control, c.arms.length);
    },
  };
}
