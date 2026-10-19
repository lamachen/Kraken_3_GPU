/**
 * Milestone 1 hat noch KEIN Gehirn. Dieser Platzhalter-Controller erzeugt einen
 * Greif-Zug-Gang, damit der Viewer etwas zeigt, das physikalisch Sinn ergibt:
 * Arme strecken → Spitze haftet (Reflex) → Arm einknicken → lösen.
 *
 * Ab Milestone 3 wird das hier durch das evolvierte Netz ersetzt; die
 * Schnittstelle (applyControl mit tone + pro Arm shoulder/curl/adhesion) bleibt.
 */
import { BODY } from '../body/kraken.js';

export function demoControl(c, step) {
  const period = 150;
  const ph = (step % period) / period;
  // asymmetrisch: lange Streckphase, kurze kräftige Zugphase
  const pull = Math.pow(Math.max(0, Math.sin(ph * Math.PI * 2)), 2.2);
  const arms = c.arms.map((arm, j) => {
    // pro Arm phasenversetzt → es hält immer mindestens einer
    const off = ((step + j * period / c.arms.length) % period) / period;
    const p = Math.pow(Math.max(0, Math.sin(off * Math.PI * 2)), 2.2);
    return {
      shoulder: Math.sin(off * Math.PI * 2) * BODY.shoulderLimit * 0.45,
      curl: p * BODY.jointLimit * 0.95,
      // Haftung nur in der Streck-/Setzphase, in der Zugphase voll
      adhesion: p > 0.25 ? 0.55 + p * 0.45 : 0.02,
    };
  });
  void ph; void pull;
  return { tone: 0.55 + 0.4 * pull, arms };
}
