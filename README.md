# Kraken 3 — GPU

Physikalisch simulierte, cephalopodenartige Kreaturen, deren Verhalten **evolviert**
statt programmiert wird. Proof of Concept für die Frage: *was ist für Spiel-KI
überhaupt lernbar und kontrollierbar?*

Vorgänger [`Kraken_v2`](https://github.com/lamachen/Kraken_v2) bleibt als Archiv
stehen und läuft weiter auf [Pages](https://lamachen.github.io/Kraken_v2/).

---

## Warum neu angefangen wurde

Nicht wegen Unordnung, sondern wegen zweier gemessener Deckel, die sich in der
alten Architektur nicht heben ließen:

1. **Lernbudget.** 5307 Gewichte, 8 Individuen × 12 Generationen = 96 Evaluationen
   pro Phase → **55 Gewichte pro Evaluation**. Das ist kein Tuning-Problem, das
   System konnte strukturell nicht konvergieren. Gemessen: Anker-Rate 7 → 12 %
   über 14 Generationen, also praktisch kein Fortschritt.
2. **Durchsatz.** 101 Sim-Schritte/s, CPU-gebunden. Der Speed-Regler brachte
   1.28×, nicht mehr. Ein Curriculum-Durchlauf: 57 Minuten.

Worker-Parallelität hätte 6× gegeben (9:1 — immer noch zu wenig). GPU-Batching
gibt 100–1000× und macht die Frage **beantwortbar**: belastbares „ja, es lernt"
*oder* belastbares „nein, diese Formulierung funktioniert nicht". Beides ist ein
Ergebnis; aktuell ist keines von beiden erreichbar.

## Design-Grundlagen (aus v2 übernommen, dort gemessen)

- **Kein Wasser.** Die Kreaturen bewegen sich auf einer Oberfläche. Keine
  Hydrodynamik, kein Paddeln, kein Auftrieb.
- **Keine Antriebskraft auf den Körper.** Gelenkmomente sind innere Kräfte
  (Actio = Reactio innerhalb der Armkette) und ändern den Gesamtimpuls nicht.
- **Der einzige Antrieb:** Spitze haftet am Untergrund → Muskeln knicken den Arm
  ein → der Arm wird kürzer → der Mantel wird herangezogen. Hält die Spitze
  stattdessen etwas Freies und Leichtes, zieht dieselbe Mechanik das Objekt heran.
- **Haftung ist ein Reflex, kein Schalter.** Das Netz steuert eine stetige
  Haftstärke `a ∈ [0,1]`, kein „an/aus". Angehaftet wird, wenn die Spitze
  relativ zum Untergrund *steht* (Slip-Gate) — der Takt des Gangs ergibt sich
  daraus physikalisch, ohne dass die KI einen Einschalt-Zeitpunkt treffen muss.
- **Begrenzte Haltekraft.** Ein Saugnapf reißt, wenn die echte Kraft die
  Adhäsionsgrenze übersteigt — geprüft über den Lagrange-Multiplikator, nicht
  über eine Dehnungsschwelle. Kein Timer.
- **Keine dedizierten Greifpunkte in der Karte.** Nur Rand und zufälliger
  Korridorraster. Der Saugnapf haftet am Untergrund überall.
- **Keine handgeschriebene Gangkurve.** Ein CPG mit geformter Sinuswelle war in
  v2 eingebaut und hat messbar geschadet: Trend auf die Netto-Distanz
  −9.34 px/Generation *mit* Kurve gegen +5.14 px/Generation *ohne*.
- **Propriozeption ist der entscheidende Eingang.** In v2 brachte sie
  +50.8 Score/Generation gegen +10.4 ohne. Ein Netz, das die Pose der eigenen
  Arme nicht sieht, kann kein Körpermodell aufbauen.

## Architektur

```
src/physics/xpbd.js    XPBD-Solver, flache typisierte Arrays. CPU-Referenz = ORAKEL.
src/world/map.js       Rechteckkarte + analytisches Signed-Distance-Field.
src/body/kraken.js     Körpermodell: Mantel + 6 Arme × 7 Segmente, 1 Saugnapf/Arm.
src/sim/core.js        DOM-freier Simulationskern: Zustand, Propriozeption, Steuerung.
src/gpu/solver.wgsl    WGSL-Port des Solvers (Jacobi, f32, sequentielle Farb-Dispatches).
src/gpu/gpuworld.js    Buffer/Bind-Groups, upload/packControls, step, download/mirrorInto.
src/sim/gputrain.js    Hybrid-Trainer: Physik GPU, Sinne/Netz/ES CPU, Zustandsspiegel.
src/render/            EINZIGE DOM-abhängige Schicht (view.js) + Platzhalter-Gang (demo.js).
tools/harness.js       56 Abnahme-Checks. Auch das Oracle für die WGSL-Portierung.
tools/profile-step.js  Zerlegt den Tier-Schritt in Phasen (µs) — Architektur-Entscheide.
tools/mocktrain.js     Hybrid-Trainer headless ohne Device (Passthrough/f32-Mock).
tools/f32solver.mjs    f32-Solver-Emulation (WGSL-Semantik per-op) — G7-Fehlerklassen.
tools/*-probe.mjs      Messproben: Lyapunov, Injektion, Bruch-Timing, Ordnung, f32-Parität.
```

`src/physics`, `src/world`, `src/body` und `src/sim` enthalten **keine einzige
DOM-Referenz** — Check 9 im Harness erzwingt das. Deshalb läuft derselbe Kern
unverändert im Browser, im Worker und headless; der Worker-/GPU-Split ist kein
Retrofit.

**Buildless.** Kein Framework, kein Bundler, keine `node_modules`. `index.html`
lädt ES-Module direkt. Deployment auf Pages ist ein Upload, kein Build.

### Warum XPBD und keine Physik-Engine

v2 benutzte matter-js. Vier voneinander unabhängige Engine-Eigenheiten haben dort
jeweils Stunden gekostet; alle vier sind hier strukturell ausgeschlossen:

| matter-js-Verhalten | gemessene Folge in v2 |
|---|---|
| `angularStiffness` default 0 → `bodyA.angle -= p` ohne `anglePrev` | Kette mit Kräften=0, Momenten=0, Verletzung=0.00000 explodierte auf `max|v|`=69 px/step, `max|ω|`=95 rad/step im ersten Update |
| `chamfer radius > min(w,h)/2` | degenerierte 1.6-px-Splitter, `inertia` 1.20 statt 10.52, Trägheitsverhältnis 250:1 |
| `constraint.pointA` rotiert nur inkrementell ab `angleA` | Schulteranker um 56.94 px versetzt, wenn die Pose nach `create()` gesetzt wurde |
| `setVelocity` normalisiert auf `_baseDelta` | Einheitenfalle bei Substeps |

XPBD projiziert Positionen und rechnet die Geschwindigkeiten **danach** aus der
tatsächlichen Positionsänderung zurück (`v = (pPred − p)/h`). Projektion und
Impuls können damit nicht auseinanderlaufen, und das Verfahren ist von Natur aus
dissipativ. Check 1 (passives Gleichgewicht) ist die direkte Konsequenz:

```
Passiv, 8 s, kein Antrieb, keine Haftung:
  Pfad          0.00e+0 px      (v2: 964 px)
  Verschiebung  0.00e+0 px
  max|ω|        4.26e-13 rad/s  (Float-Rauschen)
```

### Muskel als Compliance, nicht als Drehmoment

Ein Muskel ist die **Compliance eines Winkel-Constraints** — also Steifigkeit,
direkt von der KI gesteuert. Schlaffer Arm = hohe Compliance = gibt nach.
Gespannter Arm = niedrige Compliance = hält Form und überträgt Zug. Das ist das
Muskel-Hydrostat-Prinzip in seiner natürlichen Formulierung; in v2 musste es über
einen skalierten Drehmoment-Term nachgebildet werden.

Kalibrierung: XPBD löst mit `α̃ = α/h²`, und `α̃` muss in der Größenordnung der
effektiven Winkelmasse `wSum ≈ 0.33` liegen. Mit `h = 1/120` ist `α̃ = 14400/k`,
also `k ≈ 43.600` für `α̃ = wSum`. Der erste Entwurf hatte `k ∈ [8, 400]` —
Muskeln waren damit praktisch inert (Gelenkkorrektur 1e-7 rad pro Iteration).

### Saugnapf-Bruch über echte Kraft

`λ` ist der akkumulierte Lagrange-Multiplikator, `F = |λ|/h`. Der Bruch wird
daran geprüft, nicht an einer Dehnung. Check 4:

```
schwach gestellt (83 N Grenze)  → reißt bei 272 N Spitze
voll gestellt   (900 N Grenze)  → hält unter demselben Zug
```

## Abnahme: `node tools/harness.js`

```
56/56 Checks bestanden
  passiv        exakt in Ruhe            Pfad 0.00e+0 px        (v2: 964 px)
  slack-fixpoint Störung < slack         Δ = 0.00e+0 px, v = 0  (f32-Boden: GPU driftete 1.28 px)
  slack-corrects 25× slack korrigiert    0.050 px → 1.95e-3 px
  mirror-fields GPU→CPU-Spiegel komplett 86 Körper, 180 Constraints, Grip-force = |λ|/h
  pack-controls Steuer-Packing + Anker   lbx/lby-Sync für den Re-Grip mitten in der Episode
  prey-determinism Beute aus Seed identisch (fix: makeEntity zog Math.random())
  gputrain-identical Hybrid = CPU-Trainer Fitness bit-identisch, 2 Gen × 8 Tiere × 120 Schritte
  noise-absorbed Zustandsrauschen zerfällt 1e-3 px/Schritt → maxΔ 3.9e-3 px (kontraktiv)
  order-chaos    Solve-Ordnung gedreht (f64!) maxΔ 35.35 px — Schritt-Parität unerreichbar
  f32-class      f32-Emulation in der Klasse maxΔ 0.78 px, pathLen +0.24 %, NaN 0
  spawn         Pin-Verletzung 0         max 0.000e+0 px        (v2: 56.94 px)
  spawn-motor   Muskelziele = Spawn-Pose
  grip          verankerter Arm zieht    70.6 px = 72 %         (v2: 92…99 %)
  break         schwacher Grip reißt     272 N > 83 N Grenze
  break-hold    starker Grip hält        900 N Grenze
  long-nan      0 NaN / 1200 Schritte
  long-oob      0 außerhalb der Karte                           (v2: 129)
  long-v        peak|v| im Limit         360 px/s               (v2: 4188 px/s)
  long-av       peak|ω| im Limit         18 rad/s               (v2: 3918 rad/step)
  mass          Rumpf : Segment          11.1:1                 (v2: 250:1)
  mass-taper    Taper über den Arm       1.78:1
  determinism   gleicher Seed → identisch, bit-genau
  proprio       42 Eingänge, 29 belegt, 0 NaN, alle normiert     (v2: 0 Gelenkwinkel)
  domfree       Kern ohne DOM-Referenz
  perf-single   4161 Kraken-Schritte/s                          (v2: 101)  = 41×
  perf-linear   88 % Effizienz bei 64× Batch
  map-spawn     20/20 Karten mit voller Armfreiheit
  map-nogrip    keine dedizierten Greifpunkte

  echo-dist     Raymarch 60.8 px bei tatsächlich 60 px  (Fehler 0.80 px)
  echo-latency  Echo verzögert: ETA 0.1360 s gegen 2·d/c = 0.1351 s
  echo-cooldown Refraktärzeit blockt und ist danach wieder frei
  echo-loud     Aussendung ist hörbar (loudness 1.0) — wer ortet, verrät sich
  hear-bearing  Peilung zur Quelle: größter Fehler 0° über 6 Richtungen
  hear-falloff  60 px → 0.625,  140 px → 0.225  (quadratischer Abfall)
  chem-gradient 4/4 Blickrichtungen korrekt zur Quelle aufgelöst
  chem-range    700 px Abstand → peak 0.0000
  light-falloff Quelle 0.850 > 220 px 0.173 > fern 0.050
  light-ambient im Dunkeln nur 0.050 → Sehen ist konstruktionsbedingt intermittierend
  input-dim     85 = 42 Propriozeption + 43 Sinne
  input-finite  0 NaN/Inf über 400 Schritte
  input-range   hart auf ±1.2 geklemmt (Hard-Stop erlaubt Überschwingen bis 1.61)
  input-senses  25/43 Sinneswerte aktiv
```

### Eingangs-Layout (Teil der Schnittstelle — nicht mehr umstellen, ohne Genome zu verwerfen)

```
[0..5]   Schulterwinkel pro Arm        [6..11]  mittlere Krümmung pro Arm
[12..23] Spitze x/y im Rumpfsystem     [24..29] Zuglast am Saugnapf
[30..35] haftet gerade (0/1)           [36..41] Rumpf: Beschleunigung längs/quer,
                                                tragende Arme, Extension, Tempo, ω
[42..65] Echolot, 12 Strahlen × (Distanz, Intensität)
[66..71] Chemorezeptorbogen, 6 Werte   [72..74] Licht links/rechts/gesamt
[75..78] passives Gehör, 4 Quadranten  [79..84] Tastkontakt pro Arm
```

Zwei Harness-Checks waren zunächst selbst falsch geschrieben und sind korrigiert:
der Cooldown-Test maß nach Ablauf der Refraktärzeit (dann ist sie korrekt frei),
und der Gehör-Test las `c.x` vor dem ersten Sim-Schritt (`undefined + 60 = NaN`).

### Performance

| Tiere | Sim-Schritte/s | Kraken-Schritte/s | ms/Schritt |
|---|---|---|---|
| 1 | 4161 | 4161 | 0.24 |
| 8 | 503 | 4027 | 1.99 |
| 64 | 57 | 3652 | 17.5 |

Lineare Skalierung, 88 % Effizienz bei 64× Batch. **Das ist die Voraussetzung für
GPU-Batching:** die Kosten pro Tier dürfen mit der Batch-Größe nicht wachsen.
Der erste Entwurf war hier O(n²) — `w.step()` lief pro Krake statt pro
Sim-Schritt, 64 Tiere kosteten 1083 ms statt 18 ms. Check `perf-linear` fängt
das ab, weil es die gesamte GPU-These zerstört hätte.

Superlinearen Gewinn gibt es erst aus der räumlichen Parallelität des Shaders,
nicht aus CPU-Batching. Erwartet: 10.000–100.000 Kraken-Schritte/s.

## Milestones

1. **[fertig] Kern + Physik.** XPBD-Referenzlöser, Körpermodell, Karte,
   Propriozeption, 22 Abnahme-Checks, Viewer.
2. **[fertig] Sinne.** Echolot per SDF-Raymarching **mit Laufzeit** — kein
   Entfernungssensor, sondern ein Gehör: das Tier sieht die Vergangenheit, nach
   Entfernung gestaffelt. Chemorezeption über einen Rezeptorbogen,
   Photorezeption gegen wandernde Lichtquellen, passives Gehör über vier
   Quadranten mit Richtungsauflösung, Tastkontakt pro Arm. Dazu Entitäten
   (Beute) als Duft- und Geräuschquellen. Eingang 42 → **85 Dimensionen**.
   Im Spiel bleibt das Wellenfeld mit gegenseitigem Hören — dieselbe
   Sensor-Schnittstelle (`write(out, offset)`), zwei Implementierungen.
3. **[fertig: 3a] Gehirn + Evolution.** MLP 105→24→24→20 mit Rückkopplung der
   eigenen Ausgaben (ohne Gedächtnis kein Rhythmus — die handgeschriebene
   Gangkurve ist bewusst entfernt). Natural Evolution Strategies mit gespiegelten
   Perturbationen, nicht CMA-ES (bei 3644 Gewichten wäre die Kovarianzmatrix
   53 MB plus Eigenzerlegung pro Generation). Fitness nach den drei v2-Regeln:
   nur Netto-Fortschritt, lageunabhängig, jeder Term einzeln nicht farmbar.

   **Gemessener Trainingslauf** (32 Tiere, 500 Schritte/Gen, 20 Generationen, 91 s):

   | | Start | Ende | Trend/Gen |
   |---|---|---|---|
   | Anker-Rate | 37 % | **100 %** | +1.42 % |
   | Zug-Strecke | 9.3 px | **22.9 px** | +146 % |
   | Netto-Distanz | 12.8 px | **22.1 px** | +73 % |
   | Score | 224 | 294 | +2.96 |

   3459 Kraken-Schritte/s, 4.5 s pro Generation. **Das erste Verhalten in diesem
   Projekt, das nachweislich gelernt wurde** — in Kraken_v2 kam die Anker-Rate
   über 12 % nie hinaus.

   Bekannte Grenze: `Zugzyklen ≈ 1`. Die ES findet sofort „einmal festhalten und
   nie mehr loslassen"; höhere `gait`-Gewichte ändern das nicht. Der Grund ist
   dynamisch: Loslassen erfordert Adhäsion < 0.04 auf allen sechs Armen
   gleichzeitig, und danach müssen die Spitzen für den Re-Grip unter 150 px/s
   liegen — während der Körper gerade driftet.
4. **[fertig: 3b — 11/11 im Browser validiert] GPU-Portierung.**
   `src/gpu/solver.wgsl` ist ein 1:1-Port des CPU-Lösers (integrieren →
   Constraint-Iterationen → Kollision → Abschluss → Grip-Bruch), `src/gpu/gpuworld.js`
   die Bindings. **`gpu.html` fährt CPU und GPU auf identischen Eingaben im
   selben Browser und vergleicht sie.**

   Bit-Identität ist dabei *kein* gültiges Kriterium und wird nicht behauptet:
   die CPU löst sequentiell (Gauss-Seidel), die GPU parallel ohne Atomics
   (Jacobi). Bei mehreren Constraints am selben Körper ist das Ergebnis
   reihenfolgeabhängig. Validiert wird deshalb getrennt:

   - **EXAKT** — passives Gleichgewicht (Soll: 0.0), Spawn-Verletzung (Soll: 0),
     NaN-Freiheit, GPU-Determinismus, Grip-Bruch-Status
   - **TOLERANZ ±30 %** — grip-pull-Weg, Spitzenwerte innerhalb der Limits

   Statisch ohne GPU prüfbar (`tools/check-wgsl.js`, Check 12 im Harness):
   Struct-Feldanzahl gegen Byte-Stride, Params-Größe gegen Buffer, alle 5
   Entry-Points verdrahtet, 6 Bindings mit korrektem `read-only-storage` vs.
   `storage`, Workgroup-Größe identisch, Substep-Reihenfolge wie im Oracle, und
   alle Solver-Konstanten werden aus dem CPU-World *gelesen* statt dupliziert
   — einschließlich der Totzone (`SLACK_POS`/`SLACK_ANG`, Werte UND Verwendung
   auf beiden Seiten).

   **Messstand Browser-Validierung** (RTX 3070 Ti, Build `1d82d4e`): **11/11**.
   G1 passiv 6.4e-5 px (< 0.05), Wachstum **0.0000** px, maxV **0.0000** px/s;
   G2 Spawn-Verletzung 1e-4 px; G3 Zugweg 70.328 CPU vs. 70.339 GPU (0.015 %);
   G4 Vollast 360.0001 px/s / 18.0 rad/s — die GPU fährt wirklich ans Limit
   (vor der Control-Pfad-Korrektur: 0.1415 = Rauschboden, falsch-positiv);
   G5 deterministisch, identische Hashes 429477292 unter Last;
   G6 Durchsatz 0.3× (N=1) / 18.6× (N=64) / **90.8× (N=512) = 455 111
   Kraken-Schritte/s**. Zwei Befunde aus der 10/11-Runde (`d487532`), beide
   behoben und als Regression gepinnt:

   - **G1 Wachstum 1.2789 px** — kein Race, sondern die numerische Bodenhöhe
     von f32: ulp(position) ≈ 1.7e-4 px bei 1440-px-Koordinaten (G2 maß die
     Spawn-Verletzung nach Upload: 2.0e-4 px — exakt diese Größenordnung).
     Das Rundungszittern wurde über `v = Δp/h` zu 0.1606 px/s Rausch-
     geschwindigkeit und integrierte sich linear. Heilung: **Korrektur-
     Totzone** `SLACK_POS = 2e-3 px`, `SLACK_ANG = 2e-4 rad` in BEIDEN
     Solvern — Ruhezustände sind damit bitexakte Fixpunkte (Harness-Checks
     `slack-fixpoint`: Δ = 0.00e+0; `slack-corrects`: 0.050 px → 1.95e-3 px,
     echte Verletzungen werden weiterhin korrigiert; grip-pull unverändert
     70.6 px, Lern-Checks unverändert grün).
   - **G4 war falsch-positiv** — der Test setzte die Vollast-Motoren auf
     `a.krakens` (CPU-Welt), aber `uploadControls(gpu)` liest die Welt des
     GPU-Handles (`b.world`). Die GPU simulierte ein passives Tier: maxV
     0.1415 px/s ≈ G1-Rauschboden 0.1606 statt 360 am Limit. Behoben:
     Steuerung geht in beide Welten, und das Urteil verlangt jetzt
     ≥ 80 % des CPU-Spitzenwerts — „nur unter Limit" besteht sonst auch
     im Stillstand. G5 lädt die Steuerung seitdem ebenfalls hoch
     (Determinismus unter Muskellast, nicht nur im Ruhezustand).
4. **[fertig: 3c — auf Hardware validiert, G7-Tore neu kalibriert] Hybrid-Trainer — Training auf der GPU.**
   Profil zuerst (`tools/profile-step.js`, pop=32, µs pro Tier·Schritt):

   | Phase | µs | Anteil |
   |---|---|---|
   | Physik (world.step) | 301.9 | **91.3 %** |
   | Netz (forward+decode) | 13.7 | 4.1 % |
   | Propriozeption | 6.2 | 1.9 % |
   | Sinne nachführen (tickSenses) | 5.6 | 1.7 % |
   | Rest (Steuerung, Messwerte, Welt) | 3.4 | 1.0 % |

   → Die Physik wandert auf die GPU (90.8× gemessen), **Sinne/Netz/ES bleiben
   auf der CPU** — sie sind zusammen keine 9 %. Architektur (`src/sim/gputrain.js`):
   nach jedem GPU-Schritt spiegelt `mirrorInto()` den kompletten Zustand
   (alle Körper px/py/vx/vy/ang/av, Constraint-λ, Grip-broken, Grip-force
   = |λ|/h) in dieselbe CPU-Welt zurück. Dadurch laufen Propriozeption, Sinne,
   **der Saugnapf-Reflex** und die Fitness UNVERÄNDERT — keine duplizierte
   Logik, die auseinanderlaufen kann. `uploadControls()` synced jetzt auch
   `lbx/lby`: ohne die würde ein mitten in der Episode neu verankerter Grip
   auf der GPU am ALTEN Anker ziehen.

   Zwei echte Bugs, die der Bau gefunden hat (beide als Check gepinnt):

   - **`makeEntity()` zog `Math.random()`** für heading/seed der Beute —
     zwei `createSim`-Aufrufe mit gleichem Seed waren verschieden. Mit prey=0
     unsichtbar (M3a lief so), mit Beute wäre jede Messreihe nicht
     reproduzierbar gewesen. Gefangen vom Passthrough-Check: Drift ab
     Schritt 85, als die Beute vom deterministischen Fluchten ins Wandern
     wechselte. Fix: Beute zieht aus `sim.rnd`; Check `prey-determinism`.
   - **Constraint-Objekte waren polymorph**: `force` und `color` wurden erst
     nachträglich angehängt → Grips und Motoren bekamen verschiedene Hidden
     Classes, jeder Constraint-Loop lief langsam. Gemessen bei B=512
     (46 080 Constraints): `mirrorInto` 9.27 → 3.5 ms, `packControls`
     2.87 → 2.0 ms pro Schritt. Fix: beide Felder stehen in der Factory.

   Headless-Beweise (ohne GPU-Device, `tools/mocktrain.js`):
   `gputrain-identical` — Passthrough-Backend liefert **bit-identische**
   Fitness zum CPU-Trainer (2 Gen × 8 Tiere × 120 Schritte, inkl. Aufruf-
   Zähler); f32-Mock-Lauf (CPU-Physik + f32-Rundung des Spiegels, 8 Gen ×
   32 Tiere × 500 Schritte) lernt durch die komplette Mirror-Pipeline:
   **Anker 37 → 99 %, best 176.7 → 383.5, Zyklen 0.44 → 2.75** — dieselbe
   Signatur wie M3a.

   Browser-Beweise (gpu.html, RTX 3070 Ti, Build `fb622c8`): **G8**
   Trainer-Durchsatz **20 596 Tier-Schritte/s = 6.8×** der CPU-Kalibrierung
   (3010 Tier-Schritte/s), 512 Episoden in 12.4 s = 24.3 ms/Episode, und
   gelernt wird durch die Mirror-Pipeline: Gen 0 → 1: best 248.5 → 324.1,
   Anker 53 → 77 %, Zyklen 0.65 → 0.93. **G7** (Lockstep-Parität) FAILte
   dagegen mit den aus dem f32-Mock abgeleiteten Toren: maxΔ 32.69 px statt
   < 0.5 px, 4.07 % Grip-Flag-Flips statt < 1 % — die Nachmessung zeigte:
   die Tore kalibrierten die falsche Fehlerklasse.

   **Die zwei Fehlerklassen** (kopflos vermessen: `tools/lyap-`, `inject-`,
   `breaktiming-`, `order-`, `f32parity-probe.mjs` + `f32solver.mjs`; als
   Regression gepinnt: `node tools/harness.js --only chaos`):

   | Störung (150 NN-Schritte, seed 7, pop 8, prey 2) | maxΔ Mantel | pathLen-Δ |
   |---|---|---|
   | Anfangsstörung 1e-7 px (Lyapunov-Probe) | 7e-9 px (zerfällt!) | 0 % |
   | Zustandsrauschen 1e-3 px/Schritt, alle Körper | 0.0042 px | ~0 % |
   | Bruchkraft-Rauschen 1e-2 relativ | 0.1477 px | +0.1 % |
   | f32-Emulation per-op, gleiche Ordnung (3 Seeds) | 0.78 … 19.3 px | ≤ +2.1 % |
   | **Solve-Ordnung gedreht — reines f64!** | **35.35 px** | **+25.3 %** |
   | echte GPU (RTX 3070 Ti) | 32.69 px | **-4.0 %** |

   Die Dynamik ist **kontraktiv gegen Zustandsstörungen** (Rauschen zerfällt),
   aber **chaotisch gegen Solver-Internika** (f32-Arithmetik, Trig-ulp,
   fma-Kontraktion, Solve-Ordnung): Unterschiede IM Solve verschieben
   diskrete Ereignisse — Grip-Bruch, Slip-Gate, das 260-px-Fluchtgate der
   Beute, die Panic-Tempo-Umschaltung (1× ↔ 2.1×) — um ±1 Schritt; danach
   verfolgen zwei gültige Physiken verschiedene Trajektorien (gemessen:
   Beute-Δ bis 35 px als Folge, nicht als Ursache). Eine f64-CPU mit
   gedrehter Constraint-Ordnung weicht im Fitness-Rohwert STÄRKER ab (+25 %)
   als die echte GPU (-4 %): Schritt-Parität ist zwischen zwei gültigen
   Solver-Implementierungen so unerreichbar wie Bit-Identität. Der f32-Mock
   (0.0005 px) hatte nur die Zustandsrunde getestet — die absorbierte
   Klasse — und damit ein Tor kalibriert, das keine gültige GPU-Physik
   bestehen kann. Die Verdrahtung selbst ist sauber: pack/mirror-Offsets
   gegen das WGSL-Struct geprüft, `attachGrip`/`releaseGrip` schreiben nur
   gepackte Felder, `ppx/angP` hat keinen CPU-Verbraucher, Constraint-
   Ordnung ist CPU wie GPU identisch (core.js färbt und sortiert beim Spawn).

   **G7-Redesign** (gpu.html, 18 Urteile): In-situ-Selbstkalibrierung statt
   fester Marken — Kontrolle C = derselbe Trainer mit gedrehter Constraint-
   Ordnung (reines f64, CPU vs CPU) misst die Klassenschranke im selben
   Szenario auf derselben Maschine. Tore: **maxΔ_GPU ≤ 3× maxΔ_Kontrolle** ·
   Grip-Flags ≥ 90 % (Messung: GPU 95.9 %, Kontrolle 96.7 %) · Bruch-
   Ereignisse ±25 % · pathLen ±10 % · NaN exakt. pop 32 statt 8 mittelt das
   Realisierungs-Pech einzelner Tiere aus den Fitness-Rohgrößen heraus.
   Neu im Log: Flip-Richtung (Oracle-only vs GPU-only), 1.-Flip-Schritt,
   Beute-Δ, Control-Statistik — plus **Kopier-Button** für den Gesamtlog.

   Bekannte Grenze (Hebel für 3d): der Readback ist die VOLLWelt
   (4.6 MB/Schritt bei B=512, ~6–10 ms JS-Spiegel-Kosten) — ein kompakter
   `stateOut`-Buffer (54 Floats/Tier statt 258+90/Tier) und gebatchtes NN
   auf der GPU sind die nächsten Faktoren.
5. **Gegner.** Bewohner-Agent im Trainings-Loop mit Lärm, Duftspur und Angriff.
   In v2 war die komplette gegenseitige Wahrnehmung gebaut, stand aber unter
   keinem Selektionsdruck: `stepPhysics` hatte null Referenzen auf den Spieler.

## Lokal

```bash
node tools/harness.js          # 56 Abnahme-Checks
node tools/harness.js --only chaos          # G7-Fehlerklassen (absorbiert vs chaotisch)
node tools/check-wgsl.js       # Shader/Binding-Konsistenz (ohne GPU)
node tools/trainrun.js --gens 20 --pop 32   # CPU-Trainingslauf (~90 s)
node tools/profile-step.js 32 400           # Phasen-Profil des Tier-Schritts
node tools/mocktrain.js --gens 8            # Hybrid-Trainer headless (f32-Mock)
node tools/f32parity-probe.mjs              # f64-Oracle vs f32-Solver-Emulation
python3 -m http.server 8080    # oder: npx serve .   → http://localhost:8080
```

ES-Module brauchen einen Server (CORS bei `file://`). Kein Build, keine
Abhängigkeiten.

Tasten: `1…4` = 1/4/16/64 Tiere · `Leertaste` = Pause · `R` = Karte neu würfeln.
