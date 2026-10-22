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
src/render/            EINZIGE DOM-abhängige Schicht (view.js), Platzhalter-Gang (demo.js),
                       Viewer-Takt (pace.js — DOM-frei, Checks pace-*).
tools/harness.js       73 Abnahme-Checks. Auch das Oracle für die WGSL-Portierung.
tools/profile-step.js  Zerlegt den Tier-Schritt in Phasen (µs) — Architektur-Entscheide.
tools/mocktrain.js     Hybrid-Trainer headless ohne Device (Passthrough/f32-Mock).
tools/f32solver.mjs    f32-Solver-Emulation (WGSL-Semantik per-op, 4 Kollisions-Modi).
tools/class-spread.mjs Misst die Solver-Divergenz pro Störvariante — fand den Kollisions-Race.
tools/collprobe.mjs    Misst, was der Race physikalisch anrichtet (verworfene Korrekturen, Rest-Penetration).
tools/compile-wgsl.js  Echter WGSL-Compile (naga als WASM) — optional, in der CI Pflicht.
tools/gaitprobe.mjs    Misst den Gang: Greif-Ereignisse pro Arm gegen das, was score() sieht.
tools/strideprobe.mjs  Skript-Gänge gegen das trainierte Netz — physikalischer Kopfraum.
tools/demogait-probe.mjs Platzhalter gegen Skript-Demo-Gänge — warum legacy weiterläuft.
tools/spawnnoise-probe.mjs Spawn-Lotterie vs. Policy-Signal im ES-Gradienten — fand den Such-Flaschenhals.
tools/theta-eval.mjs     Gespeichertes θ-Genom auf fremder Karte — Generalisierung, nicht Nischenfit.
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
73/73 Checks bestanden
  passiv        exakt in Ruhe            Pfad 0.00e+0 px        (v2: 964 px)
  slack-fixpoint Störung < slack         Δ = 0.00e+0 px, v = 0  (f32-Boden: GPU driftete 1.28 px)
  slack-corrects 25× slack korrigiert    0.050 px → 1.95e-3 px
  mirror-fields GPU→CPU-Spiegel komplett 86 Körper, 180 Constraints, Grip-force = |λ|/h
  pack-controls Steuer-Packing + Anker   lbx/lby-Sync für den Re-Grip mitten in der Episode
  prey-determinism Beute aus Seed identisch (fix: makeEntity zog Math.random())
  gputrain-identical Hybrid = CPU-Trainer Fitness bit-identisch, 2 Gen × 8 Tiere × 120 Schritte
  noise-absorbed Zustandsrauschen zerfällt 1e-3 px/Schritt → maxΔ 3.9e-3 px (kontraktiv)
  order-chaos    CPU-Listenordnung gedreht (f64) maxΔ 27.87 px — die CPU ist ordnungs-chaotisch
  f32-class      f32-Emulation race-frei: maxΔ 0.032 px, pathLen -0.01 %, NaN 0
  coll-race-repro Kollisions-Race reproduziert die GPU-Divergenz: 9.41 px vs 0.032 px race-frei
  coll-race-physics Physik dahinter: 22.8 % Korrekturen verworfen, 7.35× Rest-Penetration
  fit-ratchet    Rattern zahlt nicht zweimal 129.7 px Weg → 11.1 px gutgeschrieben (vorher 30.3)
  fit-static-floor Stillstand lohnt nicht    halten −0.1 vs Gang 186.7 Punkte (vorher 30.9 vs 262.4)
  fit-stride-farm Adhäsions-Tretmühle       102 Greif-Ereignisse → 0 gewertete Schritte
  fit-ranks-travel Score folgt der Distanz   Verhältnis 0.33 vs 0.34 (vorher 0.71 vs 0.34)
  crn-pair-noise Identisches Genom im Paar: mit CRN Δ = 0 exakt, ohne Δ 238.6 Punkte
  crn-spawn-ring CRN-Spawns im Ring 26…118.5 px Wandabstand (gemessen 42.4…105.8), paaridentisch
  color-order-free Ordnung innerhalb einer Farbklasse ist frei: maxΔ 0.00e+0 — GPU-Dispatch exakt
  coll-grouping    groupCollPoints + Shader-Weg = _solveCollisions: 128 Posen, 26 356 Kontakte, Δ = 0
  wgsl-compile     solver.wgsl compiliert wirklich (naga 30 als WASM), nicht nur Textvergleich
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

  pace-realtime  1× = 60 Sim-Schritte/Echtsekunde (vorher: versteckter 40×-Turbo)
  pace-multiplier 16× = 960/s, Frame-Deckel greift
  pace-turbo-pause Turbo-Budgetmodus, Pause, kein Aufhol-Burst nach Standbild
  console-deployed train.html wird deployed und verdrahtet den Hybrid-Trainer
  build-stamps   keine handgebackenen Stempel — __BUILD_SHA__ in allen Seiten
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

   **Gemessener Trainingslauf** (32 Tiere, 500 Schritte/Gen, 20 Generationen, 91 s;
   Fitness-Stand VOR der Revision unten — Score-Zahlen sind nicht vergleichbar):

   | | Start | Ende | Trend/Gen |
   |---|---|---|---|
   | Anker-Rate | 37 % | **100 %** | +1.42 % |
   | Zug-Strecke | 9.3 px | **22.9 px** | +146 % |
   | Netto-Distanz | 12.8 px | **22.1 px** | +73 % |
   | Score | 224 | 294 | +2.96 |

   3459 Kraken-Schritte/s, 4.5 s pro Generation. **Das erste Verhalten in diesem
   Projekt, das nachweislich gelernt wurde** — in Kraken_v2 kam die Anker-Rate
   über 12 % nie hinaus.

   ── Die „bekannte Grenze `Zugzyklen ≈ 1`" war falsch diagnostiziert ──────

   Behauptet war: die ES finde „einmal festhalten und nie mehr loslassen", der
   Grund sei dynamisch (Adhäsion < 0.04 auf allen sechs Armen gleichzeitig,
   danach Re-Grip unter 150 px/s). Gemessen (`tools/gaitprobe.mjs`, 6 Gen ×
   32 Tiere × 500 Schritte, seed 7): im Schnitt haften **2.33 von 6 Armen**,
   **79 %** aller Greif-Ereignisse sind Wechselgriffe, Ø Haltezeit 10.2 Schritte,
   längste 126. Es wird also sehr wohl umgesetzt — nur nie *alles gleichzeitig*,
   und genau das verlangte `gripCycles`: nur **0.46 %** der Schritte haben null
   Anker. Der gait-Term trug **4.2 von 360** möglichen Punkten (1.4 % des
   Scores). Ein toter Gradient, kein statisches Verhalten.

   Beim Nachmessen mit Skript-Gängen (`tools/strideprobe.mjs`: 14 getaktete
   Controller, jeweils frische Trainer mit gleichem Seed ⇒ identische Welt,
   identische 32 Spawns, nur der Controller anders) sind zwei weitere
   Fehlanreize aufgefallen — beide Verletzungen von Regel 1 und Regel 3:

   | Fitness vor der Revision | `wave T=16` | `halten` | `zappeln` | trainiertes Netz |
   |---|---|---|---|---|
   | netto | 28.3 px | 9.1 px | 13.1 px | 16.3 px |
   | Weg | 342 px | 15 px | 228 px | 25 px |
   | „Zug" (`gripPullProg`) | **126.9 px** | 9.8 px | 57.7 px | 12.7 px |
   | Score | **453.9** (Platz 1) | 104.4 | 255.8 | 149.8 |

   • **Rattern war farmbar.** `gripPullProg` summierte jeden Anstieg des
     Spawn-Abstands — also auch den zweiten Vorstoß nach einem Zurückrutschen.
     4.5× mehr „Zug" als Netto-Distanz, und dieser eine Term trug 330 von 454
     Punkten: Der höchste Score der ganzen Sweep-Tabelle gehörte einem Tier,
     das 28 px weit kam.
   • **Stillstand war profitabel.** `directed` (Netto/Weg) und `headingHold`
     sind Verhältnisse ohne Fortschritts-Schwelle und damit durch Nichtstun
     maximierbar. Ein toter Krake scored 104.4, ein zappelnder 255.8 — das
     trainierte Netz nach 6 Generationen ES: 149.8. Eine Policy schlechter als
     ein toter Krake war fitness-optimal.

   **Revision** (`src/sim/fitness.js` + `postStep()`, gepinnt von 5 neuen
   Harness-Checks, `node tools/harness.js --only fitness`):

   1. **Ratchet**: `gripPullProg` zählt nur noch neue Maxima des Spawn-Abstands
      ⇒ `gripPullProg ≤ maxDisplacement`, geprüft über alle 17 Konfigurationen
      der Sonde.
   2. **`progGate`** = `min(1, disp / (0.25 · spawnReach))` multipliziert alle
      Qualitätsterme (`directed`, `headingHold`, `meanSpeed`, `efficiency`).
      Fortschrittsterme und Strafterme bleiben ungetorrt.
   3. **`gait` → `stride`**: `postStep()` zählt Wechselgriffe pro Arm mit
      durchgehender Abstützung (`c.strides` — ein Arm fasst neu, während ein
      anderer ÜBER den Schritt hinweg hält). Multiplikativ auf die
      Fortschrittsbasis: `stride = (gripPull + reach) · 0.5 · min(strides, 60)/60`.
      `scoreTerms()` ist exportiert, damit „Score 418" nicht mehr unkommentiert
      im Raum steht.

   Wirkung auf dieselben Controller (500 Schritte, 32 Spawns):

   | | `wave T=32` | `sync T=16` | `wave T=16` (Ratterer) | `zappeln` | `halten` |
   |---|---|---|---|---|---|
   | netto | 59.3 px | **64.4 px** | 28.3 px | 13.1 px | 9.1 px |
   | Score vorher | 430.3 | 341.9 | **453.9** ← Platz 1 | 255.8 | 104.4 |
   | Score nachher | **380.7** ← Platz 1 | 337.3 | 181.8 | 61.0 | 44.4 |

   Platz 1 nach Score ist jetzt derselbe Gang wie Platz 1/2 nach Distanz, der
   Stillstands-Boden ist von 104.4 auf 44.4 gefallen (bei 200 Schritten: auf
   −0.1), und das Score-Verhältnis zweier Gänge trifft ihr Distanz-Verhältnis
   auf **0.33 gegen 0.34** (vorher 0.71 gegen 0.34).

   **Der stride-Term additiv war eine Falle — gemessen.** Der erste Entwurf
   rechnete `min(strides,60) · 2.2 · progGate` additiv. Ergebnis nach 20
   Generationen: 75 Wechselgriffe pro Episode bei nur 22.7 px netto — die ES
   hatte den leichteren der beiden Wege genommen (Takt statt Reise). Erst
   multiplikativ auf den Fortschritt ist der Term nicht mehr gegen die Reise
   auszuspielen.

   **A/B über drei Seeds** (20 Gen × 32 Tiere × 500 Schritte, prey 2,
   `node tools/trainrun.js --seed …`):

   | | Seed 7 | Seed 13 | Seed 21 | Ø |
   |---|---|---|---|---|
   | netto-px, Fitness **vor** der Revision | 38.2 | 22.2 | 20.9 | **27.1** |
   | netto-px, Fitness **nach** der Revision | 29.9 | 22.3 | 19.4 | **23.9** |
   | Anker-Rate nach der Revision | 89 % | 98 % | 100 % | 96 % |
   | Wechselgriffe/Episode nach der Revision | 67.2 | 76.3 | 33.3 | **58.9** |
   | `gripCycles` (alte, blinde Metrik) vorher | 1 | 10 | 1 | 4 |

   Die Netto-Distanz ist innerhalb der Seed-Streuung gleich geblieben (die
   38.2 px des Seed-7-Laufs sind ein Ausreißer — dieselbe Fitness liefert auf
   Seed 13/21 nur 22.2/20.9 px), der Gang ist neu da: **1 Zyklus → 33–76
   Wechselgriffe pro Episode**. Nebenbefund, der für alle künftigen Messreihen
   gilt: **Ein Einseed-Lauf ist in diesem Projekt keine Evidenz.** Die Streuung
   über drei Seeds (20.9…38.2 px) ist größer als jeder Effekt, den die
   Fitness-Revision hatte.

   Offen blieb damit die **Suche** — und die Messung hat ihren Grund gefunden:
   nicht die fehlende Phasenversetzung, sondern **Spawn-Lotterie im Gradienten**
   (Abschnitt direkt unten). Die handgeschriebene Gangkurve bleibt trotzdem
   draussen (v2: Trend −9.34 px/Gen MIT Kurve gegen +5.14 ohne); die Sonde ist
   ein Messstand, kein Produktbestandteil.

   ── Der Such-Flaschenhals war Spawn-Lärm (CRN-Fix) ────────────────────────

   `resetCreatures` zog JEDER Generation frische Zufalls-Spawns — auch für die
   beiden Hälften desselben Spiegelpaars (θ+σε, θ−σε). Der Paardifferenz-
   Schätzer maß damit vor allem, welche Zufallsposition günstiger lag, nicht
   welches Genom besser ist. Gemessen (`tools/spawnnoise-probe.mjs`, pop 64,
   500 Schritte, seed 1):

   | Bedingung | Score-Spanne | Paardifferenz-sd |
   |---|---|---|
   | identisches Genom, verschiedene Spawns | 21…367 | **46.3 Punkte** = Noise-Floor |
   | ES-Generation (θ=0, σ=0.06) | 0…241 | 45.6 Punkte ≈ Noise-Floor |
   | Signal-Anteil var(B) − var(A) | — | **≈ 0** (SNR 0.000) |

   Der Generation-0-Gradient war reines Rauschen. Fix: **CRN-Spawns**
   (`crnSpawns`, Default): das Spiegelpaar teilt Spawn+Heading, jeder Slot
   behält seinen Spawn über Generationen (Common Random Numbers — die
   Standard-Varianzreduktion für ES, die hier schlicht fehlte). Die
   Spawn-Regel ist ein Ring: `SPAWN_CLEAR (26 px) ≤ Wandabstand ≤ ARM_REACH
   (118.5 px)`. Beide alten Pfade waren falsch: `findSpawn(ARM_REACH+20)`
   setzt das Tier ≥ 138 px von jeder Wand — es ist NICHTS greifbar (pathLen
   2.9 px / 150 Schritte, gemessen am order-chaos-Szenario); der alte
   Fallback `60+rnd·(w−120)` spawnte regelmäßig IN die Wand. Mit CRN:
   Noise-Floor **exakt 0.000000** Punkte, dieselbe σ-Perturbation liefert
   18.6 Punkte Signal pro Paar — gepinnt von den Checks `crn-pair-noise`
   (ohne CRN: Δ 238.6 Punkte bei identischem Genom) und `crn-spawn-ring`.
   gpu.html G7/G8 fahren denselben Trainer — die Absolutzahlen des
   19/19-Protokolls (Build `4b32a00`) stammen vom alten Spawn-Regime; die
   Tore sind relativ (maxΔ, %, ±10 %) und gelten unverändert.

   **130-Generationen-A/B** (pop 64, 500 Schritte/Gen, ~23 min/Lauf auf ~3000
   Kraken-Schritte/s; θ = ES-Mittel, evaluiert auf 64 FRISCHEN Ring-Spawns,
   Generalisierung auf Karten, die das Training nie gesehen hat
   `tools/theta-eval.mjs`; Genome via `trainrun --out`):

   | Lauf | θ netto px (Trainingskarte) | netto px (fremde Karten 777/999) | Anker | Wechselgriffe |
   |---|---|---|---|---|
   | CRN seed 1 | **185.1** | **198.3 / 177.2** | 75 % | 118 |
   | CRN seed 2 | 91.5 | 80.9 / 79.3 | 100 % | 335 |
   | ohne CRN seed 1 | 124.6 | 120.6 / 125.6 | 99 % | 425 |
   | ohne CRN seed 2 | 87.6 | 79.0 / 80.1 | 100 % | 292 |

   Drei Befunde, sauber getrennt:
   1. **Lokomotion ist lernbar.** Alle vier Läufe schlagen den besten
      Skript-Gang auf derselben Aufgabenverteilung deutlich: sync T=32 kommt
      auf 57.4/46.7 px (Karte seed 1/2), wave T=32 auf 46.4/39.8 px
      (`tools/demogait-probe.mjs`, neu auf Ring-Spawns gemessen). Das
      „24–30-px-Plateau" der Fitness-A/B-Messung war ein Artefakt von
      20-Generationen-Läufen — bei 130 Generationen lernt auch der alte Code.
   2. **θ transferiert.** Fremde Karten liegen innerhalb ±12 % der
      Trainingskarte (CRN seed 1: 185 → 198/177 px). Das ist ein Gang, keine
      Nischenanpassung an ein Terrain. `genomes/walker-crn-s1.json` liegt im Repo.
   3. **Der CRN-Effekt aufs Ergebnis ist ein Trend, kein Beweis.** Ø 138.3 px
      (CRN) vs. Ø 106.1 px (ohne), +30 % — aber bei n=2 Seeds pro Arm mit
      überlappenden Spannen (91.5…185.1 vs. 87.6…124.6). Der MECHANISMUS ist
      gepinnt (Noise-Floor 46.3 → 0, Checks), der Lernvorteil bräuchte mehr
      Seeds. Seed-Effekt dominiert beide Arme: seed 1 wird ein „Reisender"
      (Anker 75–99 %, 124–185 px netto), seed 2 ein „Anker-und-Treter"
      (Anker 100 %, ~88 px netto, 292–335 Wechselgriffe).

   Für die Leitfrage — was ist für Spiel-KI lernbar? — ist das die erste
   positive Antwort mit Zahl: ein 3644-Gewichte-Netz, das einen sechsarmigen
   Weichkörper-Kraken 3.2–3.8× weiter trägt als der beste handgetaktete Gang,
   gefunden von 8320 Evaluationen (2.3 Evaluationen pro Gewicht).
4. **[fertig: 3b — 11/11 im Browser validiert] GPU-Portierung.**
   `src/gpu/solver.wgsl` ist ein 1:1-Port des CPU-Lösers (integrieren →
   Constraint-Iterationen → Kollision → Abschluss → Grip-Bruch), `src/gpu/gpuworld.js`
   die Bindings. **`gpu.html` fährt CPU und GPU auf identischen Eingaben im
   selben Browser und vergleicht sie.**

   **Bit-Identität ist kein Kriterium — aber fast.** Der Grund steht seit der
   Race-Messung fest und war vorher nur vermutet: Beide Solver lösen dieselben
   Constraints in derselben Reihenfolge. `core.js` färbt und sortiert beim
   Spawn (`groupConstraintsByColor`), die CPU iteriert die farblich sortierte
   Liste, die GPU dispatched die Farben aufsteigend — identisch. Und
   *innerhalb* einer Farbklasse ist die Reihenfolge **messbar beliebig**
   (Harness-Check `color-order-free`: maxΔ 0.00e+0 px, 0 Grip-Flips), weil
   Farbgruppen körper-disjunkt sind. Der parallele Dispatch pro Farbe ist also
   exakt, nicht nur näherungsweise; übrig bleibt f32 gegen f64.

   Gemessene Restabweichung (race-frei, `tools/class-spread.mjs`, 150 Schritte):
   **0.36 px bei pop 32**, 0.78 px bei pop 8/seed 7, 19.3 px bei pop 8/seed 13
   — nicht Null, aber zwei Größenordnungen unter dem, was der Race produzierte.
   Validiert wird deshalb getrennt:

   - **EXAKT** — passives Gleichgewicht (Soll: 0.0), Spawn-Verletzung (Soll: 0),
     NaN-Freiheit, GPU-Determinismus, Grip-Bruch-Status
   - **TOLERANZ** — grip-pull-Weg ±30 %, Spitzenwerte innerhalb der Limits,
     und in G7 maxΔ < 5 px / Grip-Flags ≥ 99.5 % / Brüche ±10 % / pathLen ±5 %
     / Fitness ±5 % (alle aus der race-freien f32-Emulation abgeleitet, nicht
     aus einer „Klasse")

   Statisch ohne GPU prüfbar — zwei Werkzeuge, weil sie Verschiedenes können:

   - `tools/check-wgsl.js` (Textvergleich): Struct-Feldanzahl gegen Byte-Stride,
     Params-Größe gegen Buffer, alle 6 Entry-Points verdrahtet, 9 Bindings mit
     korrektem `read-only-storage` vs. `storage`, Workgroup-Größe identisch,
     Substep-Reihenfolge wie im Oracle, Solver-Konstanten aus dem CPU-World
     *gelesen* statt dupliziert (einschließlich `SLACK_POS`/`SLACK_ANG`, Werte
     UND Verwendung auf beiden Seiten), und die sechs Race-Invarianten des
     Kollisions-Dispatches (pro Körper statt pro Punkt, `collRange` vorhanden,
     Dispatch `disp(nB)`, Gruppierung über `groupCollPoints()`, ein Rückschrieb
     am Schleifenende).
   - `tools/compile-wgsl.js` (**echter Compile**, naga 30 als WASM): parst,
     validiert und serialisiert den Shader zurück. Kostet einen Browser-Start
     pro vermiedenem Tippfehler — im Race-Fix war es `let cnt` statt `var cnt`
     in einer Schleife, was kein Textvergleich der Welt findet. Optional lokal
     (`npm i --no-save naga-wasm@30.2.0`), Pflicht in der CI.

   **Messstand Browser-Validierung** (RTX 3070 Ti, Build `1d82d4e`): **11/11**.
   G1 passiv 6.4e-5 px (< 0.05), Wachstum **0.0000** px, maxV **0.0000** px/s;
   G2 Spawn-Verletzung 1e-4 px; G3 Zugweg 70.328 CPU vs. 70.339 GPU (0.015 %);
   G4 Vollast 360.0001 px/s / 18.0 rad/s — die GPU fährt wirklich ans Limit
   (vor der Control-Pfad-Korrektur: 0.1415 = Rauschboden, falsch-positiv);
   G5 deterministisch, identische Hashes 429477292 unter Last;
   G6 Durchsatz 0.3× (N=1) / 18.6× (N=64) / **90.8× (N=512) = 455 111
   Kraken-Schritte/s**.

   **Hardware-Messstand nach dem Kollisions-Race-Fix: 19/19 grün (RTX 3070 Ti,
   Build `4b32a00`, Benutzer-Protokoll).** G7 — das zuvor offene Tor — liefert
   auf echtem Silizium die Zahlen der race-freien Emulation: maxΔ **0.3530 px**
   (Ø 0.0085 px, Tor < 5 px), Grip-Flags **100.000 %**, Brüche 26 823 →
   26 823 (±10 %), pathLen 16.89 → 16.89 (**+0.02 %**), Fitness 47.12 → 47.13
   (**0.00 %**), Beute-Δ 0.03 px. Die In-situ-Kontrolle im selben Lauf
   (CPU-Listenordnung gedreht, reines f64) zeigt maxΔ 27.77 px / pathLen
   −7.91 % / Fitness −13.22 % — die Tore diskriminieren, sie sind nicht blind.
   G6: 0.3× (N=1) / 16.9× (N=64) / **72.1× (N=512) = 365 062 Kraken-
   Schritte/s**. G8: GPU-Trainer **23 648.7 Tier-Schritte/s = 4.5×** der
   CPU-Kalibrierung (5270), 512 Episoden in 10.8 s (21.1 ms/Episode), und es
   wird durch die Mirror-Pipeline gelernt: Gen 0 → 1 best 193.7 → 259.7,
   Anker 52 → 78 %, Wechselgriffe 0.1 → 0.6 (revidierte Fitness — Scores sind
   mit den Vor-Revision-Zahlen nicht vergleichbar, Durchsatz und px schon).
   Damit ist der Race-Fix nicht mehr nur headlos bewiesen (`coll-grouping`:
   128 Posen, 26 356 Kontakte, max|Δ| = 0.00e+0) und statisch abgesichert
   (`check-wgsl.js`, `compile-wgsl.js`), sondern auf der Ziel-Hardware
   bestätigt: **der GPU-Solver ist das CPU-Oracle bis 0.35 px.**

   Zwei Befunde aus der 10/11-Runde (`d487532`), beide
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
4. **[fertig: 3c — auf Hardware validiert; 3c-fix: Kollisions-Race im Shader behoben, G7-Tore dadurch wieder scharf] Hybrid-Trainer — Training auf der GPU.**
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
   dagegen: maxΔ 32.69 px statt < 0.5 px, 4.07 % Grip-Flag-Flips statt < 1 %.
   Zwei Deutungsversuche später stand die Ursache fest — ein Schreib-Race im
   Kollisions-Dispatch, kein Numerik- und kein Chaos-Problem (nächster
   Abschnitt). G8-Zahlen sind vor dem Fix gemessen und gelten weiterhin: der
   Race betraf die Kollisionskorrektur, nicht den Durchsatz.

   **Achtung beim Vergleichen mit neuen Logs:** alle Score-, Anker- und
   Zyklus-Zahlen in diesem Abschnitt sind mit der Fitness VOR der Revision aus
   Punkt 3 gemessen. `score()` hat seitdem ein Ratchet, ein `progGate` und
   `stride` statt `gait` — Score-Werte und „Zyklen" sind zwischen den Builds
   nicht vergleichbar, wohl aber der Durchsatz (Tier-Schritte/s) und die
   Netto-Distanz in px. gpu.html meldet seit der Revision zusätzlich
   `Wechselgriffe` pro Tier.

   **Die vermeintlichen „zwei Fehlerklassen" — und was wirklich dahinter war.**
   Build `529fdc7` erklärte die G7-Abweichung mit Chaos: die Dynamik sei
   kontraktiv gegen Zustandsstörungen, aber chaotisch gegen Solver-Internika
   (f32, Trig-ulp, fma, Solve-Ordnung), deshalb seien 32.69 px „Klasse" und
   Schritt-Parität unerreichbar. **Diese Erklärung war falsch**, und der
   Beweis dafür ist eine Zahl: die f32-Emulation (`tools/f32solver.mjs`),
   Operation für Operation aus `solver.wgsl` abgeschrieben, blieb bei
   **0.36 px** (pop 32, seed 7). Wäre f32 die Ursache, hätte sie bei ~22 px
   landen müssen. Die Emulation hatte also etwas nicht nachgebildet.

   `tools/class-spread.mjs` hat die fehlende Variable isoliert — die
   Kollisions-Semantik. `solveCollisions` dispatchte **einen Thread pro
   Kollisionspunkt**, aber der Mantel hat 8 Punkte und jedes Armsegment 2
   (`src/body/kraken.js`): mehrere Threads lasen dasselbe `bodies[i]`,
   korrigierten und schrieben zurück. Letzter Schreiber gewinnt, die übrigen
   Korrekturen gehen verloren. `_solveCollisions()` auf der CPU läuft
   sequentiell und **kumulativ** — der zweite Mantelpunkt sieht die Korrektur
   des ersten. Messung pop 32, seed 7, 150 Schritte:

   | Variante (headlos, `tools/class-spread.mjs`) | maxΔ | Grip-Flags | pathLen | Fitness |
   |---|---|---|---|---|
   | f32, Kollision in CPU-Ordnung | **0.36 px** | 100.00 % | **+0.01 %** | **-0.01 %** |
   | f32 + Race, letzter Punkt gewinnt | 29.77 px | 99.44 % | -14.71 % | -14.77 % |
   | f32 + Race, erster Punkt gewinnt | 51.10 px | 99.35 % | -21.83 % | -20.39 % |
   | f32 + Jacobi (alle Deltas summiert) | 30.54 px | 99.27 % | -15.34 % | -11.13 % |
   | echte RTX 3070 Ti (G7, Build 529fdc7, MIT Race) | 22.02 px | 99.35 % | -14.80 % | — |
   | **echte RTX 3070 Ti (G7, Build 4b32a00, NACH Fix)** | **0.35 px** | **100.00 %** | **+0.02 %** | **0.00 %** |

   maxΔ, Grip-Flags und pathLen sind Physik und blieben über die Fitness-
   Revision exakt gleich; die Fitness-Spalte ist mit der revidierten Fitness
   neu gemessen (`6d80abb`). Sie folgt jetzt der pathLen-Spalte statt von ihr
   abzuweichen (-14.71 % pathLen → -14.77 % Fitness, vorher -3.27 %): die
   ungetorrten Verhältnis-Terme hatten die Abweichung vorher verdünnt.
   Seed 13 zur Einordnung derselben race-freien Emulation: 1.09 px, 98.95 %
   Flags, +0.23 % pathLen, +0.49 % Fitness — die G7-Tore sind auf Seed 7
   kalibriert, weil gpu.html G7 mit Seed 7 läuft.

   Die Race-Zeile trifft die Hardware-Messung (pathLen -14.71 vs -14.80 %,
   Flags 99.44 vs 99.35 %), race-frei bleibt **40× darunter**. Es war also
   kein Chaos, sondern ein Daten-Race — laut WebGPU-Spezifikation
   undefiniertes Verhalten bei überlappenden Writes ohne Atomics oder Barriere.
   Die Nachmessung desselben Tors auf Build `4b32a00` bestätigt den Fix auf
   derselben Hardware: **0.35 px / +0.02 %** statt 22.02 px / −14.80 %
   (19/19-Protokoll, Abschnitt oben).

   **Was der Race physikalisch anrichtet** (`tools/collprobe.mjs`, derselbe
   Lauf, Race auf der CPU nachgebaut; Harness-Check `coll-race-physics`):

   | Größe (pop 32, seed 7, 150 Schritte) | sequentiell (Fix) | Race (529fdc7) |
   |---|---|---|
   | wirksame Kontaktmeldungen | 60 162 | 87 511 (**+45.5 %**) |
   | davon verworfen | 0 | **24 844 (28.4 %)** |
   | Rest-Penetration pro Schritt (Σ px) | 33.93 | **196.73 (5.80×)** |
   | Grip-Brüche | 9 | 3 |
   | Anker-Rate | 35.3 % | 38.0 % |
   | Ø pathLen | 16.886 | 14.402 (**-14.71 %**) |
   | Ø Fitness `score()` | 122.19 | 118.20 (-3.26 %) |

   Der Race findet *mehr* Kontakte, nicht dieselben: ein zu wenig
   herausgeschobener Körper steht in der nächsten Iteration und im nächsten
   Substep weiterhin im Untergrund und meldet denselben Kontakt immer wieder,
   während 28.4 % der Korrekturen im Schreibkonflikt untergehen. Netto bleibt
   die 5.8-fache Rest-Penetration stehen — die Tiere liegen weicher im Boden,
   der Zugweg schrumpft um 14.71 %.

   Zwei Lehrstücke stehen hier, weil sie sonst verloren gehen: **(1)** Die
   naheliegende Erklärung war falsch. „Weniger Korrektur → mehr Slip → mehr
   Grip-Brüche → kürzerer Pfad" klingt mechanisch und ist es nicht: die Brüche
   gehen von 9 auf 3 *zurück*, die Anker-Rate *steigt*. Ohne `collprobe` wäre
   die falsche Kausalkette im README gelandet. **(2)** Mit Zufallssteuerung
   und 4 Tieren misst dieselbe Probe **null** Kontakte (max. Penetration
   -43 px, die Tiere wandern in der offenen Kartenmitte) — die Kollision wäre
   als „unbeteiligt" durchgegangen. Erst das Trainer-Szenario (NN-gesteuert,
   pop 32, Beute) zeigt 60 162 Korrekturen, also ~0.44 pro Körper und
   Schritt: Bodenkontakt ist der Normalfall dieses Ganges, nicht die Ausnahme.

   **Fix:** ein Thread pro Körper, seine Punkte sequentiell in CPU-Ordnung
   (`solver.wgsl` `solveCollisions`, neue `collRange`-Bindung 8; Punkte werden
   in `gpuworld.js` stabil nach Körper gruppiert hochgeladen). Nebenbei
   billiger: 86 Threads statt 184 pro Kollisions-Dispatch.

   Was von der Chaos-Erklärung **übrig bleibt**, ist enger und präziser
   (Regression: `node tools/harness.js --only chaos`, 5 Checks):

   | Störung (150 NN-Schritte, seed 7, pop 8, prey 2) | maxΔ Mantel | pathLen-Δ |
   |---|---|---|
   | Anfangsstörung 1e-7 px (Lyapunov-Probe) | 7e-9 px (zerfällt!) | 0 % |
   | Zustandsrauschen 1e-3 px/Schritt, alle Körper | 0.0039 px (3.86e-3) | ~0 % |
   | Bruchkraft-Rauschen 1e-2 relativ | 0.1477 px | +0.1 % |
   | Ordnung **innerhalb** einer Farbklasse gemischt | **0.00e+0 px** | 0 % |
   | f32-Emulation per-op, race-frei (seed 7 / 13) | 0.78 px / 19.3 px | +0.24 % / +2.14 % |
   | CPU-**Listen**ordnung gedreht (reines f64!) | 35.35 px | +25.3 % |

   Zustandsstörungen zerfallen (kontraktiv), die Ordnung innerhalb einer
   Farbklasse ist **beliebig** — genau die Freiheit, die der parallele
   Farb-Dispatch der GPU hat, denn Farbgruppen sind körper-disjunkt. Chaotisch
   ist nur die Gauss-Seidel-Listenordnung der CPU über alle Farben gemischt;
   die GPU benutzt diese Ordnung nie (sie dispatched farbweise aufsteigend,
   identisch zu `groupConstraintsByColor`). Bleibt f32 als einzige echte
   Fehlerquelle: 0.4…19 px je nach Seed — und nicht der Race.

   **G7 jetzt** (gpu.html, 20 Urteile): keine Selbstkalibrierung mehr gegen
   eine „Klasse", sondern feste Tore aus der race-freien Emulation —
   **maxΔ < 5 px · Grip-Flags ≥ 99.5 % · Brüche ±10 % · pathLen ±5 % ·
   Fitness ±5 % · NaN exakt**. Fitness (`score()`) ist neu dabei: das ist die
   Größe, nach der die ES tatsächlich selektiert, pathLen ist nur ihr
   Rohbaustein. Der Lauf mit gedrehter CPU-Listenordnung bleibt als
   **Diagnose-Zeile** im Log, nicht mehr als Tor.

   Bekannte Grenze (Hebel für 3d): der Readback ist die VOLLWelt
   (4.6 MB/Schritt bei B=512, ~6–10 ms JS-Spiegel-Kosten) — ein kompakter
   `stateOut`-Buffer (54 Floats/Tier statt 258+90/Tier) und gebatchtes NN
   auf der GPU sind die nächsten Faktoren.
5. **Gegner.** Bewohner-Agent im Trainings-Loop mit Lärm, Duftspur und Angriff.
   In v2 war die komplette gegenseitige Wahrnehmung gebaut, stand aber unter
   keinem Selektionsdruck: `stepPhysics` hatte null Referenzen auf den Spieler.

## Viewer & Training-Konsole

**`index.html` — Viewer.** Der Platzhalter-Gang (`src/render/demo.js`) driftet
netto nur ~4–6 px/s, während die Arme mit einer Periode von 150 Schritten
(2.5 s Sim-Zeit) wedeln. Daraus folgen zwei Präsentations-Entscheidungen,
beide gemessen statt geraten:

- **Tempo ist explizit.** Bis Build `4b32a00` lief der Viewer de facto immer
  im Turbo (bis zu 40 Schritte/Frame ≈ 40× Echtzeit): die Arme wedelten mit
  ~16 Hz, die Drift verschwand — das Tier wirkte wie ein Kreisel, und die
  HUD-Rate zeigte frames/s × Tierzahl unter dem Label „Kraken-Schritte/s"
  (60 = Deckel? nein: Anzeige-Fehler). Jetzt: `src/render/pace.js`
  (DOM-frei, Checks `pace-*`), Standard **1× Echtzeit**, `S` schaltet
  1×/4×/16×/Turbo, und die HUD-Rate zählt echte Sim-Schritte pro
  Echtsekunde.
- **Netto-Bewegung wird sichtbar.** Taste `T`: Mantel-Spur + Spawn-Kreuz;
  HUD: **Netto** (Luftlinie vom Spawn) neben **Weg** (Pfadlänge); Taste `F`:
  Kamera folgt Tier 0 (2× Zoom).

Soll der Platzhalter durch einen „richtigen" Skript-Gang ersetzt werden?
Gemessen (`tools/demogait-probe.mjs`, identische Spawns, seed 7): **nein** —
keiner der Kandidaten läuft langfristig besser.

| Konfiguration | netto px (500 Schr.) | Umdreh. | netto px (1500 Schr.) | Umdreh. | px/Umdreh. (lang) |
|---|---|---|---|---|---|
| **legacy (Platzhalter)** | 32.8 | 0.55 | **96.1** | 1.77 | **54.3** |
| wave T=32 | 47.6 | 1.05 | 58.6 | 3.13 | 18.7 |
| wave T=64 | 33.0 | 0.85 | 64.3 | 2.48 | 25.9 |
| sync T=32 | 52.5 | 3.84 | 36.5 | 12.16 | 3.0 (Kreisel) |
| tripod T=16/32/64 | 22.4–28.4 | 1.18–1.74 | — | — | — |

Kurzfristig führt `sync T=32` (52.5 px), langfristig ist es ein Kreisel
(12.2 Umdrehungen für 36.5 px netto). Der Platzhalter hat mit Abstand den
besten Netto-Weg pro Umdrehung und 100 % Anker-Zeit — er bleibt; die
Präsentation macht seine Drift sichtbar.

Messstandard-Hinweis: die Tabelle oben stammt vom alten Lotterie-Spawn
(seed 7). Neu auf CRN-Ring-Spawns gemessen (seed 1/2, pop 64, 500 Schritte):
legacy 36.1/35.6 px, sync T=32 57.4/46.7 px, wave T=32 46.4/39.8 px —
Reihenfolge unverändert. **Echtes Laufen ist nicht mehr Aufgabe der
Evolution, sondern ihr Ergebnis**: 130-Generationen-Läufe erreichen θ-Werte
von 87.6…185.1 px auf derselben Verteilung — 1.5–3.2× der beste Skript-Gang
(Milestone 3, Abschnitt „CRN-Fix"). Das Sieger-Genom liegt als
`genomes/walker-crn-s1.json` im Repo; ein Viewer-Loader dafür fehlt noch.

**`train.html` — Training-Konsole.** Interaktiver Trainer auf denselben
Modulen wie gpu.html G8 (`src/sim/train.js`, `src/sim/gputrain.js`): Physik
auf der GPU (ein geteiltes WebGPU-Device, automatischer CPU-Fallback mit
Log-Meldung), Sinne/Netz/ES auf der CPU über den Zustandsspiegel.

- **Regler:** Population 16–256, Schritte/Generation 450/900/1800, Seed,
  Beute aus/4. Parameter-Änderung baut den Trainer neu (GPU-World wird
  zerstört und neu angelegt — die Device-Leak-Lehre aus gpu.html).
- **Lauf:** Start/Pause (Leertaste), „1 Generation" (G) läuft bis zum
  Generationsende und pausiert, Reset. Fortschrittsbalken = Schritte der
  laufenden Generation.
- **Telemetrie pro Generation:** Fitness best/Ø (Sparkline), σ, Netto-px
  Ø/Episode, Wechselgriffe Ø, Anker-Zeit Ø, Spin Ø, Geradheit, Tier-
  Schritte/s, Episoden-Zähler. „📋 kopieren" liefert Parameter + Verlauf
  (TSV) + Log — dasselbe Format wie der gpu.html-Log-Knopf.
- **🏆 Champion (C):** Zeitlupen-Replay des besten je gefundenen Genoms —
  eigener CPU-Trainer (popSize 2, beide Genome = Champion), 30 Schritte/s
  = 0.5× Echtzeit, Kamera folgt, Spur an. Bewusst derselbe Code-Pfad wie
  das Training (`trainStep`), kein separater Replay-Zweig, der abweichen
  könnte.
- **Kamera:** Überblick (ganze Population) oder folgt dem besten Tier der
  letzten Generation (Index bleibt über den Respawn stabil, weil
  `spawnCache` dieselben Startplätze wiederverwendet).

## Lokal

```bash
node tools/harness.js          # 73 Abnahme-Checks
node tools/harness.js --only chaos          # G7-Fehlerklassen (absorbiert vs chaotisch)
node tools/check-wgsl.js       # Shader/Binding-Konsistenz (Textvergleich, ohne GPU)
npm i --no-save naga-wasm@30.2.0 && node tools/compile-wgsl.js   # echter WGSL-Compile
node tools/trainrun.js --gens 20 --pop 32   # CPU-Trainingslauf (~90 s)
node tools/trainrun.js --gens 130 --pop 64 --seed 1 --out theta.json  # Siegerlauf-A/B (~23 min)
node tools/spawnnoise-probe.mjs             # Spawn-Lotterie vs. Policy-Signal (~45 s)
node tools/theta-eval.mjs --theta theta.json --seed 777   # θ auf fremder Karte (~15 s)
node tools/profile-step.js 32 400           # Phasen-Profil des Tier-Schritts
node tools/mocktrain.js --gens 8            # Hybrid-Trainer headless (f32-Mock)
node tools/f32parity-probe.mjs              # f64-Oracle vs f32-Solver-Emulation
node tools/class-spread.mjs --pop 32 --seeds 7,13 --variants ctrl,f32,f32race
node tools/collprobe.mjs                    # Physik des Kollisions-Race (4 s)
node tools/gaitprobe.mjs --gens 6           # Gang: Ereignisse pro Arm vs. Fitness-Sicht (30 s)
node tools/strideprobe.mjs                  # Skript-Gänge vs. trainiertes Netz (110 s)
node tools/demogait-probe.mjs               # Demo-Gang-Kandidaten (27 s; --longonly: 14 s)
node tools/harness.js --only fitness        # die 5 Fehlanreiz-Regressionen der Fitness
python3 -m http.server 8080    # oder: npx serve .   → http://localhost:8080
```

ES-Module brauchen einen Server (CORS bei `file://`). Kein Build, keine
Abhängigkeiten.

Viewer (`index.html`): `S` = Tempo (1×/4×/16×/Turbo) · `T` = Spur · `F` = Kamera
folgt · `1…4` = 1/4/16/64 Tiere · `Leertaste` = Pause · `R` = Karte neu würfeln.
Training-Konsole (`train.html`): `Leertaste` = Start/Pause · `G` = eine
Generation · `C` = Champion-Zeitlupe · `R` = Reset.
