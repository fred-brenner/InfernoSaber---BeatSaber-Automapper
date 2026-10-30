# TODO

- [ ] Optionaler Quickfix für BPM-Wechsel: Tempoabschnitte aus v2/v3-Maps lesen und Beat-Zeiten abschnittsweise korrekt in Sekunden umrechnen. Bis dahin betroffene Maps weiterhin mit Begründung überspringen.
- [ ] Weitere Import-Sonderfälle mit dem alten Preprocessing abgleichen und belegte Korrekturen mit Regressionstests übernehmen.
- [x] Bekannte Standard-Dateinamenpaare wie `ExpertPlusStandard.dat` / `ExpertPlus.dat` beim neuen Import in beide Richtungen auflösen, ohne Quelldateien umzubenennen.
- [x] Genau die Vorbis-Warnung `Could not update timestamps for discarded samples.` bei der Datensatzvorbereitung ausblenden; andere Meldungen und Dekodierungsfehler erhalten.

- [ ] V2: Legacy Timing-/Noten-Nachverarbeitung anhand kontrollierter Vergleiche schrittweise reduzieren; Rohmodell-Ausgabe, heuristische Ergänzungen und Multiplier getrennt bewerten.
- [ ] V2: Nach Profiling Modellfusion und TensorFlow-native Vorverarbeitung untersuchen. Aktuell bleiben neun Netze pro CPU-Worker geladen; Timing/Encoder sind gebatcht und Aufrufe kompiliert.
- [ ] V2: Slider-Turbo-Start neu prüfen. Er bleibt deaktiviert, weil er einen nicht vorhandenen Noten-Endpunkt erfindet; Arcs dürfen nur bestehende gleichhändige Noten verbinden.
- [x] V2: Repariertes Hazards-Modell technisch validieren, kalibrieren und auf Wunsch übernehmen (2026-10-20, Bombenschwelle 0.97); vollständiges vorheriges Bundle gesichert. Qualitätsprüfung fehlgeschlagen; siehe docs/baseline.md.
- [ ] V2: History-Abhängigkeit des Hazards-Modells: Experiment 2026-10-24 ausgewertet (docs/baseline.md). Arm B (Masking + Korruption 0,5) auf ausdrücklichen Wunsch in `v2_expert` installiert, Schwellen 0,98/0,87; Backup `v2_expert_before_history-repair-20261024-231200`. Live-Evaluation durch Nutzer läuft; danach behalten oder zurückrollen. `start`-Regime F1 ~0,005, Präzision ~1,6 % — nächster Ansatz repräsentationsseitig; kein prozeduraler Fallback.
- [ ] V2: Arc-Modell kann ohne vorherige positive Referenz-Arcs keine passenden Sequenzen beginnen. Beide Reparatur-Arme (Masking, Masking+Korruption) blieben bei 0/358 Validation / 0/66 Benchmark. Zusätzlich zum Exposure-Bias deckelt Datenarmut: nur ~30 Train-Songs / 6 Validation-Songs mit Arcs. Zuerst unbekannte Arc-positive Aufnahmen sammeln (siehe separater Punkt), dann Repräsentation/Datenvertrag neu abstimmen; vorheriges Bundle bleibt als Backup erhalten. Siehe docs/baseline.md.
- [ ] Optional / zurückgestellt: Eigenständiges V2-Lighting ist noch nicht für Produktion implementiert (not yet implemented for production — optional/deferred). Experimentelle Verträge, Trainer und Artefakte bleiben erhalten; Produktion verwendet ausschließlich V1-`fav_15`-Lighting. Eine spätere Übernahme braucht explizite Qualitätsprüfung; siehe docs/sparse-stage-repair.md.
- [ ] Optional / zurückgestellt: Custom-/Chroma-Lighting als eigene unterstützte Repräsentation untersuchen; der experimentelle reparierte portable Basis-Modus bewahrt und meldet ausgeschlossene Kommandos.
- [ ] V2: Unbekannte Aufnahmen mit positiven Arcs für eine saubere End-to-End-Evaluation sammeln. Der Arc-Train/Validation-Overlay enthält vorheriges Upstream-Training; TEST hat nur eine positive Aufnahmegruppe.
- [ ] V2: Arc-Clearance für native Kurven-Handles geometrisch prüfen; aktuell wird die Verbindung zwischen den vorhandenen Endpunkten abgetastet. In-Game-Prüfung bleibt separat.

- [ ] V2: Retained-Encoder-Vorverarbeitung nach Paritätsprüfung batchen. CPU-Probe mit 1.024 echten Ausschnitten: identische Patches und ca. 23 % weniger Spektrogrammzeit; größere Recording-/Edge-Parität und Gesamtlaufzeit separat prüfen.
