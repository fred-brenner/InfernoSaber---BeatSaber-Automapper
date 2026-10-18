# TODO

- [ ] Optionaler Quickfix für BPM-Wechsel: Tempoabschnitte aus v2/v3-Maps lesen und Beat-Zeiten abschnittsweise korrekt in Sekunden umrechnen. Bis dahin betroffene Maps weiterhin mit Begründung überspringen.
- [ ] Weitere Import-Sonderfälle mit dem alten Preprocessing abgleichen und belegte Korrekturen mit Regressionstests übernehmen.
- [x] Bekannte Standard-Dateinamenpaare wie `ExpertPlusStandard.dat` / `ExpertPlus.dat` beim neuen Import in beide Richtungen auflösen, ohne Quelldateien umzubenennen.
- [x] Genau die Vorbis-Warnung `Could not update timestamps for discarded samples.` bei der Datensatzvorbereitung ausblenden; andere Meldungen und Dekodierungsfehler erhalten.

- [ ] V2: Legacy Timing-/Noten-Nachverarbeitung anhand kontrollierter Vergleiche schrittweise reduzieren; Rohmodell-Ausgabe, heuristische Ergänzungen und Multiplier getrennt bewerten.
- [ ] V2: Nach Profiling Modellfusion und TensorFlow-native Vorverarbeitung untersuchen. Aktuell bleiben neun Netze pro CPU-Worker geladen; Timing/Encoder sind gebatcht und Aufrufe kompiliert.
- [ ] V2: Slider-Turbo-Start neu prüfen. Er bleibt deaktiviert, weil er einen nicht vorhandenen Noten-Endpunkt erfindet; Arcs dürfen nur bestehende gleichhändige Noten verbinden.
- [ ] V2: Seltene Lighting-Ausgaben und fehlende Arc-Anhänge mit echten positiven Fällen verbessern; keine versteckten prozeduralen Fallbacks hinzufügen.
- [ ] V2: Arc-Clearance für native Kurven-Handles geometrisch prüfen; aktuell wird die Verbindung zwischen den vorhandenen Endpunkten abgetastet. In-Game-Prüfung bleibt separat.
