# TODO

- [ ] Optionaler Quickfix für BPM-Wechsel: Tempoabschnitte aus v2/v3-Maps lesen und Beat-Zeiten abschnittsweise korrekt in Sekunden umrechnen. Bis dahin betroffene Maps weiterhin mit Begründung überspringen.
- [ ] Weitere Import-Sonderfälle mit dem alten Preprocessing abgleichen und belegte Korrekturen mit Regressionstests übernehmen.
- [x] Bekannte Standard-Dateinamenpaare wie `ExpertPlusStandard.dat` / `ExpertPlus.dat` beim neuen Import in beide Richtungen auflösen, ohne Quelldateien umzubenennen.
- [x] Genau die Vorbis-Warnung `Could not update timestamps for discarded samples.` bei der Datensatzvorbereitung ausblenden; andere Meldungen und Dekodierungsfehler erhalten.
