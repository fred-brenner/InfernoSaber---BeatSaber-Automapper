# Playability metrics — evaluation scorer (design, 2026-11-06)

## Intent

Generated v2 maps pass the hard invariants (flow ≥135°, swing-into-standing-note,
same-cell dedup) yet still contain patterns that feel unnormal in play. Four
player-visible impressions need to become measurable scores so that bundles and
generations can be compared against the human baseline on a playability axis:

1. Lateral/diagonal cut overload, especially in fast sections.
2. Notes hidden behind their predecessor (bottom→top rises with short read time).
3. Both hands performing unrelated movements simultaneously.
4. Hand paths crossing — swings invading each other's workspace, breaking the
   two matched "movement waves" human maps produce (flagship metric).

The scorer is an offline evaluation tool. Soft corpus-relative verdicts, never
hard gates: human maps legitimately contain crossovers and mirrored doubles, so
each metric is reported as a percentile within the human reference distribution
at matched note density.

## Non-goals

- No generation-time gate, repair pass, or rejection sampling (later follow-up
  once the numbers are trusted).
- No training-signal use, no lighting, bombs, arcs, or obstacle scoring.
- No single composite score in v1 — a panel of four metrics plus incidents;
  a composite is only meaningful after seeing real distributions.

## Inputs and decoding

All metrics operate on one note list: `(t_seconds, hand, x, y, d)` sorted by
time. `hand` 0 = left (red), 1 = right (blue); `x` ∈ 0..3, `y` ∈ 0..2,
`d` ∈ 0..8. Time is **seconds**, not beats — reaction time and swing physics
are wall-clock; decoders convert using each package's BPM.

- `decode_package_notes(package_dir, difficulty=None) -> (notes, bpm)`:
  reads `Info.dat` (v2) for BPM + beatmap filename, then the beatmap (v2
  `_notes` or v3 `colorNotes`), converting beats → seconds. `difficulty=None`
  picks the highest Standard difficulty listed (canonical order
  Easy<Normal<Hard<Expert<ExpertPlus); explicit name supported. Reuses
  `human_maps`' validation helpers (`_version`, `_check_timing`,
  `_package_file`); `load_reference` itself returns only onset times, so the
  beatmap is re-parsed for cells and directions.
- **Human baseline decodes only the highest difficulty per package** (user
  decision): ~7 curated ExpertPlus-level maps. Generated packages still score
  every listed Standard difficulty — each generated difficulty is a separate
  row judged against the human distribution in its NPS band.
- Same-hand notes within `eps` (0.02s) collapse to their first representative
  — a stack is one physical stroke, and this keeps package input under the
  same contract as the prepared dataset (`note_classes` is one slot per hand
  per event) and the generated decoder's existing dedup. Documented cost: the
  rare same-instant same-hand stack pattern in human maps is reduced to one
  note, slightly understating human density.
- Human baseline (primary): curated `evaluate_beats/bs_map_input` packages —
  the designated comparison set in AGENTS.md. Packages that fail validation
  (tempo changes, offsets, missing files) become skipped rows with reasons,
  like `prepare_evaluation`'s manifest.
- Human baseline (optional, wider quantiles): `--dataset` prepared pipeline_v2
  dataset via `pattern_analysis.decode_human_song` (retained contract).
- Generated input: `--generated` folder of output package dirs (same decoder
  as human packages).
- No audio decoding: `duration` for rate denominators is the note span
  `t_last − t_first`, applied identically to both corpora (avoids an aubio
  dependency and the intro/outro asymmetry between gap-filled generated maps
  and human maps). Maps with <2 notes are skipped.

## Shared geometry

Reuse the established occupancy model (same convention as
`map_creation/v2_postprocessing.py`): approach cell = `pos + step(d)`, exit
cell = `pos − step(d)`, swept cells = in-grid approach/exit. The geometry
helpers (`_DIRECTION_STEP`, `_swept_cells`, `_dot_virtual_direction`) are
**copied locally** — `v2_postprocessing` transitively imports aubio/librosa
through `sanity_check` at module level, which is too heavy for a numpy-only
metric module; the copies are ~30 lines and kept verbatim next to a comment
pointing at the source.

Per-note context, same shape as `_history_context`: same-hand predecessor
`(t, x, y, d_eff)` with dot implied direction resolved through geometry;
per-transition gap `g = t_i − t_{i−1}`.

## Metrics

All metrics are **speed-conditioned**: every impression is explicitly worse at
high density, so contributions are weighted or banded by gap/local rate, and
corpus comparison uses matched NPS bands.

### M1 — lateral load

Cut stress `w(d)`: vertical (0,1) → 0; diagonal (4–7) → 0.5; horizontal (2,3)
→ 1; dot (8) → 0 (free-direction relief). Speed weight `s(g) = max(0, 1 −
g/τ_fast)`, `τ_fast = 0.5s`.

`M1 = Σ w(d_i)·s(g_i) / Σ s(g_i)` over same-hand transitions → [0,1]; the
share of fast reversal work done on non-vertical axes. Report also per
gap-band (≤0.2s, 0.2–0.35s, >0.35s) to expose "only bad when fast".

### M2 — occlusion

For each note `j`, the occluder is the **strictly earlier** (`t_i < t_j`)
note nearest by `Δt` with `|Δx| ≤ 1` and `t_j − t_i ≤ T_vis` (0.4s) — the
note physically in front on the same sightline (any hand; color is
irrelevant to sightlines). Simultaneous partners never occlude: they share
j's depth and are excluded by the strict inequality. Once that nearest
occluder is cut, the sightline is clear for the rest of j's approach, so
older lower notes behind it don't stack penalties. `occl_j = max(0, y_j −
y_i) · (1 − |Δx|/3) · (1 − (t_j − t_i)/T_vis)`.

`M2 = Σ occl_j / N_notes`; report incident count (`occl_j ≥ 0.5`) per minute
and top-k incidents.

### M3 — bimanual divergence

Segment the event stream at gaps > `T_seg` (0.5s). Per segment, per hand:
displacement vectors between consecutive same-hand notes; dominant axis =
top eigenvector of the 2×2 scatter `Σ vvᵀ`. Skip a hand with <2 vectors or
near-isotropic motion (`λ1/(λ1+λ2) < 0.6`).

`divergence(segment) = |sin θ|` between the two hands' axes (unoriented):
same-axis or mirrored-symmetric → 0; orthogonal (V vs H) → 1.
`M3` = mean divergence weighted by `min(nvec_L, nvec_R)`.

### M4 — path interference (flagship)

Swing volume `V_n` = own cell ∪ swept cells(`d_eff`) ∪ travel corridor
(supercover line cells, prev-exit → approach, only when the same-hand gap
`≤ T_reset = 0.4s` — after an idle beat the hand resets, no travel reach).
Known v1 limitation: corridors are straight lines even when the predecessor
connects through an arc (arcs are not parsed); the error is bounded by the
4×3 grid and affects both corpora symmetrically.

For every cross-hand pair `(a left, b right)` with `|Δt| ≤ τ_x` (0.25s):

- `shared = |V_a ∩ V_b|` cells → `severity += shared · (1 − |Δt|/τ_x)`
- crossover: `x_a > x_b` → `severity += (x_a − x_b) · (1 − |Δt|/τ_x)`
- exact mirrored simultaneous pair (`x_a + x_b = 3`, same `y`, `d_b = mirror
  (d_a)`, `|Δt| ≤ eps`) → severity × 0 (the canonical human double is
  intentional, not a defect).

`M4` = Σ severity per map-minute (and per 100 notes). Report top-k incidents
(time, cells, Δt) — for a diagnostic tool the incident list is as valuable as
the scalar. Note: mirrored/ordinary doubles legitimately share swept cells, so
the human baseline is nonzero by construction — discrimination comes from the
rate×severity distribution vs corpus, not from zero.

## Calibration and report

- Compute all four metrics for every human reference map (~7 maps, highest
  difficulty only); group by measured NPS bands (`<3`, `3–5`, `5–7`, `7+`
  notes/second over the map's note span).
- Two complementary verdicts per generated map, both corpus-relative:
  - **Scalar percentile**: the map's metric value vs the distribution of
    per-map human scalars in its matched NPS band, falling back to the
    all-band distribution when a band holds <5 human maps (expected to
    trigger often at this corpus size — the verdict then reads "vs all
    curated maps").
  - **Incident rate** (the robust verdict at this corpus size): pool every
    human per-unit contribution (transitions for M1, notes for M2, segments
    for M3, pair-severities for M4), take the pooled p95 as the "notable
    incident" threshold, then report the fraction of generated units above
    it vs the human fraction. Stays meaningful where per-map distributions
    are thin and matches the "occasional incidents are fine, the rate is
    the issue" semantics.
- Percentile ≥ ~p95 or incident rate ≥ ~3× human = flag; no fixed
  thresholds anywhere.
- Console output mirrors `pattern_analysis` style: human quantile table per
  band, per-map rows (`M1..M4 + percentile`), and a worst-incidents section
  for flagged maps. `--json` writes machine-readable output.
- Include the existing `clearance_violations` residual counts as a sanity row
  when scoring generated packages (hard invariants stay visible next to the
  soft metrics).

## Files

- `evaluation/playability_metrics.py` — pure functions on note lists, numpy
  only, no config import. Small atomic functions per AGENTS.md style.
- `evaluation/playability_report.py` — CLI + decoders + corpus comparison:
  `--human-root <evaluate_beats>` (bs_map_input packages), `--dataset
  <prepared>` (optional second baseline), `--generated <folder>`,
  `--difficulty`, `--json`.
- `tests/test_playability_metrics.py` — synthetic sequences with known
  verdicts: exact mirrored double → M4 ≈ 0; crossed hands → high M4;
  bottom→top short-gap pair → M2 spike; orthogonal-hand window → M3 ≈ 1;
  dense lateral stream → high M1 vs a vertical control.

## Decisions already made (mine to correct)

- Seconds as the time base for all thresholds.
- `bs_map_input` packages are the primary baseline (7 maps, curated); the
  prepared dataset is an optional wider sample.
- Constants (`τ_fast`, `T_vis`, `T_seg`, `τ_x`, `T_reset`, isotropy 0.6) live
  as module constants; tuned after the first corpus run, not config settings.
- Metrics consider colored notes only.
- Mirrored-double discount applies only to exact simultaneous mirrors.
