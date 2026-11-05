# Post-playtest improvement backlog

Findings from a manual playtest of v2-generated maps. Each case records the
observed symptom, the proposed rule, candidate `config.py` knobs, and notes on
where the change would hook in. Status: **all seven cases implemented** —
see the 2026-11-04/05 updates in `docs/generation-v2.md` and the plans in
`docs/superpowers/plans/`.

## 1. Notes at time zero — start-of-map free zones

**Symptom:** some maps start at or near 0 s with notes. Violation.

**Proposed rule:**

- Note-free zone: the first `2.0` s of the map contains no notes. Any note in
  this window is simply discarded. Lighting events are *not* affected.
- Bomb/obstacle-free zone: the first `1.0` s contains no bombs or walls.

**Candidate knobs:**

- `v2_map_intro_no_notes_seconds = 2.0` (0 disables)
- `v2_map_intro_no_hazards_seconds = 1.0` (0 disables)

**Implementation notes (confirmed against code):**

- All v2 objects live in elapsed seconds until `serialize_beatmap`
  (`map_creation/v2_export.py`) multiplies by `bpm/60` once at the end — a
  plain time-threshold filter, no unit conversion.
- **Recommended note hook:** filter `state["notes"]` in
  `map_creation/v2_pipeline.py` between `snap_note_times` (~line 532) and the
  retained snapshot (~line 534). Dropping there is airtight (the onset snap
  can no longer pull a surviving note below the floor), the retained
  snapshot only contains survivors so the retained-invariant check stays
  intact, and hazards/arcs/calm-effects are generated afterwards and
  automatically validate against the reduced note set. `event_times` stays
  untouched, so lighting keeps its early events — matching the
  "note (not event) free zone" requirement.
- **Recommended hazard hook:** inside `validate_hazards`
  (`map_creation/v2_postprocessing.py` ~lines 1123/1160) — one check covers
  heuristic walls, model-decoded hazards, and calm-effect objects, since
  calm effects re-run the same validator.
- Likely culprit for near-zero notes today: `start_end_fill`
  (`add_start_end_beats = True`) explicitly inserts events before the first
  model event when audio energy is high.
- The turbo-start guidance arc (`slider_turbo_start`) anchors on the first
  note and keeps working when the first note is ≥2 s; its synthetic head is
  an arc object, not a wall/bomb, so it can stay.
- Calm-effect pulses already respect `v2_calm_effects_start_seconds = 1.5`;
  decide whether the new hazard floor supersedes/aligns it.

**Decided:** v2-only. The v1 legacy path (case 7) keeps its own cleanup
unchanged — no new filters there.

## 2. Swing-flow violations — consecutive same-hand cut directions

**Symptom:** sequences that are too hard and don't swing nicely. Observed
combination: (1) note in the middle, cut down-right; (2) note at the left,
cut up-right. Two problems: the second cut does not reverse the first stroke
(it goes to a *different* direction), and the swing's natural follow-through
would go up/up-left — never up-right.

**Desired:** optimal is `down → up`, or diagonals pointing opposite each
other (`down-right → up-left`).

**Current state (confirmed against code):** the clearance fixpoint already
enforces a hard rule (`_flow_legal`, `map_creation/v2_postprocessing.py`
~line 327): consecutive same-hand notes more than
`v2_cleanup_same_dir_gap` (0.1 s) apart must change direction by more than
90°, and exactly 90° is allowed when the hand travels at least two cells —
which the observed middle→left pair satisfies, so nothing is fixed today.
The recut helper (`_flow_direction`/`_rank_flow_candidates`) already prefers
the largest reversal available. Direction encoding: 0=up, 1=down, 2=left,
3=right, 4=up-left, 5=up-right, 6=down-left, 7=down-right, 8=dot; the
observed pair (down-right→up-right) is a 90° turn onto a different axis,
legal only via the two-cell-travel exemption. The legacy v1 flow pass
(`turn_notes_single`/`check_note_movement` in `sanity_check.py`) implements
a stronger movement-implied reversal rule but is disabled for v2
(`v2_cleanup_note_heuristics = False`).

**Decided:** ban all 90° turns. Every same-hand pair separated by more than
`v2_cleanup_same_dir_gap` must change direction by ≥135° — the ≥2-cell
travel exemption is removed entirely (including cardinal 90° "whip" turns).
Same mechanism as today: `_flow_legal` returns False for `steps == 2`, the
fixpoint recuts to a legal direction preferring reversal-like picks;
same-stroke pairs (≤0.1 s) and dots' implied directions are unaffected.
Candidate knob: `v2_cleanup_flow_min_turn_steps = 3` (45° steps; 2 +
exemption reproduces the old rule). Retained model notes keep their current
protection — the recut lands on the later or non-retained note, same as the
existing pass.

## 3. Too many diagonal cuts

**Symptom:** too many diagonal-direction notes in play.

**Proposed:** a postprocessing cap — at most `10%` of notes may have a
diagonal cut direction (dirs 4/5/6/7). Worst case the knob goes to 100% and
disables the pass.

**Decided mechanics:**

- Fraction measured per map over all notes.
- Which diagonals get recut: random selection via the generation RNG —
  deterministic per seed, same as bomb placement.
- Replacement direction: **movement cost first, vertical as tiebreaker.**
  Cost = grid distance the hand must travel from the previous cut's exit
  cell to the candidate's approach cell (the `_dot_virtual_direction`
  geometry). Candidates are restricted to cardinals flow-legal against
  *both* same-hand neighbours under the new ≥135° rule; among legal picks
  the cheapest wins, ties prefer the vertical direction; no previous note
  → all costs equal → vertical default.
- Legality-first: if no cardinal is flow-legal, the diagonal stays — the
  fraction may exceed the cap rather than dropping a note or creating a
  flow violation.

**Candidate knobs:**

- `v2_diagonal_max_fraction = 0.10` (1.0 disables)

**Implementation notes:**

- Diagonals are cut directions 4/5/6/7 (up-left, up-right, down-left,
  down-right); dots (8) are already forced off in production.
- Natural place: inside or right after `enforce_note_flow_and_clearance`,
  before the retained snapshot/multiplier so copies inherit the final
  direction; the post-multiplier fixpoint re-checks legality anyway.

## 4. Density gaps — "5 s on / 5 s off" cycles

**Symptom:** too many note-free areas alternating with full-speed bursts;
the alternation does not flow. Silent areas are still wanted — just shorter
and less binary.

**Proposed:** tune the beat model so it generates more beats overall and
limits quiet sections less harshly.

**Diagnosis (confirmed against code):** likely multiplicative causes —

- The timing model (P(0/1/2 notes) per 10 ms frame, conditioned on
  intensity shares + effective NPS) under-produces events ~3–4× vs
  requested density (recorded ablation); the legacy bisection fills carry
  most of the density — and fills can only bisect gaps ≤ ~1.6 s.
- `density_fill` runs **before** `silence_removal` in `postprocess_timing`:
  fills placed into quiet spans are then deleted by `remove_silent_times`.
- The `onset_snap` step (`sanity_check_timing2`) **deletes** any event more
  than ~1 s from a detected onset — quiet sections with sparse onsets lose
  model events outright.
- `v2_intensity_low_boost = 0.1` lifts quiet bins only 10% toward uniform.
- The timing model's receptive field is ~5.09 s — suspiciously matching the
  observed "5 s on / 5 s off" cycle.

**Existing levers (no code change needed for a first experiment):**

- `v2_intensity_low_boost` (0.1 → try 0.3–0.5): lifts quiet-region
  conditioning share — most direct knob.
- `v2_intensity_scale` (1.0): raises global conditioning NPS.
- `v2_intensity_high_reduction` (0.1), `v2_intensity_adjustment_strength`
  (0.8): shape bursts/blend.
- `add_beat_intensity` (95): app slider "Beat Intensity %"; drives the
  `density_fill` event-count target.
- `silence_threshold` / `silence_thresh_hard` (app slider "Silence
  Threshold %") or `add_silence_flag = False`: softens the quiet-section
  cull. A `timing_no_silence` skip arm already exists in
  `evaluation/evaluate_v2_postprocessing.py` for A/B measurement.

**Decided — all three tracks:**

1. **Config tuning (first experiment):** `v2_intensity_low_boost` ~0.3–0.5,
   optionally `v2_intensity_scale` slightly >1, optionally soften
   `silence_threshold`/`silence_thresh_hard`. Playtest new maps; measure
   per-step deltas with `evaluation/evaluate_v2_postprocessing.py`.
2. **Gap safety net (new postprocess):** `v2_max_note_gap_seconds` — insert
   beats on the BPM grid when a gap exceeds X s and the audio is not
   detected-silent. Deterministic fix for zero-probability dead zones;
   candidate source can later switch to detected onsets (see case 5).
3. **Pipeline fixes:**
   - `density_fill` currently runs before `silence_removal` and gets its
     quiet-span fills deleted — reorder, or tag fill-added events and
     exempt them (needs added/retained tagging either way).
   - `onset_snap` (`sanity_check_timing2`) deletes events >1 s from a
     detected onset — in quiet sections this kills model events outright.
     Fix: unsnapped events keep their model time instead of being dropped
     (at least in low-intensity/detected-non-silent regions).

## 5. Music beat/pitch detector as model input

**Idea:** mix a music beat/note/pitch detector into the AI model for more
consistency, without making output monotonous/robotic. User believes one
already exists (possibly improvable or replaceable by a better library).

**Existing detectors (confirmed inventory):**

- The timing model's 82-channel audio input **already includes** an
  onset-strength channel at 100 Hz alongside 80 log-mel + volume
  (`training/beat_models/features.py`) — a detector is literally wired in,
  just not a beat/pitch one.
- `find_bpm.get_file_bpm`: custom whole-song onset-periodicity BPM detector
  (periodogram + harmonic-sum scoring) — supplies the *constant* tempo used
  by timing conditions, `align_beats_on_bpm` (hard-quantizes all events to
  whole beats) and the calm-effects grid. This is probably the "one I
  already created".
- librosa spectral-flux onset peaks: used for event-level snapping
  (`sanity_check_timing2`) and note-level snapping (`snap_note_times`).
- aubio pitch tracker (`beat_prediction/find_beats.py`): legacy-only, not
  used in v2 production.
- Trained TCN beat generator (`tf_beat_gen_*.h5`): exists as a legacy
  model, runs only in `evaluation/legacy_generator.py`.
- **Missing entirely:** chroma/pitch-class features, MFCCs, tempo-curve
  tracking (`librosa.beat.beat_track`), and any beat-*phase*/subdivision
  detector (which of 8th/16th positions is musically active).

**Decided:** implement **both** inference-only approaches for A/B testing;
the retrain/feature-augmentation path is explicitly **out of scope**.

- **(a) Onsets as candidates:** the case-4 `v2_max_note_gap_seconds` safety
  net draws insertion times from detected onsets (existing librosa
  spectral-flux recipe in `_detect_onsets`, `v2_postprocessing.py`) instead
  of — or snapped to — the BPM grid. Musically-timed fills, no new assets.
- **(b) v1 beat TCN as detector:** run `tf_beat_gen_*` at inference
  (`evaluation/legacy_generator.py` already contains the full
  `find_beats` → TCN → threshold path to reuse) and merge its
  high-confidence beats into timing candidates — dead-zone fills or a
  capped union. Learned human placement; needs the model file shipped or
  downloaded on demand (ties into case 7's `model_download` path, or vendor
  just this file).
- Shared structure: both feed the same dead-zone insertion stage, so a
  single knob (`v2_gap_fill_source = "onsets" | "v1_beat" | "bpm_grid"`)
  makes them directly comparable — the "consistency vs. human-feel" trade
  becomes a playtest dial.

## 6. Slower entry after a pause and at song start

**Symptom:** note density resumes at full subdivisions immediately after a
pause (and at map start); entry should ramp slower.

**Proposed:** subdivision fills (2/3/4 divisions inserted between beats) may
only activate `1.0` s (tunable) after a section entry — where an entry is
the first event following a note gap of at least a pause threshold, or song
start. Whole-beat notes remain allowed inside the delay window.

**Candidate knobs:**

- `v2_fill_entry_delay_seconds = 1.0` (0 disables)
- `v2_fill_entry_pause_beats = 4.0` — **decided: beat-relative.** A gap of
  ≥4 beats marks a section entry; adapts to tempo (a rest means the same
  musical thing at any BPM). Song start counts as an entry.

**Implementation notes (confirmed against code):**

- There is no explicit 2/3/4-division representation: "filling" is
  `fill_map_times_scale` bisecting bounded gaps (each pass halves eligible
  gaps, iterated `map_filler_iters` times). No per-section ramp exists
  anywhere — `se_thresh` is only a global first/last-5%-of-events guard,
  and `_start_end_times` does the opposite (adds events at the edges).
- Implementation point: gate `fill_map_times_scale` passes per section —
  allow subdivision depth only when the gap starts ≥ delay after a section
  entry — or add a `section_ramp` entry to `_TIMING_STEPS` between
  `minimum_spacing` and the fills. The fill bounds widen per iteration, so
  blocking a gap in early passes leaves the whole-beat anchor but prevents
  the 1/2→1/4→1/8 cascade inside the window.
- Fill-added times are currently indistinguishable from model times (new
  candidates take the nearest frame's probabilities) — gating inside the
  fill step avoids needing an added/retained tag.
- Song start is covered by case 1's note-free zone for the first 2 s; this
  rule governs the ramp *after* that (and after mid-map pauses).
- Related legacy machinery (disabled for v2 today): `decr_speed_range=30` /
  `decr_speed_val=0.35` ramps max-speed down for the first/last ~30 events,
  but only inside the legacy flow pass.

## 7. v1 models as a selectable legacy option

**Wish:** users can switch between model generations in the app; default
stays v2. v1 models are still on HuggingFace.

**Current state (confirmed against code):**

- The v1 generation path is **complete but dormant**: it lives in
  `evaluation/legacy_generator.py` (aubio `find_beats` → TCN beat gen →
  LSTM mapper → event gen → `create_map`), verified end-to-end on
  2026-10-16 before the move. Only benchmark callers remain, and they stop
  after the timing stage — the full map-producing run is effectively dead
  code in production.
- Model selection is hard-forced to v2 at three chokepoints:
  `main_app.py` (`run_process` sets `use_mapper_selection = "v2_expert"`),
  `main.py`, `main_multi.py` (worker snapshot). The `use_model` argument
  and `inferno_model` env var are accepted but ignored.
- The app **already has** a `### Model Selection` dropdown
  (`main_app.py` ~line 682) — currently `choices=["v2_expert"]`,
  `interactive=False`. The previously released app had exactly the desired
  UI: `choices=["fav_15","pp3_15","easy_15","expert_15"]`, interactive.
- `tools/utils/huggingface.py::model_download` still works: it
  snapshot-downloads `BierHerr/InfernoSaber` at a branch-per-selection
  revision and still validates legacy bundles via the `tf_event_gen_`
  check — so HF download of v1 branches is already supported.
- The v1 lighting model (`fav_15` event gen + vocabulary) is already
  vendored inside the `v2_expert` bundle and loaded per worker.
- Per selection, v1 needs `tf_beat_gen_*`, `tf_model_enc_*`,
  `tf_model_mapper_*`, `tf_event_gen_*` + three vocab pickles;
  `get_full_model_path`/`update_model_file_paths` resolve them from
  `model/<selection>/` — all helpers intact.
- Caveats: v1 writes old-style per-difficulty `1234_*` folders via
  `create_map`, not `12345_*` combined packages — routing needs
  `write_zip`/`shutil_copy_maps` instead of `combine_packages`.
  `legacy_generator.main` calls `exit()` for >1 song. Some v1 UI flags are
  inert under v2 (Add Breaks, the flow-model and block-shifter toggles);
  Left/Right Notes Filling and Emphasize Beats already affect v2 output
  too (mirroring probability; beat-intensity shift and the relaxed legacy
  flow check), with partial/different semantics than under v1.
- Side finding: `docs/development.md` ~line 224 still says the beat
  benchmark calls `gen_beats.main` — it now calls
  `evaluation/legacy_generator.main`. Small doc fix to fold in.

**Decided:**

- Expose **all legacy presets**: `v2_expert` (default) + `fav_15`,
  `pp3_15`, `easy_15`, `hard_15`, `expert_15` — same list the old app had;
  each is already an HF branch.
- v1 runs **inside the existing worker pool** via a v1 task branch (keeps
  multi-song parallelism; `legacy_generator.main`'s `exit()` on multi-song
  gets wrapped per-song).
- v1 output stays v1-style `1234_*` per-difficulty folders, zipped via
  `write_zip`/`shutil_copy_maps` — not combined `12345_*` packages.
- Assets download on demand via `model_download(use_model)` when a
  selection's bundle is missing.

**Work items:**

- Make the model dropdown interactive (`choices` = v2 + all five v1
  presets), pass `use_model` through `run_process` instead of forcing v2
  (`main_app.py`/`main.py`/`main_multi.py` chokepoints), re-derive
  `paths.model_path` via `update_model_file_paths`, call
  `model_download(use_model)` when assets are missing.
- **Dynamic parameter UI (user requirement, UI recently changed):** the
  parameter controls must adapt to the selected model — either shown/hidden
  dynamically on selection change, or at minimum grouped under titles
  marking them v1-only / v2-only / applies-to-all. Known applicability
  split from the code:
  - **Both models:** Add Bombs, Add Obstacles, sliders/arcs, Silence
    Threshold, Beat Intensity %, dot notes, single-notes-only, sporty
    obstacles, jump-speed offset, BS mapping version, Expert auto-create.
  - **Both models, with v2 caveats:** Left/Right Notes Filling
    (`gimme_more_notes_prob` also feeds the v2 mirroring pass under
    `gimme_more_notes_flag`) and Emphasize Beats (`emphasize_beats_flag`
    shifts `add_beat_intensity` in v2 difficulty settings and feeds
    `sanity_check_notes` via the relaxed-flow pass — a partial effect
    with different semantics than under v1). These two sit ungrouped in
    the Parameters tab, not in a v1-only section.
  - **v1-only (inert under v2 today):** Add Breaks, Add Dot
    Notes %, flow model, block shifters, waveform pattern.
  - **v2-only:** Note Sample Temperature and the other `v2_*` decode/cleanup
    knobs.
  - `random_note_map_factor` applies to both but in different roles (v1 map
    variety vs. v2's v1-lighting score weighting).
  - A Gradio `gr.Column(visible=...)` toggle on the dropdown change event
    is the natural mechanism; static section titles are the fallback.
- Route v1 selections to a wrapped `legacy_generator` runner; keep v1
  cleanup/flags untouched (case 1 decision).
- Restore per-selection v1 semantics if needed (`max_speed` list, cleanup
  flags per preset).
- Small doc fix while there: `docs/development.md` still credits
  `gen_beats.main` for the beat benchmark — it's
  `evaluation/legacy_generator.main` now.

---

# Playtest round 2 — v2 fast-section quality (2026-11-05)

Source: playtest of `playtest_maps/v2_expert` + `fav_15` A/B set (8 songs, 5.0 NPS).
Reference map: `12345_BWB - No Sleep` (v2_expert). All four items are v2-only;
v1 fav_15 stays untouched. Status: **all four implemented** — `32f1b4c`
(R2-1), `1e2d6d5` (R2-2), `c7b70f0` + `3147add` (R2-3/R2-4); plan:
`docs/superpowers/plans/2026-11-05-v2-fast-section-quality.md`.

## R2-1: Opening goes full-speed from note ~3

- Symptom: left-hand stream at ~48 ms gaps (≈1/4 beat at detected BPM 309.6)
  starting at t=2.03 s — the fill density arrives before the player has a
  pattern to read.
- Diagnosis: `start_end_fill` (`add_start_end_beats`, `_start_end_times`)
  inserts events at audio-frame resolution before the first model event
  whenever energy exceeds a running threshold. It does NOT consult
  `fill_blocked()` — the `v2_fill_entry_*` gate only wraps
  `high_difficulty_fill`/`density_fill`. The 2 s intro floor then clips only
  the <2 s part of that burst; the remainder survives untouched.
  `_merge_candidates` merges at `min_time_diff`, not `min_beat_time`, so no
  stage re-clamps density after fills.
- Decision: gate fills until 4 consecutive notes exist after an entry
  (count-based ramp, pauses restart the count) — user-specified.
- Implemented (`32f1b4c`): `v2_fill_entry_min_notes = 4` — each entry's
  fill blackout now also runs until the 4th kept candidate after it
  (`_blocked_intervals`, so `high_difficulty_fill`/`density_fill` are both
  gated; fewer remaining candidates gates to map end, `0` disables the
  count gate); `v2_suppress_start_fill = True` — `start_end_fill` drops
  its pre-first-event inserts while the `v2_map_intro_no_notes_seconds`
  floor is active (end segment unchanged). Diagnostics:
  `entry_blocked_windows`, `start_fill_suppressed`.

## R2-2: Fast sections exceed playable movement speed (NPS 5)

- Symptom: the same burst runs 1–2 cells per ~48 ms ≈ 20–40 cells/s.
- Diagnosis: the movement-cost gate EXISTS — v1 `sanity_check_notes` +
  `calc_note_speed` (exit→approach distance ÷ dt + reaction buffer; deletes
  timesteps over `max_speed * get_factor_from_max_speed`, with
  `decr_speed_val=0.35` ramp over the first `decr_speed_range=30` slots).
  But it sits inside `legacy_note_flow`, which v2 skips entirely because
  `v2_cleanup_note_heuristics = False`. v2's own passes
  (flow legality, diagonal cap, clearance) never check travel/time.
- Implemented (`1e2d6d5`): `apply_speed_cap` — a standalone pass after
  multiplier expansion and the post-expansion clearance fixpoint, reusing
  the v1 cost formula (`calc_note_speed`; cap = `max_speed *
  get_factor_from_max_speed`) per hand. Over-cap pairs drop the later
  note (v1 dropped the whole timestep); each deletion re-checks the
  newly-adjacent pair for speed and `_flow_legal` (recut via
  `_flow_direction`, else drop) to a deletion-only fixpoint. Knobs:
  `v2_speed_cap_scale = 1.0` (`<= 0` disables), `v2_speed_cap_start_notes
  = 30` / `v2_speed_cap_start_factor = .35` — the v1
  `decr_speed_range`/`decr_speed_val` start ramp over combined
  chronological positions. Diagnostics under `speed_cap`.

## R2-3: Direction chaos in fast runs

- Symptom: the opening burst cycles d = 3,1,0,6,2,5,2,7,4,0,1 — unreadable
  at speed. Slow-area variety is wanted; fast runs need simpler patterns
  (not strictly up/down — a middle ground).
- Implemented (`c7b70f0` + `3147add`): `apply_fast_direction_rules` — a
  same-hand pair closer than `v2_fast_gap_seconds = .2` (`<= 0` disables
  both fast rules) is a fast pair; runs extend transitively. Each run
  member's allowed set is the predecessor's direction plus its 180°
  reverse, union a ±45° family admitted per note with probability
  `min(1, expected/local)` (`expected` = map-mean `calc_note_speed`,
  `local` = the pair's own cost): the reverse's ±45° neighbours — the
  135°/225° off-reversals, the only ±45° cuts legal between separated
  strokes, added by the `3147add` review widening — plus the
  predecessor's ±45° for same-stroke pairs. Notes outside the set are
  recut to a legal allowed direction; nothing legal → the direction
  stays. The planned `v2_fast_dir_pattern` knob did not ship — the
  speed-adaptive ±45° admission is the middle ground. Runs pre-expansion
  just before `apply_diagonal_cap`. Diagnostics: `fast_pairs`,
  `fast_simplified`, `fast_counter_recuts`, `expected_speed`.

## R2-4: Counter-directional placements in fast runs

- Symptom: note in top row with down-cut, or bottom row with up-cut —
  readable in slow areas, misleading at speed (a brief top-row note reads
  as "reach up").
- Implemented (`c7b70f0`): same pass — `v2_fast_counter_direction =
  "recut"` (`"off"`/`"keep"` disables; no `"drop"` mode shipped) recuts
  fast-run members whose cut opposes their row (y=2 & d in {1,6,7};
  y=0 & d in {0,4,5}) before the general simplify rule, preferring the
  allowed pool then any legal non-counter direction. Slow areas keep the
  cuts — the rule only applies inside fast runs.
