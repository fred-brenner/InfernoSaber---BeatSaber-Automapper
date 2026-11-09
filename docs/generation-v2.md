# V2 map generation

The application uses seven v2 stages from the evaluated local
`../Data/model/v2_expert` bundle: intensity, timing, positions, arrows, multiplier,
hazards and arcs, plus its retained encoder. Lighting comes exclusively from
the original v1 `fav_15` model and event vocabulary. Independent v2 lighting is
**not yet implemented for production — optional/deferred**; its experimental
code and artifacts remain available. Existing bundle contents are unchanged.

Settings in `tools/config/config.py` accept repository-relative or absolute paths:

- `v2_model_bundle_path`: `../Data/model/v2_expert`
- `v1_lighting_model_path`: `../Data/model/fav_15/tf_event_gen_2_12_5_5__22_18.h5`
- `v1_lighting_vocabulary_path`: `../Data/model/fav_15/onehot_encoder_events.pkl`

Application preflights validate the seven-stage production projection and require
the v1 lighting assets; they do not require the experimental v2 lighting stage.
Strict eight-stage bundle validation and experimental tools remain available.
Prediction and optional Beat Saber library paths stay separately configured in
`tools/config/paths.py`. No training datasets or historical source paths are
required. Missing assets or incompatible selected production contracts fail
explicitly; v2 bundle hashes, normalization, encoder identity and vocabularies
remain validated. V2 generation does not download models or train them.

## Existing commands

Run from the repository root in the selected CPU environment:

```bash
source /home/frede/.local/share/infernosaber-baseline/env.sh
python main_app.py
```

The existing single-song command and environment variables remain available:

```bash
max_speed=5 inferno_model=fav_15 python main.py
```

`inferno_model` and the Python `main(use_model=...)` argument select the mapper:
`v2_expert` when unset, or one of the five legacy v1 presets (`fav_15`,
`pp3_15`, `easy_15`, `hard_15`, `expert_15`). A v1 preset downloads its model
folder from Hugging Face on first use and runs the unchanged legacy generator —
no v2 postprocessing — writing per-difficulty `1234_<speed>_<name>` folders
that combine into `12345_*` packages exactly like the v2 path. `max_speed=5` on the command line
requests 5 retained colored notes per second under v2 (the legacy difficulty
scale under v1). The Python function's `diff` argument and internal
`config.max_speed` retain the historical factor of four: `diff=20`
means 5 NPS. This internal value still drives legacy cleanup, NJS and jump offset.
`main.py` continues to disable library export in its CLI defaults.

Existing multi-difficulty CLI:

```bash
diff_list='[3,5]' python main_multi.py
```

This retains the CLI's optional library-export behavior and existing worker
count defaults. Programmatic callers can select the count and disable export:

```python
from main_multi import main_multi_par
result = main_multi_par(2, [3, 5], export_results_to_bs=False)
```

A v1 preset runs the same call with `model_selection="fav_15"`; its per-difficulty
`1234_*` folders combine into the same `12345_<song>` folder + ZIP layout as v2
(the intermediates are removed after publication).

Call multiprocessing from a guarded `if __name__ == '__main__':` entry point.
The app, `main_multi`, and `main_multi_par` signatures remain available;
`main_training.py` selects the seven production v2 stages. Explicit experimental
lighting training uses `python -m training.pipeline_v2 train --stage lighting`;
see [training selection](pipeline-v2.md#small-development-launcher).

The app model dropdown lists `v2_expert` (the default) followed by the five
legacy v1 presets; parameter groups that do not apply to the selected model
stay hidden. Importing any main script does not build the
UI, check for updates, or load TensorFlow. Explicit `main_app.build_app()` builds
the UI; `build_app(check_updates=False)` also skips the startup update check.

## Runtime and workers

Each spawned CPU worker receives a serializable settings/path snapshot before
TensorFlow loads. It loads the seven v2 networks, retained encoder and v1 lighting
model/vocabulary once per worker, retains one recording's decoded features,
detected BPM, intensity and encoder latents, and replaces that song cache when
the recording changes. Timing windows and
encoder queries are batched; chronological stages retain bounded preceding
history and indexed note lookups. Model calls use reusable TensorFlow graphs.

A song's difficulties share a worker when there are enough songs. With fewer
songs than workers, difficulty groups split to use the requested parallelism.
Different workers intentionally have separate model and song caches. Memory use
therefore grows with worker count, especially for the retained large lighting
network. Settings in `config.py` are `v2_inference_threads` (2 per worker),
`v2_timing_batch_size` (8), and `v2_compile_inference` (True).

`main_multi_par` returns explicit `results`, `packages`, and `failures`. Only
successful variants from that call may be combined. A failed requested variant
prevents publication of a new combined package for that song. Previously complete
outputs remain intact, but never count as a successful current result. The app
reports failures instead of displaying unconditional success.

Run messages always print to the CLI/Pinokio console in full. The app progress
box only receives milestone lines — setup, `### ETA: ... Time per song: ... ###`
after each completed work group, variant failures, and the final summary — unless
verbose progress is enabled (`config.verbose_level > 1`), which also forwards
per-variant and per-stage worker messages.

## Production stages and controls

The order is audio/BPM → intensity/timing → timing cleanup → positions/arrows →
note cleanup → multiplier → hazards → arcs → package. V1 lighting uses
the cleaned gameplay timing sequence, with audio latents from the current
`FixedEncoder`. It preserves the legacy 16-step block convention and pairs
decoded events with the timestamp prefix. For `N` cleaned gameplay timestamps,
there are `max(floor(N/16) - 1, 0)` calls and 16 times that many events; incomplete
tails and the final complete block are omitted. Songs with fewer than 32 timings
therefore have no lighting events. Each block uses argmax type/value classes and
one-hot history, with the existing random score weighting. This is separate from
the experimental v2 independent schedule and multi-label decoder.

Beat features use the trained 16 kHz contract. The encoder independently uses
its trained 14,800 Hz preprocessing and fixed song normalization. Neither input
builder needs invented note or intensity targets. Fixed BPM remains optional;
otherwise the existing periodicity detector supplies BPM separately from NPS.

Timing cleanup keeps onset matching, filling, silence removal and alignment,
with actual timestamps in seconds. Moving events retains their count; nearby
merges use `min_time_diff`, keep the largest count, and cap it at two. Added events
use the nearest timing frame's stronger one/two-note probability. Zero density
remains empty. Mirroring, single-note mode, flow/movement, breaks, emphasis,
waveform changes and dot controls run before learned expansion. Position and
arrow categories are sampled from the model distributions at
`config.v2_note_sample_temperature` (default `0.9`; `0` restores argmax),
exposed in the app as *Note Sample Temperature*. A
one-note event samples jointly over the 24 hand/cell combinations weighted like
the count-one argmax rule; a two-note event samples each hand independently.
Sampling replaces the former `random_factor` noise mixture for note decoding,
which produced degenerate near-uniform argmax patterns; the setting still feeds
the separate original v1 lighting score weighting through
`config.random_note_map_factor`.

The 2026-11-04 timing/density update reorders the cleanup and gates its
insertions; `_TIMING_STEPS` reads `minimum_spacing`, `high_difficulty_fill`,
`bpm_alignment`, `onset_snap`, `start_end_fill`, `silence_removal`,
`density_fill`, `max_gap_fill` — silence removal now runs before the density
loop so refills cannot undo it, and a max-gap safety net trails it — the
initial/final BPM alignments bookend the sequence. Before the timing argmax the
one/two-note classes receive `v2_timing_event_bias` (default `0.3`, must stay
`< 1` so a fully silent frame can never flip; `0` restores pure argmax), a
playtest-tunable experiment lifting borderline frames into events — raised
from `0.1` after playtests showed the model under-triggering on audible hits
in loud sections (~2x raw event rate on the weakest songs at `0.3`). After map
start and every note-free pause of at least `v2_fill_entry_pause_beats` beats
(default `4.0`; `0` leaves only the map start as an entry), the subdivision fills (`high_difficulty_fill`,
`density_fill`) may not insert for `v2_fill_entry_delay_seconds` (default
`1.0`; `0` disables) — an entry ramp recomputed at each fill site on the
then-current times, extended since 2026-11-05 until the
`v2_fill_entry_min_notes`-th kept candidate after the entry (default `4`;
`0` disables the count gate; map end when fewer remain); model events
already inside the window are kept. `start_end_fill` stays ungated except
its start segment — `v2_suppress_start_fill` (default `True`) drops
inserts before the first event while the `v2_map_intro_no_notes_seconds`
floor is active. Blocked windows and suppressed inserts are counted under
`entry_blocked_windows` and `start_fill_suppressed`. With
`v2_fill_respect_silence` (default `True`) the same fills also skip detected
silent spans (silent frame times merged into spans, gaps under four hops
closed). The timing onset snap zeroes events with no onset peak in reach, and
`v2_onset_snap_unmatched_policy` (default `"non_silent"`) keeps them unless
inside a detected silent span — `"always"` is the legacy drop-all, `"never"`
keeps them all; the `"non_silent"` check keys on `add_silence_flag` alone,
deliberately independent of `v2_fill_respect_silence`. Finally
`v2_max_note_gap_seconds` (default `4.0`; `0` disables — a playtest-tunable
experiment like the bias and the low-intensity boost below) bounds the
largest tolerated note-free gap: `max_gap_fill` inserts candidate times from
`v2_gap_fill_source` — `"onsets"` (default and the proven option; the shared
librosa spectral-flux detector), `"bpm_grid"` (whole-beat lattice) or
`"v1_beat"` (experimental: the trained v1 beat-TCN under
`model/<v2_gap_fill_v1_bundle>/`, default `"fav_15"`; requires a downloaded
v1 bundle, and any failure prints a warning and falls back to `"onsets"`,
recorded under `gap_fill_source_used`; its `K.clear_session()` inside a
cached-predictor worker has never been exercised live) — inset
`v2_gap_fill_edge_seconds` (default `0.5`, floored at `min_beat_time`) from
the bounding events, still silent-span gated (unconditionally — independent
of `v2_fill_respect_silence`) and `min_beat_time` apart. The net is a
dead-zone backstop: it intentionally ignores entry-ramp windows — a gap
beyond the threshold overrides the style gate.

The 2026-11-06 tempo/density round adds three pieces. `v2_bpm_octave_correct`
(default `True`) repairs doubled auto-detected tempos at audio preparation —
the whole-song pulse tracker favours subdivisions, so detections at or above
`v2_bpm_octave_min_bpm` (default `180`) are halved while the onset-envelope
autocorrelation supports it (`ac(2T) > ac(T)`) and no energy sits at the
doubled rate (`ac(T/2)` below `v2_bpm_octave_subbeat_max`, default `0.15` —
the veto keeping genuinely fast songs intact), never below
`v2_bpm_octave_min_result` (default `90`). One correction inside
`V2Runtime.prepare_audio` keeps every consumer consistent — entry pauses,
BPM alignment, grid fills, calm pulses and the exported BPM field; manual
BPM input is never corrected and the detected value is recorded as
`bpm_detected` in song metadata and pipeline diagnostics. The v1 path reads
`get_file_bpm` directly and is untouched. Every insertion channel
(subdivision, start/end, gap-net and coverage fills) keeps the
difficulty-scaled `v2_fill_min_interval_seconds` spacing floor (default
`0.09`, calibrated at `v2_fill_min_interval_nps` `5.0`, inversely scaled with
map NPS and clamped to `v2_fill_min_interval_min_seconds` `0.035` — nps 10
maps get `0.045`), so fills cannot generate sub-playable streams the speed
cap then culls in bulk. And a new `onset_coverage` timing step (last, after
the final alignment so inserts stay on the onset) appends events at strong
detected onsets the model missed: onsets above the
`v2_onset_fill_strength_quantile` (default `0.5`) share of detected-onset
envelope strengths, farther than `v2_onset_fill_window_seconds` (default
`0.12`) from every candidate, gated by detected silence and the fill floor,
counted under `onset_coverage_added`; `v2_onset_fill_flag` (default `True`)
or the step name disables it, and a missing `audio_path` skips it silently.
A second, loudness-gated admission tier opens where an onset sits inside a
loud section — its smoothed-RMS loudness above
`v2_onset_fill_loudness_quantile` (default `0.6`) of all onset loudness —
admitting onsets down to `v2_onset_fill_loudness_strength_quantile` (default
`0.25`); `0` on the loudness quantile disables the tier and admissions are
counted separately under `onset_coverage_loud_admitted`. Coverage inserts
keep the post-entry `v2_fill_entry_delay_seconds` delay but not the
`v2_fill_entry_min_notes` runway, which exists to throttle synthetic
subdivision fills: onset hits are real audio events, and a sparse stream
would otherwise stretch the runway over exactly the loud sections coverage
exists to repair.

Arrows decoding optionally mixes in a second model: when
`v2_arrows_variety` is above `0` the runtime also loads the arrows stage of
`v2_arrows_reference_bundle` (currently the history-corrupted
`v2_expert_noterepair-c50` candidate) and each note decodes with the reference
model with probability `v2_arrows_variety`. Per-note model selection keeps each
model's internal coherence — linear softmax blending flattened disagreeing
regions and overshot side cuts — so the slider interpolates aggregate
statistics directly (`0` = production model only, `1` = reference only).

The shared legacy flow pass (`sanity_check_notes`) is relaxed only for v2 via
`v2_cleanup_*` overrides: `allow_dots` keeps sampled dot notes, `flow_recut`
and `sanity_recut` skip the movement-reversal direction rewrites (which
collapsed the 135°/dot vocabulary), `lane_shift` and `middle_shift` control
the positional block shifters, `updown_shift`/`cross_hand_adjust` gate the
remaining position movers (`shift_blocks_up_down`, `correct_notes_all`), and
`add_dots` skips invented extra dots. The new `shift_blocks_left_right_flag`,
`shift_blocks_middle_flag`, `shift_blocks_up_down_flag`,
`correct_notes_all_flag` and `sanity_direction_recut_flag` defaults preserve
the v1 cleanup unchanged.

As of 2026-10-31 `v2_cleanup_note_heuristics = False` skips the legacy v1 note
mutators entirely — mirroring (`gimme_more_notes`) and the whole legacy flow
pass including its emphasis/break/dot additions (`LEGACY_NOTE_MUTATORS` in
`v2_postprocessing.py`). The controlled comparisons below showed the flow pass
culled ~half of sampled model notes (realized NPS ~0.45× requested at all
tested difficulties) while the structural pass — per-hand dedup, retained
labeling — plus the direction recut and clearance fixpoint already leave zero
residual violations on raw model output; removing the mutators makes realized
NPS track the request (~0.8–1.2×). Set the flag `True` to restore the legacy
cleanup for A/B checks. The `v2_cleanup_*` overrides above now only matter
when the flag is enabled.

Because production skips that legacy pass, `allow_dot_notes = False` is
enforced in the v2 structuring loop instead: a candidate dot whose siblings
all offer legal cuts takes the best non-dot option, and a surviving dot is
recut to its geometry-implied direction (`_dot_virtual_direction` — the
same implied direction the clearance fixpoint uses) or, for a hand's first
note, the legacy `first_note_layer_threshold` vertical fallback. Final notes
never carry direction `8` unless dots are explicitly enabled.

Hard note invariants run as a bounded fixpoint
(`enforce_note_flow_and_clearance`) both after dedup and once more after
multiplier expansion, with retained-model notes contract-fixed whenever an
alternative fix exists:

- no two notes share `(time, hand)` — deduplication keeps the best-flowing
  candidate, and same-hand same-cell duplicates are dropped;
- consecutive same-hand notes separated by more than
  `v2_cleanup_same_dir_gap` seconds must change cut direction by at least
  `v2_flow_min_turn_steps` 45° steps (default `3` = strict ≥135° with no
  travel exemption; `<= 2` restores the legacy rule where a 90° turn is
  allowed when the hand travels at least two cells);
  dots take an implied direction from geometry — the hand rests at the
  previous cut's exit cell, so a dot on the exit ray continues the stroke
  while a dot in the return zone inherits the pendulum reversal, and
  consecutive dots keep alternating. The direction recut runs
  on the flattened stream and on structured notes, preferring reversal-like
  replacements inside the original axis group;
- with `v2_cleanup_cell_clash` enabled no two simultaneous notes may share
  one cell for any hand pair — clashes relocate the off-side, non-retained
  note to the nearest clear cell or drop it as a last resort;
- with `v2_cleanup_swing_clash` enabled none of a note's swing volume —
  the approach cell, its own cell, and the exit cell one grid step along
  the arrow axis (`get_cut_dir_xy` yields the approach vector; the saber
  sweeps `pos + step → pos → pos - step`) — may hold an *uncut* other-hand
  note at the swing instant: struck simultaneously or up to
  `v2_cleanup_swing_window` seconds later (default `.15`; an earlier note
  is already cleared by its own cut). Dots swing along their implied
  direction; a later note stacked on the swinger's own cell can only be
  relocated or dropped, never recut away. Resolution order: recut the
  swinger under the flow rule, then relocate the target, then relocate the
  swinger, then drop the target.

Counts are reported under `note_clearance` (`flow_recuts`,
`cell_relocated`/`cell_dropped`, `swing_recut`/`swing_relocated`/
`swing_dropped`) plus a `residual` block that must read zero for
`same_cell_pairs`, `swing_into_pairs` and `flow_repeats`. Chain heads stay
synced to their anchor's direction and position.

After cleanup, training reduction selects one representative per hand; heuristic
extra notes remain separate. Stable note IDs anchor expansions and arc endpoints.
Invalid/duplicate expansions and collisions are rejected without moving anchors.
Single-note mode suppresses emphasis/mirroring conflicts and multiplier expansion.
V2 supplies all bombs, walls and arcs, filtered by applicable legacy options and
clearance against final notes and chain occupancy. Ordinary arcs require existing
same-hand endpoints; the turbo-start guidance arc ends at each hand's first note
with a synthetic head at `0.3 × t_first` (arc geometry only, never a note).
Current learned
thresholds remain unchanged for legacy bundles; absent arcs are reported, with no
procedural fallback. Repaired production replacement artifacts select versioned
hazard and arc policies. Experimental v2 lighting policies do not affect
production. A candidate bundle may include checkpoint-bound validation
calibration sidecars; runtime verifies their hashes
and contracts before applying thresholds. Stage-specific compatibility keeps
unchanged upstream checkpoints usable. See [repair workflow](sparse-stage-repair.md). These conservative checks do not prove gameplay safety.

The 2026-10-19 arc-only update uses the repaired contract and a calibrated **0.18**
attachment threshold. It was installed at the user's request despite failing
autoregressive positive-case evaluation: **0/358 validation** and **0/66 curated**
arcs recovered. The disposable Insane packages also contain no arcs. Other stage
assets are unchanged, and the previous bundle remains available for rollback.
See the [evaluation and backup paths](baseline.md#arc-only-deployment-and-failed-positive-case-evaluation-2026-10-19).

The 2026-10-31 arcs revert sets `v2_arcs_heuristic = True`: production arcs again
use the pre-model implementation — adjacent same-hand endpoint pairs gated by
`slider_time_gap`, `slider_probability`, `slider_movement_minimum` and the
shared bomb/wall/note clearance in `validate_arcs`. The arcs model checkpoint
stays in the bundle but is not decoded (`v2_arcs_heuristic = False` restores the
model path); three repaired arms (E/G/H) found no usable autoregressive attach
operating point — see [baseline](baseline.md). The legacy turbo start is active
again through `slider_turbo_start`: each hand's first note receives a
`synthetic_head` guidance arc (head spec on the record, one real endpoint), so
the first cut direction is signalled before the map begins.

The 2026-10-31 walls revert sets `v2_walls_heuristic = True`: production walls
again use the pre-model `gen_obstacles` rule generator — obstacles fill spans
where a lane stayed note-free, and `validate_hazards` then applies the legacy
margins, allowed types/positions and note clearance. The hazards model stage
is skipped entirely (bomb decoding was already disabled via
`v2_hazard_bombs_enabled = False`); the checkpoint stays in the bundle and
`v2_walls_heuristic = False` restores model decoding. Live playtests showed
the model holding walls continuously at fixed spots and colliding with notes.

The heuristic path also proposes gameplay bombs (`_heuristic_bomb_candidates`
in `v2_pipeline.py`): each note rolls `v2_bombs_prob` (default `0.25`) and may
place one bomb at its own time in a cell outside the note safe area. That
area is a box — `v2_bomb_note_cell_gap` (default `1`, rings of grid cells
around every swing cell) around every note within `v2_bomb_note_time_gap`
(default `0.8` seconds on either side of the bomb) — so a bomb can never sit
next to a note, not even across the non-cut axis. `validate_hazards` enforces
the same box for every bomb family — heuristic, model-decoded and
calm-effect — on top of its geometry and wall-overlap rejection. Placement
draws from the generation RNG, so proposals are deterministic for a given
seed; notes whose surroundings are fully covered propose nothing.
`add_bombs_flag` (default `True`, surfaced as "Add Bombs" in the
app) gates every bomb family including calm-effect corner bombs, matching
how `add_obstacle_flag` gates all walls; learned hazard decoding still
emits bombs only when `v2_walls_heuristic = False` and
`v2_hazard_bombs_enabled` allow it.

The same-day wall clearance hardening closes two containment gaps: the lane
scan now gives notes off the event grid (onset-snapped notes, multiplier
copies) their own one-note groups — previously invisible, they could be
spanned by proposals — and `validate_hazards` rejects a wall when any note's
full sweep volume (not just its centre cell) intersects the wall box inside
the `obstacle_time_gap` margins, matching the bomb rule. Exported packages
verify zero swing/box overlaps.

The 2026-10-31 onset snap adds `snap_note_times` (`v2_notes_onset_snap = True`):
after note cleanup and before the retained snapshot/multiplier each note may
move onto its nearest librosa onset peak within `v2_notes_onset_snap_ioi_ratio`
(default `0.15`) of its own same-hand inter-onset interval — the smaller gap to
its same-hand neighbours, evaluated per note. A candidate move is rejected when
it would create a same-cell, swing-volume or same-hand-flow conflict against
any other note (either hand) at their current times — the same rules the
clearance fixpoint resolves — or push a chain past the song end; the note then
keeps its model time. Running before the multiplier keeps the retained-notes
invariant intact and lets expansion copies inherit snapped anchor offsets.
`event_times` stay on the timing grid: lighting and the note-free wall spans
still anchor there while shifted notes are protected by `validate_hazards`
containment instead. Diagnostics land under `onset_snap` (`moved`, `blocked`,
`no_onset`, `mean_shift_ms`, `max_shift_ms`); `v2_notes_onset_snap = False`
disables the step for A/B. The detector reuses the legacy librosa
spectral-flux recipe — a nine-method comparison (aubio, superflux, percussive,
RMS, backtrack, uniform control) found it is the consensus peak set and the
best suited for snap targets — with a shorter refractory
(`v2_notes_onset_snap_wait = 4` frames instead of the legacy 8) for coverage;
`thresh_onbeat` stays the peak threshold.

The 2026-11-04 note postprocessing update adds four knobs around that same
post-snap boundary: `v2_map_intro_no_notes_seconds` (default `2.0`) drops
notes starting before its floor and `v2_map_intro_no_hazards_seconds`
(default `1.0`) rejects walls and bombs starting before theirs inside
`validate_hazards` — generated and calm-effect hazards alike, counted under
`rejected_intro` — so the opening seconds stay readable like human maps (`0`
disables each zone); `v2_flow_min_turn_steps` (default `3`) is the strict
≥135° same-hand turn of the clearance invariant above; and
`v2_diagonal_max_fraction` (default `0.10`) caps diagonal cuts (directions
4–7, dots counted in the base) by recutting an RNG-drawn excess to the
cheapest flow-legal adjacent cardinal, vertical on ties; `>= 1` disables the
cap. The note-side filters run after `snap_note_times` and before the
retained snapshot and multiplier, so expansion copies and arcs see final
note state; the hazard
floor runs inside `validate_hazards`. The cap is best-effort — a diagonal
whose adjacent cardinals are all illegal stays, so the realized fraction may
exceed the cap. Diagnostics land under `intro_zones` (`notes_dropped`) and
`diagonal_cap` (`diagonal_before`, `diagonal_after`, `recut`,
`kept_over_cap`, `fraction_after`).

The 2026-11-05 fast-section update adds two note passes around the same
boundary. `apply_fast_direction_rules` runs just before the diagonal cap:
a same-hand pair closer than `v2_fast_gap_seconds` (default `.2`; `<= 0`
disables both fast rules) forms a fast run, inside which each note's
allowed directions are its predecessor's plus the 180° reverse, union a
±45° family admitted per note with probability `min(1, expected/local)`
— `expected` the map-mean `calc_note_speed`, `local` the pair's own cost
— made of the reverse's ±45° neighbours (the 135°/225° off-reversals,
the only ±45° cuts separated strokes may play) plus the predecessor's
±45° for same-stroke pairs; a note outside its set is recut to a legal
allowed direction or keeps its own. With `v2_fast_counter_direction =
"recut"` (`"off"`/`"keep"` disables) run members whose cut opposes their
row (top-row down-family 1/6/7, bottom-row up-family 0/4/5) are recut
first, preferring the allowed pool. `apply_speed_cap` — the v1
`sanity_check_notes` movement gate v2 otherwise skips — then runs twice:
once just before the retained snapshot while every note is still mutable,
and again after multiplier expansion and the post-expansion fixpoint with
`drop_retained=False`, so copies and heuristic notes still drop while
retained notes are never touched. Consecutive same-hand notes over
`ramped * get_factor_from_max_speed(ramped)` scaled by
`v2_speed_cap_scale` (default `2.0`; `<= 0` disables) resolve by dropping a
note — the scale is the reference at `add_beat_intensity` `100`, multiplied
linearly by the app's Beat Intensity field via `speed_cap_scale()` (reads
`add_beat_intensity_orig`, so the per-variant silence/emphasize/sporty
adjustments shift the fill budget only, not the cap): `ramped` is `max_speed` times a linear
`v2_speed_cap_start_factor` (default `.35`) → 1 ramp over the first
`v2_speed_cap_start_notes` (default `30`) combined positions, and the
factor is evaluated on the ramped value exactly like v1; each deletion
re-checks the new adjacency for speed and flow (recut or drop) to a
deletion-only fixpoint. In the post-expansion pass a violating pair whose
later note is frozen resolves through the earlier one — a mutable earlier
note drops on overspeed, and a revealed flow violation recuts or drops it —
so only retained→retained pairs count as skipped
(`speed_cap_retained_skipped` for overspeed, `speed_cap_flow_skipped` for
flow). A recut inside a fast pair keeps the counter family out of its
candidate pool. Right after, `apply_fast_counter_sweep` recuts
counter-directional *non-retained* notes inside fast pairs the
pre-snapshot rules never saw — expansion copies inherit their anchor's
cut — gated on `v2_fast_counter_direction = "recut"` and
`v2_fast_gap_seconds > 0`. Diagnostics land under `fast_directions`
(`fast_pairs`, `fast_simplified`, `fast_counter_recuts`), `speed_cap`
(`speed_cap_dropped`, `speed_cap_recuts`, `speed_cap_value`),
`speed_cap_post_expansion` (same counters plus `speed_cap_retained_skipped`
and `speed_cap_flow_skipped`) and `fast_counter_sweep`
(`fast_pairs_checked`, `fast_counter_recuts`).

The 2026-10-31 calm-effects stage (`v2_calm_effects = True`) fills note-free
stretches the model intentionally leaves empty — a requested behaviour — with
visual pulses instead of notes. `calm_area_pulses` finds gaps between gameplay
objects (notes as points, chains as full spans) of at least
`v2_calm_effects_min_gap_seconds` (default `4.0`) and emits beat-grid pulse
times (`v2_calm_effects_division` per beat, anchored on the bpm grid at zero);
`v2_calm_effects_edge_seconds` (default `1.0`) keeps pulses clear of the
boundary objects so real silence keeps its empty space and the stage never
reaches into played sections, and `v2_calm_effects_start_seconds` (default
`1.5`) additionally keeps the map opening quiet. `add_calm_effects` resolves
a per-gap placement theme — deterministic rotation through `alternate`,
`pairs`, `stagger`, `phrase`, `rest4`, `sparse` and `twin` motifs
(`v2_calm_effects_variety = False` restores the plain alternation) — placing
`v2_calm_effects_wall_beats`-long wall flashes on the edge lanes (`x` 0/3,
`v2_calm_effects_walls`) and bottom-corner bombs opposite the wall
(`v2_calm_effects_bombs`), skipping cells an accepted arc already sweeps;
`validate_hazards` then applies the same legacy margins, allowed types and
note containment as generated walls, so no decorative object can collide with
a note. `add_calm_lighting` merges matching left/right laser flash events
(`v2_calm_effects_lighting`, types 2/3 value 6, both lasers on accent pulses,
none on rest beats) into the retained v1 lighting output, skipping pulse
times a model event of the same type already occupies. Each element toggle
can disable its object family independently, and `add_obstacle_flag` still
gates all walls including decorative ones. Placement is visual only —
nothing forces movement or a cut — and the note stream is never modified;
diagnostics land under `calm_effects` (`gaps`, `pulses`, `walls`, `bombs`,
`lighting_events`, `hazard_clearance`).

The 2026-10-20 hazard-only update also selects the repaired quarter-beat policy.
Its checkpoint-bound bomb threshold is **0.97**; wall geometry remains categorical.
It was installed as requested despite failing the seven-song positive-case screen:
**0/209 supported wall sections** and **0/52 bomb cells** matched. Full CPU sample
packages do contain valid cleared walls, but reference agreement and own-history
robustness remain poor. Other stage assets, including arcs and v1 lighting, are
unchanged. See the [hazard report and rollback backup](baseline.md#hazard-only-deployment-and-cpu-validation-2026-10-20).

## Intensity postprocessing

Generation adjusts the intensity profile before timing inference. The four
`config.py` controls apply only during production generation:

| Setting | Default | Meaning |
| --- | --- | --- |
| `v2_intensity_low_boost` | `0.3` | Fraction of the deficit below uniform density to lift. Experiment value (was `0.1`) — more notes in quiet parts; playtest-tunable. |
| `v2_intensity_high_reduction` | `0.1` | Fraction of the excess above uniform density to trim. |
| `v2_intensity_scale` | `1.0` | Overall target gain; one preserves average conditioning NPS. |
| `v2_intensity_adjustment_strength` | `0.8` | Fraction of the change toward the configured target to apply. |

Boost, reduction and strength must be finite numeric values in `[0, 1]`; scale
must be finite and nonnegative. Invalid settings raise an explicit error.
Uniform shares are each bin's actual duration divided by decoded song duration,
so a partial final bin receives proportionately less mass. For raw shares `s`
and uniform shares `u`, generation computes:

```text
target = s + low_boost * max(u - s, 0) - high_reduction * max(s - u, 0)
target = target / sum(target)
mass = (1 - strength) * s + strength * scale * target
conditioning_nps = requested_nps * sum(mass)
adjusted_shares = mass / sum(mass)
```

Timing and the downstream stages receive the adjusted normalized shares and
effective conditioning NPS. The original requested NPS still controls package
identity and legacy gameplay settings. Zero mass retains the raw normalized
profile with zero conditioning NPS and skips timing prediction. Zero requested
NPS stays empty. Strength zero preserves the original conditioning exactly.

The defaults gently flatten the profile while preserving its average density:
relative intensities `0.5 / 1.5` become `0.60 / 1.40`. With both shaping controls
set to one, the same inputs become `0.9 / 1.1` at strength `0.8`. Scale `1.25`
with strength `0.8` raises average conditioning NPS by 20%. These changes affect
conditioning; the timing decoder, gameplay cleanup and note expansion still
determine the actual emitted and exported NPS.

Workers cache the raw model shares and reapply the current controls for every
generation call, including cached difficulties. Adjustments never accumulate
between calls. Reports record the controls and requested/effective NPS under
`intensity_postprocessing`; prediction NPZ files save `raw_intensity`, adjusted
`intensity`, and scalar `conditioning_nps`. Training, prepared targets and model
artifacts retain their original contracts.

## Postprocessing ablation comparisons

`evaluation/evaluate_v2_postprocessing.py` isolates the legacy timing/note
postprocessing contributions on real recordings. Both `postprocess_timing`
and `postprocess_notes` accept an optional ``skip`` set naming steps to omit
(`_TIMING_STEPS`/`_NOTE_STEPS` in `map_creation/v2_postprocessing.py`); the
tool injects arm-specific wrappers through `generate(postprocessors=...)` and
forces the multiplier's NONE class through a predictor wrapper — production
code and defaults are unchanged. Each (song, requested NPS) pair reruns all
arms with the same decoder seed so arms stay paired; raw intensity shares are
cached per song. Diagnostic-only requests may also skip ``clearance`` and
``minimum_spacing``.

```bash
python -m evaluation.evaluate_v2_postprocessing --plan --output /tmp/x
python -m evaluation.evaluate_v2_postprocessing --execute \
  --songs "BWB - Break the Horizon" "Bunker Machen Laut" --nps 3,5,8 \
  --output ../Data/training/evaluate_beats/results/v2-postprocessing
```

Arms cover the all-steps legacy baseline, timing-only subsets (minimal, no fills,
no alignment/snap, no silence), note-only subsets (minimal, no added notes),
the multiplier alone and the raw model stream. Metrics include per-step
event/note deltas, retained-vs-added events, snap distance to raw model
times, distance to detected onsets and BPM grid, note composition and
direction histograms, violations the fixpoint receives (`pre_clearance`) and
leaves (`residual`), per-stage seconds, and downstream hazard/arc counts.
Per-run note/event arrays are saved as NPZ under `arrays/`; incremental
`records.jsonl` survives partial runs. These are internal consistency
comparisons — they do not establish gameplay quality or human-map agreement.

2026-10-31 results (2 songs × NPS 3/5/8 × 10 arms, seed-paired): raw model
events already land ~21ms from detected onsets (99% within 50ms), yet the
timing model under-produces events ~3-4× versus requested density, so the
legacy fills stay load-bearing. Conversely the legacy note flow pass culled
~50-60% of sampled notes — `full` realized only 1.25/2.12/4.56 NPS at
requests 3/5/8 while `notes_minimal` (structure + clearance only) realized
3.21/4.68/9.55 — which motivated skipping `LEGACY_NOTE_MUTATORS` in
production. Mirroring/emphasis additions were ~round-tripped through the
flow cull (net +1-3%). The learned multiplier was near-inert (≤4 accepted
expansions of ~20-40 proposals per song) but costs ~1s and remains enabled.
All arms reached zero residual violations — the clearance fixpoint alone
enforces the hard invariants.

## Playability experiments

The v2 app and CLI use the final priority policy by default. The master,
`priority`, `reach` and `late_turns` switches are enabled, with slow-turn
thresholds .5s for 90 degrees and .65s for 45 degrees. Other historical component
switches remain disabled. Disabling the master selects the original app1 rules,
including strict flow on separated strokes. Historical experiments explicitly
reset every switch before selecting their profiles, independently of defaults.
The diagnostics for the primary speed-cap pass record actual note repairs.

| Setting (prefix `v2_playability_`) | Effect when enabled |
|---|---|
| `reach` | Apply the combined calibrated policy, including the hard opposite-edge cut ban and bounded joint cell/arrow repair. Use together with the other switches through the `combined` experiment. |
| `m1` | Search bounded same-hand windows for legal alternatives to fast horizontal reversals. Keep boundary arrows and use each arrival's own speed cap. |
| `isolated` | Revisit individual primary speed-cap recuts once their final neighbors are known. |
| `m2`, `m4`, `m3` | Try one-cell changes to improve sightlines, swing/travel clearance and eligible bimanual segments. Geometry precedes direction repair; the original position wins ties. |
| `slow90_seconds`, `slow45_seconds` | Permit a supported decoded turn after the specified incoming same-hand gap. A 90° entry requires adequate travel; a 45° entry also needs a compatible approach. Zero disables the angle. |
| `late_turns` | Keep early cleanup on strict flow. Try restoring eligible slow arrows on the surviving stream beyond the protected opening. Use together with the slow-turn thresholds. |
| `dwell` | Rank feasible direction windows by hand-specific cut-axis dwell, using fixed human reference summaries and local chart density. |
| `recovery_ratio` | Allow a single-note event to switch hands when the proposed alternative hand/cell has sufficient relative likelihood and gives the crowded hand time to recover. Zero disables reassignment. |

Direction repair keeps note IDs, timestamps, positions and hands. Placement
repair keeps IDs, timestamps and hands. The recovery policy acts earlier at
position decoding, so its final density and hand workload need separate checks.
Expanded notes, chain anchors, hazards and arc endpoints still pass through the
normal application validators. The generation path reads its fixed reference
tables locally. The human packages are only needed for offline comparisons.
[Paired comparison procedure](development.md#paired-playability-comparisons)

The `combined` experiment enables reach, M1–M4, slow 90°/45° restoration
at 500/650 ms and hand-specific dwell together. Its frozen reach tables
are shipped in `map_creation/v2_playability_reach_reference.json`.
Left-hand far-right cuts toward the left (including diagonals) and the
mirrored right-hand cases are forbidden. Joint repair keeps note identity,
hand and timing. It checks both neighboring movement edges, contextual
flow, opposite-hand clearance and dot-dependent directions. Infeasible
hard repairs fail with the offending note/window; they never export the
forbidden stream. Expanded copies have a protected final pass before
hazards/arcs are generated, followed by an end-of-pipeline ban check.

Combined visibility costs include direction-aware same-height upper-row
crossings. Mirrored doubles receive the same M4 overlap checks as other pairs;
there is no blanket exemption in geometry v3.
The complete combined policy first repairs mandatory cuts in reverse order,
trying the original arrow at nearby cells before changing directions. Completed
later time groups stay fixed except for bounded two-note backtracking over at
most three later groups. A failed hard repair restores the complete invocation.

One shared local acceptance snapshot covers M1, M2, M3, M4, cut-axis dwell and
reach/placement proposals. Optional edits preserve moderate movement, visibility,
interference, supported coordination and calibrated component tails. Every
accepted change becomes the next pass's baseline; whole-pass rollback restores
that baseline too. A load-reducing horizontal-run repair may use bounded
diagonal/cell bridges while preserving direction-tail cost. Repeated fast
horizontal reversals have a separate human-calibrated M1 event tail; ordinary
one or two reversals remain allowed. The search keeps small edit alternatives,
checks complete feasibility before selection, and revisits only shorter
remaining pieces of an improved run. Normal-range patterns
receive no reward for becoming uniform or centered.

The early diagonal cap retains the app1 setting and survivor selection. Later
optional repairs have a net diagonal-growth allowance of at most one percentage
point, also bounded by the frozen human upper fraction. Mandatory physical/hard
repairs are reported separately. Independent final checks record bans, clearance
and distinct-time same-hand speed residuals, including simultaneous stacks.

Frozen human tables additionally cover repeated top-row, outer-lane and
opposite-side exposure, lane/row concentration, and per-hand 45/90-degree cut-axis
runs in notes, elapsed duration and incoming gap. Placement and the joint M5
run-tail proxy fit <=10% eligible human events; specific reach and M3 segment
tails fit <=3%. These are fitted trigger budgets, not limits on the fraction of
notes allowed in the top row. Nine unsupported human packages remain explicit
skips, and sparse-band uncertainty is recorded with calibration artifacts.

Combined decoder recovery records model-supported single-note alternatives at
minimum likelihood ratio .90 while preserving original autoregressive position
history, arrow predictions and primary survivor selection. A bounded local
commit runs after the primary cap, before the postprocessing hand snapshot.
Adjacent supported singles may swap hands and transplant their existing
per-hand cuts together, preserving alternation without a model-history ripple.
It checks balance, recovery, own-side placement, cooldown, doubles, both hands'
physical edges and the shared acceptance policy; every committed assignment
must strictly reduce actual fast reversals. The original model stream's full
signature and deferred/committed counts are recorded separately. Doubles,
event timing and survivor counts remain fixed. Full-map M3 upper/lower excess
and complete per-hand M5 tails are protected in addition to local windows.
A rare repeated-M1 tail may trade ordinary M4 variation only when its
calibrated M4 tail does not increase; comfortable patterns receive no such
exception. Final M5 residuals remain explicit; recovery does not establish human-equivalent
rhythm.

The versioned evaluator `evaluation.playability_metrics_v2` shares the new
M2/M4 costs. Geometry v3 removes the mirrored-double exemption; its inherited
M4 thresholds await refitting after the metric-definition review. Archived
scorecards retain their recorded definitions and results.
[Second playtest guide](../../Data/training/evaluate_beats/results/combined-reach-playtest-20261108/playtest-final/PLAYTEST.md)

### Priority policy after the November 8 player review

The `priority` experiment arm selects the revised rules in the
[player priority review](superpowers/specs/2026-11-08-playability-priority-review.md).
It sets `v2_playability_priority` and uses a separate joint position/arrow repair;
historical `combined` and single-metric arms remain explicit profiles. The final
app/CLI default now selects the corrected priority profile; ordinary generation
does not require an experiment runner or an A/B comparison.

The shared definition is `priority-events-v2-segments-20261108`. Intersecting
straight cut/travel segments at <=250 ms are forbidden; slower intersections
through 1.5 s and near misses within half a grid cell are soft costs. Travel uses
the immediate previous same-hand onset through a 1.5 s lookback. Actual segment
geometry replaces the old shared-grid-region proxy, including unclamped endpoints;
mirrored arrows have no exemption. This remains a 2D approximation, not measured
player trajectories. Dots use the documented virtual movement estimate when a
predecessor exists; otherwise their path is a point.

Preparation uses each hand's 3x3 area and max per-edge excursion: two cells past
an edge are forbidden; a diagonal one cell outside both edges is one strong soft
excursion. Speed and recent awkward exposure compound the local penalties.
Follow-through outside alone has no cost. Axis changes use undirected cut axes:
a 90-degree base costs twice 45 degrees, isolated changes are small, frequent
changes strong and irregular frequent changes stronger. Long unchanged runs and
fast reversal counts never force variety. M1 and M3 remain monitoring outcomes;
existing placement is active at medium priority, with travel subordinate to the
primary rules. Local visibility now has its own calibrated acceptance guard.
The old competing repair stack is bypassed.

The soft search can jointly move and recut adjacent notes or a short outer-lane
phrase. It permits small isolated axis changes and slower path costs when the
preparation improvement justifies them, while guarding individual severe
preparation and axis events using the matched human NPS band. Two bounded local
rounds retain a legal incumbent; neither a map-wide average nor an easier
monitoring score can excuse a hard violation. Recovery travel remains a
subordinate cost, so rare soft travel tradeoffs still require playtesting.

Repairs preserve IDs, onsets, hands, dots and count and check physical flow,
clearance, arrival caps and affected successor paths. Repairs run before the
retained snapshot; mutable expansions receive retained-protected repair, chain
anchor synchronization and a final full-stream hard audit. Search exhaustion
raises an error and rolls back rather than returning a successful invalid map.
Existing physical exceptions must be recorded and cannot worsen; hard priority
violations receive no inherited exception. Numerical soft weights and the half-cell
near-miss margin are provisional engineering choices, distinct from calibrated
human flag thresholds and in-game comfort.

### Final local visibility and M2

`visibility-upper-center-v1-20261108` measures the mean central top-row occupancy
of the last eight same-hand onset groups within 1.5 seconds, requiring four
groups. Groups use the first onset's 20ms boundary; stacks receive fractional
weight and all eligible zero events remain in the denominator. The pure shared
implementation and fixed human reference live in `v2_playability_visibility.py`
and `v2_playability_visibility_reference.json`. The independently fitted specific
event tail budget is 3%, with pooled and equal-map checks and named skipped maps.

This catches a sustained flat upper stream even when the historical pairwise
rise M2 is zero. Current priority reports expose its local flags, maxima and
calibrated mean excess as the versioned current M2. `legacy_m2` and
`pairwise_value` retain the previous rise-only score explicitly; these values
have different definitions and must not be silently compared or added.

Soft search cannot create or worsen an individually severe local visibility
endpoint. Bounded joint position/arrow repair reduces existing exposure while
retaining the reach/path/axis and physical constraints. Final no-regression
allowances derive from original pre-hard exposure and the human cutoff. Protected
late expansions use retained-only exposure as their reference, so new copies
cannot create inherited exceptions. Reports retain native group membership and
limits for independent serialized-map validation. Ordinary human-range upper
notes remain allowed; there is no blanket top-row ban or quota.

Final-only mild residuals are separately versioned and explicit. After bounded
repair, an event originally at/below the human cutoff may remain within one
onset group's occupancy above that cutoff, provided residuals do not exceed
floor(3% of eligible events) for either hand. Over-rate hands fail entirely;
existing above-cutoff tails cannot worsen. Original event limits remain
`max(original, cutoff)`; residual metadata does not inflate them. Shared live
recomputation and the independent serialized audit verify every exception,
including its onset count and per-hand denominator. Candidate guards, metric
thresholds, reach/path hard bans and physical limits remain unchanged.

### Exact repeated-position separation

After calibrated priority and visibility repair, a single chronological pass
compares each note with the previous same-hand note's current cell. Exact
position repeats propose one cell in the current arrow's cut direction;
diagonals use only their vertical component. Left-hand destinations stay in
columns 0–2, right-hand destinations in 1–3, and all destinations in rows 0–2.
A legal top-row proposal is applied with 50% probability using Python's existing
random stream (seeded by the offline case runner). Other legal destinations are
always applied. There is no speed threshold or extra direction change.

Every proposal uses the existing speed/flow/clearance, hard reach/path and
calibrated local soft/visibility guards; unsafe proposals stay unchanged. Dot
notes, simultaneous same-hand onset groups and protected retained notes stay
intact. This runs before the final independent hard/physical/visibility audits,
and reports `repeat_position_changed`. It does not alter IDs, onsets, hands,
arrows or note counts. This November 9 addition does not rewrite the previously
delivered frozen five-song archive.

## Packages and diagnostics

Default output remains Beat Saber **3.2.0**: notes, bombs, walls, basic lighting,
arcs and native burst chains. The legacy **2.0.0** option keeps ordinary notes and
chain heads, omitting arcs, chain bodies and walls that cannot use legacy geometry.
Omissions are counted. Seconds and durations convert to beats once at export.

Metadata, cover/audio names, difficulty labels, NJS, jump offsets and ZIP layout
remain compatible. Multi-difficulty metadata is assembled as JSON. Per-variant
folders keep `1234_<internal difficulty>_<song>`; combined folders/ZIPs keep
`12345_<song>` and the `1234_*` intermediates are deleted after combining.
Ascending requested difficulties retain the Easy/Normal/Hard/
Expert/ExpertPlus labels. Optional automatic Expert uses a separate reduced-density
pass. Packages and ZIPs publish through temporary destinations only after success.

Reports live outside packages under `paths.temp_path/v2-generation/`. They include
raw/cleaned counts, heuristic changes, learned expansions, rejected objects,
legacy omissions, per-stage durations and cache metrics. Companion numeric NPZ
files contain raw/adjusted intensity shares, effective conditioning NPS, raw timing
probabilities/counts, processed events, pre-cleanup model notes and final notes;
they load with `allow_pickle=False`. The historical v1 runner
lives in `evaluation/legacy_generator.py`; the app and CLI v1 presets invoke it
unchanged through `map_creation/v1_generation.py`, and benchmarks call it directly.

See [baseline acceptance](baseline.md) for the executed checks and finished sample
packages. GPU inference, native Windows execution and in-game playability require
separate checks; this integration uses the recorded WSL CPU runtime.
