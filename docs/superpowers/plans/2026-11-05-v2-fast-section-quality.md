# v2 fast-section quality — playtest round 2

Source: `todo_after_playtest.md` R2-1..R2-4 (user decisions 2026-11-05).
All changes are v2-only, gated by new `v2_*` config knobs. v1 untouched.

## Diagnosis (verified)

- Opening burst = `start_end_fill` inserting frame-density events before the
  first model event; it does not consult `fill_blocked()` and nothing
  re-clamps spacing after fills (`_merge_candidates` merges at
  `min_time_diff`, not `min_beat_time`).
- v1's movement-cost gate (`sanity_check_notes` + `calc_note_speed`,
  `max_speed * get_factor_from_max_speed`, `decr_speed_val` ramp) is inside
  `legacy_note_flow`, skipped under `v2_cleanup_note_heuristics = False`.
  v2 has no travel/time verification anywhere.
- No fast-section direction constraints exist.

## Pipeline anchors (v2_pipeline.generate)

- `postprocess_notes` :529 · `snap_note_times` :534 · `apply_intro_floor`
  :545 · `apply_diagonal_cap` :552 · retained snapshot :554 · multiplier
  expansion :563-591 · `enforce_note_flow_and_clearance` :594 ·
  hazards :598+
- Timing: `postprocess_timing` `fill_blocked()` :263-268, fill calls :293-324,
  `start_end_fill` :312-314, `_section_entries` :120, `_blocked_intervals` :128,
  `_start_end_times` :88.

## Task 1 — count-based entry gate + start_fill suppression (timing level)

`map_creation/v2_postprocessing.py`, `tools/config/config.py`.

- `v2_fill_entry_min_notes = 4` (0 disables). Extend `fill_blocked()`:
  for each entry time from `_section_entries`, the blocked window extends
  to the time of the k-th kept candidate at/after that entry
  (`[entry, t_k)`) — union with the existing seconds-delay window.
  Applies to every call site already passing `fill_blocked()`
  (`high_difficulty_fill`, `density_fill`) — model times still always pass
  (`_filter_fill_times` kept-set semantics unchanged).
- `v2_suppress_start_fill = True`: when truthy AND
  `v2_map_intro_no_notes_seconds > 0`, `start_end_fill` drops its
  start-segment insertions (events before `times[0]`); the end segment is
  unchanged. Implement inside the `start_end_fill` call site by filtering
  `_start_end_times` additions `< times()[0]` — keep `_start_end_times`
  itself pure.
- Diagnostics: report `entry_blocked_windows` count and
  `start_fill_suppressed` count in timing diagnostics.
- Tests: count gate blocks fill inserts inside ramp window; 4th note
  releases; pause resets count; `0` preserves legacy; start segment
  suppressed only when intro floor active; end segment intact.

## Task 2 — movement speed cap (note level)

`map_creation/v2_postprocessing.py`, `map_creation/v2_pipeline.py`,
`tools/config/config.py`.

- New `apply_speed_cap(notes, chains, rng=None, logger=None) -> diagnostics`,
  called in `generate()` after `enforce_note_flow_and_clearance` (:594) and
  before `index_state`/hazards (:598). Also add it to the
  `v2_cleanup_note_heuristics`-off SimpleNamespace (:415-419).
- Cost: reuse `calc_note_speed` (v1 semantics — exit→approach distance/dt +
  `reaction_time`/`reaction_time_fact` buffer) on each hand's consecutive
  kept notes. Cap = `config.max_speed * get_factor_from_max_speed(
  config.max_speed, .5, 1) * v2_speed_cap_scale` (v1-identical when
  scale=1).
- Start ramp: first `v2_speed_cap_start_notes` (default 30) note positions
  (combined chronological index) cap at `v2_speed_cap_start_factor`
  (default .35) → 1.0 linear ramp — mirrors `decr_speed_val`/`decr_speed_range`.
- Action: drop the LATER note of a violating pair (v1 drops the timestep;
  per-hand drop preserves the other hand's note — deliberate refinement).
  Deleting a chain head drops its chain (sync `chains` list; a too-fast
  chain is unplayable). After each deletion re-check the new same-hand
  pair; iterate the whole map to fixpoint (deletion-only → converges).
  If the post-deletion pair (prev→next) violates `_flow_legal`, recut the
  next note via `_flow_direction`; if no legal recut, delete it too.
- Diagnostics: `speed_cap_dropped`, `speed_cap_chain_dropped`,
  `speed_cap_recuts`, `speed_cap_value`.
- Tests: synthetic overspeed pair → later note dropped; borderline kept;
  start ramp tightens cap on early notes; scale knob scales; disabled =
  no-op; chain-head deletion drops chain; flow preserved post-deletion.

## Task 3 — fast-run direction simplification + counter-direction (note level)

`map_creation/v2_postprocessing.py`, `map_creation/v2_pipeline.py`,
`tools/config/config.py`.

- New `apply_fast_direction_rules(notes, rng, logger=None) -> diagnostics`,
  called just before `apply_diagonal_cap` (:549-552 site) and added to the
  heuristics-off namespace. Runs pre-expansion so copies inherit simplified
  directions; `enforce_note_flow_and_clearance` remains the legality
  backstop afterwards.
- Fast pair: same-hand consecutive gap < `v2_fast_gap_seconds` (default .2).
  Runs extend through chains of fast pairs.
- R2-3: within fast runs the allowed direction set for each note is
  `{pred_dir, REVERSE(pred_dir)}` plus `pred_dir ± 45°` — the ±45° options
  are sampled only with probability
  `min(1, expected_speed / local_speed)`, where `expected_speed` =
  mean `calc_note_speed` over all same-hand pairs in the map (per-map
  "expected average at this NPS") and `local_speed` = this pair's cost.
  → at local ≈ expected, ±45° is fully allowed; at 2× expected, 50%;
  approaches 0 at extreme speed. Seeded rng.
  If the note's current direction is outside the effective allowed set,
  recut via `_flow_direction` restricted to that set; if none is
  flow-legal, keep original (legality-first, like the diagonal cap).
- R2-4: `v2_fast_counter_direction = "recut"` (`"keep"`/`"off"` disables):
  in fast runs, top-row notes (y=2) with down-family cuts {1,6,7} and
  bottom-row notes (y=0) with up-family cuts {0,4,5} are recut first —
  they get priority in the pass before the general simplify rule.
- Diagnostics: `fast_pairs`, `fast_simplified`, `fast_counter_recuts`,
  `expected_speed`.
- Tests: fast run simplified to ping-pong±45; slow run untouched; ±45
  disappears at extreme local speed (seeded); counter-direction recut in
  fast run, kept in slow run; threshold/disabled knobs honored; no legal
  candidate → keep.

## Task 4 — docs + focused verification

- `tools/config/config.py` comment block coherency; `docs/generation-v2.md`
  v2 postprocessing section; `todo_after_playtest.md` R2 items → decided.
- Regenerate `12345_BWB - No Sleep` v2_expert locally and re-run the
  metrics script: confirm opening ramp (no sub-4-note fill burst),
  diagonal %, max same-hand speed ≤ cap, direction variety in opening.
- Full `pytest -q` suite in the baseline env.

## Order & isolation

- Tasks share `v2_postprocessing.py`/`config.py` → implement sequentially.
- All knobs default to the requested behavior ON (these are the desired
  production semantics); every knob has an off/legacy escape.
- v1 path untouched (no shared code paths mutated — `sanity_check_notes`,
  `calc_note_speed`, `fill_map_times_scale` are read-only reused).

## Outcome (2026-11-05)

Tasks 1–3 shipped on `main_app_dev` in four commits:

- `32f1b4c` — Task 1: `v2_fill_entry_min_notes` count gate inside
  `fill_blocked()`/`_blocked_intervals` plus `v2_suppress_start_fill`
  dropping `start_end_fill`'s pre-first-event inserts while the intro
  note floor is active; diagnostics `entry_blocked_windows` /
  `start_fill_suppressed`.
- `1e2d6d5` — Task 2: `apply_speed_cap` post-expansion pass (v1
  `calc_note_speed` cost, `v2_speed_cap_*` knobs, per-hand later-note
  drops with flow recut fallback, deletion-only fixpoint).
- `c7b70f0` — Task 3: `apply_fast_direction_rules` — fast runs collapse
  to ping-pong with speed-adaptive ±45° admission;
  `v2_fast_counter_direction` recuts row-opposing cuts first.
- `3147add` — Task-3 review fix: the admitted ±45° family now also
  includes the reverse's own ±45° neighbours (the 135°/225°
  off-reversals — the only ±45° cuts `_flow_legal` admits between
  separated strokes), not just the predecessor's.

Task 4 docs landed with this footer (`todo_after_playtest.md`,
`docs/generation-v2.md`); config comments verified against the shipped
code. The map regeneration/metrics and full-suite items of Task 4 remain
manual follow-ups.
