# Forward Note Postprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Steps use checkbox syntax.

**Goal:** Replace production repair search with bounded chronological rules and add the approved independent outward-center move.

**Architecture:** A shared local time index guards concrete collisions. Small forward passes finish all base-note edits before expansion; a separate append-only guard checks later additions. Existing search-mode and historical/custom adapters remain intact.

**Tech Stack:** Python, existing NumPy/config/geometry helpers, pytest, CPU WSL runtime; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-09-forward-postprocessing-design.md` (approved with user clarifications).

## Global Constraints

- Preserve b65d2b4 and all unrelated working-tree changes.
- No v1, model, training, timing-generation, platform, or package-format changes.
- Use loaded production flags. Observed: master/priority/reach/late-turns true;
  separate M1/M2/M3/M4/dwell/recovery experiments false; legacy cleanup false.
- Respect enabled flow, counter, diagonal, speed, emphasis, and Single Mode settings.
- No reverse repair, convergence sweeps, joint assignment products, or candidate full-map rescoring.
- At most eight proposals per note per relevant pass; leave unresolved incumbents and report them.
- Complete base-note movement before copies/chains. Never move or recut accepted additions.
- Structural errors still fail. Remaining playability findings do not fail forward generation.
- Search/custom/historical policies retain strict validation and their existing behavior.
- Preserve each file's line endings. No commits, pushes, library export, or corpus reruns.

## Review Focus

- Live index entries must see accepted position/direction edits; rejected candidates must not mutate notes.
- A rejected atomic expansion must not pollute the index and block a later legal expansion.
- Chain samples must ignore their own anchor, but must still conflict with other notes/cuts.
- A global forward default must not accidentally weaken explicitly supplied historical/custom postprocessors.
- Disabled flags, dots, unsorted input, simultaneous groups, and preexisting violations must terminate without deleting optimization incumbents.

## File Responsibilities

- Create `map_creation/v2_forward_clearance.py`: local time index and physical collision predicate.
- Create `map_creation/v2_forward_postprocessing.py`: named bounded base-note passes.
- Create `map_creation/v2_forward_expansion.py`: append-only learned/chain/emphasis guards.
- Modify `map_creation/v2_pipeline.py`: explicit forward/search orchestration, not no-op adapter layers.
- Modify `map_creation/v2_note_emphasis.py`: optional gate factory; unchanged sampler.
- Modify `tools/config/config.py`: lead-authored mode assignment below.
- Tests: new `tests/test_v2_forward_postprocessing.py`, `tests/test_v2_forward_expansion.py`;
  add pipeline/validation cases to existing generation test files.
- Update generation documentation and AGENTS to distinguish best-effort forward from strict search.

## Lead-Owned Rule Definitions

These choices are settled. Implement them; do not invent new scoring weights,
thresholds, sampling rules, or broad-search fallback algorithms.

### Local Collision Index

`LocalNoteIndex(notes)` keeps sorted times plus live note references, using
bisect for neighborhoods; positions/directions are read live, never cached by ID.
Expose `nearby(note, window=None)`, `conflicts(note, ignore_ids=())`,
`accepts(note, ignore_ids=())`, `accepts_many(proposals, ignore_ids=())`,
and `add_many(records)`. Skip a candidate's own ID.

Default neighborhood horizon is `max(config.min_time_diff,
config.v2_cleanup_swing_window, priority_metrics.FAST_SECONDS)`.
The physical predicate is:

- Any hands: equal cell (coordinate tolerance 1e-9) at absolute gap <= min_time_diff + 1e-9 conflicts.
- Opposite hands: within the configured swing window, either center inside the
  other's approach/cut/exit swept cells conflicts.
- Opposite hands: within FAST_SECONDS (.25), intersecting straight cut segments
  conflict, including diagonal intersections with no shared integer cell.
- Cut segment is `(x+sx,y+sy)` to `(x-sx,y-sy)` using existing STEP;
  dots and chain-body samples are points. Reuse `_segment_distance`; no incoming
  travel reconstruction in this approximation.
- Ordinary same-hand stack cut overlap is allowed; same-cell occupancy is not.
- For a chain body sample, exempt its own anchor only.
- A proposal must have finite legal note geometry. Relocations additionally stay
  in that hand's 3x3 matrix. No mutation before acceptance.

### Core APIs And Sequence

`normalize_notes(notes, event_times, logger=None) -> (list, dict)` reuses
`base.postprocess_notes(skip=(*base.LEGACY_NOTE_MUTATORS, "flow_additions",
"same_direction_recut", "clearance"))`. Structural Single Mode, dedup, stable
labeling, and dot policy remain; expensive mutators/fixpoints do not run.

`repair_notes(notes, *, rng=None, cap=None, logger=None) -> dict` creates one
local index and runs direction -> reach/speed -> collision -> repeated position
-> existing visibility -> independent center. It changes only x/y/d, not order,
time, ID, hand, count, or unrelated metadata. It scopes/resets late-turn readiness.
Return per-pass changed/attempted/blocked/unresolved counts and elapsed seconds.

Direction pass: when configured flow/counter/diagonal rules need correction,
try distinct directions from `[original, reverse(previous_effective)]` followed
by all eight arrows sorted by `(diagonal_when_over_cap, angular_distance_from_original,
is_horizontal, direction_id)`, deduplicated/truncated to eight. Use existing
`_flow_legal`, `_counter_direction`, and direction helpers. Preserve dots.
Priority axis-churn approximation: if an axis change would occur within two
same-hand notes after the previous axis change, prefer the previous axis.
Only accept directions satisfying the active local predicates and physical guard.
The diagonal allowance is the existing configured fraction times total notes,
not a new calibrated threshold. Unresolvable notes remain unchanged.

Reach/speed pass: build same-hand predecessor groups once using `SpeedGroups`.
Use the existing scalar arrival speed minimum over the previous group. Use the
existing v1 ramp and `speed_cap_scale()` if no cap callback is supplied.
For each note, lexicographically minimize the following lead-authored local cost:
`(int(forbidden_cut(note) or max(dx,dy)>=2), max(0., arrival-cap(note)), max(dx,dy))`,
where `dx=max(low-(x+sx), x+sx-high, 0)` and
`dy=max(-(y+sy), y+sy-2, 0)` and `(low,high)` is the hand's 3x3 interval.
Ignore the speed term when its configured scale is disabled.
This is local arithmetic, not `preparation_events()` per candidate.
Proposals, deduplicated/truncated to eight:
original; clamp cell to hand bounds; clamp preparation into hand bounds;
one x step toward previous cell; one y step toward previous cell;
one x step toward own center column; one y step toward row 1;
own center `(1,1)` or `(2,1)`, all with the original arrow.
Choose the first physically clear strict cost improvement. No note removal.

Collision pass: if the incumbent conflicts, try x-1, x+1, y-1, y+1,
own center, and the two other in-hand columns at the same row, preserving arrow.
Deduplicate/truncate to eight, accept the first physically clear in-hand cell,
otherwise leave the incumbent and report it.

Repeated-position pass: mirror existing `separate_repeated_positions` geometry
exactly: with approach `(dx,dy)`, `x=old_x if dy else old_x-dx`,
`y=old_y-dy`. Eligible directed, sole-onset, same-hand repeats must be separated
by more than eps. In-hand and physical guards apply. A target in the top row
uses the existing 50% RNG gate only after geometry is admissible.

Existing visibility pass: priority enables the current upper-center definition,
independent of the historical experimental m2 switch. Use `visibility_costs()`
once for its unchanged cutoff and event windows/weights. Process endpoint groups
chronologically, recomputing each event's weighted central_top membership from
live notes. When severity exceeds the cutoff, only current endpoint top-center
notes are candidates. Try row 1, then the hand's outer column; preserve arrows,
require actual exposure reduction and physical clearance. No prior-group edits.
Do not enforce old global event quotas. If priority is disabled, skip this pass.

`clear_center_view(notes, *, rng=None, index=None) -> dict`: eligible hand/column
is `(0,1)` or `(1,2)`, weight `{0:0.,1:1.,2:.25}[y]`; wrong columns and bottom
consume no RNG. For each eligible note draw once and move iff
`float(rng.random()) < .8*weight`. Proposed x is `x-1` for left or `x+1` for
right. Preserve y/d/time/hand/ID/metadata, including dots. Check only in-hand
bounds and physical clearance; no flow/reach/previous/following metric. Blocked
moves consume the draw but do not mutate or try alternatives.

### Expansion And Policy Interfaces

In `v2_forward_expansion`, expose `validate_expansions(state, song, logger=None)`
and `apply_note_emphasis(state, song, *, rng=None, logger=None)` with the same
adapter signatures as existing postprocessors.
`validate_expansions(state, song, logger=None) -> dict` checks new additions as
one atomic batch against an accepted local index; never edits known notes.
Cache the guard privately on state, track accepted IDs and accepted chains,
and do not commit anything from a rejected batch. Preserve existing rejection
counter names. A final repeat call with no new additions is cheap and idempotent.
Read the current state lists, not stale list references: atomic rollback replaces
those lists. Initialize accepted notes from non-copy base notes, not from an
unchecked first incoming batch. Remove the private guard immediately after
emphasis, before hazard indexing/model input construction.
Reuse the existing chain sampling convention, checking samples against accepted
notes/chains and new batch members, ignoring only the chain's own anchor.
Never move or recut an accepted addition. Remove private runtime caches before
returning/serializing the generation result.

Change existing `apply_note_emphasis(..., gate_factory=None)` only to instantiate
the supplied factory instead of `_EmphasisGate`; retain sampler/quantile/RNG/
probability/geometry/toggle logic byte-for-byte. Forward's gate implements the
same `p2/p3`, `eligible`, `copy`, `check` protocol using the local index and exact
existing probability formulas. Synchronize previously appended emphasis notes
before subsequent checks. Enforce physical clearance, configured counter rule
and non-worsening diagonal fraction locally; no full-state scratch validators
or calibrated global rescoring. Search mode continues to use the original gate.

Lead-authored config addition: `v2_note_postprocessing_mode = "forward"`.
Only `"forward"` and `"search"` are valid. Forward selection requires ordinary
default adapters, master enabled, priority or reach enabled, and legacy cleanup
off. Never infer relaxed validation merely from global mode when custom adapters
were supplied. Do not turn other config flags off to bypass code.

In `generate`, select forward normalization and expansion/emphasis adapters;
after onset snap/intro call `repair_notes` exactly once before the retained
snapshot. Do not also call old fast/diagonal/speed/priority repair hooks.
After learned expansion do not call old flow fixpoint, speed, counter sweep,
or reach finalization. All additions happen after frozen base-note edits.

`_validate_state(state, duration, *, playability_mode="search")`: preserve all
structural validation. In forward mode collect existing forbidden-cut and
priority hard preparation/crossing findings instead of raising on them.
Strict default/custom mode still raises exactly as before. Add final diagnostic
markers distinguishing structural production validation from best-effort
playability; do not fabricate old strict-policy success fields.
Use `diagnostics["playability_policy"]` for the selected actual adapter mode,
`["playability_strict_validated"]` for enabled reach/priority audits actually
enforced, and `["playability_findings"]` for forward residuals. In forward mode
the strict marker is false; retain `production_validated=True` for structure.

### Task 1 Review Clarifications

- Preserve caller list order: process a sorted view, never `notes.sort()`.
- Use the existing four distinct undirected axes from `metrics.AXIS`.
  Track each hand's effective direction, actual previous coordinates, ordinal,
  previous axis, and last axis-change ordinal in a forward accumulator. No
  backward scans or `notes.index()` during dot propagation.
- Flow repair is enabled only when `v2_cleanup_same_dir_recut > 0`;
  counter repair needs a positive fast-gap setting. Priority axis churn is an
  independent trigger when a new change is <=2 same-hand notes after the last
  change. Diagonal excess is also an independent trigger and its denominator
  is all notes, including dots. Update axis history on every note, not just repairs.
- Reach uses `SpeedGroups.previous[i]` directly, including an empty first group.
  Repair first-note preparation and dot positions without changing dot arrows.
  Zero-cost skip is componentwise (`all(v <= 1e-9 for v in cost)`), not
  lexicographic against a tolerance tuple. Keep the specified lexicographic
  comparison only for deciding whether a proposal improves the incumbent.
- Repeated-position onset occupancy uses the local index's eps window, not
  a full-map comprehension for every note.
- Only records explicitly marked `kind="chain_sample"` receive anchor
  exemptions; ordinary copied notes never do. The exemption is symmetric.
  Samples of the same non-null chain ID do not conflict with one another.
  Sample geometry is a point regardless of its inherited arrow.
- Ordinary note cells are integers; chain sample cells may be fractional.
  Same-cell comparison uses coordinate tolerance 1e-9, including same-hand samples.
  Reject duplicate IDs inside a proposal batch.
- A chain sample has only its own point as its swept geometry. Opposite-hand
  swept-cell containment is `abs(px-cx) <= .5+1e-9 and abs(py-cy) <= .5+1e-9`;
  this conservatively includes fractional chain bodies within a swept grid cell.
  Integral ordinary note centers behave identically to integer-cell membership.
- Churn preferences are priority-gated and apply only to churn-triggered repairs.
  First-of-hand notes still participate in diagonal-cap correction.
- `changed` counts actual x/y/d changes, not successful no-op trials.
RNG contract tests use a strict scripted helper; general pass tests may use
  an explicitly named constant RNG. Do not retain dead test helpers or assertions
  that merely check that a fixed report key exists.

### Task 2 Review Clarifications

- New copies and chain samples must be checked in ONE combined atomic proposal
  batch; separate checks against the old index miss copy/chain intersections.
- `accepts_many` must index its proposal batch once and use bounded temporal
  neighborhoods for intra-batch checks too. Do not compare every sample against
  the full prefix of a long chain. Keep ID/geometry checks and purity on rejection.
- The pipeline's selected policy is a local variable initialized to `"search"`
  and set to `"forward"` ONLY inside ordinary default forward selection.
  Do not read a custom adapter's `playability_mode` attribute as authority.
  Invoke new repair/drop-guard hooks only for the actually selected forward path.
- `playability_strict_validated` is true only for search mode with the master
  enabled and at least one reach/priority hard audit enabled and completed.
  Master-off and both-flags-off paths must not claim a strict audit ran.
- On emphasis-gate construction, index ALL frozen existing notes/copies/chains,
  even if no earlier forward validator cache exists. Do not remove, move, or
  recut them. Update cached accepted IDs/chains consistently when synchronizing.
- Store first-of-hand direction state only after its final chosen recut.
  Example: diagonal cap 0, same-hand center notes `(t=0,d=4)` then
  `(t=.3,d=3)`. After the first recuts to d=0, the second must see d=0,
  not the stale d=4 prediction; it needs a legal recut rather than keeping
  the now-illegal 90-degree turn.
- Prove sequential emphasis collision awareness with two same-time anchors:
  left `(x=0,y=0,d=3)`, right `(3,2,0)`. Accept left copies at `(1,0)` and
  `(2,0)`; append them; then reject a right copy at `(3,1)` because its cut
  intersects the newly added left copy's cut, not either original anchor.
- Prove atomic cross-kind clearance with base anchors at t=0 left `(0,1,2)`
  and t=2 right `(3,2,0)`, a left chain t=0..1 from `(0,1)` to `(2,1)`,
  and a new right copy at t=.5 `(1,1,0)`. Individually clear against old
  anchors, the new chain and copy collide: reject both without index pollution.
- Fast-run membership uses the established strict `< v2_fast_gap_seconds`
  boundary in both core and emphasis; equality is not fast.
- Forward mode does not run calibrated decoder-recovery search. Do not attach
  stale `_recovery_alternatives` in forward; record recovery as not run rather
  than pretending the search ran when the experimental flag is enabled.
- Remaining final flow/clearance findings are diagnostic too. Collect the
  existing `clearance_violations(notes)` once in forward final validation and
  store under `playability_findings["clearance"]`; never rerun direction repair
  after movement. Later position passes may intentionally change slow-turn flow.
- New-copy dangling anchors are rejected with the incoming batch, before indexing.
- Retain the existing cheap `forbidden_cut` guard for NEW copies when the master
  and either reach/priority are enabled, in both learned and emphasis guards.
  Reject the optional batch/stack; never raise or repair it. Existing incumbent
  forbidden cuts remain diagnostics. This does not add a reach condition to the
  independent center rule, whose in-hand destination already avoids this ban.

## Task 1: Bounded Core And Independent Center Rule

**Files:** new clearance/core modules and `tests/test_v2_forward_postprocessing.py`.
**Consumes:** existing config, STEP/geometry, scalar speed/flow, visibility events.
**Produces:** the index, normalize_notes, repair_notes, clear_center_view APIs above.

- [ ] Write literal tests for left/right middle moves at draws .799/.8; top at
  .199/.2; bottom/wrong column zero draws; dots and metadata preserved.
- [ ] Add occupied destination, swept-cut, diagonal intersection, live-index
  update, same-hand stack, and no-mutation-on-rejection tests.
- [ ] Add flag-disabled cases, sorted/unsorted/empty notes, dense preexisting
  violations, bounded proposal counts, and separate visibility/center behavior.
- [ ] Run tests red (actual behavior failures, not collection errors).
- [ ] Implement the settled core in small patches; no pipeline integration yet.
- [ ] Run the new file plus existing priority-metrics/visibility tests green.
- [ ] Lead reviews the full diff; no commit or push.

## Task 2: Late Additions And Production Integration

**Files:** new expansion module; pipeline/emphasis/config; new expansion tests;
existing generation pipeline/postprocessing tests.
**Consumes:** Task 1 index/core APIs. **Produces:** default forward generation
with unchanged strict opt-in behavior and append-only late expansions.

- [ ] Add red tests for atomic rejected-batch rollback and index reuse,
  chain anchor exemption/real obstruction, sequential emphasis, disabled toggles,
  and no movement/recutting of learned copies or chains.
- [ ] Add default-dispatch tests that stub forbidden old search/fixpoint entry
  points to raise, plus explicit search/custom-mode tests.
- [ ] Add forward final-validation diagnostics vs strict raises for the same
  forbidden cut/crossing; structural malformed notes must still raise in both.
- [ ] Implement interfaces/config addition exactly as specified.
- [ ] Pin existing historical wiring tests explicitly to search mode when that
  is what they test; do not weaken their strict expectations.
- [ ] Run focused forward, generation, emphasis, priority and visibility tests.
- [ ] Lead reviews the complete combined diff; no commit or push.

## Task 3: Evidence, Documentation, Final Gate

**Files:** AGENTS, generation-v2 docs; external disposable artifacts only.
**Consumes:** reviewed implementation. **Produces:** verified CPU evidence.

- [ ] Update current-policy docs without rewriting historical experiment claims.
- [ ] Run the lead-authored frozen-input replay recipe once under a bounded
  timeout; retain input/output/diagnostics/timing artifacts.
- [ ] Run one disposable local-song CPU generation; validate v2/v3 serialization,
  retained snapshot, late expansions, and private-cache removal. No library export.
- [ ] Run one final `python -m pytest -q -m 'not comparison'` gate after review.
- [ ] Check line endings/diff; request independent review; report runtime,
  remaining violations and unverified gameplay. No automatic commit/push.
