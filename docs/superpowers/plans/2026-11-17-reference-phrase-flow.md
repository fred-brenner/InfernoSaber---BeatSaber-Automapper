# Reference phrase flow implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Improve connected note phrases and add safe rhythmic edge-wall accents.

**Architecture:** Extend the final mutable pattern pass and the heuristic hazard
proposal seam. Reuse existing acceptance guards and keep later notes frozen.

**Tech Stack:** Python, NumPy, pytest; retained WSL CPU environment.

**Spec:** `docs/superpowers/specs/2026-11-17-reference-phrase-flow.md`

## Global constraints

Preserve note IDs/times/hands/count and non-geometry fields. No relaxed hard
guards, convergence loops or new dependencies. CPU only. Preserve line endings.
Do not commit or push during implementation. Keep analysis artifacts outside
the repository. All thresholds and template values below are lead-authored;
report an inconsistency instead of independently changing policy.

## Review focus

- Staggered groups near epsilon must not become editable singleton phrases.
- A failed or raised candidate must restore all coordinates and arrows.
- A successful phrase must still connect to its untouched successor.
- Wall clearance includes swing cells and chain interiors, not only note heads.
- Disabled features and late expansion must preserve their existing contracts.

## Task 1: Bounded phrase repair and diagnostic reasons

Files: `map_creation/v2_forward_patterns.py`, `tests/test_v2_forward_flow.py`.
Keep existing `v2_forward_flow` shared predicates and public interface.

- [ ] Add failing focused coverage for the behavior and limits below.
- [ ] Add these module constants and exact motifs (red; mirror with `_MIRROR_D`):

```python
_PHRASE_CYCLES = (
    ("outer_bounce", ((0, 1, 0), (0, 0, 1))),
    ("bottom_rebound", ((1, 0, 0), (1, 0, 1))),
    ("diagonal_rocker", ((0, 1, 4), (1, 0, 1))),
    ("fan", ((0, 1, 4), (1, 0, 1), (1, 2, 0), (1, 0, 1))),
)
_PHRASE_TRIALS = 12
_REJECTION_EXAMPLES = 3

def _upper_down(note):
    return note["x"] in (1, 2) and note["y"] == 2 and note["d"] in (1, 6, 7)
```

- [ ] Add `report["rejections"]` and `report["rejection_examples"]` dictionaries.
  Keys are `family:reason`; family is lane/wave/phrase/local. Record the first
  failing guard of an evaluated candidate, not every possible failure.
  Examples contain only first note ID/time and candidate length, capped at
  three per key. Add `_reject(report, family, reason, notes)` returning None.
  Extend `_check_chunk(..., strict, report=None, family="wave")`. Preserve
  check order and split combined guards into entry_flow, internal_flow,
  exit_flow, direction, forbidden, overlap, collision, style,
  upper_down, neighbor_style and speed. All existing call sites pass family.

- [ ] Preserve existing lexicographic style comparison, and additionally
  reject a chunk whose raw `_upper_down` count exceeds that in `saved`.
  In `_repair_note`, reject trials with `_upper_down(trial) > _upper_down(note)`
  before mutation. Record local candidate guard failures using the same
  bounded diagnostics; keep the 12-expensive-trial limit.

- [ ] Replace fixed-length-only wave evaluation with candidate prefixes
  `sorted({len(chunk), max(4, len(chunk)-2), 4}, reverse=True)`, ignoring lengths
  exceeding the chunk. Evaluate each prefix independently using `_prepare`
  and `_check_chunk`, up to four phases. Select by returned cost, then prefer
  longer length, then phase. Save/restore each prefix in finally. Count only
  notes actually changed by the accepted prefix. Existing outer wave scan
  still consumes its original non-overlapping chunk. Bound: 12 trials/window.

- [ ] Add `_phrase_hand(context, index, cap, hand, report)` between the wave
  loop and `_local_repair`. Use `_hand_groups`; eligible singleton arrows
  must also have `len(context.group(note)) == 1`. Split at stacks/dots,
  ambiguous groups, or gaps outside `(eps, .4]`. Consume non-overlapping
  chunks of four (a final chunk of two or three is allowed). Skip chunks
  with no positive `_aggregate` cost. Candidate templates require `_regular`;
  positional transforms do not. Skip chunks adjacent to a dot or multi-note
  group, since `_prepare` is conservative there.

- [ ] `_phrase_candidates(notes, hand, regular)` yields `(name, geometry)`
  in this exact order, deduplicated and excluding the incumbent:
  1. translate every x by -1 red/+1 blue, preserving y/d, only within `_bounds`;
  2. map every x to the hand's outer column, preserving y/d;
  3. when regular, each cycle above in listed order and phases `range(len(cycle))`,
     mirrored for blue. Repeat/truncate the cycle to the candidate length.
  Thus at most 2+2+2+2+4=12 candidates. Evaluate atomically with `_check_chunk`
  and `strict=True`. Choose the lowest returned cost then candidate order.
  Store phrase_windows/proposals/applied/notes and motif name counts.

- [ ] Tests in the existing flow file must demonstrate:
  - A mixed-column center phrase can improve when a single-note move cannot;
    freeze all non-x/y/d fields and assert center count decreases, no count loss.
  - Mirrored input yields mirrored geometry and identical diagnostics counts.
  - A blocked successor or opposite-hand collision prevents application.
  - Every rejected/exceptional candidate restores the exact incumbent.
  - Existing clear vertical streams are unchanged.
  - Stacks, dots and t=(0,.015,.03) ambiguity are skipped by phrase overlays.
  - Upper-center downcut count cannot increase as center occupancy decreases.
  - A full-length wave blocked at exit can accept a shorter prefix, preserving
    flow into its actual next untouched note.
  - `phrase_proposals <= 12*phrase_windows`, `wave_proposals <= 12*wave_windows`,
    existing singleton proposal bound, and examples bounded by three per key.
  Use existing fixtures, LocalNoteIndex and direction audits; stub a guard
  only in the explicit rollback/rejection diagnostics test.

- [ ] Run the focused flow tests, then the existing forward/expansion/emphasis
  tests. Fix implementation errors without changing specified policy values.

## Task 2: Guarded active-passage edge accents

Files: new `map_creation/v2_forward_obstacles.py`,
`map_creation/v2_pipeline.py`, `tools/config/config.py`,
existing `tests/test_v2_generation_pipeline.py`.

- [ ] Add `v2_pattern_obstacles = True` directly after `v2_pattern_flow`.
- [ ] Implement
  `pattern_wall_candidates(notes) -> tuple[list[dict], dict]`.
  Return zero proposals when flow is disabled, the new flag is false,
  obstacles are disabled, or fewer than eight anchored onset groups exist.
  Build chronological all-hand onset groups anchored at the first member,
  using `min_time_diff`. Inspect non-overlapping groups of eight. Their seven
  gaps must be in [.1,.4], each within max(.015, .15*median) of the median.
  Require window start >= previous eligible window start + 4 seconds.
  Count eligible windows; advance cooldown even if both accents are blocked.
- [ ] Candidate pulse times are group 0 and group 4. Alternate preferred lane
  0,3 within each window; try the opposite only if the preferred lane is not
  clear. Geometry and duration are exact:

```python
wall = dict(kind="wall", pattern="flow_edge_accent", x=lane, y=2,
            width=1, height=1, time=t,
            end_time=t + max(.125, float(config.obstacle_min_duration) + 1e-6))
```

  Precompute `_swing_cells` once per note. Use bisect to find notes in
  `[time-obstacle_time_gap[0], end_time+obstacle_time_gap[1]]`;
  `_wall_contains` must be false for every nearby swing cell.
  At most one proposal per pulse/two per window; never move notes.
  Diagnostics: windows/proposed. Final hazard validation handles chains,
  song limits, existing walls, allowed types/positions and intro policy.
- [ ] In `_hazards_stage`'s heuristic branch only, extend existing proposals
  with the candidates before `validate_hazards`. Store diagnostics under
  `pattern_obstacles`; after validation add `accepted` by counting surviving
  walls with pattern=flow_edge_accent. Raw wall proposal count includes them.
  Leave `_heuristic_wall_candidates`'s existing public behavior unchanged.
- [ ] Tests in the pipeline test file: deterministic regular two-pulse output,
  irregular/short inputs produce none, mirrored obstruction chooses the clear
  side, whole swing blocks a wall even if its note center is clear, fixed
  cooldown/proposal bound, flags-off behavior, and retained notes unchanged.
  Use the real final validator to verify a chain crossing a proposed slab
  removes it; confirm valid edge accents survive serialization.
- [ ] Update the existing bounded-flow section of AGENTS.md with the final
  limits, motif policy, feature flag and diagnostic fields. Correct obsolete
  unconditional diagonal-cap wording only where it describes this mode.

## Verification and review

- [ ] Focused CPU tests (one combined pass after both tasks):
  `/home/frede/.venvs/infernosaber/bin/python -m pytest -q tests/test_v2_forward_flow.py tests/test_v2_forward_postprocessing.py tests/test_v2_forward_expansion.py tests/test_v2_note_emphasis.py tests/test_v2_generation_pipeline.py tests/test_v2_generation_postprocessing.py tests/test_v2_calm_effects.py`
- [ ] Lead reviews entire diff and frozen comparison evidence before accepting.
- [ ] Execute the lead-authored external `generate.py candidate` harness, then
  lead runs/checks measurement and independent audits. Do not author or modify
  measurement scripts, scoring, thresholds or sampling policies in delegation.
- [ ] One final full CPU gate:
  `/home/frede/.venvs/infernosaber/bin/python -m pytest -q -m 'not comparison'`
- [ ] Preserve generated packages outside the repo. No installation or push.
