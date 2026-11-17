# Adjacent-lane flow implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Reduce postprocessing-induced lane clustering while retaining safe flow.

**Architecture:** Prevent phrase repairs from amplifying lane repetition, add
repetition tie-breaking and one guarded adjacent-lane overlay before the retained
snapshot. Preserve the existing early repair order.

**Tech Stack:** Python, NumPy, pytest, retained WSL CPU environment.

**Spec:** `docs/superpowers/specs/2026-11-17-adjacent-lane-flow-design.md`

## Global constraints

Preserve existing hard gates, speed limits, note timing/identity/hand/count and
line endings. Overlays change x/y/d only.
No unbounded convergence, no dependencies, no unrelated cleanup, no commits or
pushes. The lead owns measurement code, policy constants and acceptance checking.

## Review focus

- Preserve the existing direction/overlap stages and their final hard audits.
- A changed predecessor can affect successor/dot context; inspect it before
  accepting any pattern edit.
- Repetition ranking must consider triples across window boundaries.
- Failed or exceptional overlays must restore every changed coordinate/direction.
- Same-hand grouping ambiguity, pauses and short intentional repeats must remain
  excluded from the overlay.

## Task 1: Direction-order experiment — rejected

The isolated six-map replay did not support early direction repair as a
countermeasure. Remove this experiment and its tests; restore the existing
direction/forward-stage files before implementing Task 2.

## Task 2: Repetition preference and guarded lane overlays

Files: new `map_creation/v2_forward_lanes.py`,
`map_creation/v2_forward_patterns.py`, existing
`tests/test_v2_forward_flow.py`, `AGENTS.md`.

- [ ] Implement `repetition_cost(context, notes) -> int` in the new module.
  `notes` are replacements keyed by existing IDs, so singleton trials need
  not mutate live context. For each affected hand, examine members from two
  before its minimum affected position through two after its maximum.
  Count only triples touching an affected ID; require three singleton arrow
  groups, gaps `(flow._eps(), flow._RUN_GAP + flow._TOL]`, and equal x.
  Use actual unaffected context members; replacements supply edited x.
- [ ] Import the module as `lanes` in patterns. In `_check_chunk`, return:

```python
return (*agg, lanes.repetition_cost(context, notes), displacement, deviation)
```

  Do not change acceptance guards or strict style-improvement semantics.
  In `_repair_note`, compute repetition cost once per possible x using
  `lanes.repetition_cost(context, [dict(note, x=x)])`. Rank cheap proposals:

```python
(cost, repetition_by_x[x], abs(x-x0)+abs(y-y0), deviation,
 abs(x-x0), x, y, d)
```

  Adjust tuple unpacking. Keep at most 12 expensive singleton checks.
- [ ] In `_phrase_hand`, capture `before_repetition =
  lanes.repetition_cost(context, notes)` before trials. After `_check_chunk`
  returns a key, reject it via `_reject(report, "phrase", "lane_repetition",
  notes)` when `key[4] > before_repetition`, regardless of style improvement.
  This is the direct countermeasure for the last commit's observed regression.
- [ ] In `_phrase_candidates`, replace `outer` with:

```python
lowered = tuple((x, 0 if x in (1, 2) and y == 1 else y, d)
                for x, y, d in saved)
```

  Append `("lower_center", lowered)` where the old `outer` proposal was added.
  Keep translation, static motifs, deduplication and the 12-phrase-trial bound.
- [ ] Implement `repair_lanes(context, index, cap, report)` in the new module.
  Import patterns inside this function to avoid an import cycle. Call it
  immediately after `_local_repair` and before the residual audit.
  Add flat report fields `lateral_windows`, `lateral_proposals`,
  `lateral_applied`, `lateral_notes`, `lateral_motifs`.
- [ ] For each hand, use existing `_hand_groups`. Maximal runs must have the
  same x, singleton arrows with `len(context.group(note)) == 1`, and every gap
  in [.1-TOL,.4+TOL]. Skip runs shorter than four. Consume non-overlapping
  prefixes of at most eight; if that prefix is irregular and longer than four,
  shorten it to four. Advance by the selected length, skip a tail shorter than
  four. Skip a chunk
  adjoining a dot or a non-singleton context group.
- [ ] Candidate order is two alternating outer-pair x assignments, preserving
  y/d, then two adaptive lane/height candidates preserving d. For each phase:

```python
x = outer[(i + phase) % 2]
y, d = note["y"], note["d"]
if x == outer[1]:
    if y == 1 and d in (0, 4, 5):
        y = 2
    elif y in (1, 2) and d in (1, 6, 7):
        y = 0
```

  Label these `adjacent_layers`. Then (only if `_regular(chunk)`) all phases of these exact
  red-hand cycles, mirrored using the existing `_MIRROR_D`:

```python
_LATERAL_CYCLES = (
    ("adjacent_rebound", ((0, 0, 1), (1, 0, 0))),
    ("adjacent_rocker", ((0, 1, 4), (1, 0, 1))),
)
```

  X-only candidates use red columns 0/1, blue columns 3/2, with two phases.
  Repeat/truncate cycles to chunk length. Deduplicate and exclude incumbent.
  After the optional regular motifs, add two `compact_bottom` candidates:

```python
for phase in range(2):
    geometry = list(saved)
    for i in range(1, len(notes) - 1):
        geometry[i] = (outer[(i + phase) % 2], 0, notes[i]["d"])
    candidates.append(("compact_bottom", tuple(geometry)))
```

  These also serve irregular timing. At most 2 x-only + 2 adaptive + 2 rebound
  + 2 rocker + 2 compact-bottom = 10 trials per window. No static arrow
  template on irregular timing. The independent original wave pass is unchanged.
- [ ] Prepare using `_prepare`; save every x/y/d. For each trial mutate
  atomically, call `_check_chunk(..., strict=False, report=report,
  family="lateral")`, and restore in finally. Additionally require a strictly
  lower repetition cost than the incumbent and no individual increase in the
  four `agg` components. Use `_reject` for repetition/style-regression reasons.
  Choose by `(new_repetition, *key, candidate_order)`, apply once, count actual
  changed notes and the winning motif. Never reconsider an accepted chunk.
- [ ] Tests in the existing flow file: repetition counts 6 for an eight-note
  constant lane and 0 for ABAB, includes crossing-window triples and excludes
  dots/stacks/near-epsilon ambiguity. Equal-style mixed candidates prefer
  lower repetition without overriding a hard guard. An eight-note outer lane
  at .2 s intervals and generous cap gains adjacent transitions, keeps all
  non-geometry fields/count, passes real audits and creates no center notes.
  Mirror the fixture and compare results/counts. Preserve three-note repeats,
  an already-alternating stream, dots, stacks and ambiguous boundary context.
  Test blocked successor/opposite hand, hard speed rejection, rollback on
  rejection and exception, no upper-down/style increase, and
  `lateral_proposals <= 10*lateral_windows` on a long stream.
  Update existing tests only where their full-repair assertion contradicts
  the newly authorized treatment of long same-lane runs; keep narrow lane-shift
  expectations intact. The previous mixed-column speed-blocked fixture that
  rewards collapsing `0,0,1,1` to a constant lane is now a negative regression:
  preserve its strict cap and assert repetition cannot increase. Add a positive
  mirrored phrase fixture with cells `(1,1,0),(0,0,1),(1,1,0),(0,0,1)` and cap
  99: center occupancy decreases, repetition stays zero, non-geometry fields
  and count stay fixed, and real direction audits pass.
  Extend the strict-cap mixed-column fixture to assert the new lower_center
  candidate clears its middle cells while preserving x and d.
  Compact-bottom fixture: red x0, y/d `(0,1),(1,0),(0,1),(1,0)`,
  times `0,.2,.4,.6`, cap15. Expected geometry
  `(0,0,1),(1,0,0),(0,0,1),(0,1,0)`, one compact_bottom application,
  unchanged endpoints/directions, zero repeated-lane triples and clean real
  direction/overlap audits; mirror for blue.
- [ ] Update AGENTS' bounded-flow section for the phrase repetition guard,
  repetition preference and final bounded lane overlay.
- [ ] Run focused gate:
  `/home/frede/.venvs/infernosaber/bin/python -m pytest -q tests/test_v2_forward_flow.py tests/test_v2_forward_postprocessing.py tests/test_v2_forward_expansion.py tests/test_v2_note_emphasis.py tests/test_v2_generation_pipeline.py tests/test_v2_generation_postprocessing.py tests/test_v2_calm_effects.py`

## Final acceptance

- [ ] Execute the lead-authored replay/generation recipe; lead checks the six
  frozen round-two candidates and a same-code baseline for the seventh
  Heeartbeeat reference, not the older round-one baseline.
- [ ] Investigate any hard-audit failure or increased hard drops. Report
  actual lane/cell/ABAB metrics, previous style targets, note counts and runtime.
- [ ] One full CPU suite, with the existing baseline Tk runtime path loaded:
  `/home/frede/.venvs/infernosaber/bin/python -m pytest -q -m 'not comparison'`.
- [ ] Lead whole-diff review and independent final review; no publishing.
