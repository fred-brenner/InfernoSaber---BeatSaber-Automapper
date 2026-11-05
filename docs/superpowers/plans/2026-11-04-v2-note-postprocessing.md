# v2 Note Postprocessing: Intro Zones, Strict Flow, Diagonal Cap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add v2-only configurable start-of-map safety zones (notes < 2 s, hazards < 1 s), ban the awkward 90° same-hand turn, and cap diagonal cut directions at a configurable fraction (default 10%).

**Architecture:** Four new `v2_*` config knobs in `tools/config/config.py`. Two new postprocessors (`apply_intro_floor`, `apply_diagonal_cap`) in `map_creation/v2_postprocessing.py`, wired into `generate()` in `map_creation/v2_pipeline.py` between `snap_note_times` and the retained-note snapshot so downstream expansion/arcs/invariants see final state. The 90° ban is a one-line rule change in `_flow_legal` plus its duplicated copy in the `postprocess_notes` dedup `rank`; hazard filtering lives inside `validate_hazards` so generated and calm-effect hazards are both covered.

**Tech Stack:** Python, numpy, pytest. No new dependencies.

**Spec:** `todo_after_playtest.md` (repo root), cases 1–3.

## Global Constraints

- **v2 generation only.** `evaluation/legacy_generator.py` and the whole v1 path stay untouched.
- All timestamps are elapsed seconds until `v2_export.py` converts once at the end. No unit conversion anywhere.
- New config knobs in `tools/config/config.py` with exact names/values:
  - `v2_map_intro_no_notes_seconds = 2.0`
  - `v2_map_intro_no_hazards_seconds = 1.0`
  - `v2_flow_min_turn_steps = 3`  (45° steps; `2` restores the legacy rule)
  - `v2_diagonal_max_fraction = 0.10`  (`>= 1` disables the cap)
- Retained-note invariant (`v2_pipeline.py:707` — full dict equality including `d`) must keep passing: all note direction/level changes land **before** `retained = deepcopy(...)` at `v2_pipeline.py:534`.
- Lighting/events untouched: `state["event_times"]` and `state["lighting"]` are never filtered.
- Deterministic randomness: the diagonal cap draws from the pipeline `rng` (`np.random.Generator` at `v2_pipeline.py:418`); tests seed it.
- New postprocessors must also be added to the `SimpleNamespace` at `v2_pipeline.py:402-413` (it is the production default when `v2_cleanup_note_heuristics` is False) and called via the `getattr(postprocessors, ..., None)` pattern like `snap_times` at `:527`.
- Tests follow `tests/test_v2_generation_postprocessing.py` conventions: `SimpleNamespace` song fixture, `monkeypatch.setattr(config, ...)`, `beats()/note()/state()/chain()` helpers.
- Run tests from the repository root: `python -m pytest tests/test_v2_generation_postprocessing.py -q`. Use the project Python environment (see `docs/development.md`); do not commit `__pycache__` or generated artifacts.

## Review Focus

1. **Duplicated 90° rule in the dedup `rank` (`v2_postprocessing.py:976-977`).** Updating only `_flow_legal` leaves `postprocess_notes`' same-hand candidate ranking still treating "90° + ≥2-cell travel" as legal. Owned by Task 2.
2. **Intro floor vs. `snap_note_times`.** Snapping runs first and can pull a ≥2 s note back under the floor; the filter must run after snap (Task 1), not inside `postprocess_notes`.
3. **Diagonal-cap recuts can reintroduce violations.** A recut must be `_flow_legal` against BOTH same-hand neighbors (predecessor and successor) and `_swing_clear` against simultaneous other-hand cells (`_other_hand_cells`), or it recreates exactly the conflicts `enforce_note_flow_and_clearance` resolved. Owned by Task 3.
4. **Walls straddling the hazard floor.** Spec pins: a wall is dropped when `wall["time"] < floor` (start-based; no clipping). A 0.9→1.5 s wall is dropped, not trimmed. Owned by Task 1.
5. **`v2_flow_min_turn_steps <= 2` must restore the exact legacy rule** (`steps >= 3 or (steps == 2 and travel >= 2)`), not "all 90° legal" — it is the regression escape hatch. Owned by Task 2.

---

### Task 1: Config knobs + start-of-map free zones (notes + hazards)

**Files:**
- Modify: `tools/config/config.py` (v2 settings block)
- Modify: `map_creation/v2_postprocessing.py` (new `apply_intro_floor`; `validate_hazards` at ~:1110-1175)
- Modify: `map_creation/v2_pipeline.py` (SimpleNamespace `:402-413`; hook after `:532`, before `retained` snapshot at `:534`)
- Test: `tests/test_v2_generation_postprocessing.py`

**Interfaces:**
- Produces: `apply_intro_floor(notes: list[dict], cutoff_seconds: float) -> int` — in-place filters `notes` to `time >= cutoff_seconds`, returns dropped count.
- Consumes: `validate_hazards(state, song, logger=None)` — reads `config.v2_map_intro_no_hazards_seconds` internally; signature unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_intro_floor_drops_notes_before_cutoff():
    notes = [note(0, .5), note(1, 1.9), note(2, 2.0), note(3, 3.)]
    assert post.apply_intro_floor(notes, 2.0) == 2
    assert [n["time"] for n in notes] == [2.0, 3.]

def test_intro_floor_zero_keeps_all():
    notes = [note(0, 0.), note(1, .5)]
    assert post.apply_intro_floor(notes, 0.) == 0
    assert [n["time"] for n in notes] == [0., .5]
```

Plus a `validate_hazards` test: `state` with a wall `time=.4, end_time=.9`, a wall `time=1.2, end_time=1.6`, a bomb `time=.9`, a bomb `time=1.5` (integer geometry, duration 10 song, no notes so no clearance interference). With `v2_map_intro_no_hazards_seconds = 1.0`: only `walls == [1.2]` and `bombs == [1.5]` survive; `rejected_intro` diagnostic reports 2. Also monkeypatch the floor to `0.0` to verify the disable path keeps all four.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q -k intro_floor or hazard`
Expected: FAIL — `apply_intro_floor` does not exist; hazards pass through unfiltered.

- [ ] **Step 3: Implement**

- `config.py`: add `v2_map_intro_no_notes_seconds = 2.0` and `v2_map_intro_no_hazards_seconds = 1.0` in the v2 block.
- `v2_postprocessing.py`:
  - `def apply_intro_floor(notes, cutoff_seconds):` — `before = len(notes); notes[:] = [n for n in notes if n["time"] >= cutoff_seconds]; return before - len(notes)` (in-place, consistent with `notes[:] = ...` at `:635`).
  - In `validate_hazards`, near `:1115`: `intro_floor = float(getattr(config, "v2_map_intro_no_hazards_seconds", 0.) or 0.)`; extend the wall early-continue at `:1123` with `or start < intro_floor`; extend the bomb filter at `:1160` with `or bomb["time"] < intro_floor`; track `rejected_intro` count and add it to the returned diagnostics dict.
- `v2_pipeline.py`:
  - Add `apply_intro_floor=base.apply_intro_floor` to the `SimpleNamespace` (`:402-413`).
  - After the `snap_times` block (`:527-532`), before `started = perf_counter()` at `:533`:

```python
    intro_fn = getattr(postprocessors, "apply_intro_floor", None)
    intro_floor = float(getattr(config, "v2_map_intro_no_notes_seconds", 0.) or 0.)
    if intro_fn is not None and intro_floor > 0:
        result.diagnostics["intro_zones"] = {
            "no_notes_seconds": intro_floor,
            "notes_dropped": intro_fn(state["notes"], intro_floor)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q`
Expected: PASS (new tests + no regressions).

- [ ] **Step 5: Commit**

```bash
git add tools/config/config.py map_creation/v2_postprocessing.py map_creation/v2_pipeline.py tests/test_v2_generation_postprocessing.py
git commit -m "Add configurable v2 intro safety zones for notes and hazards"
```

---

### Task 2: Strict same-hand flow — ban all 90° turns

**Files:**
- Modify: `tools/config/config.py` (add `v2_flow_min_turn_steps = 3`)
- Modify: `map_creation/v2_postprocessing.py` (`_flow_legal` at `:327-340`; dedup `rank` at `:976-977`)
- Test: `tests/test_v2_generation_postprocessing.py`

**Interfaces:**
- Consumes: Task 1's config-file pattern (knobs live in the same v2 block).
- Produces: `_flow_legal` now reads `config.v2_flow_min_turn_steps`; semantics — `steps >= min_turn` legal, `steps <= 1` illegal, `min_turn <= 2` restores the legacy `steps >= 3 or (steps == 2 and travel >= 2)` rule.

- [ ] **Step 1: Write the failing tests**

```python
def test_flow_legal_requires_135_degrees_by_default(monkeypatch):
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 3)
    # middle down-right -> left up-right: 90° over a 2-cell travel (was legal)
    assert not post._flow_legal(5, (0., 2, 1, 7), 0, 1, .5, .1)
    # down-right -> up is 135°, still legal
    assert post._flow_legal(0, (0., 2, 1, 7), 0, 1, .5, .1)
    # same direction stays illegal
    assert not post._flow_legal(7, (0., 0, 1, 7), 1, 1, .5, .1)
    # same-stroke exemption unchanged
    assert post._flow_legal(5, (0., 2, 1, 7), 0, 1, .05, .1)

def test_flow_min_turn_two_restores_travel_exemption(monkeypatch):
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 2)
    assert post._flow_legal(5, (0., 2, 1, 7), 0, 1, .5, .1)       # 90° + 2 cells: legacy legal
    assert not post._flow_legal(5, (0., 2, 1, 7), 1, 1, .5, .1)   # 90° + 1 cell: still illegal
```

Plus one `postprocess_notes`-level test asserting that when two same-hand candidates compete at one event time, the dedup `rank` survivor is now ≥135° from the predecessor (previously the 90°+travel option won). Build it with the existing `note()`/`postprocess_notes` helpers; `skip=post.LEGACY_NOTE_MUTATORS + ("clearance",)` keeps the run minimal.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q -k flow_legal or min_turn`
Expected: FAIL — first assertions hold today's legacy behavior.

- [ ] **Step 3: Implement**

- `config.py`: `v2_flow_min_turn_steps = 3`.
- `_flow_legal` (`:327-340`): replace `return steps >= 3 or abs(x - px) + abs(y - py) >= 2` with:

```python
    min_turn = int(getattr(config, "v2_flow_min_turn_steps", 3) or 3)
    if min_turn <= 2:
        return steps >= 3 or (steps == 2 and abs(x - px) + abs(y - py) >= 2)
    return steps >= min_turn
```

  Update the docstring (">90° rule" → "≥135° rule" wording; keep the same-stroke exemption description).
- Dedup `rank` (`:969-978`): `previous` there is already the `(time, x, y, effective_d)` tuple `_flow_legal` expects — replace the inline `illegal = steps <= 1 or (steps == 2 and travel < 2)` with `illegal = not _flow_legal(option[2], previous, option[0], option[1], time, same_stroke_gap)`. (`_step_distance` is still needed for `-steps` ranking.)
- No other call sites need changes: `recut_same_direction_notes` (`:417`), `_resolve_clashes` (`:610`), `clearance_violations` (`:666`), `_snap_violated` (`:755,761`) and `_recut_same_direction_repeats` (`:379`) all funnel through `_flow_legal`. `_flow_direction` already requires `>= 3` steps for recut candidates (`:355-356`), consistent with the new default.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/config/config.py map_creation/v2_postprocessing.py tests/test_v2_generation_postprocessing.py
git commit -m "Require >=135 degree same-hand turns in v2 flow cleanup"
```

---

### Task 3: Diagonal-cut fraction cap

**Files:**
- Modify: `tools/config/config.py` (add `v2_diagonal_max_fraction = 0.10`)
- Modify: `map_creation/v2_postprocessing.py` (new `apply_diagonal_cap`)
- Modify: `map_creation/v2_pipeline.py` (SimpleNamespace + hook between intro floor and `:534`)
- Test: `tests/test_v2_generation_postprocessing.py`

**Interfaces:**
- Consumes: Task 2's strict `_flow_legal`; `_history_context` (`:472`), `_other_hand_cells` (`:456`), `_swing_clear` (`:311`), `_DIRECTION_STEP`/`_REVERSE_DIRECTION` internals — same module.
- Produces: `apply_diagonal_cap(notes: list[dict], chains: list | None = None, rng: np.random.Generator | None = None) -> dict` — mutates `d` on chosen notes in place and syncs `chain["d"]` to its anchor (same tail behavior as `recut_same_direction_notes` at `:440-443`); returns `{"diagonal_before": int, "diagonal_after": int, "recut": int, "kept_over_cap": int, "fraction_after": float}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_diagonal_cap_limits_fraction():
    # 20 notes 0.5 s apart; each hand alternates antipodal diagonals (d5/d6,
    # 180 deg apart) so each note's adjacent cardinals sit 135 deg from both
    # same-hand neighbours -> legal under the strict Task-2 rule
    notes = [note(i, 1. + .5 * i, hand=i % 2, x=i % 4, y=1,
                 direction=5 if (i // 2) % 2 == 0 else 6) for i in range(20)]
    stats = post.apply_diagonal_cap(notes, rng=np.random.default_rng(0))
    remaining = sum(n["d"] in (4, 5, 6, 7) for n in notes)
    assert remaining <= int(0.10 * 20)
    assert stats["diagonal_before"] == 20 and stats["diagonal_after"] == remaining

def test_diagonal_cap_prefers_vertical_on_tie():
    # lone diagonal: no neighbours -> both cardinals legal, equal cost -> vertical wins
    notes = [note(0, 1., direction=7)]  # down-right -> candidates down(1) / right(3)
    post.apply_diagonal_cap(notes, rng=np.random.default_rng(0))
    assert notes[0]["d"] == 1

def test_diagonal_cap_keeps_when_no_legal_cardinal():
    # pin a diagonal between same-hand neighbours so both cardinals would be <135°
    notes = [note(0, 1., direction=7), note(1, 2., x=0, y=1, direction=5),
             note(2, 3., x=3, y=1, direction=4)]
    post.apply_diagonal_cap(notes, rng=np.random.default_rng(0))
    assert notes[1]["d"] == 5

def test_diagonal_cap_deterministic_per_seed():
    build = lambda: [note(i, 1. + .5 * i, hand=i % 2, x=i % 4, y=1,
                          direction=5 if (i // 2) % 2 == 0 else 6) for i in range(20)]
    a, b = build(), build()
    post.apply_diagonal_cap(a, rng=np.random.default_rng(42))
    post.apply_diagonal_cap(b, rng=np.random.default_rng(42))
    assert [n["d"] for n in a] == [n["d"] for n in b]

def test_diagonal_cap_disabled_at_one(monkeypatch):
    monkeypatch.setattr(config, "v2_diagonal_max_fraction", 1.0)
    notes = [note(i, 1. + .5 * i, hand=i % 2, x=i % 4, y=1,
                 direction=5 if (i // 2) % 2 == 0 else 6) for i in range(20)]
    post.apply_diagonal_cap(notes, rng=np.random.default_rng(42))
    assert [n["d"] for n in notes] == [5 if (i // 2) % 2 == 0 else 6 for i in range(20)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q -k diagonal`
Expected: FAIL — `apply_diagonal_cap` does not exist.

- [ ] **Step 3: Implement**

`apply_diagonal_cap(notes, rng=None)` in `v2_postprocessing.py`:

- `cap = float(getattr(config, "v2_diagonal_max_fraction", 1.) or 0.)`; no-op when `cap >= 1` or no notes. Clamp negative cap to 0.
- `rng = np.random.default_rng() if rng is None else rng`.
- `ordered = sorted(notes, key=lambda n: (n["time"], n["id"]))`; `pred, incoming = _history_context(ordered, eps)`; `other_cells = _other_hand_cells(ordered, eps, window)` (`eps`/`window` same config reads as `_resolve_clashes`).
- `diagonals = [n for n in ordered if n["d"] in (4, 5, 6, 7)]`; `excess = max(0, len(diagonals) - int(cap * len(notes)))`.
- Pick `excess` of them via `rng.choice(len(diagonals), size=excess, replace=False)`.
- For each chosen note: adjacent cardinals `{4: (0, 2), 5: (0, 3), 6: (1, 2), 7: (1, 3)}`; keep candidates that pass `_flow_legal(c, prev_tuple, x, y, t, gap)` AND `_flow_legal(next_effective_d, (t, x, y, c), nx, ny, next_t, gap)` AND `_swing_clear((x, y), other_cells.get(note["id"]), c)` (`gap` = `v2_cleanup_same_dir_gap`, default `.1`).
- Choose `min(legal, key=(cost, not_vertical))`: `cost` = Manhattan distance from the previous same-hand cut's exit cell to this candidate's approach cell per the spec (`todo_after_playtest.md`): `|(px - pstep_x) - (x + step_c_x)| + |(py - pstep_y) - (y + step_c_y)|` (`_DIRECTION_STEP` is the approach vector, so `pos + step` = approach cell and `pos - step` = exit cell — same convention as `calc_note_speed`); all costs equal (`0`) when no predecessor exists; `not_vertical = c not in (0, 1)` — vertical wins ties. (Controller ruling: spec formula is binding — the earlier draft's exit→successor formula was superseded.)
- If `legal` is empty, keep the diagonal and count `kept_over_cap`.
- Chain sync tail like `recut_same_direction_notes` at `:440-443`: for each `chains` entry, `chain["d"] = anchors[chain["anchor_id"]]["d"]` when the anchor survives (no chains exist at the pipeline hook yet, but the signature matches its sibling postprocessors).

`v2_pipeline.py`:

- Add `apply_diagonal_cap=base.apply_diagonal_cap` to the `SimpleNamespace`.
- Between the intro-floor block and `started = perf_counter()` at `:533` (i.e., before `retained = deepcopy(...)`):

```python
    diag_fn = getattr(postprocessors, "apply_diagonal_cap", None)
    diag_cap = float(getattr(config, "v2_diagonal_max_fraction", 1.) or 0.)
    if diag_fn is not None and state["notes"] and 0 <= diag_cap < 1:
        result.diagnostics["diagonal_cap"] = diag_fn(state["notes"], state["chains"], rng=rng)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/config/config.py map_creation/v2_postprocessing.py map_creation/v2_pipeline.py tests/test_v2_generation_postprocessing.py
git commit -m "Add configurable diagonal-direction cap to v2 note cleanup"
```

---

### Task 4: Documentation + full focused verification

**Files:**
- Modify: `docs/generation-v2.md` (postprocessing section)
- Modify: `AGENTS.md` (Direction paragraph — one clause)
- Modify: `map_creation/v2_postprocessing.py` module docstring (mention intro floor + diagonal cap in the cleanup description)

- [ ] **Step 1: Update docs**

`docs/generation-v2.md`: document the four knobs, their defaults, where the filters sit in the pipeline order (post-snap/pre-retained-snapshot for notes; inside `validate_hazards` for hazards), and that the diagonal cap is best-effort (fraction may exceed cap when no legal cardinal exists). `AGENTS.md`: extend the v2 application description with "a configurable start-of-map safety zone, strict ≥135° same-hand flow, and a diagonal-direction cap".

- [ ] **Step 2: Full focused run**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py tests/test_v2_generation_pipeline.py tests/test_v2_calm_effects.py tests/test_v2_note_onset_snap.py -q`
Expected: PASS — all four focused files green.

- [ ] **Step 3: Commit**

```bash
git add docs/generation-v2.md AGENTS.md map_creation/v2_postprocessing.py
git commit -m "Document v2 intro zones, strict flow and diagonal cap"
```
