# Playability Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An evaluation scorer that rates generated beatmaps on four playability metrics (lateral load, occlusion, bimanual divergence, path interference) as percentiles/incident-rates against the curated human baseline.

**Architecture:** One numpy-only metric module operating on a prepared note list; one report module that decodes v2/v3 beatmap packages, builds the human corpus from `bs_map_input`, scores generated packages, and prints a corpus-relative console report + optional JSON.

**Tech Stack:** Python, numpy, pytest. No tensorflow/config/aubio/librosa imports in the new modules.

**Spec:** `docs/superpowers/specs/2026-11-06-playability-metrics-design.md` — metric definitions and decisions live there; this plan pins files, signatures and tests.

## Global Constraints

- New modules import only stdlib + numpy (`math`, `json`, `argparse`, `pathlib`, `collections`, `numpy`). No `tools.config`, no audio/model deps.
- Note list format everywhere: `(t_seconds, hand, x, y, d)` tuples sorted by time; `hand` 0=left, 1=right; `x∈0..3`, `y∈0..2`, `d∈0..8`.
- Time base is **seconds**; decoders convert beats×60/bpm per package.
- Cut-direction geometry follows `get_cut_dir_xy` semantics: step = approach offset; `approach = pos+step`, `exit = pos−step`. Copied locally (see spec "Shared geometry"); do not import `map_creation.*` (pulls aubio via `sanity_check`).
- Module constants (not config settings): `EPS=0.02`, `T_FAST=0.5`, `T_VIS=0.4`, `T_SEG=0.5`, `TAU_X=0.25`, `T_RESET=0.4`, `ISO_MIN=0.6`, `NPS_BANDS=((0,3),(3,5),(5,7),(7,inf))`.
- Invalid packages become skipped rows with reasons; a corpus run never crashes on one bad package.
- Commit only the task's own new files; working tree must stay clean otherwise.

## Review Focus

1. Same-hand stacks within `EPS` must collapse to one note (contract parity) — Task 1.
2. Simultaneous (Δt≤EPS) cross-hand partners must never count as M2 occluders, and exact mirrored doubles must zero M4 severity — Tasks 3/5.
3. First-note dots / unresolved dots have `eff=None` → swing volume degenerates to own cell only; no crash — Tasks 1/5.
4. Packages with non-Standard characteristics, missing/renamed beatmap files, real tempo changes, or <2 notes → skipped rows, not exceptions — Tasks 6/7.
5. Empty bands / tiny corpus: scalar percentile falls back to all-band when `band_n<5`; empty generated folder still prints the human table — Task 7.

---

### Task 1: Note preparation + geometry (`evaluation/playability_metrics.py`)

**Files:**
- Create: `evaluation/playability_metrics.py`
- Test: `tests/test_playability_metrics.py`

**Interfaces:**
- Produces: `prepare_notes(notes) -> list[dict]` where each dict is
  `{"t", "hand", "x", "y", "d", "eff", "pred", "exit_cell"}`; `eff` = effective
  direction (dot resolved through predecessor geometry, `None` when
  unresolvable); `pred` = same-hand predecessor dict or `None`;
  `exit_cell` = `(x - STEP[eff][0], y - STEP[eff][1])` or `None`.
  Helpers other tasks/tests use: `swept_cells(x, y, eff) -> set[(x,y)]`
  (own cell excluded — approach+exit only, in-grid), `corridor(a, b) -> set`
  (supercover cells between two cells, off-grid endpoints clamped to grid),
  `MIRROR_DIR`, `STEP`.

- [ ] **Step 1: Write failing tests** in `tests/test_playability_metrics.py`
  (repo convention: `ROOT_DIR` sys.path shim like `test_human_maps.py`):
  - `test_step_table_matches_get_cut_dir_xy`: `STEP[0]==(0,-1)`,
    `STEP[3]==(-1,0)`, `STEP[2]==(1,0)`, `STEP[5]==(-1,-1)` (up/right/left/
    up-right approach offsets).
  - `test_prepare_collapses_same_hand_stacks`: two hand-0 notes 0.01s apart
    → one kept; a hand-0 + hand-1 pair 0.01s apart → both kept.
  - `test_dot_implied_direction`: seq `(0s,h0,x=1,y=1,d=1)`, dot at
    `(0.4s,h0,x=1,y=0)` — down-cut at (1,1) exits at (1,0); the dot sits on
    the exit cell → inherits the pendulum reversal → `eff == 0` (up).
    (Verify against `_dot_virtual_direction` semantics in
    `v2_postprocessing.py:519-532` — copy that function verbatim.)
  - `test_swept_cells`: `swept_cells(1,1,3) == {(2,1)}` (approach cell in
    grid), `swept_cells(3,1,3) == set()` (approach off-grid), exit cell
    (0,1) → `swept_cells(1,1,2) == {(0,1),(2,1)}`.
  - `test_corridor`: `corridor((0,0),(3,0))` ⊇ all of row 0; off-grid
    `(4,1)` clamps to `(3,1)`.

- [ ] **Step 2: Run tests** — `python -m pytest tests/test_playability_metrics.py -x -q`
  Expected: FAIL (module missing).

- [ ] **Step 3: Implement** constants, `STEP`/`MIRROR_DIR`/`REVERSE` tables,
  `_dot_virtual_direction` (verbatim copy + source comment), `swept_cells`,
  `corridor` (Bresenham over clamped endpoints), `prepare_notes`
  (stable sort by t; same-hand within EPS → first wins; attach `pred`,
  `eff`, `exit_cell`).

- [ ] **Step 4: Re-run tests** — all pass.

- [ ] **Step 5: Commit** `git add evaluation/playability_metrics.py tests/test_playability_metrics.py`
  and commit `Add playability metrics scaffold: note prep and swing geometry`.

### Task 2: M1 lateral load

**Files:** Modify `evaluation/playability_metrics.py`; same test file.

**Interfaces:** `m1_lateral(prepared) -> {"value": float, "units": np.ndarray,
"bands": {"<=0.2","0.2-0.35",">0.35": share}, "transitions": int}` — `units`
holds positive `w·s` contributions only (pooled-percentile input).

- [ ] **Step 1: Failing tests**
  - `test_m1_dense_horizontal_stream`: hand-0 notes every 0.2s alternating
    x=0/x=1 all `d=3`/`d=2` … wait — use `d=3` alternating positions;
    `value ≈ 1.0`.
  - `test_m1_vertical_control`: same timing, all `d`∈{0,1} → `value == 0`.
  - `test_m1_slow_lateral_discounted`: lateral cuts at 1.0s gaps →
    `value == 0` (s(g)=0) while `bands[">0.35"] == 1.0`.
  - `test_m1_dot_is_relief`: dot transitions contribute `w=0`.

- [ ] **Step 2–4:** `w(d)`: 0 for d∈{0,1,8}, 0.5 for d∈{4..7}, 1 for
  d∈{2,3}; `s(g)=max(0,1-g/T_FAST)`; per-hand transitions via `pred` chain;
  `value=Σw·s/Σs` (0 when Σs=0); banded = mean(w) per gap band.

- [ ] **Step 5: Commit** `Add M1 lateral-load metric`.

### Task 3: M2 occlusion

**Files:** same two files.

**Interfaces:** `m2_occlusion(prepared) -> {"value","units","incidents"}` —
`units` = positive `occl` per note; `incidents` = occl≥0.5 dicts
`{"t","note":(x,y),"occluder":(x,y),"dt","occl"}`.

- [ ] **Step 1: Failing tests**
  - `test_m2_rising_same_column`: notes `(1.0,h0,1,0,d)` then
    `(1.2,h1,1,2,d)` → target occl = `2·1·(1−0.2/0.4)=1.0`.
  - `test_m2_simultaneous_not_occluder`: same-time pair bottom/top same
    column → `units` empty.
  - `test_m2_nearest_occluder_only`: high occluder at Δt=0.05 (no occl),
    lower note at Δt=0.3 → occl==0 (nearest wins).
  - `test_m2_column_gate`: `|Δx|=2` → no occluder.

- [ ] **Step 2–4:** For each note j: candidates `i` with `EPS < t_j−t_i ≤
  T_VIS` and `|Δx|≤1`; occluder = max `t_i`;
  `occl=max(0,y_j−y_i)·(1−|Δx|/3)·(1−Δt/T_VIS)`; `value=Σoccl/N`.

- [ ] **Step 5: Commit** `Add M2 occlusion metric`.

### Task 4: M3 bimanual divergence

**Files:** same two files.

**Interfaces:** `m3_divergence(prepared) -> {"value","units","segments"}` —
`units` = per-segment divergence values (segments where both hands defined).

- [ ] **Step 1: Failing tests**
  - `test_m3_parallel_vertical`: both hands oscillating y-only over a 2s
    segment → `value ≈ 0`.
  - `test_m3_orthogonal`: left vertical wave, right horizontal wave, same
    window → `value ≈ 1`.
  - `test_m3_isotropic_skipped`: one hand's notes scattered (no dominant
    axis) → segment excluded, `segments` counts only valid ones.
  - `test_m3_idle_hand_ignored`: single-hand segment contributes nothing.

- [ ] **Step 2–4:** Events = unique times; split at gaps >T_SEG; per hand
  per segment collect displacement vectors (consecutive same-hand notes
  inside segment); need ≥2 vectors; axis = top eigenvector of `Σvvᵀ`
  (`np.linalg.eigh` on 2×2); defined iff `λ1/(λ1+λ2) ≥ ISO_MIN`;
  `div = |u_L.x·u_R.y − u_L.y·u_R.x|`; `value` = weighted mean by
  `min(nvec_L, nvec_R)`.

- [ ] **Step 5: Commit** `Add M3 bimanual divergence metric`.

### Task 5: M4 path interference + `score_map`

**Files:** same two files.

**Interfaces:** `m4_interference(prepared, minutes) -> {"value","units",
"incidents"}` — `value` = Σseverity/minute; `units` = positive severities;
`incidents` = top-`k` dicts `{"t","a":(x,y,d),"b":(x,y,d),"cells","severity",
"crossover"}` sorted desc. `score_map(notes) -> {"notes","span_seconds",
"nps","m1","m2","m3","m4"}`; `swing_volume(note) -> set` (own cell ∪
`swept_cells` ∪ gated corridor).

- [ ] **Step 1: Failing tests**
  - `test_m4_mirrored_double_zero`: `(t,0,1,1,d=2)` + `(t,1,2,1,d=3)` →
    severity 0, `units` empty.
  - `test_m4_inward_clash`: simultaneous `(0,1,1,d=3)` + `(1,2,1,d=0)` →
    left sweep covers `(2,1)` which is right's own cell → severity >0
    (deliberately non-mirrored; an exact inward pair would be discounted).
  - `test_m4_unresolved_dot_own_cell_only`: first note `d=8` → `eff=None`
    → swing volume is just its own cell, no crash.
  - `test_m4_left_cuts_right_into_active_right`: left note `(t,1,1,d=3)`
    sweeps `(2,1)` while right hand notes at `t±0.1` occupy `(2,1)`/`(2,2)`
    with corridor → severity >0 (the user's flagship example).
  - `test_m4_crossover`: left `(t,2,1)` right `(t+0.1,1,1)` → crossover
    term `(2−1)·(1−0.1/0.25)`.
  - `test_m4_corridor_reset`: same setup with a 1.0s same-hand gap → no
    corridor cells (gap>T_RESET).
  - `test_score_map_shape`: keys + `nps == notes/span`.

- [ ] **Step 2–4:** `swing_volume(n)`: `{(x,y)} ∪ swept_cells(x,y,eff) ∪
  (corridor(clamp(pred.exit_cell), clamp(approach)) if pred and
  t−pred.t≤T_RESET else ∅)`; approach = `pos+STEP[eff]` or own cell when
  `eff=None`; pred start = its `exit_cell` or own cell. Pairs: hand-0 ×
  hand-1 within TAU_X; `sev = |shared|·(1−|Δt|/TAU_X)` + crossover
  `max(0,x_a−x_b)·(1−|Δt|/TAU_X)`; exact mirror (`|Δt|≤EPS`, `x_a+x_b==3`,
  `y` equal, `d_b==MIRROR_DIR[d_a]`) → 0. `score_map` runs prepare once.

- [ ] **Step 5: Commit** `Add M4 path-interference metric and map scorer`.

### Task 6: Package decoder (`evaluation/playability_report.py`)

**Files:**
- Create: `evaluation/playability_report.py`
- Test: `tests/test_playability_report.py`

**Interfaces:**
- `DIFFICULTY_ORDER = ["Easy","Normal","Hard","Expert","ExpertPlus"]`
- `list_standard_difficulties(package) -> list[str]` (present difficulties,
  canonical order)
- `decode_package_notes(package, difficulty=None) -> {"notes":[(t,hand,x,y,d)],
  "bpm": float, "difficulty": str}` — `None` → highest listed; raises on any
  validation failure.

- [ ] **Step 1: Failing tests** (synthetic `tmp_path` packages — copy the
  `write_package` fixture style from `tests/test_human_maps.py`):
  - `test_decode_v3_colored_notes`: colorNotes c∈{0,1} → hand, beats→seconds
    at bpm 120, bombs skipped.
  - `test_decode_v2_notes`: `_notes` `_type` 0/1 kept, `_type`3 dropped.
  - `test_decode_highest_difficulty_default`: package with Hard+ExpertPlus
    → `difficulty=="ExpertPlus"`.
  - `test_decode_explicit_difficulty`, `test_decode_nonstandard_skipped`
    (OneSaber-only → error), `test_decode_tempo_change_rejected`.

- [ ] **Step 2–4:** Reuse `evaluation.human_maps` internals
  (`_read_json`, `_version`, `_check_timing`, `_package_file` — same
  package, private-name import is acceptable and keeps one validation
  implementation). Parse Info.dat (v2 only), select Standard group; parse
  beatmap v2 `_notes` or v3 `colorNotes`; `t = beat·60/bpm`; emit tuples;
  sort. (Collapse happens inside `prepare_notes`, not here.)

- [ ] **Step 5: Commit** `Add beatmap package note decoder for playability report`.

### Task 7: Corpus comparison + CLI

**Files:** Modify `evaluation/playability_report.py`; same test file.

**Interfaces:**
- `band_label(nps) -> str` → `"<3"`,`"3-5"`,`"5-7"`,`"7+"`
- `score_rows(packages, per_difficulty: bool) -> (rows, skipped)` — row:
  `{"package","difficulty","scores"}`; skipped: `{"package","error"}`
- `verdict(gen_scores, human_rows) -> {m: {"percentile","band","band_n",
  "incident_rate","human_incident_rate","flag"}}` for m in m1..m4
- `main()` CLI: `python -m evaluation.playability_report --human-root <dir>
  --generated <dir> [--dataset <prepared>] [--json out] [--top-k 5]`

- [ ] **Step 1: Failing tests**
  - `test_percentile_and_band_fallback`: human scalars [0.1,0.2,0.3] one in
    gen's band → `band_n<5` → percentile over all three.
  - `test_incident_rate`: human pooled units → p95; gen units above →
    `incident_rate = count_above/minutes`; flag at ≥3× human rate or
    percentile ≥95.
  - `test_score_rows_skip`: one bad package + one good → 1 row + 1 skipped.

- [ ] **Step 2–4:** `human root` resolves `root/bs_map_input` when present,
  else root itself; immediate subdirs only. Human side: highest difficulty
  (`difficulty=None`). Generated side: every `list_standard_difficulties`
  entry. Percentile = share of human per-map scalars ≤ gen value ×100 in
  matched band (fallback all-band when `band_n<5`). Incident rate: pooled
  positive `units` across human rows → `p95`; `rate = (units≥p95).sum() /
  span_minutes`, same for human totals. Console report: human quantile
  table per band, per-generated rows `M1..M4` value+percentile+rates,
  flags, skipped list, then top-k M4/M2 incidents for flagged maps.
  `--dataset`: optional extra human rows via
  `pattern_analysis.decode_human_song` (beats→seconds inside).

- [ ] **Step 5: Commit** `Add playability corpus report CLI`.

### Task 8: Live run + docs pointers

**Files:** Modify `docs/development.md`, `AGENTS.md` (context pointer).

- [ ] **Step 1:** Run `python -m evaluation.playability_report --human-root
  ../Data/training/evaluate_beats` (resolves `bs_map_input`) from repo root
  in the project env — human table must populate, skipped list shown.
- [ ] **Step 2:** If a generated output folder exists (check
  `../Data/prediction`/`paths`), run `--generated` against it; sanity-check
  that human rows show lower M4 than generated where expected. If none
  exists, note it and stop at the human table.
- [ ] **Step 3:** Add a short "Playability metrics" paragraph to
  `docs/development.md` (command + what it reports) and one context-pointer
  line in `AGENTS.md`.
- [ ] **Step 4:** `python -m pytest -q tests/test_playability_metrics.py
  tests/test_playability_report.py` all green; commit docs + any
  fixes.
