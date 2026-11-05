# v1 Legacy Model Option: Selectable Presets in App + Worker Routing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the five v1 mapper presets (`fav_15`, `pp3_15`, `easy_15`, `hard_15`, `expert_15`) as a selectable legacy option in the app and CLI, keeping `v2_expert` the default and v1 semantics completely untouched.

**Architecture:** `config.use_mapper_selection` flows from the app/CLI into `settings_snapshot` and thus into spawned workers; `process_song` branches per selection to either the existing v2 `generate_song` or a new `map_creation/v1_generation.py::generate_song_v1` — a thin `scoped_settings` wrapper around `legacy_generator.main([name])` (always a single-element list, avoiding its `exit()`). Missing v1 assets download on demand via the existing `model_download(selection)` (HF branches = selection names). Packaging branches: v1 keeps its per-difficulty `1234_<speed>_<name>` folders zipped directly (no `combine_packages`). The Gradio dropdown becomes interactive with dynamic parameter-group visibility.

**Tech Stack:** Python, Gradio, `tensorflow.keras` (legacy TCN loading — same profile as today), `huggingface_hub.snapshot_download` (already a dependency), pytest.

**Spec:** `todo_after_playtest.md` (repo root), case 7 + the user's dynamic-UI requirement.

## Global Constraints

- **v1 semantics untouched.** `evaluation/legacy_generator.py` is not modified at all in this plan (Plan 2's `beat_proposals` extraction is independent and must not change `main`'s behavior either). None of the v2 postprocessing/density work from the sibling plans reaches the v1 path.
- `legacy_generator.main` returns `False` on success, `1` on beat failure, an array for `debug_beats`; it calls `exit()` when `len(name_ar) > 1` — callers pass exactly one name.
- `config.use_mapper_selection` remains `"v2_expert"` by default; the selection must be a known preset (validate early).
- `update_model_file_paths()` must run **before** `settings_snapshot()` — `paths.model_path` is a snapshotted string that workers inherit (`main_multi.py:145` precedes the worker pool).
- `legacy_generator` imports TensorFlow at module top — import it lazily inside `generate_song_v1`, never at `v1_generation`/`main_multi` import time (keeps v2 worker startup unchanged and spawn-safe on Windows).
- v1 worker records keep the shape `{"status", "song", "internal_difficulty", "path", "package"}` that `main_multi` reporting consumes.
- v1 output folders are `1234_<internal_difficulty:.1f>_<name>/` (from `create_map`, `map_creation/map_creator.py:58`); package = `write_zip(folder, <speed>_<name>.zip)` + optional `shutil_copy_maps(f"{speed:.1f}_{name}")` — never `combine_packages`/`12345_*`.
- UI parameter groups must reflect model applicability (user requirement): v1-only controls hidden under `v2_expert`, v2-only controls hidden under v1 presets, shared controls always visible.
- The pre-existing worker test asserting the v2 force (`tests/test_v2_generation_workers.py` ~:127-142) must be updated, not deleted — selection now propagates by design.
- Run tests from repo root with the project env: `python -m pytest tests/test_v2_generation_workers.py -q`.

## Review Focus

1. **`use_mapper_selection` is also read by `update_model_file_paths`, `paths.model_path` and `model_download` at process level.** Every entry point (`main_app.run_process`, `main_multi._run`, `main.main`) must set it before any asset validation or snapshot — inconsistent ordering leaves `paths.model_path` pointing at the wrong bundle. Owned by Tasks 2–4.
2. **`legacy_generator.main` inverts success (`False` = ok).** The wrapper must raise on `status is not False`; `1` (too few matched beats) is a failure record, not an exception — it should surface as a failed song record, consistent with `process_song`'s error handling. Owned by Task 1.
3. **Per-variant config isolation.** `legacy_generator.main` mutates `config.add_beat_intensity`, `silence_threshold`, `obstacle_time_gap`, `thresh_*` from `*_orig` values — the wrapper must set `config.max_speed`/`max_speed_orig` and run inside `scoped_settings()` per difficulty, or one variant's mutations leak into the next. Owned by Task 1.
4. **`get_full_model_path` file-count guard (`mapper_selection.py:34-37`).** A partially downloaded bundle (>0, <15 files passes the range check but misses prefixes) must fail with a clear `FileNotFoundError`, not a confusing downstream TF error. `ensure_v1_bundle` validates all four model prefixes plus the three pickle paths before generating. Owned by Task 1.
5. **Gradio `interactive=False` removal + `.change` handler.** The dropdown is currently inert; wiring `model_selector.change` to `gr.Column(visible=...)` updates is the mechanism, but the implementer must verify the actual control list against `main_app.py` — the documented v1-only/v2-only/shared split in `todo_after_playtest.md` is the mapping source. Owned by Task 4.

---

### Task 1: `generate_song_v1` wrapper + `ensure_v1_bundle`

**Files:**
- Create: `map_creation/v1_generation.py`
- Test: `tests/test_v1_generation.py` (new)

**Interfaces:**
- Produces:
  - `V1_MAPPER_SELECTIONS = ("fav_15", "pp3_15", "easy_15", "hard_15", "expert_15")`
  - `ensure_v1_bundle(selection: str) -> None` — `update_model_file_paths()`, then validate all four model prefixes (`config.beat_gen_version`, `config.enc_version`, `config.mapper_version`, `config.event_gen_version`) via `get_full_model_path` plus the three pickle paths (`paths.notes_classify_dict_file`, `paths.beats_classify_encoder_file`, `paths.events_classify_encoder_file`); on missing → `model_download(model_branch=selection)` then re-validate, raising `FileNotFoundError` with the selection name.
  - `generate_song_v1(name: str, internal_difficulty: float, logger=None) -> dict` — returns `{"status": "ok", "song": name, "internal_difficulty": float, "path": str(folder), "package": {}}`.
- Consumes: `scoped_settings`/`_emit` from `map_creation.v2_generation`; `legacy_generator.main` (lazy import); `update_model_file_paths`, `get_full_model_path` from `tools.config.mapper_selection`; `model_download` from `tools.utils.huggingface`.

- [ ] **Step 1: Write the failing tests**

```python
def test_generate_song_v1_calls_legacy_main_with_single_name(monkeypatch, tmp_path):
    calls = []
    fake = SimpleNamespace(main=lambda names: calls.append(names) or False)
    monkeypatch.setitem(sys.modules, "evaluation.legacy_generator", fake)
    monkeypatch.setattr(paths, "new_map_path", str(tmp_path) + "/")
    (tmp_path / "1234_20.0_song").mkdir()
    result = generate_song_v1("song", 20.0)
    assert calls == [["song"]]
    assert result["status"] == "ok" and result["internal_difficulty"] == 20.0
    assert result["path"].endswith("1234_20.0_song")

def test_generate_song_v1_failure_becomes_error(monkeypatch, tmp_path):
    fake = SimpleNamespace(main=lambda names: 1)  # legacy "not enough beats"
    monkeypatch.setitem(sys.modules, "evaluation.legacy_generator", fake)
    monkeypatch.setattr(paths, "new_map_path", str(tmp_path) + "/")
    with pytest.raises(RuntimeError):
        generate_song_v1("song", 20.0)

def test_generate_song_v1_restores_config(monkeypatch, tmp_path):
    original = config.max_speed_orig
    fake = SimpleNamespace(main=lambda names: setattr(config, "max_speed", -1.) or False)
    monkeypatch.setitem(sys.modules, "evaluation.legacy_generator", fake)
    monkeypatch.setattr(paths, "new_map_path", str(tmp_path) + "/")
    (tmp_path / "1234_20.0_song").mkdir()
    generate_song_v1("song", 20.0)
    assert config.max_speed_orig == original and config.max_speed == original
```

(Note for the implementer: `monkeypatch.setitem(sys.modules, ...)` works only if the lazy import inside `generate_song_v1` runs at call time — which is exactly the design; assert `config.max_speed` restored via `scoped_settings`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v1_generation.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `map_creation/v1_generation.py`**

```python
def generate_song_v1(name, internal_difficulty, logger=None):
    from evaluation.legacy_generator import main as legacy_main  # lazy: TF at module top
    name = str(name)
    if name.lower().endswith(".egg"):
        name = name[:-4]
    difficulty = float(internal_difficulty)
    with scoped_settings():
        update_model_file_paths()
        config.max_speed = config.max_speed_orig = difficulty
        status = legacy_main([name])
    folder = Path(paths.new_map_path) / f"1234_{difficulty:.1f}_{name}"
    if status is not False or not folder.is_dir():
        raise RuntimeError(f"v1 generation failed for {name} (status={status})")
    return {"status": "ok", "song": name, "internal_difficulty": difficulty,
            "path": str(folder), "package": {}}
```

- `ensure_v1_bundle(selection)`: per the Interfaces block — `update_model_file_paths()`, validate, `model_download(model_branch=selection)` on miss, re-validate, `FileNotFoundError` naming the selection and the missing prefix.
- Import guard: `legacy_generator` must appear nowhere at module level — only inside `generate_song_v1`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v1_generation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add map_creation/v1_generation.py tests/test_v1_generation.py
git commit -m "Add worker-safe v1 song generation wrapper"
```

---

### Task 2: Worker-pool routing in `main_multi`

**Files:**
- Modify: `main_multi.py` (`main_multi_par`/`_run` signature + selection plumbing `:137-160`; `process_song` branch `:48-60`; `_combine_maps` packaging branch)
- Test: `tests/test_v2_generation_workers.py` (update the v2-force test; add v1 routing tests)

**Interfaces:**
- Consumes: Task 1's `generate_song_v1`, `ensure_v1_bundle`, `V1_MAPPER_SELECTIONS`.
- Produces: `main_multi_par(n_workers, diff_list, export_results_to_bs=True, logger_callback=None, model_selection=None)` — `model_selection=None` → `"v2_expert"`. `process_song` reads `config.use_mapper_selection` (post-snapshot) to pick the generator; no signature change needed beyond `_run` plumbing.

- [ ] **Step 1: Write the failing tests**

```python
def test_process_song_routes_to_v1_for_legacy_selection(monkeypatch):
    monkeypatch.setattr(config, "use_mapper_selection", "fav_15")
    calls = []
    monkeypatch.setitem(sys.modules, "map_creation.v1_generation",
                        SimpleNamespace(generate_song_v1=lambda n, d, logger=None: calls.append((n, d)) or {"status": "ok", "song": n, "internal_difficulty": d, "path": "p"}))
    records = process_song([("song.egg", 20.)], 1)
    assert calls == [("song.egg", 20.)] and records[0]["status"] == "ok"

def test_run_validates_v1_bundle_for_legacy_selection(monkeypatch):
    seen = []
    monkeypatch.setattr(main_multi, "ensure_v1_bundle", lambda sel: seen.append(sel))
    monkeypatch.setattr(main_multi, "validate_production_bundle",
                        lambda *a, **kw: seen.append("v2_bundle"))
    monkeypatch.setattr(main_multi, "check_music_files", lambda files, path: [])
    main_multi._run(1, [6.0], False, None, parallel=False, model_selection="fav_15")
    assert seen == ["fav_15"]  # v1 path: ensure, not v2 bundle validation

def test_run_keeps_v2_validation_by_default(monkeypatch):
    seen = []
    monkeypatch.setattr(main_multi, "ensure_v1_bundle", lambda sel: seen.append(sel))
    monkeypatch.setattr(main_multi, "validate_production_bundle",
                        lambda *a, **kw: seen.append("v2_bundle"))
    monkeypatch.setattr(main_multi, "check_music_files", lambda files, path: [])
    main_multi._run(1, [6.0], False, None, parallel=False)
    assert seen == ["v2_bundle"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v2_generation_workers.py -q`
Expected: FAIL — `main_multi_par` has no `model_selection` param; `process_song` always calls v2.

- [ ] **Step 3: Implement**

- `main_multi_par(..., model_selection=None)` → `_run(n_workers, diff_list, export_results_to_bs, logger_callback, parallel, model_selection)`.
- In `_run`, replace `:152`'s hard-force:

```python
    selection = model_selection or "v2_expert"
    if selection != "v2_expert" and selection not in V1_MAPPER_SELECTIONS:
        raise ValueError(f"Unknown mapper selection: {selection}")
    if selection == "v2_expert":
        validate_production_bundle(snapshot["config"]["v2_model_bundle_path"])
    else:
        ensure_v1_bundle(selection)  # calls update_model_file_paths internally
    snapshot = settings_snapshot()  # AFTER ensure — workers inherit paths.model_path
    snapshot["config"]["use_mapper_selection"] = selection
```

  (Move the `settings_snapshot()` call at `:145` to after the validation/`ensure` block; `validate_production_bundle` runs before snapshot as today.)
- `process_song` (`:48-60`): branch inside the loop — `generator = generate_song_v1 if config.use_mapper_selection != "v2_expert" else generate_song` (lazy `from map_creation.v1_generation import generate_song_v1` inside the branch).
- Packaging (`_combine_maps`/`_run` combine loop `:207-213`): for v1 selections, per song take the run's ok records, `write_zip(record["path"], f"{new_map_path}{record['internal_difficulty']:.1f}_{name}.zip")`, and when `export_results_to_bs` `shutil_copy_maps(f"{record['internal_difficulty']:.1f}_{name}")`; append `{"song": name, "variants": [{"path": r["path"], "zip": zip_path} for r in records]}` to `packages`. Keep `auto_move_song_afterwards` handling identical.
- Import `V1_MAPPER_SELECTIONS`/`ensure_v1_bundle`/`generate_song_v1` at `main_multi` top EXCEPT `generate_song_v1` is fine at top (`v1_generation` itself is TF-free); `legacy_generator` stays lazy inside it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v2_generation_workers.py tests/test_v1_generation.py -q`
Expected: PASS — including the updated former v2-force test now asserting `"v2_expert"` is only the default.

- [ ] **Step 5: Commit**

```bash
git add main_multi.py tests/test_v2_generation_workers.py
git commit -m "Route legacy mapper selections through the worker pool"
```

---

### Task 3: App wiring — honor the selection

**Files:**
- Modify: `main_app.py` (`run_process` ~`:313`; `model_selector` dropdown `:682-684`)
- Test: `tests/test_v2_generation_workers.py` or a small `tests/test_app_model_selection.py`

**Interfaces:**
- Consumes: `V1_MAPPER_SELECTIONS`, `ensure_v1_bundle`, Task 2's `main_multi_par(..., model_selection=)`.
- Produces: `MAPPER_CHOICES = ["v2_expert", *V1_MAPPER_SELECTIONS]` (module-level in `main_app.py` or imported); `run_process` passes `model_selection=use_model`.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_process_passes_selected_model(monkeypatch):
    captured = {}
    monkeypatch.setattr(main_app, "main_multi_par",
                        lambda *a, **kw: captured.update(kw) or {"status": "ok"})
    run_process(3, "fav_15", 2, 4, 6, None, None)
    assert captured["model_selection"] == "fav_15"
    assert config.use_mapper_selection == "fav_15"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app_model_selection.py -q`
Expected: FAIL — `use_model` is ignored today (`:313` forces v2).

- [ ] **Step 3: Implement**

- `run_process`: replace `config.use_mapper_selection = "v2_expert"` with `config.use_mapper_selection = use_model or "v2_expert"`; for v2 keep `update_model_file_paths()` + `validate_production_bundle()`; for v1 call `ensure_v1_bundle(use_model)` and log `"Downloading/validating model: <sel>"` via the progress callback. Pass `model_selection=use_model` into `main_multi_par`.
- Dropdown `:682-684`: `choices=MAPPER_CHOICES, value="v2_expert", interactive=True`, label stays "Model", info text: "v2_expert is the current production model; the rest are legacy v1 presets kept for comparison".

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app_model_selection.py tests/test_v2_generation_workers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main_app.py tests/test_app_model_selection.py
git commit -m "Honor legacy mapper selections in the app run path"
```

---

### Task 4: Dynamic parameter groups in the UI

**Files:**
- Modify: `main_app.py` (wrap parameter controls in `gr.Column(visible=...)` groups; add `model_selector.change` handler)
- Test: `tests/test_app_model_selection.py` (pure helper test)

**Interfaces:**
- Produces: `parameter_group_visibility(selection: str) -> dict` — `{"v1_only": selection != "v2_expert", "v2_only": selection == "v2_expert"}`; the change handler maps it to `gr.Column(visible=...)` updates.
- Consumes: the control→config mapping already in `main_app.py`; the documented split in `todo_after_playtest.md` case 7.

- [ ] **Step 1: Write the failing test**

```python
def test_parameter_group_visibility():
    assert parameter_group_visibility("v2_expert") == {"v1_only": False, "v2_only": True}
    assert parameter_group_visibility("fav_15") == {"v1_only": True, "v2_only": False}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — helper doesn't exist.

- [ ] **Step 3: Implement**

- Wrap the parameter controls into three groups inside the existing layout (no visual restructure beyond the columns):
  - `v1_only` column (`visible=False`): Left/Right Notes Filling (`gimme_more_notes_prob`), Emphasize Beats (`emphasize_beats_flag`), and any other controls whose config is inert under v2 per the doc's split (legacy emphasis/break/dot/flow/block-shifter/waveform controls present in the UI).
  - `v2_only` column (`visible=True`): Note Sample Temperature and any other `v2_*`-only controls.
  - Everything else stays ungrouped (shared): bombs, obstacles, sliders/arcs, silence threshold, beat intensity, dot notes, single-notes-only, sporty obstacles, jump-speed offset, BS mapping version, automatic Expert.
  - Add a `gr.Markdown` title inside each column: `"Applies to legacy v1 models"` / `"Applies to v2 only"` — visible even before the dynamic toggle is exercised.
- `model_selector.change(fn=lambda sel: parameter_group_visibility(sel), inputs=model_selector, outputs=[v1_col, v2_col])` where `fn` returns `gr.Column(visible=result["v1_only"]), gr.Column(visible=result["v2_only"])` — Gradio accepts `gr.update(visible=...)` dicts; use whichever form the installed Gradio version supports (check `gradio.__version__` usage elsewhere in the file).
- Implementer must map the ACTUAL controls — read the app parameter section; the doc's split is the reference, not an assumption.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app_model_selection.py -q` + manual smoke: `python main_app.py` → switch dropdown → groups toggle (manual check only if env allows; otherwise note unverified).

- [ ] **Step 5: Commit**

```bash
git add main_app.py tests/test_app_model_selection.py
git commit -m "Show only model-applicable parameters in the app"
```

---

### Task 5: `main.py` CLI routing + docs

**Files:**
- Modify: `main.py` (`main()` at `:25-26`)
- Modify: `docs/generation-v2.md` / `Readme.md` (legacy option note), `AGENTS.md` (selection support)
- Test: extend `tests/test_app_model_selection.py`

- [ ] **Step 1: Write the failing test**

```python
def test_main_uses_v1_wrapper_for_legacy_env(monkeypatch):
    monkeypatch.setenv("inferno_model", "fav_15")
    # main() calls generate_song_v1, not generate_song; assert via stub
```

- [ ] **Step 2-3: Implement**

- `main()`: `config.use_mapper_selection = use_model or "v2_expert"` (replaces `:26`'s force, still inside `scoped_settings`); for v1 → `ensure_v1_bundle(use_model)` before `check_music_files`, then per song `generate_song_v1(song_name, config.max_speed)` + the existing `write_zip`/`shutil_copy_maps` block (folder naming is identical for both paths).
- Docs: `docs/generation-v2.md` note that v1 remains selectable and unchanged; `AGENTS.md` direction paragraph gains "v1 presets remain selectable legacy options"; fix the stale `gen_beats.main` benchmark reference in `docs/development.md` (recorded follow-up).

- [ ] **Step 4: Verification run**

Run: `python -m pytest tests/test_v1_generation.py tests/test_v2_generation_workers.py tests/test_app_model_selection.py -q`
Expected: PASS. Then a manual smoke if env allows: `inferno_model=fav_15 python main.py` with one sample song (mark unverified if the runtime isn't available).

- [ ] **Step 5: Commit**

```bash
git add main.py main_app.py docs/generation-v2.md AGENTS.md docs/development.md tests/test_app_model_selection.py
git commit -m "Restore v1 mapper selection across app and CLI"
```
