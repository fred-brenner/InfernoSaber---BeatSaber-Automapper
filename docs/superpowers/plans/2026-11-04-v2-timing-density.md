# v2 Timing & Density: Entry Ramp, Silence Ordering, Gap Safety Net

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the "5 s on / 5 s off" density cycling in v2 maps, slow down note entries after song start and mid-song pauses, and mix an optional beat source (onsets or the trained v1 beat TCN) into gap filling — all inference-only, no retraining.

**Architecture:** All changes sit inside `postprocess_timing` (`map_creation/v2_postprocessing.py:121-229`) except one argmax-bias knob in `infer_beats` (`map_creation/v2_pipeline.py:91`) and a mechanical helper extraction in `evaluation/legacy_generator.py`. The subdivision fills (`high_difficulty_fill`, `density_fill`) get a shared post-filter that drops inserted times inside blocked intervals (entry-ramp windows and silent spans). `silence_removal` moves before `density_fill`. A new `max_gap_fill` step inserts candidate times into residual large gaps from a pluggable source. No model or architecture changes.

**Tech Stack:** Python, numpy, pytest; optional lazy path uses the existing aubio/TensorFlow TCN pipeline already in the repo. No new dependencies.

**Spec:** `todo_after_playtest.md` (repo root), cases 4–6.

## Global Constraints

- **v2 generation only.** The legacy `legacy_generator.main` keeps its exact semantics; the only permitted edit there is extracting its beat-prediction block into a callable helper (Task 6) — no behavior change.
- All timestamps are elapsed seconds until export; `frame_times`/`probabilities`/`counts` contracts in `postprocess_timing` stay valid (inserted times get counts from the nearest frame via `replace_times`' non-moving path, `v2_postprocessing.py:189-191`).
- New config knobs in `tools/config/config.py` with exact names/values:
  - `v2_fill_entry_delay_seconds = 1.0`
  - `v2_fill_entry_pause_beats = 4.0`  (pause detection is beat-relative: `pause_seconds = pause_beats * 60 / bpm`)
  - `v2_fill_respect_silence = True`
  - `v2_onset_snap_unmatched_policy = "non_silent"`  (`"always"` | `"non_silent"` | `"never"`)
  - `v2_timing_event_bias = 0.1`  (added to classes 1 and 2 before argmax; `0` restores pure argmax)
  - `v2_intensity_low_boost = 0.3`  (was `0.1` — tuning experiment, flag as such)
  - `v2_max_note_gap_seconds = 4.0`  (`0` disables)
  - `v2_gap_fill_edge_seconds = 0.5`
  - `v2_gap_fill_source = "onsets"`  (`"onsets"` | `"bpm_grid"` | `"v1_beat"`)
  - `v2_gap_fill_v1_bundle = "fav_15"`
- `_TIMING_STEPS` must stay a complete enumeration of executed step names (the `skip` parameter and `evaluate_v2_postprocessing` depend on it); final order: `("minimum_spacing", "high_difficulty_fill", "bpm_alignment", "onset_snap", "start_end_fill", "silence_removal", "density_fill", "max_gap_fill")`.
- `legacy_features()` decode must remain lazy — skipping every step that needs it must still skip the aubio/PIL decode entirely (existing contract, `v2_postprocessing.py:165-171`).
- Deterministic where randomness appears; no `random.random()` in new code.
- Tests follow `tests/test_v2_generation_postprocessing.py` conventions (`timing_defaults` fixture monkeypatches `_legacy_timing_features` and `sanity_check_timing2`; `beats()`/`song()` helpers). Run from repo root: `python -m pytest tests/test_v2_generation_postprocessing.py -q`.

## Review Focus

1. **Entry detection timing.** Entries must be recomputed at each gated fill site on the then-current `times()` — computing once at the top misses pauses `silence_removal` carves out later, and computing only after `onset_snap` misses model pauses the snap deleted. Spec: compute right before `high_difficulty_fill` and again right before `density_fill`. Owned by Task 1/2.
2. **`v1_beat` source failure paths.** `find_beats` requires `songs_pred/<name>.egg`; the bundle dir may not exist. Any failure must fall back to `"onsets"` with a diagnostics note — generation must never crash because a fill source is missing. Owned by Task 6.
3. **Silent-span semantics.** `silent_times` is an array of silent *frame times* (≈34.6 ms spacing at `hop = 512/14800`); a fill point between two silent frames is still inside silence — merge consecutive frames into spans (`_silent_spans`), don't do nearest-point matching. Owned by Task 2.
4. **`v2_timing_event_bias` vs zero density.** With `conditioning_nps == 0` probabilities are `(1,0,0)`; bias must be validated `< 1` so class 0 always keeps winning — zero requested NPS must still produce no events. Owned by Task 4.
5. **max_gap_fill vs case-1 intro floor.** Fills may place events < 2 s; the plan deliberately allows this (the note-level intro floor drops the resulting notes later while lighting stays). Test that this does not crash or violate `min_beat_time` spacing. Owned by Task 5.

---

### Task 1: Entry ramp — gate subdivision fills after pauses/startup

**Files:**
- Modify: `tools/config/config.py` (add `v2_fill_entry_delay_seconds`, `v2_fill_entry_pause_beats`)
- Modify: `map_creation/v2_postprocessing.py` (`postprocess_timing` fill call sites at `:203-206` and `:215-219`; new helpers)
- Test: `tests/test_v2_generation_postprocessing.py`

**Interfaces:**
- Produces:
  - `_section_entries(times: np.ndarray, pause_seconds: float) -> np.ndarray` — sorted entry times: `times[0]` plus every `t` where `t - prev >= pause_seconds`.
  - `_blocked_intervals(entries: np.ndarray, delay: float) -> list[tuple[float, float]]` — `[(e, e + delay) for e in entries]`, empty when `delay <= 0`.
  - `_filter_fill_times(previous: np.ndarray, new_times: np.ndarray, blocked) -> np.ndarray` — returns `new_times` minus times that are inside a blocked interval AND not in `previous` (model events inside a blocked window are kept; only fill insertions are gated).
- Consumes: `fill_map_times_scale(times(), index)` results; `postprocess_timing`'s `replace_times`/`times()` closures.

- [ ] **Step 1: Write the failing tests**

```python
def test_entry_ramp_drops_fill_inside_delay_window(song, timing_defaults, monkeypatch):
    # 120 bpm -> 4 beats = 2.0 s pause threshold; entries: 1.0 and 5.0 (gap 4.0)
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 1.0)
    monkeypatch.setattr(config, "v2_fill_entry_pause_beats", 4.0)
    insert = np.asarray([1., 5., 5.5, 6.2, 10.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, report = post.postprocess_timing(song, beats([1., 5., 10.], [1, 1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    # 5.5 is inside (5.0, 6.0) -> dropped; 6.2 outside -> kept; model times kept
    assert [item["time"] for item in candidates] == [1., 5., 6.2, 10.]

def test_entry_ramp_first_event_is_an_entry(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 1.0)
    insert = np.asarray([3., 3.4, 8.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, _ = post.postprocess_timing(song, beats([3., 8.], [1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    # 3.4 inside (3.0, 4.0) -> dropped
    assert [item["time"] for item in candidates] == [3., 8.]

def test_entry_ramp_disabled_at_zero_delay(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 0.0)
    insert = np.asarray([1., 5., 5.5, 10.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, _ = post.postprocess_timing(song, beats([1., 5., 10.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 5., 5.5, 10.]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q -k entry_ramp`
Expected: FAIL — fills land unfiltered.

- [ ] **Step 3: Implement**

- `config.py`: add `v2_fill_entry_delay_seconds = 1.0`, `v2_fill_entry_pause_beats = 4.0`.
- `v2_postprocessing.py`: add `_section_entries`, `_blocked_intervals`, `_filter_fill_times` helpers (module level, near the timing section).
- In `postprocess_timing`: define a `fill_blocked()` closure returning blocked intervals given current `times()`:

```python
    def fill_blocked():
        pause = float(getattr(config, "v2_fill_entry_pause_beats", 4.0) or 0.) * 60. / bpm
        delay = float(getattr(config, "v2_fill_entry_delay_seconds", 0.) or 0.)
        entries = _section_entries(times(), pause) if pause > 0 else np.asarray([times()[0]])
        return _blocked_intervals(entries, delay)
```

- Wrap every `fill_map_times_scale` call site: `replace_times(name, _filter_fill_times(times(), fill_map_times_scale(times(), index), fill_blocked()))`. Applies to both `high_difficulty_fill` (`:206`) and `density_fill` (`:219`) — the only subdivision fills. `start_end_fill` is an edge fill, not a subdivision: leave ungated (spec).
- Compute `fill_blocked()` fresh at each call site so later deletions (onset_snap/silence) create correctly gated new pauses.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/config/config.py map_creation/v2_postprocessing.py tests/test_v2_generation_postprocessing.py
git commit -m "Gate v2 subdivision fills behind a post-pause entry ramp"
```

---

### Task 2: Silence ordering + silent-aware fills

**Files:**
- Modify: `tools/config/config.py` (add `v2_fill_respect_silence = True`)
- Modify: `map_creation/v2_postprocessing.py` (`_TIMING_STEPS` order; `postprocess_timing` step order at `:215-222`; new `_silent_spans` helper)
- Test: `tests/test_v2_generation_postprocessing.py`

**Interfaces:**
- Produces:
  - `_silent_spans(silent_times) -> list[tuple[float, float]]` — merges silent frame times (≈34.6 ms spacing) into `(lo, hi)` spans, closing gaps `< 4 * hop` ≈ 138 ms (`hop = config.hop_size / config.samplerate_music` = 512/14800).
  - `_inside_spans(t: float, spans) -> bool` — membership test shared with Tasks 3 and 5.
  - `silent_spans()` — a lazy closure inside `postprocess_timing` computing `_silent_spans(legacy_features()[2])` once, only when `config.add_silence_flag and config.v2_fill_respect_silence` (keeps the lazy-decode contract). (Shipped refinement: signature is `silent_spans(force=False)` — the cache keys on `add_silence_flag` alone, `v2_fill_respect_silence` applies per call, and detection-level consumers pass `force=True`; see controller rulings.)
- Consumes: Task 1's `_filter_fill_times` — extend its signature: `_filter_fill_times(previous, new_times, blocked, silent_spans=())` drops insertions inside either kind of interval.
- `legacy_features()` third return is the `silent_times` array (frame times in seconds).

- [ ] **Step 1: Write the failing tests**

```python
def test_density_fill_skips_silent_spans(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_respect_silence", True)
    silent_frames = np.asarray([4.0 + i * (config.hop_size / config.samplerate_music)
                                for i in range(30)])  # silent span around t=4.0..4.3
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), silent_frames))
    insert = np.asarray([1., 4.15, 6., 10.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, report = post.postprocess_timing(song, beats([1., 6., 10.], [1, 1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 6., 10.]  # 4.15 silent -> dropped

def test_silence_removal_runs_before_density_fill(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), [2.]))
    _, report = post.postprocess_timing(song, beats([1., 2., 8.], [1, 1, 1]),
                                        "unused.egg", skip=("bpm_alignment",))
    stages = [op["stage"] for op in report["operations"]]
    assert stages.index("silence_removal") < stages.index("density_fill_0")
```

(The second test needs `map_filler_iters > 0` and a sub-threshold density so `density_fill_0` actually runs — mirror `test_filler_uses_strongest_nonzero_class_at_nearest_frame`'s setup.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q -k "silence or density_fill"`
Expected: FAIL — `density_fill` currently precedes `silence_removal`; silent-span fill lands.

- [ ] **Step 3: Implement**

- `config.py`: `v2_fill_respect_silence = True`.
- `_TIMING_STEPS` (`:117-118`): move `"silence_removal"` before `"density_fill"` (final order per Global Constraints).
- `_silent_spans(silent_times)` helper: sorted frame times → merge runs with inter-frame gap `< 4 * (config.hop_size / config.samplerate_music)` into `(first, last)` spans; return `[]` for empty input.
- In `postprocess_timing`: move the `silence_removal` block (`:220-222`) above the `density_fill` loop. Add a `silent_spans()` lazy closure computing `_silent_spans(legacy_features()[2])` once, only when `config.add_silence_flag and config.v2_fill_respect_silence` — reuse it in the fill filters AND the Task-3 snap policy (snap/gap-net calls use `force=True` — detection is independent of the fill gate).
- Extend `_filter_fill_times` with `silent_spans=()`; pass `silent_spans=silent_spans()` at both fill call sites.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q`
Expected: PASS — check `test_timing_diagnostics_record_alignment_and_silence_steps` still passes (it uses a stage dict, order-insensitive).

- [ ] **Step 5: Commit**

```bash
git add tools/config/config.py map_creation/v2_postprocessing.py tests/test_v2_generation_postprocessing.py
git commit -m "Run v2 silence removal before density fill and keep fills out of silent spans"
```

---

### Task 3: Onset-snap unmatched policy — stop culling sparse-region events

**Files:**
- Modify: `tools/config/config.py` (add `v2_onset_snap_unmatched_policy = "non_silent"`)
- Modify: `map_creation/v2_postprocessing.py` (onset-snap block at `:210-211`)
- Test: `tests/test_v2_generation_postprocessing.py`

**Interfaces:**
- Consumes: `sanity_check_timing2` returns same-length array with `0` marking unmatched events; `drop_zero` in `replace_times` removes them. Task 2's `silent_spans()` closure.
- Produces: policy values `"always"` (legacy drop-all), `"non_silent"` (default — keep unmatched events unless inside a silent span), `"never"` (keep all unmatched).

- [ ] **Step 1: Write the failing tests**

```python
def test_onset_snap_unmatched_non_silent_keeps_event(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "non_silent")
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), [8.]))
    # second event unmatched (0); t=4.0 not inside silent span -> restored to original
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    candidates, _ = post.postprocess_timing(song, beats([1., 4., 8.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1.02, 4., 8.01]

def test_onset_snap_unmatched_non_silent_drops_in_silence(song, timing_defaults, monkeypatch):
    # same setup but silent span covers 4.0 -> event still dropped
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "non_silent")
    frames = np.asarray([4.0 + i * (config.hop_size / config.samplerate_music) for i in range(30)])
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), frames))
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    candidates, _ = post.postprocess_timing(song, beats([1., 4., 8.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1.02, 8.01]

def test_onset_snap_unmatched_always_is_legacy(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "always")
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    candidates, _ = post.postprocess_timing(song, beats([1., 4., 8.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1.02, 8.01]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q -k unmatched`
Expected: FAIL — zero-marked events currently always drop.

- [ ] **Step 3: Implement**

In `postprocess_timing`, replace the onset-snap line (`:211`) with:

```python
    if candidates and "onset_snap" not in skip:
        snapped = sanity_check_timing2(None, times(), audio_path=audio_path)
        policy = getattr(config, "v2_onset_snap_unmatched_policy", "always")
        if policy != "always":
            spans = silent_spans() if policy == "non_silent" else ()
            snapped = np.asarray([original if marker <= 0 and (policy == "never" or not _inside_spans(original, spans))
                                  else marker
                                  for original, marker in zip(times(), snapped)])
        replace_times("onset_snap", snapped, moving=True, drop_zero=True)
```

(`_inside_spans(t, spans)` — tiny membership helper shared with Task 2's fill filter; name it once in Task 2 and reuse.) `replace_times` keeps `moving=True` (length preserved) and `drop_zero=True` (remaining zeros still drop). Reject unknown policy values with `ValueError`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q`
Expected: PASS.

**Required existing-test update:** `test_rejected_onset_zero_does_not_absorb_a_valid_early_onset` (`tests/test_v2_generation_postprocessing.py:70-73`) will fail under the new `"non_silent"` default — its `timing_defaults` fixture has `add_silence_flag=False`, so `silent_spans()` is empty and the formerly-dropped event is now restored. Update that test to set `v2_onset_snap_unmatched_policy="always"` explicitly, pinning the legacy behavior it was written to document.

- [ ] **Step 5: Commit**

```bash
git add tools/config/config.py map_creation/v2_postprocessing.py tests/test_v2_generation_postprocessing.py
git commit -m "Keep unsnapped v2 timing events outside detected silence"
```

---

### Task 4: Beat-extraction bias + intensity boost tuning

**Files:**
- Modify: `tools/config/config.py` (`v2_timing_event_bias = 0.1`; `v2_intensity_low_boost` 0.1 → 0.3)
- Modify: `map_creation/v2_pipeline.py` (`infer_beats` at `:91-92`)
- Test: `tests/test_v2_generation_pipeline.py` (add bias test alongside existing pipeline tests)

**Interfaces:**
- Consumes: `infer_beats`'s `probabilities` array `(frames, 3)`; `counts = probabilities.argmax(axis=-1)` at `:91`.
- Produces: `v2_timing_event_bias` — nonnegative float `< 1`, added to `probabilities[:, 1:3]` before argmax.

- [ ] **Step 1: Write the failing test**

```python
def test_timing_event_bias_adds_borderline_events(song_and_stubs):
    # predictors stub returns probabilities (0.55, 0.30, 0.15) per frame
    # bias 0 -> frame silent; bias 0.10 -> class 1 wins -> event
    assert count_events(bias=0.0) < count_events(bias=0.1)

def test_timing_event_bias_preserves_zero_density(song_and_stubs):
    # conditioning_nps == 0 -> probabilities (1, 0, 0); bias < 1 keeps class 0
    assert count_events(bias=0.9, conditioning_nps=0) == 0
```

(Exact stub mechanics follow the existing pipeline-test fixtures — the assertions are the contract.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v2_generation_pipeline.py -q -k event_bias`
Expected: FAIL — no bias applied today.

- [ ] **Step 3: Implement**

- `config.py`: `v2_timing_event_bias = 0.1`; change `v2_intensity_low_boost` default `0.1` → `0.3` with a comment marking it an experiment value.
- `v2_pipeline.py:91`: validate `0 <= bias < 1` (`ValueError` otherwise), then `counts = (probabilities + [0, bias, bias]).argmax(axis=-1)` — non-mutating (keep `probabilities` pristine for `frame_probabilities` diagnostics).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v2_generation_pipeline.py tests/test_v2_generation_postprocessing.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/config/config.py map_creation/v2_pipeline.py tests/test_v2_generation_pipeline.py
git commit -m "Add v2 timing event bias and raise default low-intensity boost"
```

---

### Task 5: Max-gap safety net with onset/BPM-grid sources

**Files:**
- Modify: `tools/config/config.py` (`v2_max_note_gap_seconds = 4.0`, `v2_gap_fill_edge_seconds = 0.5`, `v2_gap_fill_source = "onsets"`, `v2_gap_fill_v1_bundle = "fav_15"`)
- Modify: `map_creation/v2_postprocessing.py` (`_TIMING_STEPS` + `postprocess_timing` tail at `:223-224`; new `_gap_candidate_times` helper)
- Test: `tests/test_v2_generation_postprocessing.py`

**Interfaces:**
- Produces: `_gap_candidate_times(source: str, audio_path, bpm: float, duration: float) -> np.ndarray` — `"onsets"` → `_detect_onsets(audio_path)` (existing, `:710`); `"bpm_grid"` → `np.arange(step, duration, 60/bpm)`; `"v1_beat"` → Task 6's provider with onsets fallback (until Task 6 lands, the branch may fall back unconditionally — keep the dispatch signature stable).
- Consumes: `times()`, `replace_times`, `silent_spans()`/`_inside_spans` from Tasks 2–3.

- [ ] **Step 1: Write the failing tests**

```python
def test_max_gap_fill_inserts_onset_times(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_edge_seconds", 0.5)
    monkeypatch.setattr(post, "_detect_onsets", lambda _: np.asarray([2.6, 3.3, 4.9, 7.2]))
    candidates, report = post.postprocess_timing(song, beats([1., 8.], [1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    # gap 1->8 exceeds 4 s: onsets in (1.5, 7.5) inserted -> 2.6, 3.3, 4.9, 7.2
    assert [item["time"] for item in candidates] == [1., 2.6, 3.3, 4.9, 7.2, 8.]

def test_max_gap_fill_respects_silence_and_edge(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_edge_seconds", 1.0)
    frames = np.asarray([3. + i * (config.hop_size / config.samplerate_music) for i in range(30)])
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), frames))
    monkeypatch.setattr(post, "_detect_onsets", lambda _: np.asarray([1.6, 3.1, 7.8]))
    candidates, _ = post.postprocess_timing(song, beats([1., 9.], [1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    # interior (2.0, 8.0): 1.6 outside edge, 3.1 silent, 7.8 kept
    assert [item["time"] for item in candidates] == [1., 7.8, 9.]

def test_max_gap_fill_bpm_grid_source(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_source", "bpm_grid")
    candidates, _ = post.postprocess_timing(song, beats([1., 8.], [1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    # 120 bpm -> 0.5 s grid inside (1.5, 7.5)
    assert [item["time"] for item in candidates] == [1., 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 8.]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q -k max_gap`
Expected: FAIL — no `max_gap_fill` step exists.

- [ ] **Step 3: Implement**

- `config.py`: the four gap knobs.
- `_TIMING_STEPS`: append `"max_gap_fill"` after `"density_fill"`.
- In `postprocess_timing`, after the `density_fill` loop (before `final_bpm_alignment`):

```python
    if candidates and "max_gap_fill" not in skip:
        max_gap = float(getattr(config, "v2_max_note_gap_seconds", 0.) or 0.)
        if max_gap > 0:
            edge = float(getattr(config, "v2_gap_fill_edge_seconds", .5) or 0.)
            source = getattr(config, "v2_gap_fill_source", "onsets")
            pool = _gap_candidate_times(source, audio_path, bpm, duration)
            current, inserted = times(), []
            spans = silent_spans()
            for a, b in zip(current, current[1:]):
                if b - a <= max_gap:
                    continue
                for t in pool[(pool > a + edge) & (pool < b - edge)]:
                    if _inside_spans(t, spans) or (inserted and t - inserted[-1] < config.min_beat_time):
                        continue
                    inserted.append(float(t))
            if inserted:
                replace_times("max_gap_fill", np.sort(np.r_[current, inserted]))
```

- `_gap_candidate_times`: dispatch `"onsets"` → `_detect_onsets`, `"bpm_grid"` → `np.arange(step, duration, step)`, `"v1_beat"` → `predict_beat_times` provider (Task 6) with fallback; unknown source → `ValueError`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/config/config.py map_creation/v2_postprocessing.py tests/test_v2_generation_postprocessing.py
git commit -m "Add v2 max-gap fill with onset and BPM-grid sources"
```

---

### Task 6: v1 beat-TCN as a gap-fill source

**Files:**
- Modify: `evaluation/legacy_generator.py` (extract beat pipeline `:102-162` into `beat_proposals(name_ar) -> dict`; `main` calls it — no behavior change)
- Modify: `map_creation/v2_postprocessing.py` (`_gap_candidate_times` `"v1_beat"` branch)
- Test: `tests/test_v2_generation_postprocessing.py`

**Interfaces:**
- Produces: `beat_proposals(name_ar: list[str]) -> dict` in `evaluation/legacy_generator.py` — returns `{"timing_ar": np.ndarray, "silent_times": list, "pitch_input": list, "pitch_times": list, "song_input": list, "x_volume": np.ndarray, "x_onset": np.ndarray}`; `main()` consumes all fields, the v2 gap source uses only `timing_ar` (elapsed seconds, post-`sanity_check_beat`, pre-BPM-alignment).
- Consumes: `paths.dir_path + f"model/{config.v2_gap_fill_v1_bundle}/"` supplies `tf_beat_gen*`; `settings_snapshot`/`scoped_settings` from `map_creation.v2_generation` for the temporary `paths.model_path` override.

- [ ] **Step 1: Write the failing tests**

```python
def test_gap_fill_v1_beat_source_inserts_predicted_times(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_source", "v1_beat")
    monkeypatch.setattr(post, "_v1_beat_times", lambda path: np.asarray([3.0, 5.5]))
    candidates, _ = post.postprocess_timing(song, beats([1., 8.], [1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 3., 5.5, 8.]

def test_gap_fill_v1_beat_falls_back_to_onsets(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_source", "v1_beat")
    def boom(path):
        raise FileNotFoundError("missing bundle")
    monkeypatch.setattr(post, "_v1_beat_times", boom)
    monkeypatch.setattr(post, "_detect_onsets", lambda _: np.asarray([4.0]))
    candidates, report = post.postprocess_timing(song, beats([1., 8.], [1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 4., 8.]
    assert report.get("gap_fill_source_used") == "onsets"  # fallback recorded
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q -k v1_beat`
Expected: FAIL — `_v1_beat_times` does not exist.

- [ ] **Step 3: Implement**

- `evaluation/legacy_generator.py`: move `main`'s block from `song_input, pitch_input = find_beats(...)` (`:102`) through the `timing_ar` seconds computation (`:160-162`) into `def beat_proposals(name_ar) -> dict` returning the dict above; `main` calls it and uses the fields. Pure extraction — no behavior change.
- `v2_postprocessing.py`: `_v1_beat_times(audio_path)` — lazy `from evaluation.legacy_generator import beat_proposals`; `name = Path(audio_path).stem`; inside `with scoped_settings():` set `paths.model_path = paths.dir_path + f"model/{config.v2_gap_fill_v1_bundle}/"`; return `np.asarray(beat_proposals([name])["timing_ar"])`.
- `_gap_candidate_times` `"v1_beat"` branch: `try: return _v1_beat_times(audio_path) except Exception: log + fall back to _detect_onsets` — record `"gap_fill_source_used"` in timing diagnostics.
- Add `"gap_fill_source_used": source_used` to the timing `diagnostics` dict.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py -q`
Expected: PASS. Also run the legacy-eval sanity: `python -m pytest tests/test_v2_note_onset_snap.py -q` unaffected; optionally `python -c "from evaluation.legacy_generator import beat_proposals"` to confirm import shape (import pulls TF — only run in the project env).

- [ ] **Step 5: Commit**

```bash
git add evaluation/legacy_generator.py map_creation/v2_postprocessing.py tests/test_v2_generation_postprocessing.py
git commit -m "Reuse the trained v1 beat TCN as a v2 gap-fill source"
```

---

### Task 7: Documentation + focused verification

**Files:**
- Modify: `docs/generation-v2.md` (timing postprocessing section: new step order, entry ramp, unmatched-onset policy, gap safety net + sources, new knobs)
- Modify: `AGENTS.md` (one clause in the v2 description)
- Modify: `map_creation/v2_postprocessing.py` (`postprocess_timing` docstring — new step list)

- [ ] **Step 1: Update docs** — as in Plan 1 Task 4; document all ten knobs with defaults, mark `v2_intensity_low_boost`/`v2_timing_event_bias`/`v2_max_note_gap_seconds` as playtest-tunable experiments, note `v1_beat` requires a downloaded v1 bundle.

- [ ] **Step 2: Full focused run**

Run: `python -m pytest tests/test_v2_generation_postprocessing.py tests/test_v2_generation_pipeline.py tests/test_v2_calm_effects.py tests/test_v2_note_onset_snap.py tests/test_v2_generation_workers.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/generation-v2.md AGENTS.md map_creation/v2_postprocessing.py
git commit -m "Document v2 timing density and entry-ramp controls"
```
