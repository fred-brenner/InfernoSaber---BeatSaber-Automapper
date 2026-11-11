"""Forward expansion guards and forward/search pipeline dispatch."""

import numpy as np
import pytest

from tools.config import config


def _note(id, time, hand, x, y, d, **extra):
    return {"id": id, "time": time, "hand": hand, "x": x, "y": y, "d": d, **extra}


def _retained(id, time, hand, x, y, d):
    return _note(id, time, hand, x, y, d, kind="retained")


class _ScriptedDraws:
    def __init__(self, *values):
        self.values = list(values)
        self.count = 0

    def random(self):
        self.count += 1
        if not self.values:
            raise AssertionError("unscripted RNG draw")
        return self.values.pop(0)


@pytest.fixture(autouse=True)
def expansion_defaults(monkeypatch):
    for key, value in {"min_time_diff": .02, "v2_cleanup_swing_window": .15,
                       "v2_fast_gap_seconds": .2,
                       "v2_fast_counter_direction": "recut",
                       "v2_diagonal_max_fraction": 1.,
                       "v2_playability_priority": True,
                       "v2_playability_reach": True,
                       "v2_playability_enabled": True,
                       "emphasize_beats_flag": True,
                       "single_notes_only_flag": False,
                       "emphasize_beats_3": 1., "emphasize_beats_3_fact": 0.,
                       "emphasize_beats_2": 0., "emphasize_beats_2_fact": 0.,
                       "emphasize_beats_quantile": .5,
                       "max_speed": 20., "reaction_time": 1.15,
                       "reaction_time_fact": .013, "cdf": 1.2, "cdf_lr": 1.15,
                       "v2_speed_cap_scale": 0.}.items():
        monkeypatch.setattr(config, key, value)
    yield


def _emphasis_state(count=8):
    times = [.4, .6, .8, 1., 1.2, 2.5, 3.8, 5.1]
    notes = [_retained(i, times[i], 0, 1, 1, i % 2) for i in range(count)]
    return {"notes": notes, "chains": []}


def test_validate_accepts_legal_batch_and_is_idempotent():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2),
                       _note(1, .5, 0, 2, 1, 3, kind="copy", anchor_id=0)],
             "chains": []}
    out = validate_expansions(state, None)
    assert out["rejected_copies"] == 0 and len(state["notes"]) == 2
    again = validate_expansions(state, None)
    assert again["rejected_copies"] == 0 and again["rejected_chains"] == 0


def test_atomic_rejection_unblocks_later_legal_expansion():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2),
                       _note(1, .005, 1, 0, 1, 2, kind="copy", anchor_id=0),
                       _note(2, .5, 0, 1, 1, 3, kind="copy", anchor_id=0)],
             "chains": []}
    out = validate_expansions(state, None)
    assert out["rejected_copies"] == 2 and len(state["notes"]) == 1
    state["notes"].append(_note(3, .5, 0, 1, 1, 3, kind="copy", anchor_id=0))
    out = validate_expansions(state, None)
    assert out["rejected_copies"] == 0
    assert [n["id"] for n in state["notes"]] == [0, 3]


def test_chain_anchor_exempt_but_other_obstruction_blocks():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2)], "chains": [
        {"id": 20, "anchor_id": 0, "time": 0., "end_time": .5,
         "hand": 0, "x": 0, "y": 1, "tx": 2, "ty": 1, "d": 2, "slices": 4}]}
    out = validate_expansions(state, None)
    assert out["rejected_chains"] == 0 and len(state["chains"]) == 1
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2), _retained(1, .25, 0, 1, 1, 0)],
             "chains": [{"id": 20, "anchor_id": 0, "time": 0., "end_time": .5,
                         "hand": 0, "x": 0, "y": 1, "tx": 2, "ty": 1, "d": 2,
                         "slices": 4}]}
    out = validate_expansions(state, None)
    assert out["rejected_chains"] == 1 and state["chains"] == []


def test_stationary_chain_samples_self_allowed():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2)], "chains": [
        {"id": 20, "anchor_id": 0, "time": .02, "end_time": .6,
         "hand": 0, "x": 1, "y": 1, "tx": 1, "ty": 1, "d": 2, "slices": 8}]}
    out = validate_expansions(state, None)
    assert out["rejected_chains"] == 0


def test_validate_never_moves_or_recuts_additions():
    from map_creation.v2_forward_expansion import validate_expansions
    copy = _note(1, .5, 0, 1, 1, 3, kind="copy", anchor_id=0)
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2), copy], "chains": []}
    before = dict(copy)
    validate_expansions(state, None)
    assert copy == before


def test_forward_emphasis_applies_and_sees_prior_stack():
    from map_creation.v2_forward_expansion import apply_note_emphasis
    state = _emphasis_state()
    draws = _ScriptedDraws(.99, .99, .99)
    out = apply_note_emphasis(state, None, rng=draws)
    assert out["applied"] and out["added_notes"] >= 1
    assert all(n["id"] is not None for n in state["notes"])


def test_emphasis_toggles_disable(monkeypatch):
    from map_creation.v2_forward_expansion import apply_note_emphasis
    monkeypatch.setattr(config, "emphasize_beats_flag", False)
    state = _emphasis_state()
    out = apply_note_emphasis(state, None, rng=_ScriptedDraws())
    assert out["applied"] is False and len(state["notes"]) == 8
    monkeypatch.setattr(config, "emphasize_beats_flag", True)
    monkeypatch.setattr(config, "single_notes_only_flag", True)
    out = apply_note_emphasis(_emphasis_state(), None, rng=_ScriptedDraws())
    assert out["applied"] is False


def test_forward_emphasis_gate_factory_protocol():
    from map_creation import v2_note_emphasis
    seen = {}

    class _Probe:
        p3, p2 = 0., 0.

        def __init__(self, state, song, eps):
            seen["built"] = (state, song, eps)

        def eligible(self, anchor):
            return False

        def copy(self, anchor, cell, identity):
            raise AssertionError("unused")

        def check(self, proposed):
            raise AssertionError("unused")

    out = v2_note_emphasis.apply_note_emphasis(
        _emphasis_state(), None, rng=_ScriptedDraws(*([.5] * 4)),
        gate_factory=_Probe)
    assert seen["built"][2] == pytest.approx(.02)
    assert out["applied"] and out["added_notes"] == 0


def _generate_state_setup(monkeypatch):
    from map_creation import v2_postprocessing as post
    monkeypatch.setattr(post, "sanity_check_timing2",
                        lambda name, times, audio_path=None: np.asarray(times))
    monkeypatch.setattr(post, "align_beats_on_bpm", lambda times, bpm: np.asarray(times))
    monkeypatch.setattr(post, "fill_map_times_scale",
                        lambda times, index=5: np.asarray(times))
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda audio_path: (np.zeros((2, 2)), np.zeros((2, 2)), np.empty(0)))
    monkeypatch.setattr(post, "remove_silent_times",
                        lambda times, silent: np.asarray(times))
    monkeypatch.setattr(post, "_start_end_times",
                        lambda times, volume, onset: np.asarray(times))


def _song_and_parts():
    from training.beat_models.contracts import CORE, Song
    song = Song("forward", {"duration": 11., "bpm": 120.},
                {"features": np.zeros((1100, 82), np.float32),
                 "coarse_features": np.zeros((11, 82), np.float32),
                 "frame_times": np.arange(1100, dtype=np.float64) / 100,
                 "bin_durations": np.ones(11, np.float32)})
    vocab = {"multiplier": [{"kind": "none"},
                            {"kind": "copies", "copies": [{"dx": 1, "dy": 0, "dt": .05}]}],
             "walls": [{"kind": "none"}],
             "lighting": [{"type": 0, "value": 1}]}

    class _Timing:
        status = "dummy"

        def __call__(self, inputs):
            values = np.zeros((len(inputs["audio"]), CORE, 3), np.float32)
            values[..., 0] = 1
            for frame in (50, 100, 300):
                values[:, frame, 0] = 0
                values[:, frame, 1] = 1
            return {"counts": values}

    def lighting(times, encoder):
        return ([{"time": float(time), "type": 0, "value": 1} for time in times],
                {"backend": "v1", "queries": len(times)})

    return song, vocab, _Timing(), lighting


def _run_generate(monkeypatch, force_expansion=False, **flags):
    from map_creation import v2_pipeline
    from training.pipeline_v2.pipeline import dummy_predictors
    from training.pipeline_v2.settings import preparation_settings
    _generate_state_setup(monkeypatch)
    for key, value in {"v2_map_intro_no_notes_seconds": 0.,
                       "v2_map_intro_no_hazards_seconds": 0.,
                       "v2_diagonal_max_fraction": 1.,
                       "v2_max_note_gap_seconds": 0.,
                       "v2_cleanup_note_heuristics": False,
                       "v2_notes_onset_snap": False,
                       "emphasize_beats_flag": False,
                       "v2_note_postprocessing_mode": "forward"}.items():
        monkeypatch.setattr(config, key, value)
    for key, value in flags.items():
        monkeypatch.setattr(config, key, value)
    song, vocab, timing, lighting = _song_and_parts()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = timing
    if force_expansion:
        vocab["multiplier"] = [{"kind": "copies",
                                "copies": [{"dx": 1, "dy": 0, "dt": .05}]}]
    settings = preparation_settings()
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    return v2_pipeline.generate(
        song, 4., predictors,
        lambda times: np.zeros((len(times), 16), np.float32),
        settings, vocab, lighting_generator=lighting,
        random_factor=0, logger=lambda message: None)


def test_generate_forward_never_calls_old_repair_entry_points(monkeypatch):
    from map_creation import v2_postprocessing as post
    for name in ("apply_fast_direction_rules", "apply_diagonal_cap",
                 "apply_speed_cap", "enforce_note_flow_and_clearance",
                 "apply_fast_counter_sweep", "finalize_playability_reach"):
        monkeypatch.setattr(post, name,
                            lambda *a, **k: pytest.fail(f"forbidden hook {name}"))
    result = _run_generate(monkeypatch)
    assert result.diagnostics["playability_policy"] == "forward"
    assert "forward_postprocessing" in result.diagnostics


def test_generate_search_mode_when_flag_disables(monkeypatch):
    from map_creation import v2_postprocessing as post
    calls = []
    real = post.apply_fast_direction_rules

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(post, "apply_fast_direction_rules", spy)
    result = _run_generate(monkeypatch, v2_cleanup_note_heuristics=True)
    assert calls
    assert result.diagnostics["playability_policy"] == "search"


def test_generate_invalid_mode_raises(monkeypatch):
    with pytest.raises(ValueError):
        _run_generate(monkeypatch, v2_note_postprocessing_mode="bogus")


def test_validate_state_forward_collects_vs_strict_raises():
    from map_creation.v2_pipeline import _validate_state
    state = {"notes": [_note(0, .5, 0, 3, 1, 2, kind="retained")],
             "chains": [], "walls": [], "bombs": [], "arcs": [], "lighting": []}
    findings = _validate_state(state, 4., playability_mode="forward")
    assert findings["forbidden_cut_ids"] == [0]
    with pytest.raises(ValueError):
        _validate_state(state, 4., playability_mode="search")


def test_validate_state_structural_errors_raise_in_both_modes():
    from map_creation.v2_pipeline import _validate_state
    state = {"notes": [_note(0, .5, 0, 4, 1, 2, kind="retained")],
             "chains": [], "walls": [], "bombs": [], "arcs": [], "lighting": []}
    for mode in ("forward", "search"):
        with pytest.raises(ValueError):
            _validate_state(state, 4., playability_mode=mode)


def test_forward_diagnostics_markers_not_strict_success(monkeypatch):
    result = _run_generate(monkeypatch)
    diagnostics = result.diagnostics
    assert diagnostics["playability_policy"] == "forward"
    assert diagnostics["playability_strict_validated"] is False
    assert "playability_findings" in diagnostics
    assert diagnostics["production_validated"] is True
    assert diagnostics["gameplay_validated"] is False

def test_validate_rejects_cross_kind_batch_atomically():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2), _retained(1, 2., 1, 3, 2, 7),
                       _note(5, .5, 1, 1, 1, 0, kind="copy", anchor_id=1)],
             "chains": [{"id": 20, "anchor_id": 0, "time": 0., "end_time": 1.,
                         "hand": 0, "x": 0, "y": 1, "tx": 2, "ty": 1, "d": 2,
                         "slices": 4}]}
    out = validate_expansions(state, None)
    assert out["rejected_copies"] == 1 and out["rejected_chains"] == 1
    assert len(state["notes"]) == 2 and state["chains"] == []
    state["notes"].append(_note(6, .6, 0, 0, 0, 5, kind="copy", anchor_id=0))
    assert validate_expansions(state, None)["rejected_copies"] == 0
    assert [n["id"] for n in state["notes"]] == [0, 1, 6]


def test_validate_rejects_dangling_anchor_and_forbidden_copy(monkeypatch):
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 1, 0),
                       _note(5, .5, 0, 1, 1, 0, kind="copy", anchor_id=99)],
             "chains": []}
    assert validate_expansions(state, None)["rejected_copies"] == 1
    assert len(state["notes"]) == 1
    state = {"notes": [_retained(0, 0., 0, 0, 1, 0),
                       _note(5, .5, 0, 1, 1, 0, kind="copy")],
             "chains": []}
    assert validate_expansions(state, None)["rejected_copies"] == 1
    assert len(state["notes"]) == 1
    state = {"notes": [_retained(0, 0., 0, 0, 1, 0),
                       _note(5, .5, 0, 3, 1, 6, kind="copy", anchor_id=0)],
             "chains": []}
    assert validate_expansions(state, None)["rejected_copies"] == 1
    monkeypatch.setattr(config, "v2_playability_reach", False)
    monkeypatch.setattr(config, "v2_playability_priority", False)
    state = {"notes": [_retained(0, 0., 0, 0, 1, 0),
                       _note(5, .5, 0, 3, 1, 6, kind="copy", anchor_id=0)],
             "chains": []}
    assert validate_expansions(state, None)["rejected_copies"] == 0


def test_emphasis_sequential_checks_see_accepted_stack():
    from map_creation.v2_forward_expansion import apply_note_emphasis
    notes = [_retained(i, .2 * i, 0, 0, 1, 6) for i in range(4)]
    notes += [_retained(4, 3., 0, 0, 0, 3), _retained(5, 3., 1, 3, 2, 0)]
    state = {"notes": notes, "chains": []}
    out = apply_note_emphasis(state, None, rng=_ScriptedDraws(.99, .99))
    assert out["applied"]
    left = [n for n in state["notes"] if n.get("anchor_id") == 4]
    assert [(n["x"], n["y"]) for n in left] == [(1, 0), (2, 0)]
    assert out["rejected"].get("cell_occupied", 0) >= 1
    assert not any(n.get("anchor_id") == 5 for n in state["notes"])


def test_emphasis_gate_sees_frozen_incumbents():
    from map_creation.v2_forward_expansion import apply_note_emphasis
    notes = [_retained(i, .2 * i, 0, 0, 1, 0) for i in range(4)]
    notes += [_retained(4, 3., 0, 0, 0, 3),
              _note(9, 3., 1, 1, 0, 0, kind="copy", anchor_id=0)]
    state = {"notes": notes, "chains": []}
    out = apply_note_emphasis(state, None, rng=_ScriptedDraws(.99))
    assert out["applied"] and out["rejected"].get("cell_occupied", 0) >= 1


def test_emphasis_forbidden_copy_rejected():
    from map_creation.v2_forward_expansion import apply_note_emphasis
    notes = [_retained(i, .2 * i, 0, 0, 1, 0) for i in range(4)]
    notes += [_retained(4, 3., 0, 2, 1, 2)]
    state = {"notes": notes, "chains": []}
    out = apply_note_emphasis(state, None, rng=_ScriptedDraws(.99))
    assert out["applied"] and out["rejected"].get("forbidden_cut", 0) >= 1


def test_emphasis_disabled_still_drops_guard():
    from map_creation.v2_forward_expansion import apply_note_emphasis, _PRIVATE
    import map_creation.v2_note_emphasis as emphasis_module
    import pytest
    state = _emphasis_state()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "emphasize_beats_flag", False)
        apply_note_emphasis(state, None, rng=_ScriptedDraws())
    assert _PRIVATE not in state


def test_direction_fast_boundary_exclusive(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_fast_gap_seconds", .2)
    notes = [_note(0, 0., 0, 0, 1, 0), _note(1, .2, 0, 1, 2, 1)]
    repair_notes(notes, cap=lambda n: 9., rng=_ScriptedDraws(.9, .9))
    assert notes[1]["d"] == 1


def test_direction_diag_only_trigger_keeps_incumbent(monkeypatch):
    from map_creation.v2_forward_postprocessing import repair_notes
    monkeypatch.setattr(config, "v2_diagonal_max_fraction", .1)
    notes = [_note(0, 0., 0, 1, 1, 4), _note(1, .3, 0, 1, 1, 7),
             _note(2, .6, 0, 1, 1, 4), _note(3, .9, 0, 1, 1, 7)]
    out = repair_notes(notes, cap=lambda n: 9., rng=_ScriptedDraws(.9, .9, .9, .9))
    assert out["direction"]["unresolved"] >= 0
    assert sum(n["d"] not in (4, 5, 6, 7) for n in notes) >= 1


def test_generate_rejects_expansion_flow_violation_and_drops_guard(monkeypatch):
    """The forced 50 ms same-direction copy is a hard flow violation now."""
    result = _run_generate(monkeypatch, force_expansion=True)
    assert result.diagnostics["final_counts"]["learned_copies"] == 0
    assert result.diagnostics["expansion_clearance"]["rejected_copies"] > 0
    assert "_forward_guard" not in result.state
    assert result.diagnostics["playability_policy"] == "forward"
    assert result.diagnostics["production_validated"] is True


def test_custom_spoof_namespace_stays_strict(monkeypatch):
    from map_creation import v2_pipeline
    from test_v2_generation_pipeline import processors, audio_song, vocabulary, lighting, Timing
    from training.pipeline_v2.pipeline import dummy_predictors
    from training.pipeline_v2.settings import preparation_settings
    _generate_state_setup(monkeypatch)
    for key, value in {"v2_cleanup_note_heuristics": False,
                       "v2_note_postprocessing_mode": "forward",
                       "emphasize_beats_flag": False}.items():
        monkeypatch.setattr(config, key, value)
    spoof = processors()
    spoof.playability_mode = "forward"
    spoof.postprocess_notes = lambda values, event_times, logger=None: (
        [_note(0, .5, 0, 3, 1, 2, kind="retained")], {})
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    settings = preparation_settings()
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    with pytest.raises(ValueError, match="forbidden cuts"):
        v2_pipeline.generate(song, 4., predictors,
                             lambda times: np.zeros((len(times), 16), np.float32),
                             settings, vocab, lighting_generator=lighting,
                             postprocessors=spoof, random_factor=0,
                             logger=lambda message: None)


def test_strict_marker_matrix(monkeypatch):
    result = _run_generate(monkeypatch, v2_playability_enabled=False)
    assert result.diagnostics["playability_policy"] == "search"
    assert result.diagnostics["playability_strict_validated"] is False
    result = _run_generate(monkeypatch, v2_playability_reach=False,
                           v2_playability_priority=False)
    assert result.diagnostics["playability_policy"] == "search"
    assert result.diagnostics["playability_strict_validated"] is False


def test_accepts_many_bounded_index_work():
    import pytest
    from map_creation.v2_forward_clearance import LocalNoteIndex
    work = {"count": 0}
    real = LocalNoteIndex.add_many

    def spy(self, records):
        records = list(records)
        work["count"] += len(self.entries) + len(records)
        return real(self, records)

    proposals = [_note(100 + i, .01 * i, 0, 1, 1, 8, kind="chain_sample",
                       anchor_id=0, chain_id=7) for i in range(1000)]
    index = LocalNoteIndex([_retained(0, 0., 0, 1, 1, 2)])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(LocalNoteIndex, "add_many", spy)
        assert index.accepts_many(proposals)
    assert work["count"] <= 3 * len(proposals)

def test_expansion_rejects_staggered_flow_breaking_copy():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2), _retained(1, 1., 0, 0, 2, 2),
                       _note(5, .5, 0, 1, 1, 2, kind="copy", anchor_id=0)],
             "chains": []}
    out = validate_expansions(state, None)
    assert out["rejected_copies"] == 1
    assert len(state["notes"]) == 2
    state["notes"].append(_note(6, .5, 0, 1, 1, 5, kind="copy", anchor_id=0))
    assert validate_expansions(state, None)["rejected_copies"] == 0


def test_expansion_rejects_copy_making_incumbent_counter():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 1., 0, 0, 0, 0),
                       _note(5, .85, 0, 1, 1, 6, kind="copy", anchor_id=0)],
             "chains": []}
    out = validate_expansions(state, None)
    assert out["rejected_copies"] == 1 and len(state["notes"]) == 1


def test_expansion_same_time_stack_allowed():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2), _retained(1, 1., 0, 0, 0, 2),
                       _note(5, 1., 0, 1, 0, 5, kind="copy", anchor_id=1)],
             "chains": []}
    assert validate_expansions(state, None)["rejected_copies"] == 0


def test_expansion_rejects_outgoing_flow_breaking_copy():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2), _retained(1, 1., 0, 0, 2, 2),
                       _note(5, .5, 0, 1, 1, 5, kind="copy", anchor_id=0)],
             "chains": []}
    assert validate_expansions(state, None)["rejected_copies"] == 0
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2), _retained(1, .5, 0, 0, 2, 3),
                       _note(5, .25, 0, 1, 1, 5, kind="copy", anchor_id=0)],
             "chains": []}
    out = validate_expansions(state, None)
    assert out["rejected_copies"] == 1 and len(state["notes"]) == 2


def test_expansion_direction_rejection_keeps_indexes_clean():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 1, 2), _retained(1, 1., 0, 0, 2, 2),
                       _note(5, .5, 0, 1, 1, 2, kind="copy", anchor_id=0)],
             "chains": []}
    assert validate_expansions(state, None)["rejected_copies"] == 1
    state["notes"].append(_note(6, .5, 0, 1, 1, 5, kind="copy", anchor_id=0))
    assert validate_expansions(state, None)["rejected_copies"] == 0
    assert [n["id"] for n in state["notes"]] == [0, 1, 6]


def test_expansion_dot_run_propagation_rejects():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 0, 0, 2),
                       _retained(1, .15, 0, 0, 0, 8),
                       _retained(2, .3, 0, 2, 0, 2),
                       _note(5, .45, 0, 1, 0, 2, kind="copy", anchor_id=2)],
             "chains": []}
    out = validate_expansions(state, None)
    assert out["rejected_copies"] == 1

def test_fast_ids_covers_later_stack_members():
    from map_creation.v2_forward_directions import _fast_ids
    ordered = [_note(0, 0., 0, 1, 1, 0), _note(1, .15, 0, 1, 1, 0),
               _note(2, .15, 0, 1, 1, 0)]
    assert _fast_ids(ordered, .02, .2) == {0, 1, 2}


def test_expansion_rejects_same_onset_fast_counter_copy():
    from map_creation.v2_forward_expansion import validate_expansions
    state = {"notes": [_retained(0, 0., 0, 1, 1, 1),
                       _retained(1, .15, 0, 2, 1, 0),
                       _note(5, .15, 0, 2, 0, 0, kind="copy", anchor_id=1)],
             "chains": []}
    assert validate_expansions(state, None)["rejected_copies"] == 1
    state = {"notes": [_retained(0, 0., 0, 1, 1, 1),
                       _retained(1, .15, 0, 2, 1, 0),
                       _note(5, .15, 0, 0, 1, 0, kind="copy", anchor_id=1)],
             "chains": []}
    assert validate_expansions(state, None)["rejected_copies"] == 0


def test_guard_preserves_incumbent_residual_findings():
    from map_creation.v2_forward_directions import DirectionGuard
    incumbents = [_retained(0, 0., 0, 2, 2, 0), _retained(1, .15, 0, 1, 0, 0)]
    guard = DirectionGuard(incumbents)
    proposed = [_note(5, .3, 0, 0, 1, 6, kind="copy", anchor_id=1)]
    assert guard.check(proposed) is None


def test_guard_rejects_staggered_same_direction_copy():
    """A 50 ms-staggered copy repeating the cut is a hard flow violation."""
    from map_creation.v2_forward_directions import DirectionGuard
    guard = DirectionGuard([_retained(0, 1., 0, 0, 1, 2)])
    copy = _note(1, 1.05, 0, 1, 1, 2, kind="copy", anchor_id=0)
    assert guard.check([copy]) == "direction"


def test_guard_overlap_rejects_same_cell_windows_and_batch_pairs():
    from map_creation.v2_forward_directions import DirectionGuard
    guard = DirectionGuard([_retained(0, 1., 0, 0, 1, 2)])
    # same-hand same cell inside the 0.1 s window; the reversal is flow-legal
    copy = _note(1, 1.05, 0, 0, 1, 3, kind="copy", anchor_id=0)
    assert guard.check([copy]) == "overlap"
    # cross-hand member on the cell inside the 0.15 s window
    cross = _note(2, 1.1, 1, 0, 1, 0, kind="copy", anchor_id=0)
    assert guard.check([cross]) == "overlap"
    # proposed-vs-proposed same-cell pairs are caught too
    batch = [_note(3, 1.3, 0, 1, 1, 3, kind="copy", anchor_id=0),
             _note(4, 1.35, 0, 1, 1, 2, kind="copy", anchor_id=0)]
    assert guard.check(batch) == "overlap"


def test_guard_accepts_same_time_different_cell_stacks():
    from map_creation.v2_forward_directions import DirectionGuard
    guard = DirectionGuard([_retained(0, 1., 0, 0, 1, 2)])
    copy = _note(1, 1., 0, 1, 1, 2, kind="copy", anchor_id=0)
    assert guard.check([copy]) is None
    cross = _note(2, 1., 1, 2, 1, 3, kind="copy", anchor_id=0)
    assert guard.check([copy, cross]) is None
