"""Controlled postprocessing ablation arms and their injected variants."""

from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.evaluate_v2_postprocessing import (ARMS, NOTE_MINIMAL,
                                                 TIMING_HEURISTIC_STEPS,
                                                 _none_multiplier, arm_postprocessors)
from map_creation.v2_postprocessing import postprocess_notes, postprocess_timing
from tools.config import config
from training.beat_models.contracts import Song
from training.pipeline_v2.pipeline import dummy_predictors
from training.pipeline_v2.settings import preparation_settings


def audio_song(frames=1100):
    duration = frames / 100
    bins = int(np.ceil(duration))
    return Song("audio-only", {"duration": duration, "bpm": 120.},
                {"features": np.zeros((frames, 82), np.float32),
                 "coarse_features": np.zeros((bins, 82), np.float32),
                 "frame_times": np.arange(frames, dtype=np.float64) / 100,
                 "bin_durations": np.minimum(1., duration - np.arange(bins)).astype(np.float32)})


def beats(counts, frames=1100):
    counts = np.asarray(counts, np.int64)
    event_frames = np.flatnonzero(counts)
    probabilities = np.zeros((frames, 3), np.float32)
    probabilities[:, 0] = 1.
    for frame in event_frames:
        probabilities[frame, 0] = 0.
        probabilities[frame, counts[frame]] = 1.
    return {"counts": counts, "event_frames": event_frames,
            "event_times": event_frames / 100, "frame_probabilities": probabilities}


def model_notes(rows):
    return [{"id": index, "time": time, "hand": hand, "x": x, "y": y, "d": direction}
            for index, (time, hand, x, y, direction) in enumerate(rows)]


def vocabulary():
    return {"multiplier": [{"kind": "none"}, {"kind": "copies", "copies": [{"dx": 0, "dy": -1, "dt": .05}]}],
            "walls": [{"kind": "none"}, {"kind": "wall", "x": 0, "y": 0, "width": 1, "height": 3}],
            "lighting": [{"type": 0, "value": 1}, {"type": 1, "value": 2}]}


class Timing:
    status = "dummy"

    def __call__(self, inputs):
        batch = len(inputs["audio"])
        from training.beat_models.contracts import CORE
        values = np.zeros((batch, CORE, 3), np.float32)
        values[..., 0] = 1
        values[:, 50, 0] = 0
        values[:, 50, 2] = 1
        return {"counts": values}


def lighting(times, encoder):
    return ([{"time": float(time), "type": 0, "value": 2000000804} for time in times],
            {"backend": "v1", "queries": len(times)})


def test_unknown_timing_skip_step_is_rejected():
    song = audio_song()
    counts = np.zeros(1100, np.int64)
    counts[100] = 1
    with pytest.raises(ValueError, match="Unknown timing postprocessing"):
        postprocess_timing(song, beats(counts), None, skip=("not_a_step",))


def test_unknown_note_skip_step_is_rejected():
    with pytest.raises(ValueError, match="Unknown note postprocessing"):
        postprocess_notes(model_notes([(1., 0, 0, 0, 0)]), [1.], skip=("not_a_step",))


def test_minimal_timing_keeps_only_minimum_spacing_and_needs_no_audio():
    song = audio_song()
    counts = np.zeros(1100, np.int64)
    counts[[100, 103, 200, 500]] = [1, 1, 2, 1]
    candidates, diagnostics = postprocess_timing(song, beats(counts), None,
                                                 skip=TIMING_HEURISTIC_STEPS)
    assert [operation["stage"] for operation in diagnostics["operations"]] == ["minimum_spacing"]
    assert [candidate["time"] for candidate in candidates] == [1., 2., 5.]
    assert diagnostics["minimum_spacing_removed"] == 1
    assert sorted(diagnostics["skipped_steps"]) == sorted(TIMING_HEURISTIC_STEPS)


def test_each_timing_step_is_independently_skippable(monkeypatch):
    from map_creation import v2_postprocessing as postprocessors

    calls = []
    monkeypatch.setattr(postprocessors, "sanity_check_timing2",
                        lambda name, times, audio_path=None: calls.append("onset_snap") or np.asarray(times))
    monkeypatch.setattr(postprocessors, "align_beats_on_bpm",
                        lambda times, bpm: calls.append("bpm_alignment") or np.asarray(times))
    monkeypatch.setattr(postprocessors, "fill_map_times_scale",
                        lambda times, index=5: calls.append(f"fill_{index}") or np.asarray(times))
    monkeypatch.setattr(postprocessors, "_legacy_timing_features",
                        lambda audio_path: calls.append("features") or
                        (np.zeros((2, 2)), np.zeros((2, 2)), np.empty(0)))
    monkeypatch.setattr(postprocessors, "remove_silent_times",
                        lambda times, silent: calls.append("silence_removal") or np.asarray(times))
    monkeypatch.setattr(postprocessors, "_start_end_times",
                        lambda times, volume, onset: calls.append("start_end_fill") or np.asarray(times))
    song = audio_song()
    counts = np.zeros(1100, np.int64)
    counts[[200, 500]] = 1
    monkeypatch.setattr(config, "max_speed", 32.)
    # the fixture gap (2.0->5.0 s) is below the default net threshold; pin it
    # off so a future smaller value cannot send "x.egg" into librosa — same
    # for the onset-coverage decode
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)
    monkeypatch.setattr(config, "v2_onset_fill_flag", False)

    calls.clear()
    postprocess_timing(song, beats(counts), "x.egg", skip=("onset_snap",))
    assert "onset_snap" not in calls and "bpm_alignment" in calls and "silence_removal" in calls

    calls.clear()
    postprocess_timing(song, beats(counts), "x.egg",
                       skip=("bpm_alignment", "onset_snap", "start_end_fill",
                             "density_fill", "high_difficulty_fill", "silence_removal"))
    assert calls == []


def test_minimal_notes_skip_mutators_and_mark_everything_retained():
    notes = model_notes([(1., 0, 0, 0, 0), (1., 1, 3, 2, 1), (2., 0, 1, 1, 8),
                       (3., 0, 0, 2, 0), (4., 1, 2, 0, 3)])
    structured, diagnostics = postprocess_notes(notes, [1., 2., 3., 4.],
                                              skip=NOTE_MINIMAL)
    assert [operation["stage"] for operation in diagnostics["operations"]] == ["duplicate_removal"]
    assert len(structured) == 5
    assert all(note["kind"] == "retained" for note in structured)
    assert diagnostics["heuristic_notes"] == 0 and diagnostics["retained_notes"] == 5
    assert diagnostics["pre_clearance"]["same_cell_pairs"] >= 0
    assert diagnostics["note_clearance"]["residual"] == {"same_cell_pairs": 0,
                                                         "swing_into_pairs": 0,
                                                         "flow_repeats": 0}


def test_flow_additions_skip_disables_in_flow_adders(monkeypatch):
    from map_creation import v2_postprocessing as postprocessors

    seen = {}

    def spy(values, timings):
        seen.update(emphasis=config.emphasize_beats_flag,
                    breaks=config.add_breaks_flag, dots=config.add_dot_notes)
        return values

    monkeypatch.setattr(postprocessors, "sanity_check_notes", spy)
    notes = model_notes([(1., 0, 0, 0, 0), (2., 1, 3, 2, 1)])
    postprocess_notes(notes, [1., 2.], skip=("mirroring", "flow_additions"))
    assert seen == {"emphasis": False, "breaks": False, "dots": 0}

    seen.clear()
    postprocess_notes(notes, [1., 2.], skip=("mirroring",))
    assert seen["emphasis"] is True and config.emphasize_beats_flag is True


def test_clearance_is_skippable_for_diagnostics():
    notes = model_notes([(1., 0, 0, 0, 0), (1.5, 0, 1, 0, 0), (2., 1, 3, 2, 4)])
    structured, diagnostics = postprocess_notes(notes, [1., 1.5, 2.],
                                              skip=(*NOTE_MINIMAL, "clearance"))
    assert diagnostics["note_clearance"] is None
    assert diagnostics["same_direction_recut_notes"] is None
    assert len(structured) == 3


def test_arm_postprocessors_expose_the_full_injected_api():
    shim = arm_postprocessors(("onset_snap",), ("mirroring",))
    for name in ("postprocess_timing", "postprocess_notes", "validate_expansions",
                 "validate_hazards", "validate_arcs", "enforce_note_flow_and_clearance"):
        assert callable(getattr(shim, name))


def test_none_multiplier_forces_the_none_class():
    def predictor(inputs):
        return {"red": np.array([[.1, .9]]), "blue": np.array([[.8, .2]])}

    output = _none_multiplier(predictor)({"audio": np.zeros((1, 1))})
    assert output["red"].tolist() == [[1., 0.]] and output["blue"].tolist() == [[1., 0.]]


def test_raw_arm_runs_generate_end_to_end_without_audio_or_expansion():
    from map_creation.v2_pipeline import generate

    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    predictors["multiplier"] = _none_multiplier(predictors["multiplier"])
    settings = preparation_settings()
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    shim = arm_postprocessors(TIMING_HEURISTIC_STEPS, NOTE_MINIMAL)
    result = generate(song, 4., predictors, lambda times: np.zeros((len(times), 16), np.float32),
                      settings, vocab, lighting_generator=lighting, postprocessors=shim,
                      random_factor=0, logger=lambda message: None)
    assert result.status == "dummy" and result.diagnostics["export_ready"]
    assert result.diagnostics["raw_decisions"]["multiplier_non_none"] == 0
    assert result.diagnostics["final_counts"]["heuristic_notes"] == 0
    assert result.diagnostics["final_counts"]["learned_copies"] == 0
    assert result.diagnostics["final_counts"]["retained_notes"] == \
        result.diagnostics["raw_decisions"]["position_notes"]
    post = result.diagnostics["note_postprocessing"]
    assert post["pre_clearance"] is not None and post["skipped_steps"] == sorted(NOTE_MINIMAL)


def test_arm_definitions_reference_existing_steps():
    from map_creation.v2_postprocessing import _NOTE_STEPS, _TIMING_STEPS
    for name, arm in ARMS.items():
        assert set(arm["timing_skip"]) <= set(_TIMING_STEPS), name
        assert set(arm["notes_skip"]) <= set(_NOTE_STEPS), name
        assert isinstance(arm["multiplier"], bool)


def _patch_timing_externals(monkeypatch):
    from map_creation import v2_postprocessing as postprocessors

    monkeypatch.setattr(postprocessors, "sanity_check_timing2",
                        lambda name, times, audio_path=None: np.asarray(times))
    monkeypatch.setattr(postprocessors, "align_beats_on_bpm",
                        lambda times, bpm: np.asarray(times))
    monkeypatch.setattr(postprocessors, "fill_map_times_scale",
                        lambda times, index=5: np.asarray(times))
    monkeypatch.setattr(postprocessors, "_legacy_timing_features",
                        lambda audio_path: (np.zeros((2, 2)), np.zeros((2, 2)), np.empty(0)))
    monkeypatch.setattr(postprocessors, "remove_silent_times",
                        lambda times, silent: np.asarray(times))
    monkeypatch.setattr(postprocessors, "_start_end_times",
                        lambda times, volume, onset: np.asarray(times))


def _generate_with_production_postprocessors(monkeypatch, **overrides):
    from map_creation.v2_pipeline import generate

    _patch_timing_externals(monkeypatch)
    # no audio_path in these runs — keep the max-gap safety net off
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    predictors["multiplier"] = _none_multiplier(predictors["multiplier"])
    settings = preparation_settings()
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    return generate(song, 4., predictors, lambda times: np.zeros((len(times), 16), np.float32),
                    settings, vocab, lighting_generator=lighting,
                    random_factor=0, logger=lambda message: None, **overrides)


def test_production_default_skips_legacy_note_mutators(monkeypatch):
    monkeypatch.setattr(config, "v2_cleanup_note_heuristics", False)
    post = _generate_with_production_postprocessors(monkeypatch).diagnostics["note_postprocessing"]
    assert post["skipped_steps"] == ["clearance", "flow_additions", "legacy_note_flow",
                                     "mirroring", "same_direction_recut"]
    assert all(operation["stage"] != "legacy_note_flow" for operation in post["operations"])


def test_flag_true_restores_full_legacy_note_cleanup(monkeypatch):
    monkeypatch.setattr(config, "v2_cleanup_note_heuristics", True)
    post = _generate_with_production_postprocessors(monkeypatch).diagnostics["note_postprocessing"]
    assert post["skipped_steps"] == []
    assert any(operation["stage"] == "legacy_note_flow" for operation in post["operations"])


def test_app1_control_matches_explicit_legacy_notes_geometry_and_rng(monkeypatch):
    from evaluation.evaluate_v2_postprocessing import _arm_processors
    from evaluation.playability_experiments import experiment_policy

    first_rng, second_rng = (np.random.default_rng(71) for _ in range(2))
    with experiment_policy('app1_control'):
        direct = _generate_with_production_postprocessors(monkeypatch, rng=first_rng)
    with experiment_policy('app1_control'):
        control = _generate_with_production_postprocessors(
            monkeypatch, rng=second_rng, postprocessors=_arm_processors('app1_control', ARMS['app1_control']))
    assert first_rng.bit_generator.state == second_rng.bit_generator.state
    np.testing.assert_array_equal(direct.state['raw_model_notes'], control.state['raw_model_notes'])
    for key in ('notes', 'chains', 'walls', 'bombs', 'arcs', 'lighting'):
        assert direct.state[key] == control.state[key]
    for key in ('speed_cap', 'speed_cap_post_expansion', 'diagonal_cap', 'note_clearance', 'intro_zones'):
        assert direct.diagnostics.get(key) == control.diagnostics.get(key)
