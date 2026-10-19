"""Production composition contracts with injected predictors, never training."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from map_creation.v2_pipeline import _atomic_expand, generate, infer_beats
from tools.config import config
from training.beat_models.contracts import CORE, Song, timing_window
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


def vocabulary():
    return {"multiplier": [{"kind": "none"}, {"kind": "copies", "copies": [{"dx": 0, "dy": -1, "dt": .05}]}],
            "walls": [{"kind": "none"}, {"kind": "wall", "x": 0, "y": 0, "width": 1, "height": 3}],
            "lighting": [{"type": 0, "value": 1}, {"type": 1, "value": 2}]}


class Timing:
    status = "dummy"

    def __init__(self):
        self.batches = []

    def __call__(self, inputs):
        batch = len(inputs["audio"])
        self.batches.append(batch)
        values = np.zeros((batch, CORE, 3), np.float32)
        values[..., 0] = 1
        values[:, 50, 0] = 0
        values[:, 50, 2] = 1
        return {"counts": values}


def lighting(times, encoder):
    """Injected retained-lighting boundary; decoder parity has dedicated tests."""
    return ([{"time": float(time), "type": 0, "value": 2000000804} for time in times],
            {"backend": "v1", "queries": len(times)})


def processors():
    def timing(song, beats, audio_path, logger=None):
        return [{"time": float(time), "count": int(beats["counts"][frame])}
                for frame, time in zip(beats["event_frames"], beats["event_times"])], {}

    def notes(values, event_times, logger=None):
        return values, {}

    def validate(state, song, logger=None):
        return {}

    return SimpleNamespace(postprocess_timing=timing, postprocess_notes=notes,
                           validate_expansions=validate, validate_hazards=validate, validate_arcs=validate)


def test_timing_windows_are_target_free_and_batching_preserves_tails():
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    sequential, batched = Timing(), Timing()
    predictors["timing"] = sequential
    first = infer_beats(song, 4., predictors, batch_size=1)
    predictors["timing"] = batched
    second = infer_beats(song, 4., predictors, batch_size=2)
    assert sequential.batches == [1, 1, 1]
    assert batched.batches == [2, 1]
    np.testing.assert_array_equal(first["counts"], second["counts"])
    window, valid = timing_window(song, second["conditions"], 1024)
    assert valid.sum() == 76
    assert window["audio"].shape == (1, 1020, 82)
    assert set(song.arrays) == {"features", "coarse_features", "frame_times", "bin_durations"}


def test_production_pipeline_uses_only_audio_and_keeps_all_output_families(monkeypatch):
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    calls = []
    predictors["lighting"] = lambda inputs: pytest.fail("Production must use v1 lighting")

    def encoder(times):
        calls.append(np.asarray(times).copy())
        return np.zeros((len(times), 16), np.float32)

    settings = preparation_settings()
    settings.pop("stage_policies", None)  # Exercise the retained legacy decoder.
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    result = generate(song, 4., predictors, encoder, settings, vocab,
                      lighting_generator=lighting, postprocessors=processors(), random_factor=0, logger=lambda message: None)
    assert result.status == "dummy"
    assert result.diagnostics["export_ready"]
    assert result.stages["lighting"]["backend"] == "v1"
    np.testing.assert_array_equal([event["time"] for event in result.state["lighting"]],
                                  result.state["event_times"])
    assert set(result.stages) == {"intensity", "timing", "positions", "arrows", "multiplier", "hazards", "arcs", "lighting"}
    assert all(result.state[family] for family in ("notes", "walls", "bombs", "arcs", "lighting"))
    assert result.stages["multiplier"]["added_notes"] > 0
    assert any(len(query) > 1 for query in calls)
    assert "_tracks" not in result.state and "_occupancy" not in result.state
    assert result.review_stops == []
    raw, final = result.diagnostics["raw_decisions"], result.diagnostics["final_counts"]
    assert raw["position_events"] == 3 and raw["position_notes"] == raw["arrow_notes"] == 6
    assert raw["multiplier_non_none"] == final["learned_copies"] == 3
    assert raw["wall_proposals"] == final["walls"]
    assert raw["bomb_proposals"] == final["bombs"]
    assert raw["arc_attachments"] == final["arcs"]
    assert raw["lighting_commands"] == final["lighting_events"]
    assert final["colored_notes"] == 9 and final["unique_onsets"] == 6
    assert result.state["raw_model_notes"].shape == (6, 6)
    assert not result.state["raw_model_notes"].dtype.hasobject


def test_rejected_expansion_is_atomic_and_does_not_mutate_anchor():
    anchor = {"id": 4, "time": 1., "hand": 0, "x": 0, "y": 2, "d": 1, "kind": "retained"}
    state = {"notes": [anchor], "chains": []}
    original, rejected = deepcopy(state), []
    template = {"kind": "copies", "copies": [{"dx": 1, "dy": 0, "dt": 0}, {"dx": -1, "dy": 0, "dt": 0}]}
    assert not _atomic_expand(state, anchor, template, 10., rejected)
    assert state == original and rejected
    assert not _atomic_expand(state, anchor, {"kind": "copies", "copies": [{"dx": 0, "dy": 0, "dt": 0}]}, 10., rejected)
    assert state == original


def test_single_mode_cannot_be_undone_by_multiplier(monkeypatch):
    monkeypatch.setattr(config, "single_notes_only_flag", True)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    result = generate(song, 4., predictors, lambda times: np.zeros((len(times), 16), np.float32),
                      preparation_settings(), vocab, lighting_generator=lighting, postprocessors=processors(), random_factor=0,
                      logger=lambda message: None)
    assert not result.state["chains"]
    assert not any(note["kind"] == "copy" for note in result.state["notes"])
    assert result.diagnostics["raw_decisions"]["multiplier_non_none"] == 3
    assert result.diagnostics["final_counts"]["learned_copies"] == 0


def test_multiplier_histories_record_rejected_outputs_as_none(monkeypatch):
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    seen = []

    class Multiplier:
        status = "dummy"

        def __call__(self, inputs):
            seen.append(inputs["history"][0][inputs["history_mask"][0]].copy())
            return {"red": np.array([[0., 1.]], np.float32), "blue": np.array([[1., 0.]], np.float32)}

    predictors["multiplier"] = Multiplier()
    hooks = processors()

    def reject(state, song, logger=None):
        state["notes"] = [note for note in state["notes"] if note.get("kind") != "copy"]
        return {}

    hooks.validate_expansions = reject
    result = generate(song, 4., predictors, lambda times: np.zeros((len(times), 16), np.float32),
                      preparation_settings(), vocab, lighting_generator=lighting, postprocessors=hooks, random_factor=0,
                      logger=lambda message: None)
    assert len(seen) == 3
    assert (seen[1][-1] == [0, 0]).all()
    assert not any(note["kind"] == "copy" for note in result.state["notes"])
    assert len(result.diagnostics["rejected_expansions"]) == 3
    assert result.diagnostics["raw_decisions"]["multiplier_non_none"] == 3
    assert result.diagnostics["final_counts"]["learned_copies"] == 0


def test_raw_hazard_and_arc_choices_survive_clearance_rejection_in_diagnostics(monkeypatch):
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    settings = preparation_settings()
    settings.pop("stage_policies", None)  # Exercise the retained legacy decoder.
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    hooks = processors()

    def reject_hazards(state, song, logger=None):
        counts = {"rejected_walls": len(state["walls"]), "rejected_bombs": len(state["bombs"])}
        state["walls"], state["bombs"] = [], []
        return counts

    def reject_arcs(state, song, logger=None):
        count = len(state["arcs"])
        state["arcs"] = []
        return {"rejected_arcs": count}

    hooks.validate_hazards, hooks.validate_arcs = reject_hazards, reject_arcs
    result = generate(song, 4., predictors, lambda times: np.zeros((len(times), 16), np.float32),
                      settings, vocab, lighting_generator=lighting, postprocessors=hooks, random_factor=0, logger=lambda message: None)
    raw, final = result.diagnostics["raw_decisions"], result.diagnostics["final_counts"]
    assert raw["wall_proposals"] > 0 and raw["bomb_proposals"] > 0 and raw["arc_attachments"] > 0
    assert final["walls"] == final["bombs"] == final["arcs"] == 0
    assert result.diagnostics["hazard_clearance"]["rejected_walls"] == raw["wall_proposals"]
    assert result.diagnostics["hazard_clearance"]["rejected_bombs"] == raw["bomb_proposals"]
    assert result.diagnostics["arc_clearance"]["rejected_arcs"] == raw["arc_attachments"]


@pytest.mark.parametrize("nps", [-1., float("nan"), True])
def test_invalid_density_is_rejected(nps):
    with pytest.raises(ValueError, match="NPS"):
        generate(audio_song(), nps, {}, None, preparation_settings(), vocabulary(), lighting_generator=lighting, postprocessors=processors())
