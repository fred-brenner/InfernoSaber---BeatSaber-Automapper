"""Production composition contracts with injected predictors, never training."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from map_creation.v2_pipeline import _atomic_expand, generate, infer_beats
from tools.config import config
from training.beat_models.contracts import CORE, FRAME_SECONDS, Song, frame_conditions, timing_window
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

    def validate(state, song, logger=None, **_ignored):
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


def test_timing_event_bias_adds_borderline_events(monkeypatch):
    # Borderline frames (0.38, 0.32, 0.30) stay silent under pure argmax; a
    # 0.10 bias on the event classes flips class 1 into an event without
    # mutating the probabilities kept for diagnostics.
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)

    class Borderline:
        status = "dummy"

        def __call__(self, inputs):
            batch = len(inputs["audio"])
            values = np.zeros((batch, CORE, 3), np.float32)
            values[...] = [.38, .32, .30]
            return {"counts": values}

    predictors["timing"] = Borderline()
    monkeypatch.setattr(config, "v2_timing_event_bias", 0.)
    silent = infer_beats(song, 4., predictors)
    monkeypatch.setattr(config, "v2_timing_event_bias", .1)
    biased = infer_beats(song, 4., predictors)
    assert not silent["counts"].any()
    assert len(silent["event_frames"]) < len(biased["event_frames"]) == 1100
    np.testing.assert_array_equal(silent["frame_probabilities"], biased["frame_probabilities"])
    assert (biased["frame_probabilities"].argmax(axis=-1) == 0).all()


def test_timing_event_bias_preserves_zero_density(monkeypatch):
    # conditioning_nps == 0 leaves probabilities at (1, 0, 0): any bias < 1
    # keeps class 0 ahead, so zero requested NPS still produces no events.
    monkeypatch.setattr(config, "v2_timing_event_bias", .9)
    predictors = dummy_predictors(vocabulary())
    predictors["timing"] = lambda inputs: pytest.fail("zero density must skip timing")
    beats = infer_beats(audio_song(), 0., predictors)
    assert beats["conditioning_nps"] == 0
    assert len(beats["event_frames"]) == 0


@pytest.mark.parametrize("bias", [-.01, 1., 1.5, float("nan"), True, "0.1"])
def test_timing_event_bias_rejects_out_of_range(monkeypatch, bias):
    monkeypatch.setattr(config, "v2_timing_event_bias", bias)
    with pytest.raises(ValueError, match="v2_timing_event_bias"):
        infer_beats(audio_song(), 4., dummy_predictors(vocabulary()))


def test_production_pipeline_uses_only_audio_and_keeps_all_output_families(monkeypatch):
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    monkeypatch.setattr(config, "v2_walls_heuristic", False)  # model decode: exercises the hazards stage
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
    settings["hazard_feasibility_input"] = False
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


def test_heuristic_walls_fill_only_note_free_lane_spans():
    from map_creation.v2_pipeline import _heuristic_wall_candidates

    events = np.asarray([0., 1., 2., 6., 7.])
    notes = [{"time": float(time), "x": lane, "y": 1, "hand": 0, "d": 0}
             for time in events for lane in (0, 1)]
    walls = _heuristic_wall_candidates(notes, events)
    assert len(walls) == 1
    wall = walls[0]
    assert wall["kind"] == "wall" and wall["x"] == 0 and wall["width"] == 1
    assert 0 <= wall["y"] <= 2 and 1 <= wall["height"] <= 5
    assert 2. < wall["time"] < wall["end_time"] < 6.
    # Off-grid notes (snapped times, multiplier copies) anchor their own
    # groups: the free span must split around the shifted note instead of
    # spanning across it.
    shifted = notes + [dict(notes[0], time=3.3)]
    walls = _heuristic_wall_candidates(shifted, events)
    assert len(walls) == 2
    assert all(wall["end_time"] <= 3.3 or wall["time"] >= 3.3 for wall in walls)
    assert not any(wall["time"] < 3.3 < wall["end_time"] for wall in walls)
    # Dense streams and empty inputs leave no span for obstacles.
    dense = [{"time": float(time), "x": lane, "y": 1, "hand": 0, "d": 0}
             for time in np.arange(0., 7.) for lane in (0, 1, 2, 3)]
    assert _heuristic_wall_candidates(dense, np.arange(0., 7.)) == []
    assert _heuristic_wall_candidates([], events) == []


def test_validate_hazards_rejects_walls_overlapping_note_swings():
    from map_creation.v2_postprocessing import validate_hazards

    song = audio_song()
    wall = {"kind": "wall", "x": 0, "y": 1, "width": 1, "height": 1,
            "time": 1.8, "end_time": 2.2}
    note = {"time": 2.0, "x": 1, "y": 1, "hand": 0, "d": 3, "id": 0}
    # A horizontal cut sweeps the whole row: the note centre sits outside the
    # wall but its swing crosses it — rejected like the bomb rule.
    state = {"notes": [dict(note)], "chains": [], "walls": [dict(wall)], "bombs": []}
    out = validate_hazards(state, song)
    assert state["walls"] == [] and out["rejected_walls"] == 1
    # A dot at the same cell sweeps nothing — the wall stays.
    state = {"notes": [dict(note, d=8)], "chains": [], "walls": [dict(wall)], "bombs": []}
    validate_hazards(state, song)
    assert len(state["walls"]) == 1
    # A vertical cut sweeps the other column — the wall stays.
    state = {"notes": [dict(note, d=0)], "chains": [], "walls": [dict(wall)], "bombs": []}
    validate_hazards(state, song)
    assert len(state["walls"]) == 1


def test_heuristic_walls_bypass_the_hazards_predictor(monkeypatch):
    monkeypatch.setattr(config, "v2_walls_heuristic", True)
    monkeypatch.setattr(config, "v2_bombs_prob", 0.)  # no bomb proposals
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    del predictors["hazards"]  # any lookup would raise — proves the model is unused
    settings = preparation_settings()
    result = generate(song, 4., predictors, lambda times: np.zeros((len(times), 16), np.float32),
                      settings, vocab, lighting_generator=lighting, postprocessors=processors(),
                      random_factor=0, logger=lambda message: None)
    assert result.stages["hazards"]["mode"] == "heuristic"
    assert result.state["bombs"] == []
    assert result.diagnostics["raw_decisions"]["wall_proposals"] == len(result.state["walls"])
    assert all(wall["kind"] == "wall" and 0 <= wall["x"] < 4 and 0 <= wall["y"] <= 4
               and wall["width"] >= 1 and wall["height"] >= 1
               and 0 <= wall["time"] < wall["end_time"] for wall in result.state["walls"])
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


def test_heuristic_arc_candidates_pair_adjacent_same_hand_notes():
    from map_creation.v2_pipeline import _heuristic_arc_candidates

    notes = [{"id": 0, "time": 1., "hand": 0, "x": 0, "y": 0, "d": 8},
             {"id": 1, "time": 2., "hand": 0, "x": 1, "y": 1, "d": 0},
             {"id": 2, "time": 3., "hand": 0, "x": 2, "y": 0, "d": 1},
             {"id": 3, "time": 1.5, "hand": 1, "x": 3, "y": 2, "d": 4},
             {"id": 4, "time": 3., "hand": 1, "x": 0, "y": 2, "d": 5}]
    candidates = _heuristic_arc_candidates(notes, 1.25)
    assert [(c["endpoint_ids"], round(c["gap"], 10)) for c in candidates] == [
        ([0], .7), ([3], 1.05), ([0, 1], 1.), ([3, 4], 1.5), ([1, 2], 1.)]
    turbo = candidates[0]
    assert turbo["synthetic_head"] == {"x": 0, "y": 0, "d": 8}
    assert turbo["time"] == pytest.approx(.3) and turbo["end_time"] == 1.
    assert turbo["anchor_mode"] == 1 and candidates[1]["anchor_mode"] == 1
    assert all(c["kind"] == "arc" and c["mu"] == c["tmu"] == 1.25
               and (c["anchor_mode"] == (1 if "synthetic_head" in c else 0))
               for c in candidates)
    same_time = [{"id": 0, "time": 1., "hand": 0, "x": 0, "y": 0, "d": 8},
                 {"id": 1, "time": 1., "hand": 0, "x": 1, "y": 0, "d": 8}]
    assert _heuristic_arc_candidates(same_time, 1.)[0]["endpoint_ids"] == [0]
    assert all("synthetic_head" in c for c in _heuristic_arc_candidates(same_time, 1.))


def test_heuristic_arcs_skip_the_model_and_validate_with_legacy_gap(monkeypatch):
    monkeypatch.setattr(config, "v2_arcs_heuristic", True)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    predictors["lighting"] = lambda inputs: pytest.fail("Production must use v1 lighting")
    predictors["arcs"] = lambda inputs: pytest.fail("Heuristic arcs never query the model")
    hooks = processors()
    seen = {}

    def validate(state, song, logger=None, settings=None, probability=None):
        seen["settings"] = settings
        return {}

    hooks.validate_arcs = validate
    settings = preparation_settings()
    settings["hazard_feasibility_input"] = False
    result = generate(song, 4., predictors, lambda times: np.zeros((len(times), 16), np.float32),
                      settings, vocab, lighting_generator=lighting, postprocessors=hooks,
                      random_factor=0, logger=lambda message: None)
    assert result.stages["arcs"]["mode"] == "heuristic"
    assert seen["settings"] == {"stage_policies": {}}
    assert result.diagnostics["raw_decisions"]["arc_attachments"] == len(result.state["arcs"])
    assert all(arc["mu"] == arc["tmu"] == config.slider_radius_multiplier
               and len(arc["endpoint_ids"]) == (1 if "synthetic_head" in arc else 2)
               for arc in result.state["arcs"])


@pytest.mark.parametrize("repaired_contract", [False, True])
def test_intensity_adjustment_conditions_timing_and_all_downstream_stages(monkeypatch, repaired_contract):
    from map_creation import v2_pipeline
    from training.pipeline_v2 import contracts

    monkeypatch.setattr(config, "v2_intensity_scale", 1.5)
    monkeypatch.setattr(config, "v2_intensity_adjustment_strength", .8)
    monkeypatch.setattr(config, "v2_arcs_heuristic", False)  # model decode: exercises arcs build_inputs
    monkeypatch.setattr(config, "v2_walls_heuristic", False)  # model decode: exercises hazards build_inputs
    base, vocab = audio_song(), vocabulary()
    predictors, timing = dummy_predictors(vocab), Timing()
    shares = np.arange(1, len(base.arrays["bin_durations"]) + 1, dtype=np.float32)
    shares /= shares.sum()
    predictors["intensity"] = lambda inputs: {"shares": shares[None]}
    windows, stages = [], set()

    def predict_timing(inputs):
        windows.append(inputs["conditions"].copy())
        return timing(inputs)

    original = v2_pipeline.build_inputs

    def build(stage, song, state, candidate, *args, **kwargs):
        inputs = original(stage, song, state, candidate, *args, **kwargs)
        frame = min(int(np.floor(candidate["time"] / FRAME_SECONDS + .5)), len(state["conditions"]) - 1)
        np.testing.assert_allclose(inputs["conditions"][0, 2:], state["conditions"][frame], rtol=1e-6)
        assert state["requested_nps"] == 4.
        assert state["conditioning_nps"] == pytest.approx(5.6)
        stages.add(stage)
        return inputs

    predictors["timing"] = predict_timing
    monkeypatch.setattr(v2_pipeline, "build_inputs", build)
    monkeypatch.setattr(contracts, "build_inputs", build)
    settings = preparation_settings()
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    if repaired_contract:
        settings["stage_policies"]["arcs"]["max_gap_seconds"] = 6.
    else:
        settings.pop("stage_policies", None)
        settings["hazard_feasibility_input"] = False
    result = generate(base, 4., predictors, lambda times: np.zeros((len(times), 16), np.float32),
                      settings, vocab, lighting_generator=lighting, postprocessors=processors(),
                      random_factor=0, logger=lambda message: None)
    assert stages == {"positions", "arrows", "multiplier", "hazards", "arcs"}
    np.testing.assert_array_equal(result.state["raw_intensity"], shares)
    expected = frame_conditions(base, result.state["intensity"], 5.6)
    np.testing.assert_allclose(result.state["conditions"], expected, rtol=1e-6)
    np.testing.assert_allclose(windows[0], np.concatenate([timing_window(base, expected, start)[0]["conditions"]
                                                        for start in range(0, 1100, CORE)]), rtol=1e-6)
    assert result.diagnostics["intensity_postprocessing"]["scale"] == 1.5
    assert result.diagnostics["requested_retained_nps"] == 4.


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
    monkeypatch.setattr(config, "v2_walls_heuristic", False)  # model decode: raw proposals come from predictors
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    settings = preparation_settings()
    settings.pop("stage_policies", None)  # Exercise the retained legacy decoder.
    settings["hazard_feasibility_input"] = False
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    hooks = processors()

    def reject_hazards(state, song, logger=None, **_ignored):
        counts = {"rejected_walls": len(state["walls"]), "rejected_bombs": len(state["bombs"])}
        state["walls"], state["bombs"] = [], []
        return counts

    def reject_arcs(state, song, logger=None, **_ignored):
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


def test_sampled_position_decode_preserves_count_and_cells():
    from map_creation.v2_pipeline import _decode_positions_sampled
    outputs = {"red": np.array([[.6, .05, .05, .05, .05, .05, .05, .05, .05, .0, .0, .0, .0]], np.float32),
               "blue": np.array([[.4, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .5, .1]], np.float32)}
    rng = np.random.default_rng(5)
    singles = [_decode_positions_sampled(outputs, 1, .9, rng) for _ in range(400)]
    assert all((red == 0) != (blue == 0) for red, blue in singles)
    assert all(1 <= max(red, blue) <= 12 for red, blue in singles)
    assert len(set(singles)) > 2  # non-degenerate sampling explores the tail
    doubles = [_decode_positions_sampled(outputs, 2, .9, rng) for _ in range(200)]
    assert all(1 <= red <= 12 and 1 <= blue <= 12 for red, blue in doubles)
    argmaxed = [_decode_positions_sampled(outputs, 1, 0, rng) for _ in range(10)]
    assert len(set(argmaxed)) == 1  # zero temperature keeps the argmax contract


def test_note_sampling_flows_through_generate():
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = Timing()
    predictors["lighting"] = lambda inputs: pytest.fail("Production must use v1 lighting")
    settings = preparation_settings()
    settings.pop("stage_policies", None)
    settings["hazard_feasibility_input"] = False
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    settings["note_sample_temperature"] = 0.9
    counts = {}
    for seed in range(6):
        result = generate(song, 4., predictors, lambda times: np.zeros((len(times), 16), np.float32),
                          settings, vocab, lighting_generator=lighting, postprocessors=processors(),
                          random_factor=0, rng=np.random.default_rng(seed), logger=lambda message: None)
        assert result.diagnostics["note_sample_temperature"] == 0.9
        retained = [note for note in result.state["notes"] if note.get("kind") == "retained"]
        assert len(retained) == result.diagnostics["raw_decisions"]["position_notes"]
        for note in retained:
            assert 0 <= note["x"] < 4 and 0 <= note["y"] < 3 and 0 <= note["d"] <= 8
        counts[tuple((note["x"], note["y"]) for note in result.state["notes"])] = 1
    assert result.status == "dummy"


def _note(note_id, time, hand, direction, kind="heuristic"):
    return {"id": note_id, "time": time, "hand": hand, "x": 1, "y": 1,
            "d": direction, "kind": kind}


def test_same_direction_recut_flips_repeat():
    from map_creation.v2_postprocessing import recut_same_direction_notes, _step_distance
    notes = [_note(0, 0., 0, 1), _note(1, .2, 0, 1), _note(2, .4, 0, 1),
             _note(3, .2, 1, 0), _note(4, .4, 1, 0)]
    flipped = recut_same_direction_notes(notes)
    assert flipped == 2
    assert _step_distance(notes[0]["d"], notes[1]["d"]) >= 3
    assert _step_distance(notes[1]["d"], notes[2]["d"]) >= 3
    assert _step_distance(notes[3]["d"], notes[4]["d"]) >= 3


def test_same_direction_recut_handles_near_repeats_and_exemptions(monkeypatch):
    from map_creation.v2_postprocessing import recut_same_direction_notes, _step_distance
    monkeypatch.setattr(config, "v2_playability_enabled", False)
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 3)
    # 45° neighbour repeats violate the ≥135° rule for far-apart notes
    notes = [_note(0, 0., 0, 1), _note(1, .5, 0, 6)]
    assert recut_same_direction_notes(notes) == 1
    assert _step_distance(notes[0]["d"], notes[1]["d"]) >= 3
    # 90° changes are still too hectic for separated notes under the strict
    # rule, at one cell or with two-cell travel
    notes = [_note(0, 0., 0, 1), _note(1, .5, 0, 2)]
    assert recut_same_direction_notes(notes) == 1
    assert _step_distance(notes[0]["d"], notes[1]["d"]) >= 3
    notes = [_note(0, 0., 0, 1), _note(1, .5, 0, 2)]
    notes[1]["x"], notes[1]["y"] = 3, 1
    assert recut_same_direction_notes(notes) == 1
    assert _step_distance(notes[0]["d"], notes[1]["d"]) >= 3
    # v2_flow_min_turn_steps <= 2 restores the legacy rule: 90° is allowed
    # when the hand travels at least two cells
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 2)
    notes = [_note(0, 0., 0, 1), _note(1, .5, 0, 2)]
    notes[1]["x"], notes[1]["y"] = 3, 1
    assert recut_same_direction_notes(notes) == 0
    # 135° changes are fine as-is
    notes = [_note(0, 0., 0, 1), _note(1, .5, 0, 4)]
    assert recut_same_direction_notes(notes) == 0
    # near-simultaneous notes share one stroke and keep the direction
    notes = [_note(0, 0., 0, 1), _note(1, .05, 0, 1)]
    assert recut_same_direction_notes(notes) == 0
    # dots may repeat freely
    notes = [_note(0, 0., 0, 8), _note(1, .5, 0, 8)]
    assert recut_same_direction_notes(notes) == 0
    # a same-cell dot is hit by the return swing: down → dot → down is legal
    notes = [_note(0, 0., 0, 1), _note(1, .2, 0, 8), _note(2, .4, 0, 1)]
    assert recut_same_direction_notes(notes) == 0
    assert notes[1]["d"] == 8
    # a dot in the continuation zone continues the stroke: down → dot → down repeats
    notes = [_note(0, 0., 0, 1), _note(1, .2, 0, 8), _note(2, .4, 0, 1)]
    notes[0]["y"], notes[1]["y"] = 2, 0
    assert recut_same_direction_notes(notes) == 1
    assert notes[1]["d"] == 8 and _step_distance(notes[0]["d"], notes[2]["d"]) >= 3


def test_same_direction_recut_protects_retained():
    from map_creation.v2_postprocessing import recut_same_direction_notes, _step_distance
    # retained note preceded by same-direction copy: recut the earlier note
    notes = [_note(0, 0., 0, 4, kind="copy"), _note(1, .2, 0, 4, kind="retained")]
    assert recut_same_direction_notes(notes) == 1
    assert notes[1]["d"] == 4 and _step_distance(notes[0]["d"], 4) >= 3
    # retained/retained pairs are contract-fixed and left alone
    notes = [_note(0, 0., 0, 1, kind="retained"), _note(1, .2, 0, 1, kind="retained")]
    assert recut_same_direction_notes(notes) == 0
    assert notes[1]["d"] == 1
    # unprotected mode recuts the later note unconditionally
    assert recut_same_direction_notes(notes, protect_retained=False) == 1
    assert _step_distance(notes[0]["d"], notes[1]["d"]) >= 3


def test_same_direction_recut_syncs_chain_heads():
    from map_creation.v2_postprocessing import recut_same_direction_notes
    notes = [_note(0, 0., 0, 2, kind="copy"), _note(1, .2, 0, 2, kind="copy")]
    chain = {"anchor_id": 1, "d": 2}
    assert recut_same_direction_notes(notes, chains=[chain]) == 1
    assert notes[1]["d"] == chain["d"] and notes[1]["d"] != 2


def test_simultaneous_same_cell_is_resolved():
    from map_creation.v2_postprocessing import clearance_violations, enforce_note_flow_and_clearance
    # different-hand notes on one cell at one timestep are physically impossible
    notes = [_note(0, .5, 0, 1), _note(1, .5, 1, 0)]
    stats = enforce_note_flow_and_clearance(notes)
    assert clearance_violations(notes)["same_cell_pairs"] == 0
    assert len(notes) == 2 and stats["cell_relocated"] == 1
    # the same-hand duplicate is dropped outright
    notes = [_note(0, .5, 0, 1), _note(1, .5, 0, 0)]
    stats = enforce_note_flow_and_clearance(notes)
    assert len(notes) == 1 and stats["cell_dropped"] == 1


def test_swing_into_other_hand_is_resolved():
    from map_creation.v2_postprocessing import clearance_violations, enforce_note_flow_and_clearance
    # left cuts into the simultaneous right note's approach cell: recut or relocate
    notes = [_note(0, .5, 0, 2), _note(1, .5, 1, 0)]
    notes[1]["x"], notes[1]["y"] = 2, 1
    stats = enforce_note_flow_and_clearance(notes)
    assert clearance_violations(notes)["swing_into_pairs"] == 0
    assert stats["swing_recut"] + stats["swing_relocated"] >= 1
    # the exported-map violation: right note (2,0) arrow-left exits through the
    # simultaneous left note at (1,0)
    notes = [_note(0, .5, 1, 2), _note(1, .5, 0, 0)]
    notes[0]["x"], notes[0]["y"], notes[1]["x"], notes[1]["y"] = 2, 0, 1, 0
    stats = enforce_note_flow_and_clearance(notes)
    assert clearance_violations(notes) == {"same_cell_pairs": 0, "swing_into_pairs": 0, "flow_repeats": 0}
    assert stats["swing_recut"] + stats["swing_relocated"] + stats["swing_dropped"] >= 1
    # a retained swinger cannot be recut, so the target moves instead
    notes = [_note(0, .5, 0, 2, kind="retained"), _note(1, .5, 1, 0)]
    notes[1]["x"], notes[1]["y"] = 2, 1
    stats = enforce_note_flow_and_clearance(notes)
    assert notes[0]["d"] == 2 and stats["swing_relocated"] == 1
    assert (notes[1]["x"], notes[1]["y"]) != (2, 1)
    assert clearance_violations(notes)["swing_into_pairs"] == 0


def test_swing_into_later_note_is_resolved():
    from map_creation.v2_postprocessing import clearance_violations, enforce_note_flow_and_clearance
    # a later other-hand note still stands in the swept cell when the blade
    # passes through: right (2,0) arrow-left exits where the left note lands .08s later
    notes = [_note(0, .5, 1, 2), _note(1, .58, 0, 0)]
    notes[0]["x"], notes[0]["y"], notes[1]["x"], notes[1]["y"] = 2, 0, 1, 0
    stats = enforce_note_flow_and_clearance(notes)
    assert clearance_violations(notes) == {"same_cell_pairs": 0, "swing_into_pairs": 0, "flow_repeats": 0}
    assert stats["swing_recut"] + stats["swing_relocated"] + stats["swing_dropped"] >= 1
    # a later note standing on the swinger's own cell is a real clash too and
    # can only be relocated/dropped, never recut away
    notes = [_note(0, .5, 1, 1), _note(1, .58, 0, 0)]
    notes[0]["x"], notes[0]["y"], notes[1]["x"], notes[1]["y"] = 2, 0, 2, 0
    stats = enforce_note_flow_and_clearance(notes)
    assert clearance_violations(notes)["swing_into_pairs"] == 0
    assert stats["swing_recut"] == 0
    # notes struck before the swing are already gone: .08s before the exit cell
    # is not a clash
    notes = [_note(0, .42, 0, 0), _note(1, .5, 1, 2)]
    notes[0]["x"], notes[0]["y"], notes[1]["x"], notes[1]["y"] = 1, 0, 2, 0
    stats = enforce_note_flow_and_clearance(notes)
    assert clearance_violations(notes)["swing_into_pairs"] == 0
    assert stats["swing_recut"] + stats["swing_relocated"] + stats["swing_dropped"] == 0


def test_swing_recut_respects_same_hand_flow(monkeypatch):
    from map_creation.v2_postprocessing import (clearance_violations, _step_distance,
                                                enforce_note_flow_and_clearance)
    # legacy rule keeps this pair flow-legal; the strict default would let the
    # flow pass fix it before the swing clash is even checked
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 2)
    # swinger is flow-legal against its predecessor (90° with 2-cell travel),
    # so the clash fix - not the flow pass - must supply the new direction;
    # any horizontal recut still sweeps the blocked cell, so it must leave the
    # original axis entirely
    notes = [_note(0, 0., 0, 0), _note(1, .5, 0, 2), _note(2, .5, 1, 0)]
    notes[0]["x"], notes[0]["y"] = 0, 0
    notes[2]["x"], notes[2]["y"] = 2, 1
    stats = enforce_note_flow_and_clearance(notes)
    assert stats["flow_recuts"] == 0 and stats["swing_recut"] == 1
    assert notes[1]["d"] == 1  # vertical reversal is the furthest legal family
    assert clearance_violations(notes) == {"same_cell_pairs": 0, "swing_into_pairs": 0, "flow_repeats": 0}
    assert _step_distance(notes[0]["d"], notes[1]["d"]) >= 2


def test_v2_mirroring_also_mirrors_cut_direction(monkeypatch):
    from map_creation.v2_postprocessing import _v2_mirroring
    monkeypatch.setattr(config, "gimme_more_notes_prob", 1.)
    monkeypatch.setattr(config, "max_speed", 4.)
    # single-hand section: mirrored copy must flip lane, hand and horizontal cut
    assert _v2_mirroring([[1, 1, 0, 2]]) == [[1, 1, 0, 2, 2, 1, 1, 3]]
    # vertical cuts and dots survive the mirror unchanged
    assert _v2_mirroring([[0, 0, 0, 1]]) == [[0, 0, 0, 1, 3, 0, 1, 1]]
    assert _v2_mirroring([[2, 2, 0, 8]]) == [[2, 2, 0, 8, 1, 2, 1, 8]]
    # mixed-hand sections are not doubled
    assert _v2_mirroring([[1, 1, 0, 2, 2, 1, 1, 3]]) == [[1, 1, 0, 2, 2, 1, 1, 3]]


def test_dot_bridges_flow_with_implied_direction():
    from map_creation.v2_postprocessing import clearance_violations, enforce_note_flow_and_clearance
    # reported violation: up-right, dot in the return zone, down-left again -
    # the dot already carries the down-left pendulum motion, so the third note
    # repeats it and must be recut back to the up-right family
    notes = [_note(0, 0., 1, 5), _note(1, .25, 1, 8), _note(2, .5, 1, 6)]
    notes[0]["x"], notes[0]["y"] = 2, 0
    notes[1]["x"], notes[1]["y"] = 1, 0
    notes[2]["x"], notes[2]["y"] = 3, 0
    enforce_note_flow_and_clearance(notes)
    assert notes[1]["d"] == 8
    assert notes[2]["d"] == 5
    assert clearance_violations(notes)["flow_repeats"] == 0
    # consecutive dots alternate the implied direction through the run
    notes = [_note(0, 0., 1, 5), _note(1, .25, 1, 8), _note(2, .5, 1, 8), _note(3, .75, 1, 0)]
    notes[0]["x"], notes[0]["y"] = 2, 0
    notes[1]["x"], notes[1]["y"] = 1, 0
    notes[2]["x"], notes[2]["y"] = 0, 0
    notes[3]["x"], notes[3]["y"] = 3, 0
    enforce_note_flow_and_clearance(notes)
    assert notes[3]["d"] == 1
    assert clearance_violations(notes)["flow_repeats"] == 0
