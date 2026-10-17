"""Synthetic composition contracts, with no neural models or corpus comparisons."""

from copy import deepcopy

import numpy as np
import pytest

from training.pipeline_v2.pipeline import ReviewRequired, dummy_predictors, generate
from training.pipeline_v2.scheduling import arc_candidates, hazard_candidates, lighting_candidates


@pytest.fixture(scope="module")
def dummy_dataset():
    from training.pipeline_v2.data import make_dummy_dataset
    return make_dummy_dataset()


def _settings():
    return {"schedule": {"arc_max_gap_seconds": 1.2}}


def test_default_stop_returns_raw_timing_without_changing_source(dummy_dataset):
    song = dummy_dataset.songs[0]
    original = deepcopy(song.objects)
    with pytest.raises(ReviewRequired) as caught:
        generate(song, 2., vocabularies=dummy_dataset.vocabularies)
    assert caught.value.boundary == "bpm_alignment"
    partial = caught.value.partial_result
    assert set(partial.stages) == {"intensity", "timing"}
    assert partial.state["notes"] == []
    assert partial.diagnostics["export_ready"] is False
    assert song.objects == original


def test_full_dummy_contract_exercises_all_stages_and_preserves_anchors(dummy_dataset):
    song = dummy_dataset.songs[0]
    original = deepcopy(song.objects)
    result = generate(song, 2., settings=_settings(), contract_mode=True,
                      vocabularies=dummy_dataset.vocabularies)
    assert result.status == "dummy"
    assert set(result.stages) == {"intensity", "timing", "positions", "arrows", "multiplier", "hazards", "arcs", "lighting"}
    for family in ("notes", "walls", "bombs", "arcs", "lighting"):
        assert result.state[family], family
    assert result.stages["multiplier"]["added_notes"] > 0
    assert result.stages["positions"]["retained_notes"] == result.stages["timing"]["retained_notes"]
    retained = {note["id"]: note for note in result.state["notes"] if note["kind"] == "retained"}
    for note in result.state["notes"]:
        assert 0 <= note["time"] < song.base.metadata["duration"]
        assert 0 <= note["x"] < 4 and 0 <= note["y"] < 3
        if note["kind"] == "copy":
            anchor = retained[note["anchor_id"]]
            assert (note["hand"], note["d"]) == (anchor["hand"], anchor["d"])
    for arc in result.state["arcs"]:
        assert all(endpoint in {note["id"] for note in result.state["notes"]} for endpoint in arc["endpoint_ids"])
    assert all(stop["bypassed"] for stop in result.review_stops)
    assert result.diagnostics["safety_reviewed"] is False
    assert result.diagnostics["export_ready"] is False
    assert song.objects == original


def test_inference_does_not_read_reference_labels_or_objects(dummy_dataset):
    song = dummy_dataset.songs[0]
    poisoned = deepcopy(song)
    poisoned.objects = {"notes": [{"unexpected": "must never be read"}]}
    poisoned.examples = {"invalid": object()}
    for key in ("intensity", "timing", "note_classes", "original_notes", "retained_indices", "event_times"):
        if key in poisoned.base.arrays:
            poisoned.base.arrays[key] = np.zeros_like(poisoned.base.arrays[key])
    first = generate(song, 2., settings=_settings(), contract_mode=True, vocabularies=dummy_dataset.vocabularies)
    second = generate(poisoned, 2., settings=_settings(), contract_mode=True, vocabularies=dummy_dataset.vocabularies)
    for family in ("notes", "chains", "walls", "bombs", "arcs", "lighting"):
        assert first.state[family] == second.state[family]


def test_multiplier_inputs_keep_retained_context_after_earlier_expansions(dummy_dataset, monkeypatch):
    from training.pipeline_v2 import contracts, pipeline
    original_build, original_expand = contracts.build_inputs, pipeline._expand
    seen = {"copy_added": False, "queries_after_copy": 0, "retained_ids": None}

    def expand(state, anchor, template, duration, rejected):
        original_expand(state, anchor, template, duration, rejected)
        seen["copy_added"] |= any(note["kind"] == "copy" for note in state["notes"])

    def build(stage, song, state, candidate, own_history, encoder=None, settings=None):
        if stage == "multiplier":
            assert all(note["kind"] == "retained" for note in state["notes"])
            assert state["chains"] == []
            identities = [note["id"] for note in state["notes"]]
            if seen["retained_ids"] is None:
                seen["retained_ids"] = identities
            assert identities == seen["retained_ids"]
            if seen["copy_added"]:
                seen["queries_after_copy"] += 1
        return original_build(stage, song, state, candidate, own_history, encoder=encoder, settings=settings)

    monkeypatch.setattr(pipeline, "_expand", expand)
    monkeypatch.setattr(contracts, "build_inputs", build)
    result = generate(dummy_dataset.songs[0], 2., settings=_settings(), contract_mode=True,
                      vocabularies=dummy_dataset.vocabularies)
    assert result.stages["multiplier"]["added_notes"] > 0
    assert seen["queries_after_copy"] > 0


def test_encoder_is_queried_independently_and_history_precedes_current_query(dummy_dataset, monkeypatch):
    from training.pipeline_v2 import contracts
    original = contracts.build_inputs
    observed, queries = [], []

    def encoder(times):
        queries.append(np.asarray(times).copy())
        return np.zeros((len(times), 16), dtype=np.float32)

    def build(stage, song, state, candidate, own_history, encoder=None, settings=None):
        assert all(record["time"] < candidate["time"] for record in own_history)
        observed.append(stage)
        return original(stage, song, state, candidate, own_history, encoder=encoder, settings=settings)

    monkeypatch.setattr(contracts, "build_inputs", build)
    result = generate(dummy_dataset.songs[0], 2., settings=_settings(), contract_mode=True,
                      encoder=encoder, vocabularies=dummy_dataset.vocabularies)
    assert set(observed) == {"positions", "arrows", "multiplier", "hazards", "arcs", "lighting"}
    assert len(queries) >= len(observed)
    assert len(result.state["lighting"]) >= result.stages["lighting"]["queries"]


def test_neural_predictors_cannot_claim_dummy_contract_mode(dummy_dataset):
    predictors = dummy_predictors(dummy_dataset.vocabularies)
    predictors["positions"].status = "untrained"
    with pytest.raises(ValueError, match="dummy"):
        generate(dummy_dataset.songs[0], 2., predictors=predictors, contract_mode=True,
                 vocabularies=dummy_dataset.vocabularies)


def test_zero_density_keeps_independent_lighting_schedule(dummy_dataset):
    result = generate(dummy_dataset.songs[0], 0., settings=_settings(), contract_mode=True,
                      vocabularies=dummy_dataset.vocabularies)
    assert result.state["notes"] == []
    assert result.state["lighting"]


def test_lighting_uses_configured_block_length_and_retains_partial_tail(dummy_dataset, monkeypatch):
    from training.pipeline_v2 import contracts
    original = contracts.build_inputs
    masks = []

    def build(stage, song, state, candidate, own_history, encoder=None, settings=None):
        inputs = original(stage, song, state, candidate, own_history, encoder=encoder, settings=settings)
        if stage == "lighting":
            assert inputs["audio"].shape[1] == 8
            assert inputs["history"].shape[1] == 8
            masks.append(inputs["mask"].copy())
        return inputs

    monkeypatch.setattr(contracts, "build_inputs", build)
    result = generate(dummy_dataset.songs[0], 0., settings={"lighting_block_length": 8},
                      contract_mode=True, vocabularies=dummy_dataset.vocabularies)
    queries = result.stages["lighting"]["queries"]
    assert result.stages["lighting"]["blocks"] == (queries + 7) // 8
    assert sum(np.count_nonzero(mask) for mask in masks) == queries
    assert result.stages["lighting"]["tail_queries"] == queries % 8
    assert len(result.state["lighting"]) == queries * len(dummy_dataset.vocabularies["lighting"])


def test_schedulers_are_deterministic_and_keep_existing_arc_endpoints():
    notes = [{"id": 4, "time": .5, "hand": 0, "x": 0, "y": 2},
             {"id": 7, "time": 1., "hand": 0, "x": 1, "y": 1},
             {"id": 8, "time": 2., "hand": 1, "x": 3, "y": 2}]
    before = deepcopy(notes)
    hazards = hazard_candidates(notes, 3., 120.)
    assert hazards == hazard_candidates(notes, 3., 120.)
    assert {query["kind"] for query in hazards} == {"wall", "bomb"}
    arcs = arc_candidates(notes)
    assert arcs == [{"kind": "arc", "time": .5, "end_time": 1., "hand": 0,
                     "endpoint_ids": [4, 7], "gap": .5}]
    assert notes == before
    with pytest.raises(ValueError, match="max_candidates"):
        hazard_candidates([], 10., 120., {"max_candidates": 2})


def test_lighting_scheduler_is_independent_of_notes_and_retains_tail(dummy_dataset):
    song = dummy_dataset.songs[0]
    first = lighting_candidates(song)
    alternate = deepcopy(song)
    alternate.objects = {}
    assert first == lighting_candidates(alternate)
    assert first[0]["time"] == 0
    assert first[-1]["time"] < song.base.metadata["duration"]
    result = generate(song, 0., contract_mode=True, vocabularies=dummy_dataset.vocabularies)
    assert result.stages["lighting"]["queries"] == len(first)


def test_strobes_are_note_aware_and_zero_grid_disables_them():
    notes = [{"id": 0, "time": 1., "hand": 0, "x": 0, "y": 1}]
    active = hazard_candidates(notes, 3., 120., {"wall_strobe_grid_beats": 1.})
    strobes = [query for query in active if query.get("strobe")]
    assert strobes
    assert all(not query["time"] - .3 <= 1. <= query["end_time"] + .3 for query in strobes)
    disabled = hazard_candidates(notes, 3., 120., {"wall_strobe_grid_beats": 0.})
    assert not any(query.get("strobe") for query in disabled)
    assert any(query["kind"] == "wall" for query in disabled)


def test_arc_decoder_uses_configured_shape_bounds(dummy_dataset):
    class ArcShape:
        status = "dummy"
        def predict(self, inputs, verbose=0):
            return {"attach": np.asarray([[0., 1.]], np.float32),
                    "shape": np.asarray([[2.5, 2.5]], np.float32),
                    "anchor": np.asarray([[1., 0., 0.]], np.float32)}

    predictors = dummy_predictors(dummy_dataset.vocabularies)
    predictors["arcs"] = ArcShape()
    settings = {**_settings(), "arc_shape_bounds": [0., 3.]}
    result = generate(dummy_dataset.songs[0], 2., predictors=predictors, settings=settings,
                      contract_mode=True, vocabularies=dummy_dataset.vocabularies)
    assert result.state["arcs"]
    assert all(arc["mu"] == arc["tmu"] == 2.5 for arc in result.state["arcs"])
    settings["arc_shape_bounds"] = [0., 2.]
    with pytest.raises(ValueError, match="configured bounds"):
        generate(dummy_dataset.songs[0], 2., predictors=predictors, settings=settings,
                 contract_mode=True, vocabularies=dummy_dataset.vocabularies)
