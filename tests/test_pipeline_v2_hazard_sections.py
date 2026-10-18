"""Sparse hazard supervision and causal wall-section composition contracts."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from training.pipeline_v2.hazard_sections import (
    ClearanceIndex, SectionDecoder, build_examples, candidates, coalesce,
    policy, policy_defaults, rollout,
)
from training.pipeline_v2.contracts import V2Song


VOCAB = {"walls": [{"kind": "none"},
                    {"kind": "wall", "x": 0, "y": 0, "width": 1, "height": 3},
                    {"kind": "wall", "x": 3, "y": 0, "width": 1, "height": 3}]}


def options(**clearance):
    frozen = policy_defaults()
    frozen["clearance"].update(clearance)
    return {"stage_policies": {"hazards": frozen}, "history_length": 32}


def song(walls=(), bombs=(), ignored=(), duration=2., bpm=120.):
    return V2Song(
        base=SimpleNamespace(metadata={"duration": duration, "bpm": bpm, "difficulty_nps": 0.},
                             arrays={"frame_times": np.arange(0, duration, .01),
                                     "bin_durations": np.asarray([duration], np.float32)}),
        objects={"walls": list(walls), "bombs": list(bombs), "ignored": list(ignored)},
        diagnostics={"source_objects_complete": True})


def wall(identity, start, end, x=0, **extra):
    return {"id": identity, "time": start, "end_time": end,
            "x": x, "y": 0, "width": 1, "height": 3, **extra}


def test_schedule_uses_every_section_and_boundary_without_note_or_source_queries():
    queries = candidates(.31, 120., options())
    assert [(item["kind"], item["time"], item["end_time"]) for item in queries] == [
        ("wall", 0., .125), ("bomb", 0., 0.), ("wall", .125, .25), ("bomb", .125, .125),
        ("wall", .25, .31), ("bomb", .25, .25)]
    assert all(item["origin"] == "scheduler" for item in queries)
    limited = options()
    limited["stage_policies"]["hazards"]["max_candidates"] = 5
    with pytest.raises(ValueError, match="no queries were discarded"):
        candidates(.31, 120., limited)


def test_frozen_policy_does_not_read_later_live_clearance_changes(monkeypatch):
    from tools.config import config

    frozen = options()
    monkeypatch.setattr(config, "add_obstacle_flag", False)
    monkeypatch.setattr(config, "obstacle_min_duration", 99.)
    decoder = SectionDecoder({}, .3, VOCAB, frozen)
    for query in candidates(.3, 120., frozen):
        decoder.step(query, wall=1)
    assert len(decoder.finish()["walls"]) == 1
    legacy = {"stage_policies": {"hazards": {"grid_beats": .25}}}
    with pytest.raises(ValueError, match="version 2"):
        policy(legacy)


def test_minimum_duration_after_merge_and_none_always_breaks_run():
    opts = options()
    queries = candidates(.25, 240., opts)
    decoder = SectionDecoder({}, .25, VOCAB, opts)
    for query in queries:
        category = 1 if query["time"] in (0., .0625, .1875) else 0
        decoder.step(query, wall=category)
    final = decoder.finish()
    assert [(item["time"], item["end_time"]) for item in final["walls"]] == [(0., .125)]
    assert final["diagnostics"]["accepted_wall_sections"] == 3
    assert final["diagnostics"]["final_rejections"] == {"short_merged_wall": 1}
    # The final short run remains a preceding cleared decision in history.
    assert decoder.history[-2]["value"][0] == 1
    assert len(coalesce([wall("a", 0., .1), wall("b", .2, .3)])) == 2


def test_endpoint_rounding_uses_earlier_tie_preserves_source_and_unions_geometry():
    # Halfway endpoints choose .125 and .5, respectively.
    source = song([wall("a", .1875, .5625), wall("b", .375, .75)])
    before = deepcopy(source.objects)
    examples, diagnostic = build_examples(source, {}, VOCAB, options())
    positive = (examples.targets["wall"] > 0) & (examples.weights["wall"] > 0)
    np.testing.assert_array_equal(examples.times[positive], [.125, .25, .375, .5, .625])
    errors = diagnostic["hazard_quantization_errors"][0]
    assert (errors["start_error_seconds"], errors["end_error_seconds"]) == (-.0625, -.0625)
    assert diagnostic["hazard_supported_reference"]["final_walls"] == 1
    assert diagnostic["hazard_supported_reference"]["wall_source_objects"] == 2
    assert source.objects == before
    assert len(examples.times) == len(candidates(2., 120., options()))


def test_overlapping_different_wall_geometry_masks_only_ambiguous_sections():
    source = song([wall("a", .25, .75), wall("b", .5, 1., x=3)])
    examples, diagnostic = build_examples(source, {}, VOCAB, options())
    for time in (.5, .625):
        row = np.flatnonzero((examples.times == time) & (np.arange(len(examples.times)) % 2 == 0))[0]
        assert examples.weights["wall"][row] == 0
    assert diagnostic["hazard_exclusion_counts"]["ambiguous_geometry_overlap"] == 2
    assert diagnostic["hazard_supported_reference"]["fully_supported_wall_source_objects"] == 0
    assert diagnostic["hazard_supported_reference"]["final_walls"] == 2


def test_original_short_invalid_and_unknown_walls_mask_instead_of_becoming_none():
    source = song([wall("short", .25, .34), wall("invalid", .5, .75, width=0),
                   wall("unknown", 1., 1.25, height=2)])
    examples, diagnostic = build_examples(source, {}, VOCAB, options())
    for time in (.25, .5, .625, 1., 1.125):
        row = np.flatnonzero((examples.times == time) & (np.arange(len(examples.times)) % 2 == 0))[0]
        assert examples.weights["wall"][row] == 0
    assert diagnostic["hazard_exclusion_counts"] == {
        "short_original_wall": 1, "invalid_geometry": 1, "template_outside_training_vocabulary": 1}
    assert diagnostic["hazard_raw_source_counts"]["walls"] == 3
    assert diagnostic["hazard_supported_reference"]["wall_source_objects"] == 0


def test_bombs_assign_once_combine_cells_and_mask_swing_collision_per_cell():
    source = song(bombs=[{"id": "left", "time": .49, "x": 0, "y": 0},
                         {"id": "right", "time": .51, "x": 3, "y": 0},
                         {"id": "collision", "time": .5, "x": 1, "y": 1}])
    state = {"notes": [{"time": .5, "x": 1, "y": 2, "d": 0, "hand": 0}]}
    examples, diagnostic = build_examples(source, state, VOCAB, options())
    row = np.flatnonzero((examples.times == .5) & (np.arange(len(examples.times)) % 2 == 1))[0]
    assert examples.targets["bombs"].sum() == 2
    np.testing.assert_array_equal(np.flatnonzero(examples.targets["bombs"][row]), [0, 3])
    assert examples.weights["bombs"].shape == (len(examples.times), 12)
    assert examples.weights["bombs"][row, 5] == 0
    assert examples.weights["bombs"][row, 0] == 1
    assert diagnostic["hazard_raw_source_counts"]["bombs"] == 3
    assert diagnostic["hazard_supported_reference"]["bomb_source_objects"] == 2
    assert diagnostic["hazard_supported_reference"]["final_bombs"] == 2


def test_chain_sweep_clearance_matches_legacy_and_never_moves_upstream_notes():
    chain = {"id": 0, "anchor_id": 0, "time": .3, "end_time": .7,
             "x": 0, "y": 0, "tx": 3, "ty": 0, "d": 8, "hand": 0, "slices": 2}
    state = {"notes": [{"id": 0, "time": .3, "x": 0, "y": 0, "d": 8, "hand": 0}], "chains": [chain]}
    before = deepcopy(state)
    index = ClearanceIndex(state, 2., policy_defaults()["clearance"])
    # Dense sweep catches x=1, although a two-slice-only sample would miss it.
    assert not index.bomb_cells(.433)[1]
    assert index.wall_reason(wall("end", .6, .8, x=3)) == "note_or_chain_collision"
    assert state == before


def test_reference_targets_replay_through_same_decoder_and_history():
    source = song([wall("wall", .25, .75)], bombs=[{"id": "bomb", "time": 1., "x": 3, "y": 1}])
    opts = options()
    examples, diagnostic = build_examples(source, {}, VOCAB, opts)
    seen = []
    cursor = iter(range(len(examples.times)))

    def predict(inputs):
        index = next(cursor)
        for key in ("history", "history_age", "history_mask", "interval"):
            np.testing.assert_array_equal(inputs[key], examples.inputs[key][index:index + 1])
        seen.append(index)
        return {"wall": np.eye(len(VOCAB["walls"]))[examples.targets["wall"][index:index + 1]],
                "bombs": examples.targets["bombs"][index:index + 1]}

    state = {"notes": [], "chains": []}
    result = rollout(source, state, VOCAB, opts, predict)
    assert len(seen) == len(examples.times)
    assert result["final_walls"] == diagnostic["hazard_supported_reference"]["final_walls"] == 1
    assert result["final_bombs"] == diagnostic["hazard_supported_reference"]["final_bombs"] == 1
    assert [(item["time"], item["end_time"]) for item in state["walls"]] == [(.25, .75)]


def test_decoder_rejects_colliding_wall_and_bomb_without_suppressing_other_lane():
    opts = options(obstacle_time_gap=[0., 0.])
    state = {"notes": [{"time": .05, "x": 0, "y": 1, "hand": 0, "d": 8}]}
    decoder = SectionDecoder(state, .25, VOCAB, opts)
    queries = candidates(.25, 120., opts)
    assert decoder.step(queries[0], wall=1)["wall"] == 0
    assert decoder.step(queries[1], bombs=np.ones(12))["bombs"][4] == 0
    assert decoder.history[-1]["value"][4] == 1  # free cell (3,0)
    assert decoder.step(queries[2], wall=2)["wall"] == 2
    with pytest.raises(ValueError, match="chronological"):
        decoder.step(queries[0])


def test_ignored_bomb_with_known_cell_masks_only_that_cell():
    source = song(ignored=[{"id": "bad", "family": "bombs", "time": .5, "x": 2, "y": 0}])
    examples, _ = build_examples(source, {}, VOCAB, options())
    row = np.flatnonzero((examples.times == .5) & (np.arange(len(examples.times)) % 2 == 1))[0]
    assert examples.weights["bombs"][row, 2] == 0
    assert examples.weights["bombs"][row].sum() == 11


def test_both_equal_time_queries_receive_full_strictly_preceding_history():
    source = song(duration=2.3)
    examples, _ = build_examples(source, {}, VOCAB, options())
    warmed = examples.times >= 2.
    assert (examples.inputs["history_mask"][warmed].sum(axis=-1) == 32).all()
    assert (examples.inputs["history_age"][warmed, :, 0] > 0).all()


@pytest.mark.parametrize("raw", [{"b": 1., "d": 2., "x": "invalid", "w": 1, "h": 5},
                                {"_time": 1., "_duration": 2., "_lineIndex": "invalid", "_width": 1}])
def test_ignored_wall_recovers_full_raw_duration_for_masking_without_editing_source(raw):
    record = {"id": "invalid-geometry", "family": "walls", "time": .5, "raw": raw}
    source = song(ignored=[record])
    original = deepcopy(source.objects)
    examples, _ = build_examples(source, {}, VOCAB, options())
    wall_rows = np.arange(len(examples.times)) % 2 == 0
    affected = wall_rows & (examples.times >= .5) & (examples.times < 1.5)
    unaffected = wall_rows & ((examples.times < .5) | (examples.times >= 1.5))
    assert affected.sum() == 8
    assert not examples.weights["wall"][affected].any()
    assert examples.weights["wall"][unaffected].all()
    assert source.objects == original


def test_ignored_wall_with_unknown_duration_masks_remaining_timeline():
    source = song(ignored=[{"id": "unknown-span", "family": "walls", "time": .5,
                            "raw": {"b": 1., "d": "invalid"}}])
    examples, _ = build_examples(source, {}, VOCAB, options())
    affected = (np.arange(len(examples.times)) % 2 == 0) & (examples.times >= .5)
    assert not examples.weights["wall"][affected].any()


@pytest.mark.parametrize("record", [{"duration": 0.}, {"duration": -.125},
                                   {"raw": {"d": 0.}}, {"raw": {"_duration": -3.}}])
def test_ignored_wall_with_finite_nonpositive_duration_masks_only_onset(record):
    source = song(ignored=[{"id": "invalid-span", "family": "walls", "time": .5, **record}])
    examples, _ = build_examples(source, {}, VOCAB, options())
    wall_rows = np.arange(len(examples.times)) % 2 == 0
    np.testing.assert_array_equal(examples.times[wall_rows & (examples.weights["wall"] == 0)], [.5])


def test_ignored_legacy_bomb_is_reported_as_bomb_without_rewriting_source_or_masks():
    record = {"id": "_notes:3", "family": "notes", "time": .5,
              "raw": {"_time": 1., "_type": 3, "_lineIndex": "invalid", "_lineLayer": 1}}
    source = song(ignored=[record])
    before = deepcopy(source.objects)
    examples, diagnostic = build_examples(source, {}, VOCAB, options())
    row = np.flatnonzero((examples.times == .5) & (np.arange(len(examples.times)) % 2 == 1))[0]
    assert not examples.weights["bombs"][row].any()
    assert diagnostic["hazard_raw_source_counts"]["bombs"] == 1
    assert diagnostic["hazard_parsed_source_counts"]["bombs"] == 0
    assert diagnostic["hazard_source_exclusions"] == [{"id": "_notes:3", "family": "bombs", "reason": "ignored_source_object"}]
    assert source.objects == before


def test_ignored_legacy_bomb_with_known_valid_cell_masks_only_that_cell():
    source = song(ignored=[{"id": "_notes:3", "family": "notes", "time": .5, "x": 2, "y": 1,
                            "raw": {"_time": 1., "_type": 3}}])
    examples, diagnostic = build_examples(source, {}, VOCAB, options())
    row = np.flatnonzero((examples.times == .5) & (np.arange(len(examples.times)) % 2 == 1))[0]
    assert examples.weights["bombs"][row, 6] == 0
    assert examples.weights["bombs"][row].sum() == 11
    assert diagnostic["hazard_raw_source_counts"]["bombs"] == 1


def test_repaired_wall_vocabulary_preserves_legacy_layer_widths_and_masks_unsupported():
    from training.pipeline_v2.data import read_objects
    from training.pipeline_v2.fixtures import make_dummy_songs
    from training.pipeline_v2.models import build_hazards_model
    from training.pipeline_v2.settings import preparation_settings
    from training.pipeline_v2.targets import fit_vocabularies

    songs = make_dummy_songs()
    for base in songs:
        base.metadata["raw_beatmap"]["obstacles"] = [
            {"b": .5, "d": .5, "x": 1, "y": 0, "w": 1, "h": 5},
            {"b": 1., "d": .05, "x": 3, "y": 0, "w": 1, "h": 5},
            {"b": 2., "d": 1., "x": 3, "y": 0, "w": 20, "h": 5},
            {"b": 3., "d": 1., "x": 0, "y": 0, "w": 1, "h": 5}]
    settings = preparation_settings()
    legacy = {**deepcopy(settings), "stage_policies": {}}
    original_vocab = fit_vocabularies(songs, legacy)
    repaired_vocab = fit_vocabularies(songs, settings)
    assert repaired_vocab["walls"] == original_vocab["walls"]
    assert repaired_vocab["counts"]["walls"] == original_vocab["counts"]["walls"]
    assert len(repaired_vocab["walls"]) == 5
    original_model = build_hazards_model(original_vocab)
    repaired_model = build_hazards_model(repaired_vocab)
    assert original_model.output_shape["wall"] == repaired_model.output_shape["wall"] == (None, 5)
    assert original_model.get_layer("wall_history_embedding").input_dim == repaired_model.get_layer("wall_history_embedding").input_dim == 7
    assert original_model.count_params() == repaired_model.count_params()
    objects, diagnostic = read_objects(songs[0])
    source = V2Song(songs[0], objects, diagnostics=diagnostic)
    examples, diagnostics = build_examples(source, {}, repaired_vocab, settings)
    excluded = {record["id"]: record["reason"] for record in diagnostics["hazard_source_exclusions"]}
    assert excluded["obstacles:0"] == "unsupported_geometry_mode"
    assert excluded["obstacles:1"] == "short_original_wall"
    assert excluded["obstacles:2"] == "invalid_geometry"
    decoder = SectionDecoder({}, 2., repaired_vocab, settings)
    category = next(index for index, item in enumerate(repaired_vocab["walls"]) if item.get("width") == 20)
    assert decoder.step(candidates(2., 120., settings)[0], wall=category)["wall"] == 0
    assert examples.weights["wall"].sum() > 0


@pytest.mark.parametrize("sporty", [False, True])
def test_clearance_index_matches_frozen_legacy_geometry_chain_and_swing_rules(monkeypatch, sporty):
    from tools.config import config
    from map_creation.v2_postprocessing import validate_hazards

    opts = options(sporty_obstacles=sporty)
    clearance = opts["stage_policies"]["hazards"]["clearance"]
    for key, value in clearance.items():
        monkeypatch.setattr(config, key, deepcopy(value))
    source = song(duration=12.)
    upstream = {"notes": [{"id": 0, "time": 1., "hand": 0, "x": 0, "y": 1, "d": 0}],
                "chains": [{"id": 0, "anchor_id": 0, "time": 1., "end_time": 1.4,
                            "x": 0, "y": 1, "tx": 3, "ty": 0, "d": 0, "hand": 0, "slices": 2}]}
    index = ClearanceIndex(upstream, 12., clearance)
    for time in (0., .75, 1., 1.2, 1.5, 2., 10.):
        availability = index.bomb_cells(time)
        for cell in range(12):
            bomb = {"time": time, "x": cell % 4, "y": cell // 4}
            state = {**deepcopy(upstream), "walls": [], "bombs": [bomb]}
            validate_hazards(state, source)
            assert bool(state["bombs"]) == bool(availability[cell])
        for x, width, y, height in ((0, 1, 0, 3), (1, 1, 0, 5), (2, 1, 1, 3),
                                    (3, 1, 2, 3), (0, 4, 2, 3), (0, 1, 0, 6)):
            item = wall("probe", time, time + .125, x=x, width=width, y=y, height=height)
            state = {**deepcopy(upstream), "walls": [item], "bombs": []}
            validate_hazards(state, source)
            assert bool(state["walls"]) == (index.wall_reason(item, minimum=True) is None)
