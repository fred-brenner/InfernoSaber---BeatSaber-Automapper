"""Arc-only history, gap eligibility and whole-recording validation repair."""

from copy import deepcopy

import numpy as np
import pytest

from training.pipeline_v2.arc_policy import (ARC_POLICY_V2, arc_candidates,
                                           arc_policy, build_split_overlay,
                                           eligible_gap, history_value,
                                           stage_split, validate_split_overlay)


def _options(**overrides):
    return {"stage_policies": {"arcs": {**deepcopy(ARC_POLICY_V2), **overrides}}}


@pytest.mark.parametrize("record", [
    {"attach": 0, "shape": [1.2, 1.4], "anchor": 2},
    {"value": [0, 1.2, 1.4, 2]},
    {"attach": 1, "shape": [1.2, 1.4], "anchor": 2, "rejected": True},
    {"value": [1, 1.2, 1.4, 2], "accepted": False},
    {"attach": 1, "mu": 1.2, "tmu": 1.4, "anchor": 2, "status": "rejected"},
    {"attach": 0, "value": [1, 1.2, 1.4, 2]},
    {"value": [0, float("nan"), float("nan"), 2]},
])
def test_none_history_canonicalizes_unused_geometry(record):
    assert history_value(record) == [0, 0, 0, 0]


@pytest.mark.parametrize("record", [
    {"attach": 1, "shape": [.7, 1.3], "anchor": 2},
    {"value": [1, .7, 1.3, 2]},
    {"attach": 1, "mu": .7, "tmu": 1.3, "anchor": 2},
])
def test_attached_history_preserves_shape_and_anchor(record):
    assert history_value(record) == [1, .7, 1.3, 2]


def test_shared_input_builder_uses_canonical_none_and_excludes_equal_time_siblings():
    from types import SimpleNamespace

    from training.pipeline_v2.contracts import build_inputs

    base = SimpleNamespace(metadata={"duration": 3., "difficulty_nps": 1., "bpm": 120.},
                           arrays={"bin_durations": np.asarray([3.]), "frame_times": np.arange(300) * .01})
    state = {"notes": [{"id": 0, "time": 1., "hand": 0, "x": 0, "y": 0, "d": 1},
                       {"id": 1, "time": 1.5, "hand": 0, "x": 1, "y": 0, "d": 1}],
             "shares": np.asarray([1.]), "requested_nps": 1.}
    candidate = {"time": 1., "end_time": 1.5, "endpoint_ids": [0, 1], "gap": .5, "hand": 0}
    history = [{"time": .5, "value": [0, 1.4, 1.8, 2]},
               {"time": 1., "value": [1, .7, 1.3, 2]}]
    inputs = build_inputs("arcs", base, state, candidate, history)
    assert inputs["history_mask"].sum() == 1
    np.testing.assert_array_equal(inputs["history"][0, 0], [0, 0, 0, 0])


@pytest.mark.parametrize("gap, expected", [(0, False), (.149999, False), (.15, True),
                                          (.5, True), (.8, True), (.800001, False)])
def test_v2_inclusive_gap_boundaries(gap, expected):
    assert eligible_gap(gap, _options()) is expected


def test_optional_legacy_gap_is_strict_and_versioned():
    options = _options(use_legacy_gap=True, legacy_gap=[.2, .7])
    assert not eligible_gap(.2, options)
    assert eligible_gap(.5, options)
    assert not eligible_gap(.7, options)
    assert eligible_gap(.15, _options())
    assert arc_policy({}) is None
    with pytest.raises(ValueError, match="version"):
        arc_policy(_options(version=3))


def test_candidates_share_existing_adjacent_endpoints_and_explicit_gap_policy():
    notes = [{"id": "red-start", "time": 0., "hand": 0},
             {"id": "red-min", "time": .15, "hand": 0},
             {"id": "blue-start", "time": 0., "hand": 1},
             {"id": "blue-max", "time": .8, "hand": 1},
             {"id": "blue-too-far", "time": 2., "hand": 1}]
    options = _options()
    options["schedule"] = {"arc_min_gap_seconds": .4, "arc_max_gap_seconds": .6}
    queries = arc_candidates(notes, options)
    assert [item["endpoint_ids"] for item in queries] == [["red-start", "red-min"], ["blue-start", "blue-max"]]
    assert arc_candidates(notes, {"schedule": options["schedule"]}) == []
    assert [item["endpoint_ids"] for item in arc_candidates(notes, _options(use_legacy_gap=True))] == [
        ["blue-start", "blue-max"]]


def test_candidate_limit_remains_explicit():
    notes = [{"id": index, "time": index * .2, "hand": 0} for index in range(5)]
    options = _options()
    options["schedule"] = {"max_candidates": 2}
    with pytest.raises(ValueError, match="max_candidates"):
        arc_candidates(notes, options)


def test_optional_legacy_eligibility_is_applied_before_candidate_limit():
    notes = [{"id": index, "time": when, "hand": 0} for index, when in enumerate((0., .2, .4, 1.))]
    options = _options(use_legacy_gap=True)
    options["schedule"] = {"max_candidates": 1}
    assert [query["endpoint_ids"] for query in arc_candidates(notes, options)] == [[2, 3]]


def _summaries():
    records = []
    for index in range(6):
        for variant in ("expert", "expert-plus"):
            records.append({"song_id": f"group-{index}-{variant}", "group_id": f"group-{index}",
                            "split": "train", "positive_count": index + 1 if variant == "expert" else 0,
                            "audio_sha256": f"compressed-{index}", "decoded_sha256": f"decoded-{index}"})
    records.extend([
        {"song_id": "original-val", "group_id": "original-val", "split": "validation", "positive_count": 0},
        {"song_id": "test", "group_id": "test", "split": "test", "positive_count": 8},
    ])
    return records


def test_split_repair_is_seeded_and_transfers_every_variant_without_touching_test():
    records = _summaries()
    before = deepcopy(records)
    overlay = build_split_overlay(records)
    expected = sorted(np.random.default_rng(3).permutation([f"group-{i}" for i in range(6)])[:2].tolist())
    assert overlay["moved_group_ids"] == expected
    assert overlay == build_split_overlay(list(reversed(records)))
    assert records == before
    for item in records:
        split = "validation" if item["group_id"] in expected else item["split"]
        assert overlay["records"][item["song_id"]] == split
    assert overlay["splits"]["test"] == ["test"]
    assert "original-val" in overlay["splits"]["validation"]
    assert overlay["counts"]["train"]["groups"] == 4
    assert overlay["counts"]["validation"]["groups"] == 3
    assert overlay["counts"]["validation"]["songs"] == 5
    assert overlay["counts"]["validation"]["positive_count"] == sum(int(group[-1]) + 1 for group in expected)
    exposure = overlay["provenance"]["previous_upstream_exposure"]
    assert exposure["group_ids"] == expected
    assert exposure["split"] == "train"


@pytest.mark.parametrize("identity_key", ["group_id", "audio_sha256", "decoded_sha256"])
def test_split_repair_rejects_existing_recording_alias_leakage(identity_key):
    records = _summaries()
    records[-1][identity_key] = records[0][identity_key]
    with pytest.raises(ValueError, match="identity leaks"):
        build_split_overlay(records)


@pytest.mark.parametrize("identity_key", ["audio_sha256", "decoded_sha256"])
def test_repaired_split_checks_audio_aliases_across_different_group_ids(identity_key):
    records = _summaries()
    moved = build_split_overlay(records)["moved_group_ids"]
    kept = next(item["group_id"] for item in records if item["split"] == "train" and item["group_id"] not in moved)
    for item in records:
        if item["group_id"] in (moved[0], kept):
            item[identity_key] = "same-audio-alias"
    with pytest.raises(ValueError, match="identity leaks"):
        build_split_overlay(records)


def test_audio_alias_tokens_match_across_compressed_and_decoded_hash_fields():
    records = _summaries()
    records[-1]["audio_sha256"] = records[0]["decoded_sha256"]
    with pytest.raises(ValueError, match="identity leaks"):
        build_split_overlay(records)


def test_real_repair_requires_positive_support_and_dummy_mode_is_explicit():
    records = [item for item in _summaries() if item["group_id"] not in ("group-2", "group-3", "group-4", "group-5")]
    with pytest.raises(ValueError, match="at least three positive"):
        build_split_overlay(records)
    overlay = build_split_overlay(records, structural_dummy=True)
    assert overlay["mode"] == "structural_dummy"
    assert overlay["moved_group_ids"] == []
    assert overlay["records"] == {item["song_id"]: item["split"] for item in records}


def test_only_effectively_supervised_positive_groups_are_eligible_for_transfer():
    records = _summaries()
    for item in records:
        if item["group_id"] in ("group-0", "group-1", "group-2"):
            item["positive_count"] = 0
    overlay = build_split_overlay(records)
    assert overlay["positive_train_group_ids"] == ["group-3", "group-4", "group-5"]
    assert set(overlay["moved_group_ids"]) <= {"group-3", "group-4", "group-5"}


def test_overlay_validation_and_stage_routing_keep_shared_splits_immutable():
    records = _summaries()
    overlay = build_split_overlay(records)
    manifest = {"songs": deepcopy(records), "stage_splits": {"arcs": overlay}}
    assert validate_split_overlay(manifest)
    assert validate_split_overlay({})
    moved = next(item for item in records if item["group_id"] in overlay["moved_group_ids"])
    assert stage_split(manifest, "arcs", moved) == "validation"
    assert stage_split(manifest, "positions", moved) == "train"
    assert moved["split"] == "train"
    assert stage_split({}, "arcs", moved) == "train"
    tampered = deepcopy(manifest)
    tampered["stage_splits"]["arcs"]["records"][moved["song_id"]] = "train"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_split_overlay(tampered)
    changed_source = deepcopy(manifest)
    changed_source["songs"][0]["split"] = "test"
    with pytest.raises(ValueError, match="source recording"):
        validate_split_overlay(changed_source)
    with pytest.raises(ValueError, match="Song identity"):
        stage_split(manifest, "arcs", {**moved, "song_id": "unknown"})
