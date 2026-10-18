"""Synthetic raw-object, frozen-vocabulary and target-isolation contracts."""

from copy import deepcopy
import json

import numpy as np
import pytest

from training.pipeline_v2.contracts import NEW_STAGES, build_inputs, validate
from training.pipeline_v2.data import make_dummy_dataset, prepare_dataset, read_objects
from training.pipeline_v2.fixtures import make_dummy_songs


@pytest.fixture(scope="module")
def dataset():
    return make_dummy_dataset()


def test_dummy_has_positive_targets_for_every_stage_and_multilabel_lighting(dataset):
    assert dataset.manifest["status"] == "structural_dummy"
    for song in dataset.songs:
        assert validate(song)
        assert set(song.examples) == set(NEW_STAGES)
    song = dataset.songs[0]
    assert any(template["kind"] == "copies" for template in dataset.vocabularies["multiplier"])
    assert any(template["kind"] == "chain" for template in dataset.vocabularies["multiplier"])
    assert song.examples["hazards"].targets["wall"].max() > 0
    assert song.examples["hazards"].targets["bombs"].max() == 1
    assert song.examples["arcs"].targets["attach"].max() == 1
    assert song.examples["arcs"].weights["shape"].sum() > 0
    assert song.examples["lighting"].targets["events"].sum(axis=-1).max() == 2
    for examples in song.examples.values():
        assert not examples.inputs["audio"].any()


def test_preparation_preserves_base_arrays_source_json_and_later_retained_notes():
    songs = make_dummy_songs()
    before = deepcopy(songs)
    dataset = prepare_dataset(songs)
    for old, base, prepared in zip(before, songs, dataset.songs):
        assert old.metadata == base.metadata
        for key in old.arrays:
            np.testing.assert_array_equal(old.arrays[key], base.arrays[key])
            np.testing.assert_array_equal(old.arrays[key], prepared.base.arrays[key])
        retained = [record for record in prepared.objects["notes"] if record["retained"]]
        assert len(retained) == base.metadata["retained_colored_note_count"]
        assert any(record["time"] == 1.5 for record in retained)
        relationships = prepared.diagnostics["expansion_relationships"]
        assert any(record["chain_ids"] and record["reason"] is None for record in relationships)


def test_vocabularies_fit_train_only_and_unknown_targets_are_masked():
    songs = make_dummy_songs()
    validation = next(song for song in songs if song.metadata["split"] != "train")
    validation.metadata["raw_beatmap"]["basicBeatmapEvents"].append({"b": 0, "et": 999, "i": 123, "f": 1})
    validation.metadata["raw_beatmap"]["obstacles"][0]["w"] = 9
    result = prepare_dataset(songs, settings={})
    assert {record["type"] for record in result.vocabularies["lighting"]} == {0, 1}
    assert not any(record.get("width") == 9 for record in result.vocabularies["walls"])
    prepared = next(song for song in result.songs if song.base.song_id == validation.song_id)
    assert prepared.examples["lighting"].weights["events"][0, 0] == 0
    assert prepared.diagnostics["targets"]["unknown_lighting"]
    target = prepared.examples["hazards"]
    wall_query = np.flatnonzero(np.isclose(target.times, 2.3))[0]
    assert target.weights["wall"][wall_query] == 0
    history_times = target.times[:, None] - target.inputs["history_age"][:, :, 0]
    assert not np.any(target.inputs["history_mask"] & np.isclose(history_times, 2.3))
    assert prepared.diagnostics["targets"]["unscheduled_walls"]


def test_v2_bombs_walls_lighting_seconds_and_ignored_records_are_preserved():
    base = make_dummy_songs()[0]
    colored = base.metadata["raw_beatmap"]["colorNotes"]
    notes = [{"_time": note["b"], "_type": note["c"], "_lineIndex": note["x"], "_lineLayer": note["y"], "_cutDirection": note["d"]} for note in colored]
    bomb = {"_time": .5, "_type": 3, "_lineIndex": 2, "_lineLayer": 1, "_cutDirection": 8}
    base.metadata["raw_beatmap"] = {"_version": "2.2.0", "_notes": [*notes, bomb],
                                    "_obstacles": [{"_time": 1., "_lineIndex": 0, "_type": 1, "_duration": 2., "_width": 4}],
                                    "_events": [{"_time": .5, "_type": 1, "_value": 3}],
                                    "_customEvents": [{"unhandled": "preserve me"}]}
    before = deepcopy(base.metadata["raw_beatmap"])
    objects, diagnostics = read_objects(base)
    assert objects["bombs"][0]["time"] == .25
    assert objects["bombs"][0]["raw"] == bomb
    wall = objects["walls"][0]
    assert (wall["time"], wall["duration"], wall["y"], wall["height"]) == (.5, 1., 2, 3)
    assert objects["lighting"][0]["time"] == .25
    assert objects["ignored"][0]["raw"] == [{"unhandled": "preserve me"}]
    assert diagnostics["source_objects_complete"]
    assert before == base.metadata["raw_beatmap"]


def test_raw_source_hash_changes_and_bad_maps_are_reported_individually(tmp_path):
    songs = make_dummy_songs()
    bad = next(song for song in songs if song.metadata["split"] != "train")
    raw = bad.metadata.pop("raw_beatmap")
    path = tmp_path / "source.dat"
    path.write_text(json.dumps(raw), encoding="utf-8")
    bad.metadata["sources"] = [{"role": "beatmap", "path": str(path), "sha256": "wrong-hash"}]
    result = prepare_dataset(songs, settings={})
    assert len(result.songs) == len(songs) - 1
    assert result.manifest["skipped"][0]["song_id"] == bad.song_id
    assert "changed" in result.manifest["skipped"][0]["reason"]


def test_streaming_preparation_does_not_collect_expanded_inputs(dataset):
    emitted = []
    result = prepare_dataset([song.base for song in dataset.songs], emit=lambda song: emitted.append(song.base.song_id), collect=False)
    assert result.songs == []
    assert len(emitted) == len(dataset.songs) == result.manifest["song_count"]
    assert result.vocabularies["fingerprint"] == dataset.vocabularies["fingerprint"]


def test_all_test_evaluation_uses_frozen_training_vocabulary_without_refit(dataset):
    bases = deepcopy([song.base for song in dataset.songs[:1]])
    bases[0].metadata["split"] = "test"
    bases[0].metadata["raw_beatmap"]["basicBeatmapEvents"].append({"b": 0, "et": 999, "i": 7})
    original = deepcopy(dataset.vocabularies)
    result = prepare_dataset(bases, settings={}, vocabularies=dataset.vocabularies)
    assert result.vocabularies == original == dataset.vocabularies
    assert result.manifest["vocabulary_origin"] == "explicit_frozen_train"
    assert result.songs[0].examples["lighting"].weights["events"][0, 0] == 0
    altered = deepcopy(original)
    altered["lighting"].append({"type": 999, "value": 7})
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_dataset(bases, vocabularies=altered)


def test_native_hazard_positives_augment_scheduler_and_keep_coverage_separate():
    songs = make_dummy_songs()
    for base in songs:
        raw = base.metadata["raw_beatmap"]
        raw["obstacles"][0].update(b=4.8, d=1.2)
        raw["bombNotes"][0]["b"] = .246
    result = prepare_dataset(songs, settings={})
    for song in result.songs:
        diagnostic = song.diagnostics["targets"]
        assert diagnostic["hazard_scheduler_coverage"]["walls"]["matched_source_objects"] == 0
        assert diagnostic["hazard_scheduler_coverage"]["bombs"]["matched_source_objects"] == 0
        assert diagnostic["hazard_positive_query_coverage"]["walls"] == 1
        assert diagnostic["hazard_positive_query_coverage"]["bombs"] == 1
        assert diagnostic["hazard_positive_query_coverage"]["source_only_queries"] == 2
        provenance = diagnostic["hazard_query_provenance"]
        assert len({(row["kind"], row["time"], row["end_time"]) for row in provenance}) == len(provenance)
        examples = song.examples["hazards"]
        for index, query in enumerate(provenance):
            if query["origin"] == "source":
                key = "wall" if query["kind"] == "wall" else "bombs"
                assert examples.weights[key][index] == 1
                assert examples.targets[key][index].max() > 0


def test_shared_builder_uses_strict_history_and_hides_future_arrow_labels(dataset):
    song = dataset.songs[0]
    state = {"notes": [{"id": 0, "time": .5, "hand": 0, "x": 0, "y": 2, "d": 1},
                       {"id": 1, "time": 1.5, "hand": 0, "x": 1, "y": 1, "d": 2}]}
    history = [{"time": .1, "value": [1, 2]}, {"time": 1., "value": [3, 4]}, {"time": 2., "value": [5, 6]}]
    first = build_inputs("arrows", song, state, {"time": 1.}, history)
    assert first["history_mask"].sum() == 1
    assert first["history_mask"][0, 0]
    assert not first["history_mask"][0, 1:].any()
    np.testing.assert_array_equal(first["history"][0, 0], [1, 2])
    state["notes"][1]["d"] = 7
    second = build_inputs("arrows", song, state, {"time": 1.}, history)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_lighting_partial_current_mask_preserves_preceding_history(dataset):
    song = dataset.songs[0]
    history = [{"time": .1 * index, "value": [1., 0.]} for index in range(1, 10)]
    inputs = build_inputs("lighting", song, {"light_classes": 2},
                          {"time": 1., "times": [1., 1.25], "mask": [True, True] + [False] * 14}, history)
    assert inputs["mask"].sum() == 2
    assert inputs["history_mask"].sum() == 9
    assert inputs["history"][0, -1, 0] == 1
    assert not inputs["audio"][0, 2:].any()
