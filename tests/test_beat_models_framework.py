"""CPU-only contracts and deterministic inference; no learning or TensorFlow imports."""

import json
from copy import deepcopy
import subprocess
import sys

import numpy as np
import pytest

from training.beat_models.artifacts import (load_dataset, load_model_manifest, statistics_hash,
                                            validate_compatibility, write_dataset, write_model_manifest)
from training.beat_models.contracts import (CLASSES, CORE, FEATURES, FRAME_SECONDS, HALO,
                                           REPRESENTATION, SCHEMA_VERSION, Song, decode_note)
from training.beat_models.metrics import match_onsets, timing_scores
from training.beat_models.pipeline import DummyIntensity, decode_pair, generate


def statistics_fixture():
    channels = {"mean": [0] * FEATURES, "scale": [1] * FEATURES, "count": 100}
    return {"version": 1, "fit_split": "train", "song_ids": ["fixture"],
            "features": dict(channels), "coarse_features": dict(channels)}


def song_fixture(duration=10.31):
    frames = np.arange(int(np.ceil(duration / FRAME_SECONDS))) * FRAME_SECONDS
    frames = frames[frames < duration]
    bins = np.minimum(1, duration - np.arange(int(np.ceil(duration)))).astype(np.float32)
    events = np.array([0, len(frames) // 2, len(frames) - 1], dtype=np.int64)
    counts = np.array([1, 2, 1], dtype=np.int32)
    classes = np.array([[1, 0], [5, 30], [0, 35]], dtype=np.int32)
    timing = np.zeros(len(frames), dtype=np.int32)
    timing[events] = counts
    original = []
    for event, row in zip(events, classes):
        for color, category in enumerate(row):
            if category:
                original.append([len(original), frames[event] * 2, frames[event], color, *decode_note(category), 0])
    intensity = np.zeros(len(bins), dtype=np.float32)
    for event, count in zip(events, counts):
        intensity[int(frames[event])] += count / counts.sum()
    features = np.zeros((len(frames), FEATURES), dtype=np.float32)
    features[:, 0] = np.arange(len(frames))  # observable absolute frame position for seam checks
    arrays = {"features": features, "coarse_features": np.zeros((len(bins), FEATURES), dtype=np.float32),
              "frame_times": frames, "bin_durations": bins, "intensity": intensity, "timing": timing,
              "event_times": frames[events], "event_frames": events, "note_classes": classes,
              "original_notes": np.asarray(original, dtype=float), "retained_indices": np.arange(4, dtype=np.int64),
              "original_event_indices": np.array([0, 1, 1, 2], dtype=np.int64)}
    return Song("fixture", {"duration": duration, "bpm": 120., "split": "train", "group_id": "fixture",
                            "normalization_version": 1, "normalization_sha256": statistics_hash(statistics_fixture()),
                            "intensity_target_valid": True, "sources": []}, arrays)


class ExactTiming:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def predict(self, inputs, verbose=0):
        self.calls.append(inputs)
        frames = inputs["audio"][0, HALO:HALO + CORE, 0].astype(int)
        mask = inputs["mask"][0, HALO:HALO + CORE]
        output = np.zeros((1, CORE, 3), dtype=np.float32)
        output[0, :, 0] = 1
        for index in np.flatnonzero(mask):
            count = self.events.get(int(frames[index]), 0)
            output[0, index] = np.eye(3)[count]
        return output


class RecordingNotes:
    def __init__(self):
        self.calls = []

    def predict(self, inputs, verbose=0):
        self.calls.append(inputs)
        red, blue = np.full((1, CLASSES), 0.001), np.full((1, CLASSES), 0.001)
        red[0, 8], red[0, 0] = 0.8, 0.1
        blue[0, 39], blue[0, 0] = 0.6, 0.3
        return {"red": red / red.sum(axis=1, keepdims=True), "blue": blue / blue.sum(axis=1, keepdims=True)}


def predictors_for(song):
    timing = ExactTiming(dict(zip(song.arrays["event_frames"], song.arrays["timing"][song.arrays["event_frames"]])))
    notes = RecordingNotes()
    return {"intensity": DummyIntensity(), "timing": timing, "notes": notes}


def test_pipeline_stitches_halo_and_partial_tail_without_shifts():
    song = song_fixture()
    predictors = predictors_for(song)
    result = generate(song, 4, predictors)
    np.testing.assert_array_equal(result.counts, song.arrays["timing"])
    np.testing.assert_array_equal(result.event_times, song.arrays["event_times"])
    assert len(predictors["timing"].calls) == 3
    assert result.event_frames[-1] == len(song.arrays["features"]) - 1
    assert result.metrics["timing"]["f1"] == 1
    assert result.metrics["event_count_mismatches"] == 0
    assert result.intensity[-1] == pytest.approx(song.arrays["bin_durations"][-1] / song.metadata["duration"])


def test_note_history_is_generated_strictly_before_current_event():
    song = song_fixture()
    predictors = predictors_for(song)
    result = generate(song, 3, predictors)
    first, second, third = predictors["notes"].calls
    assert first["history"][0, 0, 0] == 109
    np.testing.assert_array_equal(second["history"][0, 0], result.note_classes[0])
    np.testing.assert_array_equal(third["history"][0, :2], result.note_classes[:2])
    assert second["history_mask"].sum() == 1
    assert third["history_mask"].sum() == 2
    assert np.all(third["history_age"][third["history_mask"]] > 0)


def test_zero_requested_density_skips_timing_and_notes():
    song = song_fixture()
    predictors = predictors_for(song)
    result = generate(song, 0, predictors)
    assert not predictors["timing"].calls and not predictors["notes"].calls
    assert not len(result.event_times) and result.note_classes.shape == (0, 2)
    assert not result.counts.any() and not result.local_nps.any()


def test_count_decoder_chooses_joint_likelihood_and_never_changes_count():
    red, blue = np.zeros(CLASSES), np.zeros(CLASSES)
    red[0], red[1] = .9, .1
    blue[0], blue[3] = .1, .9
    assert decode_pair(red, blue, 1) == (0, 3)
    assert decode_pair(red, blue, 2) == (1, 3)
    with pytest.raises(ValueError):
        decode_pair(red, blue, 0)


def test_dummy_outputs_are_deterministic_and_probabilities_are_normalized():
    song = song_fixture()
    first, second = generate(song, 5), generate(song, 5)
    np.testing.assert_array_equal(first.note_classes, second.note_classes)
    np.testing.assert_allclose(first.frame_probabilities.sum(axis=-1), 1)
    np.testing.assert_allclose(first.intensity.sum(), 1)


def test_metrics_keep_original_reduced_and_unique_counts_distinct():
    song = song_fixture()
    result = generate(song, 4, predictors_for(song))
    assert result.metrics["original_colored_notes"] == 4
    assert result.metrics["reduced_colored_notes"] == 4
    assert result.metrics["reference_unique_onsets"] == 3
    assert result.metrics["generated_colored_notes"] == 4
    assert result.metrics["generated_unique_onsets"] == 3


def test_matching_is_one_to_one_and_uses_seconds():
    assert timing_scores([1, 2], [1.02, 2.02])["f1"] == 1
    assert timing_scores([1, 2], [1.021, 2.021])["f1"] == 0
    assert match_onsets([1, 1.01], [1.005]).shape == (1, 2)
    assert timing_scores([], [1])["f1"] == 0
    assert timing_scores([], [])["f1"] == 1


def test_dataset_roundtrip_hashes_pickle_rejection_and_refusal_to_overwrite(tmp_path):
    root = tmp_path / "prepared"
    stats = statistics_fixture()
    manifest = write_dataset(root, [song_fixture()], stats, {"dummy": True})
    songs, loaded = load_dataset(root)
    np.testing.assert_array_equal(songs[0].arrays["timing"], song_fixture().arrays["timing"])
    assert loaded["representation"] == REPRESENTATION
    assert loaded["normalization_sha256"] == statistics_hash(stats)
    assert manifest["splits"]["train"] == ["fixture"]
    with pytest.raises(FileExistsError):
        write_dataset(root, songs, stats)
    with (root / "song_00000.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(root)


def test_checkpoints_require_explicit_matching_representation_stats_and_stage(tmp_path):
    stats = statistics_fixture()
    dataset = {"schema_version": SCHEMA_VERSION, "representation": REPRESENTATION,
               "normalization": stats, "normalization_sha256": statistics_hash(stats)}
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_bytes(b"placeholder, never loaded by framework")
    write_model_manifest(tmp_path, "intensity", checkpoint, dataset)
    path, manifest = load_model_manifest(tmp_path, "intensity", dataset)
    assert path == checkpoint
    with pytest.raises(ValueError, match="wrong stage"):
        load_model_manifest(tmp_path, "notes", dataset)
    with pytest.raises(ValueError, match="statistics"):
        validate_compatibility(manifest, {"mean": [1] * FEATURES})
    manifest["representation"] = {"feature_version": 99}
    with pytest.raises(ValueError, match="representation"):
        validate_compatibility(manifest)


def test_imports_and_help_never_load_tensorflow():
    code = "import sys; from training.beat_models.__main__ import parser; parser().format_help(); assert not any(k == 'tensorflow' or k.startswith('tensorflow.') for k in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pipeline_rejects_invalid_probabilities():
    song = song_fixture()
    predictors = predictors_for(song)
    predictors["intensity"] = lambda inputs: np.zeros((1, len(song.arrays["bin_durations"])))
    with pytest.raises(ValueError, match="Unnormalized"):
        generate(song, 3, predictors)


def test_partial_diagnostic_cap_is_explicit_and_conserves_decoded_counts():
    song = song_fixture()
    result = generate(song, 3, predictors_for(song), max_events=2)
    assert result.metrics["partial_diagnostic"]
    assert result.metrics["uncapped_unique_onsets"] == 3
    assert result.metrics["event_limit"] == 2
    assert len(result.note_classes) == 2
    np.testing.assert_array_equal(result.event_frames, np.flatnonzero(result.counts))
    assert np.count_nonzero(result.note_classes) == result.counts.sum()


@pytest.mark.parametrize("channel,field,value", [("features", "scale", [0] * FEATURES),
                                                 ("coarse_features", "mean", [0] * 3),
                                                 ("features", "count", 0)])
def test_rehashed_invalid_statistics_are_rejected(tmp_path, channel, field, value):
    root = tmp_path / "dataset"
    write_dataset(root, [song_fixture()], statistics_fixture())
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["normalization"][channel][field] = value
    manifest["normalization_sha256"] = statistics_hash(manifest["normalization"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="normalization"):
        load_dataset(root)


@pytest.mark.parametrize("identity", ["group_id", "decoded_sha256", "audio_sha256"])
def test_dataset_rejects_recording_identity_across_splits(tmp_path, identity):
    first, second = song_fixture(), deepcopy(song_fixture())
    second.song_id = "fixture-two"
    first.metadata[identity] = second.metadata[identity] = "same-recording"
    second.metadata["split"] = "test"
    if identity != "group_id":
        second.metadata["group_id"] = "different-spelling"
    with pytest.raises(ValueError, match="leaks"):
        write_dataset(tmp_path / "bad", [first, second], statistics_fixture())
    assert not (tmp_path / "bad").exists()


def test_curated_sources_cannot_enter_learning_splits(tmp_path):
    song = song_fixture()
    song.metadata["sources"] = [{"role": "audio", "path": "C:\\Data\\evaluate_beats\\bs_map_input\\map\\song.egg",
                                 "sha256": "a" * 64}]
    with pytest.raises(ValueError, match="Curated"):
        write_dataset(tmp_path / "bad", [song], statistics_fixture())


def test_explicit_checkpoint_filename_cannot_select_another_file(tmp_path):
    dataset = {"schema_version": SCHEMA_VERSION, "representation": REPRESENTATION,
               "normalization": statistics_fixture(), "normalization_sha256": statistics_hash(statistics_fixture())}
    selected = tmp_path / "selected.keras"
    selected.write_bytes(b"selected")
    other = tmp_path / "other.keras"
    other.write_bytes(b"other")
    write_model_manifest(tmp_path, "timing", selected, dataset)
    with pytest.raises(ValueError, match="filename disagrees"):
        load_model_manifest(other, "timing", dataset)


def test_prepare_all_skipped_persists_individual_reasons(tmp_path, monkeypatch):
    from training.beat_models.__main__ import main
    from training.beat_models import data
    report = {"skipped": [{"path": "unsupported-map", "reason": "Unsupported beatmap syntax"}],
              "reserved_skipped": [{"path": "missing-audio", "reason": "Audio file does not exist"}]}
    monkeypatch.setattr(data, "prepare_songs", lambda **kwargs: ([], report))
    output = tmp_path / "empty-run"
    with pytest.raises(SystemExit, match="No accepted songs"):
        main(["prepare", "--source-root", str(tmp_path), "--output", str(output)])
    assert json.loads((output / "preparation-report.json").read_text()) == report
    assert not (output / "manifest.json").exists()


def test_evaluation_only_requires_saved_training_normalization(tmp_path):
    from training.beat_models.__main__ import main
    with pytest.raises(SystemExit) as error:
        main(["prepare", "--dummy", "--evaluation-only", "--output", str(tmp_path / "evaluation")])
    assert error.value.code == 2
    assert not (tmp_path / "evaluation").exists()


def test_dummy_prepare_validate_and_smoke_do_not_import_tensorflow(tmp_path):
    code = """import sys
from training.beat_models.__main__ import main
root = sys.argv[1]
main(['prepare', '--dummy', '--output', root + '/dataset'])
main(['validate', '--dataset', root + '/dataset'])
main(['smoke', '--backend', 'dummy', '--dataset', root + '/dataset', '--output', root + '/smoke'])
main(['evaluate', '--backend', 'dummy', '--dataset', root + '/dataset', '--output', root + '/evaluation'])
assert not any(name == 'tensorflow' or name.startswith('tensorflow.') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text())
    assert manifest["status"] == "dummy"
    assert "runtime" in manifest and "code_identity" in manifest
    report = json.loads((tmp_path / "evaluation" / "report.json").read_text())
    assert report["evaluated_split"] == "test"
    assert report["summary"]["songs"] == len(manifest["splits"]["test"])
    assert all(song["split"] == "test" and song["group_id"] for song in report["songs"])


def test_song_normalization_provenance_cannot_be_reassigned(tmp_path):
    song = song_fixture()
    song.metadata["normalization_sha256"] = "unrelated-statistics"
    with pytest.raises(ValueError, match="normalization provenance"):
        write_dataset(tmp_path / "bad", [song], statistics_fixture())
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("field", ["splits", "sources"])
def test_dataset_rejects_missing_split_or_source_manifest(tmp_path, field):
    root = tmp_path / "dataset"
    write_dataset(root, [song_fixture()], statistics_fixture())
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest[field]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest mismatch"):
        load_dataset(root)
