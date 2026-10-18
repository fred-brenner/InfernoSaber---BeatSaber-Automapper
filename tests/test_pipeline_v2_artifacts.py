"""Artifact and command plumbing checks; no model comparisons or training."""

from copy import deepcopy

import numpy as np
import pytest

from training.pipeline_v2.artifacts import (DatasetWriter, compatibility, load_checkpoint_manifest,
                                          load_dataset, save_checkpoint_manifest, write_dataset)
from training.pipeline_v2.data import make_dummy_dataset


def test_dataset_roundtrip_preserves_targets_and_objects(tmp_path):
    original = make_dummy_dataset()
    write_dataset(tmp_path / "dataset", original)
    restored = load_dataset(tmp_path / "dataset")
    assert restored.vocabularies == original.vocabularies
    assert len(restored.songs) == len(original.songs)
    for before, after in zip(original.songs, restored.songs):
        assert before.objects == after.objects
        for key, array in before.base.arrays.items():
            np.testing.assert_array_equal(array, after.base.arrays[key])
        for stage in before.examples:
            for family in ("inputs", "targets", "weights"):
                for key, array in getattr(before.examples[stage], family).items():
                    np.testing.assert_array_equal(array, getattr(after.examples[stage], family)[key])


def test_dataset_tampering_is_rejected_on_lazy_read(tmp_path):
    dataset = make_dummy_dataset()
    write_dataset(tmp_path / "dataset", dataset)
    restored = load_dataset(tmp_path / "dataset")
    (tmp_path / "dataset/song_00000.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        restored.songs[0]


def test_writer_rejects_group_split_leakage(tmp_path):
    dataset = make_dummy_dataset()
    first = deepcopy(dataset.songs[0])
    other = deepcopy(first)
    other.base.song_id += "-different-difficulty"
    other.base.metadata["split"] = "test" if first.base.metadata["split"] == "train" else "train"
    writer = DatasetWriter(tmp_path / "dataset")
    writer.add(first)
    with pytest.raises(ValueError, match="leaks"):
        writer.add(other)


def test_checkpoint_identity_includes_encoder_and_vocab(tmp_path):
    dataset = make_dummy_dataset()
    # This exercises manifest bookkeeping only; no executable model is loaded.
    (tmp_path / "best.keras").write_bytes(b"non-model fixture")
    save_checkpoint_manifest(tmp_path, "multiplier", dataset, "untrained")
    root, manifest = load_checkpoint_manifest(tmp_path, "multiplier", dataset)
    assert root == tmp_path and manifest["status"] == "untrained"
    changed = deepcopy(dataset)
    changed.manifest["encoder_identity"]["checkpoint_fingerprint"] = "another-encoder"
    assert compatibility(dataset, "multiplier") != compatibility(changed, "multiplier")
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint_manifest(tmp_path, "multiplier", changed)


def test_dataset_destination_cannot_overwrite_existing(tmp_path):
    with pytest.raises(FileExistsError):
        write_dataset(tmp_path, make_dummy_dataset())


def test_preparation_reports_keep_every_skip_without_comparing_models():
    from training.pipeline_v2.evaluation import preparation_reports

    manifest = {
        "base_preparation": {
            "skipped": [{"path": "broken-map", "reason": "unsupported version"}],
            "reserved_skipped": [{"path": "reservation", "reason": "missing audio"}],
        },
        "skipped": [{"song_id": "late-failure", "reason": "invalid objects"}],
    }
    original = deepcopy(manifest)
    report = preparation_reports(manifest)
    assert [row["reason"] for row in report["skipped"]] == [
        "unsupported version", "missing audio", "invalid objects"]
    assert report["skip_counts"] == {
        "base_preparation.skipped": 1, "base_preparation.reserved_skipped": 1, "skipped": 1}
    assert [row["report_origin"] for row in report["skipped"]] == list(report["skip_counts"])
    report["skipped"][0]["reason"] = "changed report"
    assert manifest == original


@pytest.mark.parametrize("reader", ["song", "stage"])
def test_recording_alias_cannot_disagree_with_index_even_with_updated_file_hash(tmp_path, reader):
    from training.beat_models.artifacts import read_json, write_json, file_hash
    from training.pipeline_v2.training_data import TrainingData
    original = make_dummy_dataset()
    write_dataset(tmp_path / "dataset", original)
    restored = load_dataset(tmp_path / "dataset")
    record = restored.songs.records[0]
    record["decoded_sha256"] = "indexed-original-recording"
    path = restored.songs.root / record["metadata_file"]
    metadata = read_json(path)
    metadata["metadata"]["decoded_sha256"] = "different-recording"
    import json
    path.write_text(json.dumps(metadata), encoding="utf-8")
    record["metadata_sha256"] = file_hash(path)
    # Test the reader binding independently of the overlay's own guard.
    restored.manifest.pop("stage_splits", None)
    with pytest.raises(ValueError, match="recording alias"):
        if reader == "song":
            restored.songs[0]
        else:
            TrainingData(restored, "positions")


def test_repaired_lighting_compatibility_binds_onset_candidate_policy():
    dataset = make_dummy_dataset()
    before = compatibility(dataset, "lighting")
    dataset.manifest["data_settings"]["schedule"]["lighting_onset_threshold"] += .1
    assert compatibility(dataset, "lighting") != before


def test_legacy_arc_none_history_is_normalized_without_changing_stored_arrays(tmp_path):
    from training.beat_models.artifacts import file_hash
    from training.pipeline_v2.training_data import TrainingData
    dataset = make_dummy_dataset(settings={})
    original = dataset.songs[0].examples["arcs"].inputs["history"]
    original[0, 0] = [0, 1.2, .8, 2]
    write_dataset(tmp_path / "dataset", dataset)
    restored = load_dataset(tmp_path / "dataset")
    path = restored.songs.root / restored.songs.records[0]["file"]
    before = file_hash(path)
    np.testing.assert_array_equal(restored.songs[0].examples["arcs"].inputs["history"][0, 0], [0, 0, 0, 0])
    cached = TrainingData(restored, "arcs", ram_cache_mib=64)
    streamed = TrainingData(restored, "arcs", ram_cache_mib=0)
    for reader in (cached, streamed):
        np.testing.assert_array_equal(next(reader.examples()).inputs["history"][0, 0], [0, 0, 0, 0])
    assert file_hash(path) == before
    np.testing.assert_allclose(original[0, 0], [0, 1.2, .8, 2])
