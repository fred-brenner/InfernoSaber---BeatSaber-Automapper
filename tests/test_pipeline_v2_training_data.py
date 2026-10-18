"""Stage-specific CPU training input contracts; no models or optimizers."""

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training.beat_models.artifacts import file_hash
from training.pipeline_v2.artifacts import load_dataset, write_dataset
from training.pipeline_v2.contracts import NEW_STAGES
from training.pipeline_v2.data import make_dummy_dataset


@pytest.fixture(scope="module")
def dataset():
    return make_dummy_dataset()


@pytest.fixture
def disk_dataset(tmp_path, dataset):
    root = tmp_path / "dataset"
    write_dataset(root, dataset)
    return load_dataset(root)


def test_first_stage_read_only_decompresses_selected_stage(disk_dataset, dataset, monkeypatch):
    from training.pipeline_v2.training_data import iter_stage_examples

    original = np.lib.npyio.NpzFile.__getitem__
    read = []

    def selected_only(archive, key):
        assert key.startswith("positions::"), key
        read.append(key)
        return original(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", selected_only)
    examples = next(iter_stage_examples(disk_dataset, "positions", "train"))
    expected = next(song.examples["positions"] for song in dataset.songs
                    if song.base.metadata["split"] == "train")
    np.testing.assert_array_equal(examples.targets["red"], expected.targets["red"])
    assert read


def test_ram_cache_reuses_validated_arrays_without_archive_or_json_reads(disk_dataset, monkeypatch):
    from training.pipeline_v2.training_data import TrainingData

    reader = TrainingData(disk_dataset, "positions", ram_cache_mib=1)
    assert reader.counts == {
        "train": {"songs": 2, "examples": 10, "supervised_values": 20},
        "validation": {"songs": 1, "examples": 5, "supervised_values": 10},
        "test": {"songs": 1, "examples": 5, "supervised_values": 10},
    }
    assert reader.report["mode"] == "ram"
    assert 0 < reader.report["retained_bytes"] <= reader.report["budget_bytes"]
    json.dumps(reader.report)

    def forbidden(*args, **kwargs):
        pytest.fail("A warm RAM reader reopened an archive or metadata")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    first = list(reader.examples("train"))
    second = list(reader.examples("train"))
    assert len(first) == 2
    assert all(old is new for old, new in zip(first, second))


def test_indexed_memory_collection_does_not_require_len_or_load_other_splits(dataset):
    from training.pipeline_v2.training_data import iter_stage_examples

    class IndexedSongs:
        records = [{"split": song.base.metadata["split"]} for song in dataset.songs]

        def __getitem__(self, index):
            assert index == 0
            return dataset.songs[index]

    selected = SimpleNamespace(songs=IndexedSongs())
    assert len(list(iter_stage_examples(selected, "positions", "validation"))) == 1


def _same_examples(actual, expected):
    np.testing.assert_array_equal(actual.times, expected.times)
    for name in ("inputs", "targets", "weights"):
        assert set(getattr(actual, name)) == set(getattr(expected, name))
        for key, array in getattr(actual, name).items():
            np.testing.assert_array_equal(array, getattr(expected, name)[key])


@pytest.mark.parametrize("stage", NEW_STAGES)
def test_cached_streamed_and_memory_examples_have_identical_arrays_and_order(stage, disk_dataset, dataset):
    from training.pipeline_v2.training_data import TrainingData, iter_stage_examples

    cached = TrainingData(disk_dataset, stage, ram_cache_mib=1)
    streamed = TrainingData(disk_dataset, stage, ram_cache_mib=0)
    memory = TrainingData(dataset, stage, ram_cache_mib=0)
    assert streamed.report["mode"] == memory.report["mode"] == "stream"
    assert streamed.report["retained_bytes"] == memory.report["retained_bytes"] == 0
    assert cached.counts == streamed.counts == memory.counts
    for split in (None, "train", "validation", "test"):
        expected = [song.examples[stage] for song in dataset.songs
                    if split is None or song.base.metadata["split"] == split]
        for reader in (cached, streamed, memory):
            for _ in range(2):
                examples = list(iter_stage_examples(reader, stage, split))
                assert len(examples) == len(expected)
                for actual, reference in zip(examples, expected):
                    _same_examples(actual, reference)


def test_small_positive_budget_falls_back_without_retaining_partial_cache(disk_dataset):
    from training.pipeline_v2.training_data import TrainingData

    reader = TrainingData(disk_dataset, "positions", ram_cache_mib=.001)
    assert reader.report["mode"] == "stream"
    assert reader.report["retained_bytes"] == 0
    assert reader.report["estimated_stage_bytes"] > reader.report["budget_bytes"]
    assert len(list(reader.examples())) == 4


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), True, "1", None])
def test_invalid_cache_budget_rejected_before_reading(dataset, budget):
    from training.pipeline_v2.training_data import TrainingData

    with pytest.raises(ValueError, match="budget"):
        TrainingData(dataset, "positions", ram_cache_mib=budget)


def test_unknown_stage_and_split_rejected(dataset):
    from training.pipeline_v2.training_data import TrainingData, iter_stage_examples

    with pytest.raises(ValueError, match="stage"):
        TrainingData(dataset, "unknown")
    with pytest.raises(ValueError, match="stage"):
        next(iter_stage_examples(dataset, "unknown"))
    with pytest.raises(ValueError, match="split"):
        list(TrainingData(dataset, "positions").examples("unknown"))


@pytest.mark.parametrize("field", ["file", "metadata_file"])
def test_checksum_tampering_is_rejected(disk_dataset, field):
    from training.pipeline_v2.training_data import TrainingData

    path = disk_dataset.songs.root / disk_dataset.songs.records[0][field]
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="checksum"):
        TrainingData(disk_dataset, "positions")


def _rewrite_metadata(dataset, index, change):
    record = dataset.songs.records[index]
    path = dataset.songs.root / record["metadata_file"]
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    record["metadata_sha256"] = file_hash(path)


def test_metadata_record_split_mismatch_is_rejected(disk_dataset):
    from training.pipeline_v2.training_data import TrainingData

    _rewrite_metadata(disk_dataset, 0, lambda meta: meta["metadata"].update(split="test"))
    with pytest.raises(ValueError, match="identity/split"):
        TrainingData(disk_dataset, "positions")


def test_recording_audio_cannot_leak_across_splits(disk_dataset):
    from training.pipeline_v2.training_data import TrainingData

    _rewrite_metadata(disk_dataset, 0, lambda meta: meta["metadata"].update(audio_sha256="same-recording"))
    _rewrite_metadata(disk_dataset, 1, lambda meta: meta["metadata"].update(audio_sha256="same-recording"))
    # Keep the declared aliases consistent so this exercises cross-split leakage,
    # independently of the stricter indexed-alias integrity check.
    for record in disk_dataset.songs.records[:2]:
        record["audio_sha256"] = "same-recording"
    disk_dataset.manifest.pop("stage_splits", None)
    with pytest.raises(ValueError, match="leaks"):
        TrainingData(disk_dataset, "positions")


def test_curated_benchmark_source_cannot_enter_training(disk_dataset):
    from training.pipeline_v2.training_data import TrainingData

    _rewrite_metadata(disk_dataset, 1, lambda meta: meta["metadata"].update(sources=[{
        "path": "C:\\data\\evaluate_beats\\bs_map_input\\recording\\song.egg", "role": "audio"}]))
    with pytest.raises(ValueError, match="Curated"):
        TrainingData(disk_dataset, "positions")


@pytest.mark.parametrize("field", ["file", "metadata_file"])
def test_artifact_paths_cannot_escape_dataset_directory(disk_dataset, field):
    from training.pipeline_v2.training_data import TrainingData

    disk_dataset.songs.records[0][field] = "../" + disk_dataset.songs.records[0][field]
    with pytest.raises(ValueError, match="local basenames"):
        TrainingData(disk_dataset, "positions")


def test_selected_stage_arrays_still_receive_deep_validation(disk_dataset):
    from training.pipeline_v2.training_data import TrainingData

    record = disk_dataset.songs.records[0]
    path = disk_dataset.songs.root / record["file"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["positions::inputs::audio"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    record["sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="Invalid example array"):
        TrainingData(disk_dataset, "positions")


def test_real_training_guard_rejects_structural_dummy_before_archive_read(disk_dataset, monkeypatch):
    from training.pipeline_v2.training_data import TrainingData

    def forbidden(*args, **kwargs):
        pytest.fail("Structural dummy data was opened before the origin guard")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="real prepared"):
        TrainingData(disk_dataset, "positions", require_real=True)


def test_claimed_retained_encoder_does_not_admit_synthetic_recordings(dataset):
    from training.pipeline_v2.training_data import TrainingData

    claimed = deepcopy(dataset)
    claimed.manifest["status"] = "prepared"
    claimed.manifest["encoder_identity"].update(status="retained", checkpoint_fingerprint="claimed")
    with pytest.raises(ValueError, match="Dummy examples"):
        TrainingData(claimed, "positions", require_real=True)


@pytest.mark.parametrize("field", ["file", "metadata_file"])
def test_streamed_source_changes_fail_before_reloading(disk_dataset, field):
    from training.pipeline_v2.training_data import TrainingData

    reader = TrainingData(disk_dataset, "positions", ram_cache_mib=0)
    path = disk_dataset.songs.root / disk_dataset.songs.records[0][field]
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="source changed"):
        next(reader.examples())


def test_streamed_epochs_use_pickle_safe_selected_reads_without_json_or_hash_reopens(disk_dataset, monkeypatch):
    from training.pipeline_v2.training_data import TrainingData

    reader = TrainingData(disk_dataset, "positions", ram_cache_mib=0)
    original_load = np.load
    original_get = np.lib.npyio.NpzFile.__getitem__

    def pickle_safe(path, **kwargs):
        assert kwargs == {"allow_pickle": False}
        return original_load(path, **kwargs)

    def selected_only(archive, key):
        assert key.startswith("positions::"), key
        return original_get(archive, key)

    def forbidden(*args, **kwargs):
        pytest.fail("Epoch reopened source metadata or rehashed an artifact")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(np, "load", pickle_safe)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", selected_only)
    assert len(list(reader.examples())) == len(list(reader.examples())) == 4


def test_iterator_checks_one_selected_song_at_a_time(disk_dataset):
    from training.pipeline_v2.training_data import iter_stage_examples

    later = next(record for record in reversed(disk_dataset.songs.records) if record["split"] == "train")
    with (disk_dataset.songs.root / later["file"]).open("ab") as stream:
        stream.write(b"\n")
    examples = iter_stage_examples(disk_dataset, "positions", "train")
    assert len(next(examples).times) == 5
    with pytest.raises(ValueError, match="checksum"):
        next(examples)


@pytest.mark.parametrize("budget", [0, 1])
def test_initial_scan_only_decompresses_active_stage(disk_dataset, budget, monkeypatch):
    from training.pipeline_v2.training_data import TrainingData

    original_get = np.lib.npyio.NpzFile.__getitem__

    def selected_only(archive, key):
        assert key.startswith("positions::"), key
        return original_get(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", selected_only)
    reader = TrainingData(disk_dataset, "positions", ram_cache_mib=budget)
    assert reader.counts["train"]["examples"] == 10


def test_scan_reports_stage_and_bounded_cache_progress(disk_dataset, capsys):
    from training.pipeline_v2.training_data import TrainingData

    TrainingData(disk_dataset, "positions", verbose=1)
    output = capsys.readouterr().out
    assert "positions: scanned 1/4 songs" in output
    assert "positions: scanned 4/4 songs" in output
    assert "retained=" in output and "budget=" in output


def test_public_batches_on_lazy_artifact_only_load_requested_stage(disk_dataset, monkeypatch):
    from training.pipeline_v2.runner import batches

    original_get = np.lib.npyio.NpzFile.__getitem__

    def selected_only(archive, key):
        assert key.startswith("positions::"), key
        return original_get(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", selected_only)
    examples = next(batches(disk_dataset, "positions", "train", 128))
    assert len(examples.times) == 10
