"""Selective prepared-input checks without TensorFlow or optimizer updates."""

from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from training.beat_models import artifacts, runner
from training.beat_models.contracts import CORE, FEATURES, FRAME_SECONDS
from training.beat_models.data import build_song
from training.beat_models.training_data import STAGE_ARRAYS, load_training_dataset


def _statistics():
    channels = {"mean": [0.] * FEATURES, "scale": [1.] * FEATURES, "count": 10}
    return {"version": 1, "fit_split": "train", "song_ids": ["train"],
            "features": deepcopy(channels), "coarse_features": deepcopy(channels)}


def _song(identifier, duration, split, empty=False):
    times = np.arange(int(np.ceil(duration / FRAME_SECONDS))) * FRAME_SECONDS
    times = times[times < duration]
    bins = np.minimum(1., duration - np.arange(int(np.ceil(duration)))).astype(np.float32)
    features = np.repeat(times[:, None], FEATURES, axis=1).astype(np.float32)
    rows = []
    if not empty:
        for index, frame in enumerate(np.unique([1, len(times) // 2, len(times) - 1])):
            for color in ((0, 1) if index == 1 else (index % 2,)):
                rows.append([len(rows), times[frame] * 2, times[frame], color, index % 4, 1, 3, 0])
    metadata = {"duration": duration, "bpm": 120., "split": split, "group_id": identifier,
                "normalization_version": 1, "normalization_sha256": artifacts.statistics_hash(_statistics()),
                "sources": [{"path": f"untouched/{identifier}/song.egg", "role": "audio",
                             "sha256": ("a" if identifier == "train" else "b") * 64}]}
    return build_song(identifier, rows, {"features": features, "frame_times": times,
                                       "coarse_features": features[:len(bins)].copy(),
                                       "bin_durations": bins}, metadata)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "prepared"
    songs = [_song("train", 6.137, "train"), _song("short", .073, "validation"),
             _song("silent", 2.011, "test", empty=True)]
    # Distinct recordings are required even when the source file basenames match.
    songs[-1].metadata["sources"][0]["sha256"] = "c" * 64
    artifacts.write_dataset(root, songs, _statistics())
    return root


def _edit_manifest(root, change):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


def _edit_arrays(root, change):
    path = root / "song_00000.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    change(arrays)
    np.savez_compressed(path, **arrays)
    _edit_manifest(root, lambda manifest: manifest["songs"][0].update(sha256=artifacts.file_hash(path)))


def _assert_equal(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_equal(first[key], second[key])
    elif isinstance(first, tuple):
        for left, right in zip(first, second):
            _assert_equal(left, right)
    else:
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("stage", STAGE_ARRAYS)
def test_reads_only_selected_members_once_and_keeps_original_json(dataset, monkeypatch, stage):
    original_files = {path.name: path.read_bytes() for path in dataset.iterdir()}
    full, manifest = artifacts.load_dataset(dataset)
    getitem = np.lib.npyio.NpzFile.__getitem__
    hash_file = artifacts.file_hash
    reads, hashes, decoded = [], [], []

    def tracked(archive, key):
        reads.append(key)
        value = getitem(archive, key)
        decoded.append(value)
        return value

    def hashed(path):
        hashes.append(path.name)
        return hash_file(path)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", tracked)
    monkeypatch.setattr("training.beat_models.training_data.file_hash", hashed)
    songs, selected = load_training_dataset(dataset, stage, verbose=0)
    assert reads == list(STAGE_ARRAYS[stage]) * len(songs)
    assert hashes == [f"song_{index:05d}.npz" for index in range(len(songs))]
    assert all(value is decoded[index] for index, value in enumerate(
        array for song in songs for array in song.arrays.values()))
    assert all(set(song.arrays) == set(STAGE_ARRAYS[stage]) for song in songs)
    assert not {"original_notes", "retained_indices", "original_event_indices"}.intersection(reads)
    if stage == "intensity":
        assert "features" not in reads and "event_times" not in reads and "note_classes" not in reads
    assert {key: value for key, value in selected.items() if key != "training_input"} == manifest
    assert selected["songs"][0]["metadata"] is songs[0].metadata
    assert "original_note_records" in songs[0].metadata
    report = selected["training_input"]
    assert report["stage"] == stage and report["array_keys"] == list(STAGE_ARRAYS[stage])
    assert report["songs"] == len(songs) and report["seconds"] >= 0
    assert report["mode"] == "ram" and report["integrity_passes"] == 1
    assert report["retained_bytes"] == sum(array.nbytes for song in songs for array in song.arrays.values())
    assert report["retained_bytes"] < sum(array.nbytes for song in full for array in song.arrays.values())
    json.dumps(report, allow_nan=False)
    assert original_files == {path.name: path.read_bytes() for path in dataset.iterdir()}


@pytest.mark.parametrize("stage", STAGE_ARRAYS)
def test_training_batches_equal_full_reader_including_silence_short_and_fractional_tails(dataset, stage):
    full, manifest = artifacts.load_dataset(dataset)
    selected, selected_manifest = load_training_dataset(dataset, stage, verbose=0)
    assert runner._normalized(selected, selected_manifest) == selected
    full_profiles = runner._profiles(full) if stage == "timing" else {}
    selected_profiles = runner._profiles(selected) if stage == "timing" else {}
    _assert_equal(full_profiles, selected_profiles)
    full_refs = runner._references(full, stage)
    selected_refs = runner._references(selected, stage)
    assert [(song.song_id, index) for song, index in full_refs] == [
        (song.song_id, index) for song, index in selected_refs]
    _assert_equal(runner._batch(stage, full_refs, full_profiles),
                  runner._batch(stage, selected_refs, selected_profiles))
    if stage == "intensity":
        assert len(selected_refs) == 2
        inputs, _, _ = runner._batch(stage, selected_refs, {})
        assert not inputs["mask"][1, 1:].any()
        assert inputs["durations"][1, 0] == pytest.approx(.073)
    else:
        assert len(selected_refs) == sum(int(np.ceil(len(song.arrays["features"]) / CORE)) for song in full)
        inputs, labels, mask = runner._sample(stage, selected[1], 0, selected_profiles)
        assert mask.sum() == 8 and not labels[~mask].any()
        assert not inputs["audio"][~inputs["mask"]].any()
    assert selected_manifest["normalization_sha256"] == manifest["normalization_sha256"]


@pytest.mark.parametrize("stage", STAGE_ARRAYS)
def test_checksum_corruption_rejected_before_reading_arrays(dataset, monkeypatch, stage):
    with (dataset / "song_00000.npz").open("ab") as stream:
        stream.write(b"tamper")
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Corrupt archive was opened"))
    with pytest.raises(ValueError, match="checksum"):
        load_training_dataset(dataset, stage, verbose=0)


@pytest.mark.parametrize("filename", ["../outside.npz", "sub/song.npz", r"sub\song.npz",
                                      r"C:\outside.npz", "/outside.npz", "song.npy", "C:song.npz",
                                      "song:stream.npz"])
def test_unsafe_array_paths_rejected(dataset, filename):
    _edit_manifest(dataset, lambda manifest: manifest["songs"][0].update(file=filename))
    with pytest.raises(ValueError, match="filename"):
        load_training_dataset(dataset, "intensity", verbose=0)


def test_symlink_to_outside_arrays_rejected(dataset, tmp_path):
    path = dataset / "song_00000.npz"
    outside = tmp_path / "external.npz"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        load_training_dataset(dataset, "intensity", verbose=0)


def test_numpy_array_disguised_as_npz_rejected(dataset):
    path = dataset / "song_00000.npz"
    with path.open("wb") as stream:
        np.save(stream, np.zeros(1))
    _edit_manifest(dataset, lambda manifest: manifest["songs"][0].update(sha256=artifacts.file_hash(path)))
    with pytest.raises(ValueError, match="NPZ archives"):
        load_training_dataset(dataset, "intensity", verbose=0)


@pytest.mark.parametrize("stage", STAGE_ARRAYS)
@pytest.mark.parametrize("change,expected", [
    (lambda m: m.update(schema_version=999), "schema"),
    (lambda m: m["representation"].update(target_version=999), "representation"),
    (lambda m: m.update(normalization_sha256="0" * 64), "normalization"),
    (lambda m: m["songs"][0]["metadata"].pop("normalization_version"), "normalization"),
    (lambda m: m["songs"][0]["metadata"].update(normalization_sha256="0" * 64), "normalization"),
    (lambda m: m["songs"][0]["metadata"].update(split="test"), "Normalization training"),
    (lambda m: m["songs"][1].update(song_id="train"), "Duplicate"),
    (lambda m: m["splits"].update(train=[]), "split manifest"),
    (lambda m: m.update(sources=[]), "source manifest"),
    (lambda m: m["songs"][0]["metadata"]["sources"][0].update(sha256="bad"), "SHA256"),
    (lambda m: m["songs"][0]["metadata"]["sources"][0].update(
        path=r"C:\Data\evaluate_beats\bs_map_input\map\song.egg"), "Curated"),
    (lambda m: m.update(preparation={"reserved": [{"sources": [{"role": "audio", "sha256": "a" * 64}]}]}), "Curated"),
])
def test_manifest_integrity_guards(dataset, stage, change, expected):
    _edit_manifest(dataset, change)
    with pytest.raises(ValueError, match=expected):
        load_training_dataset(dataset, stage, verbose=0)


@pytest.mark.parametrize("identity", ["group_id", "audio_sha256", "decoded_sha256"])
def test_cross_split_recording_leaks_rejected(dataset, identity):
    def change(manifest):
        for record in manifest["songs"][:2]:
            record["metadata"][identity] = "same-recording"
    _edit_manifest(dataset, change)
    with pytest.raises(ValueError, match="leaks"):
        load_training_dataset(dataset, "intensity", verbose=0)


@pytest.mark.parametrize("field,value", [("fit_split", "test"), ("song_ids", []), ("version", 2)])
def test_rehashed_invalid_statistics_rejected(dataset, field, value):
    def change(manifest):
        manifest["normalization"][field] = value
        manifest["normalization_sha256"] = artifacts.statistics_hash(manifest["normalization"])
    _edit_manifest(dataset, change)
    with pytest.raises(ValueError, match="normalization|Normalization"):
        load_training_dataset(dataset, "intensity", verbose=0)


@pytest.mark.parametrize("stage", STAGE_ARRAYS)
@pytest.mark.parametrize("change,expected", [
    (lambda a: a.update(coarse_features=a["coarse_features"].astype(np.float64)), "float32"),
    (lambda a: a["coarse_features"].__setitem__((0, 0), np.nan), "Invalid array"),
    (lambda a: a.update(coarse_features=a["coarse_features"][:, :2]), "shape"),
    (lambda a: a.update(bin_durations=a["bin_durations"][:, None]), "shape"),
    (lambda a: a["bin_durations"].__setitem__(0, 0), "durations"),
    (lambda a: a["bin_durations"].__setitem__(-1, .5), "durations"),
    (lambda a: a["intensity"].__setitem__(0, -.1), "target mass"),
    (lambda a: a.update(intensity=np.zeros_like(a["intensity"])), "target mass"),
    (lambda a: a.update(intensity=a["intensity"].astype(object)), "Object arrays"),
    (lambda a: a.pop("intensity"), "Missing"),
])
def test_selected_intensity_array_contracts(dataset, stage, change, expected):
    _edit_arrays(dataset, change)
    with pytest.raises(ValueError, match=expected):
        load_training_dataset(dataset, stage, verbose=0)


@pytest.mark.parametrize("field,value", [("duration", 0), ("bpm", -1), ("bpm", True),
                                       ("intensity_target_valid", None), ("intensity_target_valid", 1),
                                       ("intensity_target_valid", False)])
def test_invalid_training_metadata_rejected(dataset, field, value):
    _edit_manifest(dataset, lambda manifest: manifest["songs"][0]["metadata"].update({field: value}))
    with pytest.raises(ValueError, match="positive|target mass"):
        load_training_dataset(dataset, "intensity", verbose=0)


@pytest.mark.parametrize("change,expected", [
    (lambda a: a.update(features=a["features"].astype(np.float64)), "float32"),
    (lambda a: a["features"].__setitem__((0, 0), np.inf), "Invalid array"),
    (lambda a: a["frame_times"].__setitem__(0, .001), "time base"),
    (lambda a: a.update(timing=a["timing"].astype(np.float32)), "integer categories"),
    (lambda a: a["timing"].__setitem__(0, 3), "integer categories"),
    (lambda a: a["note_classes"].__setitem__((0, 0), 109), "integer categories"),
    (lambda a: a["event_frames"].__setitem__(0, 0), "nearest frames"),
    (lambda a: a["event_times"].__setitem__(1, a["event_times"][0]), "ordered"),
    (lambda a: a["event_times"].__setitem__(-1, 10), "inside the song"),
    (lambda a: a["timing"].__setitem__(a["event_frames"][0], 2), "counts disagree"),
    (lambda a: a.update(intensity=np.roll(a["intensity"], 1)), "retained colored-note counts"),
    (lambda a: a.update(event_frames=a["event_frames"][:, None]), "shape"),
])
def test_selected_timing_array_contracts(dataset, change, expected):
    _edit_arrays(dataset, change)
    with pytest.raises(ValueError, match=expected):
        load_training_dataset(dataset, "timing", verbose=0)


def test_progress_is_bounded_and_can_be_disabled(dataset, capsys):
    load_training_dataset(dataset, "intensity")
    output = capsys.readouterr().out
    assert "1/3 songs checked" in output and "3/3 songs checked" in output
    assert len(output.splitlines()) == 2
    load_training_dataset(dataset, "intensity", verbose=0)
    assert not capsys.readouterr().out


def test_only_beat_pair_stages_supported(dataset):
    with pytest.raises(ValueError, match="intensity and timing"):
        load_training_dataset(dataset, "notes", verbose=0)


def test_import_does_not_load_tensorflow():
    code = ("import sys; import training.beat_models.training_data; "
            "assert not any(k == 'tensorflow' or k.startswith('tensorflow.') for k in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
