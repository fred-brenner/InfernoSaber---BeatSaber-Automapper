"""Load only the persisted arrays needed by intensity or timing training.

NPZ checksums retain the prepared artifact's provenance guarantee. Validation
here checks the selected stage's targets; the ordinary artifact reader remains
responsible for full original-note provenance and evaluation validation.
"""

from numbers import Real
from pathlib import Path, PureWindowsPath
from time import perf_counter

import numpy as np

from .artifacts import _validate_splits, file_hash, read_json, validate_compatibility
from .contracts import BIN_SECONDS, CLASSES, FEATURES, FRAME_SECONDS, Song


STAGE_ARRAYS = {
    "intensity": ("coarse_features", "bin_durations", "intensity"),
    "timing": ("features", "coarse_features", "frame_times", "bin_durations",
               "intensity", "timing", "note_classes", "event_times", "event_frames"),
}


def _manifest_songs(manifest):
    """Keep the full reader's cross-song and declared-index integrity guards."""
    validate_compatibility(manifest)
    records = manifest.get("songs")
    if manifest.get("kind") != "prepared_dataset" or not isinstance(records, list) or not records:
        raise ValueError("Not a prepared dataset")
    songs = []
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("song_id"), str)
                or not record["song_id"] or not isinstance(record.get("metadata"), dict)):
            raise ValueError("Invalid dataset song identity/metadata")
        songs.append(Song(record["song_id"], record["metadata"]))
    if len({song.song_id for song in songs}) != len(songs):
        raise ValueError("Duplicate song identifiers")
    _validate_splits(songs, manifest.get("preparation"), manifest["normalization"])
    expected_splits = {split: [song.song_id for song in songs if song.metadata.get("split") == split]
                       for split in ("train", "validation", "test")}
    if manifest.get("splits") != expected_splits:
        raise ValueError("Dataset split manifest mismatch")
    expected_sources = [source for song in songs for source in song.metadata.get("sources", [])]
    if manifest.get("sources") != expected_sources:
        raise ValueError("Dataset source manifest mismatch")
    return songs


def _array_path(root, filename):
    if (not isinstance(filename, str) or not filename.endswith(".npz") or ":" in filename
            or Path(filename).name != filename or PureWindowsPath(filename).name != filename
            or PureWindowsPath(filename).drive):
        raise ValueError("Invalid dataset array filename")
    path = root / filename
    if path.resolve().parent != root.resolve():
        raise ValueError("Dataset array filename escapes its directory")
    return path


def _validate_arrays(song, stage):
    arrays, metadata = song.arrays, song.metadata
    duration, bpm = metadata.get("duration"), metadata.get("bpm")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
           or value <= 0 for value in (duration, bpm)):
        raise ValueError("Song requires a positive duration and BPM")
    for key in STAGE_ARRAYS[stage]:
        array = arrays[key]
        if (not isinstance(array, np.ndarray) or array.dtype.kind not in "fiu"
                or not np.isfinite(array).all()):
            raise ValueError(f"Invalid array: {key}")
    durations, shares = arrays["bin_durations"], arrays["intensity"]
    bins = len(durations) if durations.ndim == 1 else 0
    if (not bins or arrays["coarse_features"].shape != (bins, FEATURES)
            or shares.shape != (bins,)):
        raise ValueError("Wrong intensity/bin/coarse_features shape")
    for key in ("coarse_features", "bin_durations", "intensity"):
        if arrays[key].dtype != np.float32:
            raise ValueError(f"{key} must use float32")
    expected_bins = int(np.ceil(duration / BIN_SECONDS))
    if (bins != expected_bins or np.any(durations <= 0)
            or np.any(durations > BIN_SECONDS + 1e-6)
            or not np.isclose(durations.sum(), duration, rtol=1e-6, atol=1e-5)
            or not np.allclose(durations, np.minimum(BIN_SECONDS, duration - np.arange(bins) * BIN_SECONDS),
                               rtol=1e-6, atol=1e-6)):
        raise ValueError("Invalid bin durations")
    valid = metadata.get("intensity_target_valid")
    if (not isinstance(valid, bool) or np.any(shares < 0)
            or not np.isclose(shares.sum(), 1.0 if valid else 0.0, rtol=1e-6, atol=1e-6)):
        raise ValueError("Invalid intensity target mass/mask")
    if stage == "timing":
        _validate_timing(arrays, duration, valid)


def _validate_timing(arrays, duration, intensity_valid):
    times, events = arrays["frame_times"], arrays["event_times"]
    frames = len(times) if times.ndim == 1 else 0
    count = len(events) if events.ndim == 1 else -1
    expected_shapes = {"features": (frames, FEATURES), "frame_times": (frames,),
                       "timing": (frames,), "event_times": (count,),
                       "event_frames": (count,), "note_classes": (count, 2)}
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(f"Wrong {key} shape")
    if arrays["features"].dtype != np.float32:
        raise ValueError("features must use float32")
    if (not frames or np.any(times < 0) or np.any(times >= duration)
            or frames * FRAME_SECONDS < duration - 1e-9
            or not np.allclose(times, np.arange(frames) * FRAME_SECONDS, rtol=0, atol=1e-9)):
        raise ValueError("Frames must cover the song on the sample-derived time base")
    for key, upper in (("timing", 3), ("note_classes", CLASSES), ("event_frames", frames)):
        if arrays[key].dtype.kind not in "iu" or np.any(arrays[key] < 0) or np.any(arrays[key] >= upper):
            raise ValueError(f"Invalid integer categories: {key}")
    event_frames = arrays["event_frames"]
    if (np.any(np.diff(events) <= 0) or np.any(events < 0) or np.any(events >= duration)
            or np.any(np.diff(event_frames.astype(np.int64, copy=False)) <= 0)):
        raise ValueError("Events must be ordered, distinct and inside the song")
    nearest = np.minimum(np.floor(events / FRAME_SECONDS + .5).astype(np.int64), frames - 1)
    if not np.array_equal(nearest, event_frames):
        raise ValueError("Event labels must use nearest frames with half-frame ties later")
    counts = np.count_nonzero(arrays["note_classes"], axis=1)
    if (np.any(counts == 0) or not np.array_equal(np.flatnonzero(arrays["timing"]), event_frames)
            or not np.array_equal(arrays["timing"][event_frames], counts)):
        raise ValueError("Timing and note counts disagree")
    if intensity_valid != (count > 0):
        raise ValueError("Invalid intensity target mass/mask")
    expected_intensity = np.zeros(len(arrays["intensity"]), dtype=np.float64)
    bins = np.minimum(np.floor(events / BIN_SECONDS).astype(np.int64), len(expected_intensity) - 1)
    np.add.at(expected_intensity, bins, counts)
    if expected_intensity.sum():
        expected_intensity /= expected_intensity.sum()
    if not np.allclose(expected_intensity, arrays["intensity"], rtol=1e-6, atol=1e-7):
        raise ValueError("Intensity target disagrees with retained colored-note counts")


def load_training_dataset(path, stage, verbose=1):
    """Check each NPZ once and retain only arrays used by the selected stage.

    Global reference shuffling reuses this RAM cache for every epoch. Selected
    NPZ members already own their decoded storage, so no second array copy is
    needed. No prepared files or original map records are modified.
    """
    if stage not in STAGE_ARRAYS:
        raise ValueError("Selective training input supports intensity and timing only")
    started = perf_counter()
    root = Path(path)
    manifest = read_json(root / "manifest.json")
    songs = _manifest_songs(manifest)
    retained_bytes = 0
    for index, (record, song) in enumerate(zip(manifest["songs"], songs), 1):
        array_path = _array_path(root, record.get("file"))
        if file_hash(array_path) != record.get("sha256"):
            raise ValueError(f"Dataset checksum mismatch: {record['file']}")
        archive = np.load(array_path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("Prepared song arrays must use NPZ archives")
        with archive:
            if not all(key in archive.files for key in STAGE_ARRAYS[stage]):
                raise ValueError(f"Missing {stage} training arrays: {record['file']}")
            song.arrays = {key: archive[key] for key in STAGE_ARRAYS[stage]}
        _validate_arrays(song, stage)
        retained_bytes += sum(array.nbytes for array in song.arrays.values())
        if verbose and (index == 1 or index % 25 == 0 or index == len(songs)):
            print(f"Training input ({stage}): {index}/{len(songs)} songs checked; "
                  f"{retained_bytes / 1024 ** 2:.1f} MiB retained", flush=True)
    result = dict(manifest)
    result["training_input"] = {"stage": stage, "array_keys": list(STAGE_ARRAYS[stage]),
                                "retained_bytes": retained_bytes, "songs": len(songs),
                                "seconds": perf_counter() - started, "mode": "ram",
                                "integrity_passes": 1}
    return songs, result
