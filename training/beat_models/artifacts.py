"""Versioned JSON/NPZ artifacts, independent of TensorFlow and application paths."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .contracts import FEATURES, REPRESENTATION, SCHEMA_VERSION, STAGES, Song, validate_song


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def statistics_hash(statistics):
    encoded = json.dumps(json_value(statistics), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def new_run_directory(path):
    """Every operation needs a fresh explicit destination; never replace a run."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    return root


def execution_identity():
    packages = {}
    for package in ("numpy", "librosa", "aubio", "tensorflow", "tf-keras"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    package_root = Path(__file__).parent
    return {"runtime": {"python": sys.version, "platform": platform.platform(), "packages": packages},
            "code_identity": {"package": "training.beat_models",
                              "files_sha256": {path.name: file_hash(path) for path in sorted(package_root.glob("*.py"))}}}


def validate_statistics(statistics):
    if not isinstance(statistics, dict) or statistics.get("version") != 1 or statistics.get("fit_split") != "train":
        raise ValueError("Unsupported normalization statistics")
    identifiers = statistics.get("song_ids")
    if not isinstance(identifiers, list) or not identifiers or any(not isinstance(identifier, str) or not identifier for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Normalization must record its training songs")
    for key in ("features", "coarse_features"):
        record = statistics.get(key, {})
        mean, scale = np.asarray(record.get("mean", [])), np.asarray(record.get("scale", []))
        if mean.dtype.kind not in "fiu" or scale.dtype.kind not in "fiu":
            raise ValueError("Nonnumeric normalization channel statistics")
        if mean.shape != (FEATURES,) or scale.shape != (FEATURES,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid normalization channel statistics")
        count = record.get("count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Invalid normalization observation count")
    return True


def validate_compatibility(manifest, statistics=None):
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("representation") != REPRESENTATION:
        raise ValueError("Incompatible artifact schema or representation")
    stored = manifest.get("normalization")
    validate_statistics(stored)
    if stored is None or manifest.get("normalization_sha256") != statistics_hash(stored):
        raise ValueError("Missing or corrupt normalization statistics")
    if statistics is not None and statistics_hash(stored) != statistics_hash(statistics):
        raise ValueError("Incompatible normalization statistics")
    return True


def _validate_splits(songs, preparation=None, statistics=None):
    identities = {}
    reserved_hashes = {source.get("sha256") for record in (preparation or {}).get("reserved", [])
                       for source in record.get("sources", []) if source.get("role") == "audio"}
    for song in songs:
        meta, split = song.metadata, song.metadata.get("split")
        if statistics and (meta.get("normalization_version") != statistics["version"] or
                           meta.get("normalization_sha256") != statistics_hash(statistics)):
            raise ValueError("Song normalization provenance disagrees with manifest statistics")
        if split not in ("train", "validation", "test") or not meta.get("group_id"):
            raise ValueError("Every prepared song needs a recording group and explicit split")
        for field in ("group_id", "audio_sha256", "decoded_sha256"):
            identity = meta.get(field)
            if identity:
                token = ("group" if field == "group_id" else "audio", identity)
                if token in identities and identities[token] != split:
                    raise ValueError("Recording identity leaks across dataset splits")
                identities[token] = split
        for source in meta.get("sources", []):
            digest = source.get("sha256", "")
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("Source records require SHA256 provenance")
            path = str(source.get("path", "")).replace("\\", "/").lower()
            curated = "/evaluate_beats/bs_map_input/" in f"/{path.strip('/')}/"
            if split != "test" and (curated or digest in reserved_hashes):
                raise ValueError("Curated evaluation recording cannot enter learning splits")
        if statistics and song.song_id in statistics["song_ids"] and split != "train":
            raise ValueError("Normalization training songs occur in another split")


def write_dataset(path, songs, stats=None, metadata=None):
    songs = list(songs)
    if not songs:
        raise ValueError("A prepared dataset must contain songs")
    if stats is None:
        raise ValueError("Prepared datasets require training normalization statistics")
    validate_statistics(stats)
    for song in songs:
        validate_song(song)
    if len({song.song_id for song in songs}) != len(songs):
        raise ValueError("Duplicate song identifiers")
    _validate_splits(songs, metadata, stats)
    root = new_run_directory(path)
    records = []
    for index, song in enumerate(songs):
        filename = f"song_{index:05d}.npz"
        np.savez_compressed(root / filename, **song.arrays)
        records.append({"song_id": song.song_id, "file": filename, "sha256": file_hash(root / filename),
                        "metadata": song.metadata})
    manifest = {"kind": "prepared_dataset", "schema_version": SCHEMA_VERSION,
                "representation": REPRESENTATION, "normalization": stats,
                "normalization_sha256": statistics_hash(stats), "songs": records,
                "splits": {split: [song.song_id for song in songs if song.metadata.get("split") == split]
                           for split in ("train", "validation", "test")},
                "sources": [source for song in songs for source in song.metadata.get("sources", [])],
                "preparation": metadata or {},
                "status": "dummy" if all(song.metadata.get("provenance", {}).get("adapter") == "deterministic_dummy" for song in songs) else "prepared",
                **execution_identity()}
    write_json(root / "manifest.json", manifest)
    return json_value(manifest)


def load_dataset(path):
    root = Path(path)
    manifest = read_json(root / "manifest.json")
    validate_compatibility(manifest)
    if manifest.get("kind") != "prepared_dataset" or not manifest.get("songs"):
        raise ValueError("Not a prepared dataset")
    songs = []
    for record in manifest["songs"]:
        filename = record["file"]
        if Path(filename).name != filename or not filename.endswith(".npz"):
            raise ValueError("Invalid dataset array filename")
        array_path = root / filename
        if file_hash(array_path) != record["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {filename}")
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        song = Song(record["song_id"], record["metadata"], arrays)
        validate_song(song)
        songs.append(song)
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
    return songs, manifest


def load_statistics(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json" if path.is_dir() else path)
    validate_compatibility(manifest)
    return manifest["normalization"]


def write_model_manifest(path, stage, checkpoint, dataset_manifest, metadata=None):
    """Record an explicitly saved checkpoint beside its compatible data contract."""
    if stage not in STAGES:
        raise ValueError("Unknown model stage")
    validate_compatibility(dataset_manifest)
    path, checkpoint = Path(path), Path(checkpoint)
    if not checkpoint.is_file() or checkpoint.parent.resolve() != path.resolve():
        raise ValueError("Checkpoint must exist directly inside the explicit run directory")
    status = (metadata or {}).get("status", "untrained")
    if status not in ("untrained", "trained"):
        raise ValueError("Checkpoint status must explicitly identify untrained or trained weights")
    manifest = {"kind": "model_checkpoint", "schema_version": SCHEMA_VERSION,
                "representation": REPRESENTATION, "stage": stage,
                "checkpoint": checkpoint.name, "checkpoint_sha256": file_hash(checkpoint),
                "normalization": dataset_manifest["normalization"],
                "normalization_sha256": dataset_manifest["normalization_sha256"],
                "sources": dataset_manifest.get("sources", []), "splits": dataset_manifest.get("splits", {}),
                "training": metadata or {}, "status": status, **execution_identity()}
    write_json(path / "manifest.json", manifest)
    return manifest


def load_model_manifest(path, stage=None, dataset_manifest=None):
    """No discovery or 'latest' fallback: the caller identifies the checkpoint run."""
    supplied = Path(path)
    root = supplied.parent if supplied.is_file() else supplied
    manifest = read_json(root / "manifest.json")
    validate_compatibility(manifest, dataset_manifest["normalization"] if dataset_manifest else None)
    if manifest.get("kind") != "model_checkpoint" or manifest.get("stage") not in STAGES:
        raise ValueError("Not a stage checkpoint artifact")
    if manifest.get("status") not in ("untrained", "trained"):
        raise ValueError("Checkpoint needs an explicit training status")
    if stage is not None and manifest["stage"] != stage:
        raise ValueError("Checkpoint has the wrong stage")
    filename = manifest.get("checkpoint", "")
    if not filename or Path(filename).name != filename:
        raise ValueError("Invalid checkpoint filename")
    if supplied.is_file() and supplied.name != filename:
        raise ValueError("Explicit checkpoint filename disagrees with manifest")
    checkpoint = root / filename
    if file_hash(checkpoint) != manifest.get("checkpoint_sha256"):
        raise ValueError("Checkpoint checksum mismatch")
    return checkpoint, manifest
