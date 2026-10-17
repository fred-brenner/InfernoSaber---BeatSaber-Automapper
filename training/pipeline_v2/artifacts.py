"""Explicit v2 datasets/checkpoints with lazy songs and JSON/NPZ provenance."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from training.beat_models.artifacts import (file_hash, json_value, new_run_directory,
                                           read_json, statistics_hash, write_json,
                                           execution_identity as beat_execution_identity)
from training.beat_models.contracts import Song, validate_song
from .contracts import SCHEMA_VERSION, StageExamples, V2Dataset, V2Song, validate
from .settings import STAGES, architecture_settings


def execution_identity():
    identity = beat_execution_identity()
    identity["code_identity"] = {"package": "training.pipeline_v2",
        "files_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "config_sha256": file_hash(Path(__file__).parents[2] / "tools/config/config.py")}
    return identity


def _local(root, filename, suffix):
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(suffix):
        raise ValueError("Artifact files must be local basenames of the expected type")
    path = root / filename
    if path.resolve().parent != root.resolve():
        raise ValueError("Artifact file escapes its directory")
    return path


class DatasetWriter:
    """Write one song at a time after fitting vocabularies on training groups."""

    def __init__(self, path):
        self.root = new_run_directory(path)
        self.records = []
        self.identities = {}
        self.ids = set()

    def add(self, song):
        validate(song)
        base = song.base
        if base.song_id in self.ids:
            raise ValueError("Duplicate song ID")
        split = base.metadata.get("split")
        if split not in ("train", "validation", "test") or not base.metadata.get("group_id"):
            raise ValueError("A song needs a recording group and explicit split")
        for key in ("group_id", "audio_sha256", "decoded_sha256"):
            identity = base.metadata.get(key)
            if identity:
                token = ("group" if key == "group_id" else "audio", identity)
                if self.identities.get(token, split) != split:
                    raise ValueError("Recording identity leaks between splits")
                self.identities[token] = split
        for source in base.metadata.get("sources", []):
            path = str(source.get("path", "")).replace("\\", "/").lower()
            if "/evaluate_beats/bs_map_input/" in path and split != "test":
                raise ValueError("Curated benchmark recordings cannot enter training/validation")
        index = len(self.records)
        arrays = {f"base::{key}": value for key, value in base.arrays.items()}
        for stage, examples in song.examples.items():
            arrays[f"{stage}::times"] = examples.times
            for family in ("inputs", "targets", "weights"):
                arrays.update({f"{stage}::{family}::{key}": value
                               for key, value in getattr(examples, family).items()})
        for key, value in song.latents.items():
            if isinstance(value, np.ndarray):
                arrays[f"latents::{key}"] = value
        if any(not isinstance(a, np.ndarray) or a.dtype.hasobject for a in arrays.values()):
            raise ValueError("V2 array artifacts cannot contain executable/object payloads")
        filename, metadata_file = f"song_{index:05d}.npz", f"song_{index:05d}.json"
        np.savez_compressed(self.root / filename, **arrays)
        write_json(self.root / metadata_file, {"song_id": base.song_id, "metadata": base.metadata,
                   "objects": song.objects, "diagnostics": song.diagnostics,
                   "stages": list(song.examples), "latents_metadata": {
                       k: v for k, v in song.latents.items() if not isinstance(v, np.ndarray)}})
        self.records.append({"song_id": base.song_id, "split": split, "group_id": base.metadata["group_id"],
                             "file": filename, "sha256": file_hash(self.root / filename),
                             "metadata_file": metadata_file, "metadata_sha256": file_hash(self.root / metadata_file)})
        self.ids.add(base.song_id)

    def finish(self, vocabularies, manifest):
        if not self.records:
            write_json(self.root / "preparation-report.json", manifest)
            raise ValueError("No accepted songs; preparation-report.json contains the diagnostics")
        result = {**manifest, "kind": "pipeline_v2_dataset", "schema_version": SCHEMA_VERSION,
                  "songs": self.records, "vocabularies": vocabularies,
                  "vocabularies_sha256": statistics_hash(vocabularies),
                  "splits": {split: [r["song_id"] for r in self.records if r["split"] == split]
                             for split in ("train", "validation", "test")}, **execution_identity()}
        write_json(self.root / "manifest.json", result)
        return json_value(result)


def write_dataset(path, dataset):
    writer = DatasetWriter(path)
    for song in dataset.songs:
        writer.add(song)
    return writer.finish(dataset.vocabularies, dataset.manifest)


class _Songs(Sequence):
    def __init__(self, root, records):
        self.root, self.records = root, records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        record = self.records[index]
        path = _local(self.root, record["file"], ".npz")
        meta_path = _local(self.root, record["metadata_file"], ".json")
        if file_hash(path) != record["sha256"] or file_hash(meta_path) != record["metadata_sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {record['song_id']}")
        meta = read_json(meta_path)
        if (meta["song_id"] != record["song_id"] or meta["metadata"]["split"] != record["split"]
                or meta["metadata"]["group_id"] != record["group_id"]):
            raise ValueError("Dataset song identity/split mismatch")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        base = Song(meta["song_id"], meta["metadata"], {
            key.split("::", 1)[1]: value for key, value in arrays.items() if key.startswith("base::")})
        stages = {}
        for stage in meta["stages"]:
            families = [{key.split("::", 2)[2]: value for key, value in arrays.items()
                         if key.startswith(f"{stage}::{family}::")} for family in ("inputs", "targets", "weights")]
            stages[stage] = StageExamples(*families, arrays[f"{stage}::times"])
        latents = dict(meta.get("latents_metadata", {}))
        latents.update({key.split("::", 1)[1]: value for key, value in arrays.items() if key.startswith("latents::")})
        song = V2Song(base, meta["objects"], stages, meta["diagnostics"], latents)
        validate(song)
        return song


def load_dataset(path, lazy=True):
    root = Path(path)
    manifest = read_json(root / "manifest.json")
    if manifest.get("kind") != "pipeline_v2_dataset" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Not a compatible v2 dataset")
    vocabularies = manifest["vocabularies"]
    if statistics_hash(vocabularies) != manifest.get("vocabularies_sha256"):
        raise ValueError("V2 vocabulary fingerprint mismatch")
    records = manifest.get("songs", [])
    if not records or len({r["song_id"] for r in records}) != len(records):
        raise ValueError("Dataset song list is empty or contains duplicate IDs")
    groups = {}
    for record in records:
        split = record["split"]
        if split not in ("train", "validation", "test") or groups.get(record["group_id"], split) != split:
            raise ValueError("Invalid or leaking recording split")
        groups[record["group_id"]] = split
    expected = {s: [r["song_id"] for r in records if r["split"] == s] for s in ("train", "validation", "test")}
    if expected != manifest.get("splits"):
        raise ValueError("Split index disagrees with song records")
    songs = _Songs(root, records)
    return V2Dataset(songs if lazy else list(songs), vocabularies, manifest)


class BeatSongs(Sequence):
    """Read an existing immutable beat dataset one recording at a time."""

    def __init__(self, path):
        from training.beat_models.artifacts import validate_compatibility
        self.root = Path(path)
        self.manifest = read_json(self.root / "manifest.json")
        validate_compatibility(self.manifest)
        if self.manifest.get("kind") != "prepared_dataset":
            raise ValueError("Expected an existing beat prepared_dataset")

    def __len__(self):
        return len(self.manifest["songs"])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        record = self.manifest["songs"][index]
        path = _local(self.root, record["file"], ".npz")
        if file_hash(path) != record["sha256"]:
            raise ValueError("Base beat dataset checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            song = Song(record["song_id"], record["metadata"], {key: archive[key] for key in archive.files})
        validate_song(song)
        return song


def compatibility(dataset, stage):
    if stage not in STAGES:
        raise ValueError("Unknown v2 checkpoint stage")
    manifest = dataset.manifest if hasattr(dataset, "manifest") else dataset
    vocabularies = dataset.vocabularies if hasattr(dataset, "vocabularies") else manifest["vocabularies"]
    relevant = {"multiplier": "multiplier", "hazards": "walls", "lighting": "lighting"}.get(stage)
    result = {"schema_version": SCHEMA_VERSION, "stage": stage,
              "beat_representation": manifest.get("beat_representation"),
              "normalization_sha256": manifest.get("normalization_sha256"),
              "encoder_identity": manifest.get("encoder_identity"),
              "data_settings": manifest.get("data_settings"),
              "vocabulary": vocabularies.get(relevant) if relevant else None,
              "architecture": architecture_settings(stage)}
    result = json_value(result)
    result["sha256"] = statistics_hash(result)
    return result


def save_checkpoint_manifest(output, stage, dataset, status, checkpoint="best.keras", extra=None):
    if status not in ("untrained", "trained"):
        raise ValueError("Checkpoint status must be untrained or trained")
    root = Path(output)
    path = _local(root, checkpoint, ".keras")
    if not path.is_file():
        raise ValueError("Checkpoint must exist before recording its manifest")
    manifest = {"kind": "pipeline_v2_checkpoint", "schema_version": SCHEMA_VERSION, "stage": stage,
                "status": status, "checkpoint": checkpoint, "checkpoint_sha256": file_hash(path),
                "compatibility": compatibility(dataset, stage), "vocabularies": dataset.vocabularies,
                "training": extra or {}, **execution_identity()}
    write_json(root / "manifest.json", manifest)
    return manifest


def load_checkpoint_manifest(path, stage, dataset=None):
    supplied = Path(path)
    root = supplied.parent if supplied.is_file() else supplied
    manifest = read_json(root / "manifest.json")
    if (manifest.get("kind") != "pipeline_v2_checkpoint" or manifest.get("stage") != stage
            or manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") not in ("untrained", "trained")):
        raise ValueError("Wrong stage, status or version for v2 checkpoint")
    model_path = _local(root, manifest["checkpoint"], ".keras")
    if supplied.is_file() and supplied.name not in ("manifest.json", model_path.name):
        raise ValueError("Selected checkpoint differs from the manifest")
    if file_hash(model_path) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    saved = dict(manifest["compatibility"])
    digest = saved.pop("sha256", None)
    if saved.get("stage") != stage or saved.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Checkpoint and compatibility identities disagree")
    relevant = {"multiplier": "multiplier", "hazards": "walls", "lighting": "lighting"}.get(stage)
    if relevant and saved.get("vocabulary") != manifest.get("vocabularies", {}).get(relevant):
        raise ValueError("Checkpoint vocabulary disagrees with its compatibility identity")
    if digest != statistics_hash(saved) or saved.get("architecture") != json_value(architecture_settings(stage)):
        raise ValueError("Checkpoint architecture/configuration fingerprint mismatch")
    if dataset is not None and compatibility(dataset, stage) != manifest["compatibility"]:
        raise ValueError("Checkpoint does not match the selected encoder, vocabulary or dataset contract")
    return root, manifest
