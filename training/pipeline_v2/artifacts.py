"""Explicit v2 datasets/checkpoints with lazy songs and JSON/NPZ provenance."""

from collections.abc import Sequence
from collections import Counter
from copy import deepcopy
import hashlib
import json
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


def _audit_reference(reference):
    if not isinstance(reference, dict):
        raise ValueError("Preparation audit requires a file and SHA256 reference")
    filename, digest = reference.get("file"), reference.get("sha256")
    if "version" in reference and (type(reference["version"]) is not int or reference["version"] != 1):
        raise ValueError("Unsupported preparation audit reference version")
    if (not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".json")
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError("Invalid preparation audit file or SHA256 reference")
    return {"version": 1, "file": filename, "sha256": digest}


def _external_audit(audit):
    return isinstance(audit, dict) and ("file" in audit or "sha256" in audit)


def _lighting_audit_summary(lighting):
    if not isinstance(lighting, dict):
        return lighting
    summary = {key: deepcopy(value) for key, value in lighting.items()
               if key not in ("candidate_schedules", "selected", "unsupported_records")}
    for key in ("candidate_schedules", "selected"):
        if key not in lighting:
            continue
        records = lighting[key] if key == "candidate_schedules" else [lighting[key]]
        compact = [{name: deepcopy(value) for name, value in record.items() if name not in ("songs", "collisions")}
                   for record in records]
        summary[key] = compact if key == "candidate_schedules" else compact[0]
    if "unsupported_records" in lighting:
        records = lighting["unsupported_records"]
        summary["unsupported_record_count"] = len(records)
        summary["unsupported_reasons"] = dict(Counter(record.get("reason", "unspecified") for record in records))
    return summary


def compact_preparation_metadata(manifest, audit_reference):
    """Return compact metadata referencing an already preserved detailed audit.

    This pure helper never writes files or changes the supplied metadata. File
    existence and checksum validation belong to the writer or explicit loader.
    """
    reference = _audit_reference(audit_reference)
    audit = manifest.get("preparation_audit")
    if not isinstance(audit, dict) or not audit:
        raise ValueError("Cannot compact metadata without a detailed preparation audit")
    if _external_audit(audit):
        if _audit_reference(audit) != reference:
            raise ValueError("Already compact metadata refers to a different preparation audit")
        summary = deepcopy(audit)
    else:
        records = audit.get("songs", [])
        summary = {**reference, "song_count": len(records),
                   "lighting": _lighting_audit_summary(audit.get("lighting")),
                   "support": deepcopy(audit.get("support", {})),
                   "source_exclusion_count": sum(len(record.get("source_exclusions", [])) for record in records)}
    result = {**manifest, "preparation_audit": summary}
    if "preparation_songs" in manifest:
        result["preparation_songs"] = [{key: value for key, value in record.items() if key != "source_exclusions"}
                                       for record in manifest["preparation_songs"]]
    return result


def _verified_audit_path(root, reference):
    reference = _audit_reference(reference)
    path = _local(Path(root), reference["file"], ".json")
    if not path.is_file():
        raise ValueError("Missing detailed preparation audit sidecar")
    if file_hash(path) != reference["sha256"]:
        raise ValueError("Preparation audit sidecar checksum mismatch")
    return path


def load_preparation_audit(root, manifest=None):
    """Explicitly read detailed audit evidence, checking an external reference.

    Ordinary dataset loading leaves this potentially large sidecar unopened.
    Legacy inline audits and datasets without an audit remain readable.
    """
    root = Path(root)
    manifest = read_json(root / "manifest.json") if manifest is None else manifest
    audit = manifest.get("preparation_audit", {})
    if not _external_audit(audit):
        return audit
    detailed = read_json(_verified_audit_path(root, audit))
    if not isinstance(detailed, dict) or _external_audit(detailed):
        raise ValueError("Detailed preparation audit sidecar must contain the full audit")
    return detailed


def _write_preparation_audit(root, audit):
    """Stream the full audit without recursively cloning its exclusion lists."""
    path = _local(root, "preparation-audit.json", ".json")
    if path.exists():
        digest = hashlib.sha256()
        encoder = json.JSONEncoder(indent=2, sort_keys=True, allow_nan=False, default=json_value)
        for piece in encoder.iterencode(audit):
            digest.update(piece.encode("utf-8"))
        digest.update(b"\n")
        if file_hash(path) != digest.hexdigest():
            raise ValueError("Existing detailed preparation audit differs; it was not overwritten")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False, default=json_value)
            stream.write("\n")
    return {"file": path.name, "sha256": file_hash(path)}


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
                             "audio_sha256": base.metadata.get("audio_sha256"),
                             "decoded_sha256": base.metadata.get("decoded_sha256"),
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
        audit = result.get("preparation_audit")
        if isinstance(audit, dict) and audit:
            if _external_audit(audit):
                _verified_audit_path(self.root, audit)
                reference = audit
            else:
                reference = _write_preparation_audit(self.root, audit)
            result = compact_preparation_metadata(result, reference)
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
        for key in ("audio_sha256", "decoded_sha256"):
            if record.get(key) and meta["metadata"].get(key) != record[key]:
                raise ValueError("Dataset recording alias disagrees with indexed identity")
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
        if "arcs" in song.examples:
            from .arc_policy import canonical_history_inputs
            song.examples["arcs"].inputs = canonical_history_inputs(song.examples["arcs"].inputs)
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
    from .arc_policy import validate_split_overlay
    validate_split_overlay(manifest)
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
    from .policies import stage_data
    if stage not in STAGES:
        raise ValueError("Unknown v2 checkpoint stage")
    manifest = dataset.manifest if hasattr(dataset, "manifest") else dataset
    vocabularies = dataset.vocabularies if hasattr(dataset, "vocabularies") else manifest["vocabularies"]
    relevant = {"multiplier": "multiplier", "hazards": "walls", "lighting": "lighting"}.get(stage)
    result = {"schema_version": SCHEMA_VERSION, "stage": stage,
              "beat_representation": manifest.get("beat_representation"),
              "normalization_sha256": manifest.get("normalization_sha256"),
              "encoder_identity": manifest.get("encoder_identity"),
              "contract_version": 2,
              "stage_data": stage_data(manifest.get("data_settings") or {}, stage),
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
    if dataset is not None:
        from .policies import semantic_identity
        if semantic_identity(compatibility(dataset, stage)) != semantic_identity(manifest["compatibility"]):
            raise ValueError("Checkpoint does not match the selected encoder, vocabulary or dataset contract")
    return root, manifest
