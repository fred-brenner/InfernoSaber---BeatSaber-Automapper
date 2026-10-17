"""Immutable v2/v3 object ingestion and two-pass train-vocabulary preparation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.human_maps import _check_timing, _number, _read_json, _version
from training.beat_models.contracts import REPRESENTATION
from training.beat_models.data import hash_file
from .contracts import V2Dataset, V2Song, preparation_defaults, validate
from .encoder import ENCODER_REPRESENTATION


def _integer(value, name):
    value = _number(value, name)
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _geometry(record):
    custom = record.get("customData", record.get("_customData", {}))
    if not isinstance(custom, dict):
        raise ValueError("Custom data must be an object")
    forbidden = {"position", "rotation", "localrotation", "cutdirection", "scale", "coordinates"}
    if any(key.lstrip("_").lower() in forbidden for key in custom):
        raise ValueError("Unsupported extended object geometry")


def _source(base):
    if "raw_beatmap" in base.metadata:
        return deepcopy(base.metadata["raw_beatmap"]), True
    records = [source for source in base.metadata.get("sources", []) if source.get("role") == "beatmap"]
    if records:
        record = records[0]
        path = Path(record["path"])
        if record.get("sha256") and hash_file(path) != record["sha256"]:
            raise ValueError("Source beatmap changed after beat dataset preparation")
        return _read_json(path), True
    # Old intermediate datasets retain colored objects, but absence of other
    # families cannot be interpreted as negative supervision.
    return {"version": "3.2.0", "colorNotes": deepcopy(base.metadata.get("original_note_records", []))}, False


def read_objects(base):
    """Return normalized object records and untouched per-object source JSON."""
    raw, complete = _source(base)
    version = _version(raw, ("2", "3"), "beatmap")
    _check_timing(raw)
    bpm, duration = base.metadata["bpm"], base.metadata["duration"]
    factor = 60 / bpm
    objects = {key: [] for key in ("notes", "bombs", "walls", "arcs", "chains", "lighting", "ignored")}
    indices = base.arrays["original_notes"][:, 0].astype(int)
    note_rows = {int(source): row for row, source in enumerate(indices)}
    retained = set(base.arrays["retained_indices"].tolist())
    fields = {"_notes": "notes", "_obstacles": "walls", "_events": "lighting"} if version == "2" else {
        "colorNotes": "notes", "bombNotes": "bombs", "obstacles": "walls", "sliders": "arcs",
        "burstSliders": "chains", "basicBeatmapEvents": "lighting"}
    for field, family in fields.items():
        items = raw.get(field, [])
        if not isinstance(items, list):
            objects["ignored"].append({"id": field, "family": family, "raw": deepcopy(items), "reason": "Expected an object array"})
            continue
        for index, item in enumerate(items):
            identity = f"{field}:{index}"
            record = {"id": identity, "source_index": index, "source_field": field, "raw": deepcopy(item)}
            try:
                if not isinstance(item, dict):
                    raise ValueError("Expected a source object")
                _geometry(item)
                v2 = version == "2"
                get = lambda old, new, default=0: item.get(old if v2 else new, default)
                record["time"] = _number(get("_time", "b"), "Object time") * factor
                if not 0 <= record["time"] < duration:
                    raise ValueError("Object onset is outside decoded audio")
                destination = family
                if family in ("notes", "bombs"):
                    record.update(x=_integer(get("_lineIndex", "x"), "x"), y=_integer(get("_lineLayer", "y"), "y"))
                    if not 0 <= record["x"] < 4 or not 0 <= record["y"] < 3:
                        raise ValueError("Object is outside the 4-by-3 grid")
                    hand = _integer(get("_type", "c"), "Hand") if family == "notes" else 3
                    destination = "bombs" if hand == 3 else "notes"
                    if destination == "notes":
                        if hand not in (0, 1) or index not in note_rows:
                            raise ValueError("Colored object has no existing beat-target source relationship")
                        row = note_rows[index]
                        record.update(hand=hand, d=_integer(get("_cutDirection", "d"), "Direction"),
                                      event_index=int(base.arrays["original_event_indices"][row]),
                                      original_row=row, retained=row in retained)
                elif family == "walls":
                    if v2:
                        kind = _integer(item.get("_type", 0), "Wall type")
                        if kind not in (0, 1):
                            raise ValueError("Unsupported v2 wall type")
                        y, height = (0, 5) if kind == 0 else (2, 3)
                    else:
                        y, height = _integer(item.get("y", 0), "Wall y"), _integer(item.get("h", 0), "Wall height")
                    record.update(x=_integer(get("_lineIndex", "x"), "Wall x"), y=y,
                                  width=_integer(get("_width", "w"), "Wall width"), height=height,
                                  duration=_number(get("_duration", "d"), "Wall duration") * factor)
                    if record["width"] <= 0 or height <= 0 or record["duration"] <= 0:
                        raise ValueError("Wall dimensions/duration must be positive")
                    record["end_time"] = record["time"] + record["duration"]
                    if record["end_time"] > duration + 1e-7:
                        raise ValueError("Wall end is outside decoded audio")
                elif family in ("arcs", "chains"):
                    record.update(hand=_integer(item.get("c", 0), "Hand"),
                                  x=_integer(item.get("x", 0), "Head x"), y=_integer(item.get("y", 0), "Head y"),
                                  d=_integer(item.get("d", 0), "Head direction"),
                                  end_time=_number(item.get("tb", 0), "Tail time") * factor,
                                  tx=_integer(item.get("tx", 0), "Tail x"), ty=_integer(item.get("ty", 0), "Tail y"))
                    if record["hand"] not in (0, 1) or not record["time"] < record["end_time"] < duration:
                        raise ValueError("Invalid same-hand head/tail times")
                    if any(not 0 <= record[key] < maximum for key, maximum in (("x", 4), ("tx", 4), ("y", 3), ("ty", 3), ("d", 9))):
                        raise ValueError("Unsupported head/tail geometry")
                    if family == "arcs":
                        record.update(td=_integer(item.get("tc", 0), "Tail direction"),
                                      mu=_number(item.get("mu", 1), "Head multiplier"),
                                      tmu=_number(item.get("tmu", 1), "Tail multiplier"),
                                      anchor=_integer(item.get("m", 0), "Anchor mode"))
                        if not 0 <= record["td"] <= 8 or record["anchor"] not in (0, 1, 2):
                            raise ValueError("Unsupported arc direction/anchor mode")
                    else:
                        record.update(slices=_integer(item.get("sc", 0), "Slice count"), squish=_number(item.get("s", 1), "Squish"))
                        if record["slices"] < 2 or record["squish"] <= 0:
                            raise ValueError("Invalid chain slices/squish")
                else:
                    record.update(type=_integer(get("_type", "et"), "Event type"), value=_integer(get("_value", "i"), "Event value"),
                                  float_value=_number(item.get("_floatValue", 1) if v2 else item.get("f", 1), "Event float value"))
                    if record["type"] == 100:
                        raise ValueError("Unsupported tempo-change event")
                objects[destination].append(record)
            except (ValueError, TypeError, OverflowError) as error:
                objects["ignored"].append({**record, "family": family, "reason": str(error)})
    known = set(fields) | {"_version", "version"}
    for key, value in raw.items():
        if key not in known and value not in ([], {}, None, False, ""):
            objects["ignored"].append({"id": key, "family": "unsupported_field", "raw": deepcopy(value), "reason": "Outside v2 target families"})
    return objects, {"source_objects_complete": complete, "source_version": version,
                     "ignored_records": len(objects["ignored"])}


def prepare_dataset(songs, *, settings=None, encoder=None, emit=None, collect=True, vocabularies=None):
    """Fit vocabularies on TRAIN only, then emit one song's examples at a time.

    ``encoder`` is an explicit factory(base_song)->FixedEncoder, or a mapping
    song ID -> encoder provider. Omitting it produces marked structural zeros.
    Lazy/re-iterable base datasets are supported; no expanded inputs are retained
    between songs when ``collect=False`` and ``emit`` writes them immediately.
    """
    from .targets import build_examples, fit_vocabularies, validate_vocabularies

    options = preparation_defaults(settings)
    if iter(songs) is songs:
        raise ValueError("Preparation requires a re-iterable base-song collection")
    skipped = []
    if vocabularies is None:
        vocab = fit_vocabularies(songs, options, skipped)
    else:
        validate_vocabularies(vocabularies)
        vocab = deepcopy(vocabularies)
    failed_ids = {record["song_id"] for record in skipped}
    result, records, encoder_ids = [], [], set()
    normalization_ids = set()
    for base in songs:
        if base.song_id in failed_ids:
            continue
        try:
            objects, diagnostics = read_objects(base)
            song = V2Song(base, objects, diagnostics=diagnostics)
            provider = (encoder.get(base.song_id) if isinstance(encoder, dict) else encoder(base)) if encoder is not None else None
            if encoder is not None and provider is None:
                raise ValueError(f"Missing explicit encoder provider for {base.song_id}")
            if provider is not None:
                metadata = deepcopy(getattr(provider, "metadata", {}))
                checkpoint = metadata.get("encoder_fingerprint")
                if not checkpoint:
                    raise ValueError("Encoder providers must expose metadata.encoder_fingerprint")
                song.diagnostics["encoder"] = metadata
            else:
                checkpoint = None
                song.diagnostics["encoder"] = {"status": "structural_dummy"}
            song.examples = build_examples(song, vocab, options, provider)
            validate(song)
        except (ValueError, OSError, RuntimeError) as error:
            skipped.append({"song_id": base.song_id, "sources": deepcopy(base.metadata.get("sources", [])),
                            "phase": "examples", "reason": f"{type(error).__name__}: {error}"})
            continue
        if checkpoint is not None:
            encoder_ids.add(checkpoint)
        normalization_ids.add(base.metadata.get("normalization_sha256"))
        records.append({"song_id": base.song_id, "split": base.metadata.get("split"),
                        "coverage": deepcopy(song.diagnostics.get("coverage", {})),
                        "examples": {stage: len(examples.times) for stage, examples in song.examples.items()}})
        if emit is not None:
            emit(song)
        if collect:
            result.append(song)
    if len(encoder_ids) > 1 or len(normalization_ids) > 1:
        raise ValueError("A dataset must use one encoder checkpoint and one beat-feature normalization identity")
    manifest = {"schema_version": 1, "status": "prepared" if encoder is not None else "structural_dummy",
                "encoder_identity": {"status": "retained" if encoder is not None else "structural_dummy",
                                     "checkpoint_fingerprint": next(iter(encoder_ids), None),
                                     "representation": ENCODER_REPRESENTATION,
                                     "grid_seconds": options["encoder_grid_seconds"]},
                "data_settings": options, "beat_representation": REPRESENTATION,
                "normalization_sha256": next(iter(normalization_ids), None),
                "vocabularies_fingerprint": vocab["fingerprint"], "songs": records,
                "vocabulary_origin": "fitted_train" if vocabularies is None else "explicit_frozen_train",
                "song_count": len(records), "skipped": skipped,
                "target_policy": "preserve_retained_beat_targets_mask_unsupported"}
    return V2Dataset(result, vocab, manifest)


def make_dummy_dataset(settings=None):
    from .fixtures import make_dummy_songs
    return prepare_dataset(make_dummy_songs(), settings=settings)
