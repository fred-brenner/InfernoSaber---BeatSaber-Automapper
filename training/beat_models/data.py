"""Immutable beatmap ingestion, first-cut targets, and recording-level splits."""

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from evaluation.human_maps import _check_timing, _number, _package_file, _read_json, _version
from tools.config import config
from .contracts import FRAME_SECONDS, SAMPLE_RATE, SEED, Song, encode_note, synchronize_events, validate_song
from .features import decode_audio, extract_features, fit_normalization, normalize_songs


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _objects(data, key, optional=False):
    items = data.get(key, [] if optional else None)
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"Expected {key} array of objects")
    return items


def _info(path):
    info = _read_json(path)
    _version(info, ("2",), "Info.dat")
    _check_timing(info)
    bpm = _number(info.get("_beatsPerMinute"), "BPM")
    if bpm <= 0:
        raise ValueError("BPM must be positive")
    return info, bpm


def _note_records(path, bpm):
    """Read original colored records in source order; v3 scalars default to zero."""
    data = _read_json(Path(path))
    version = _version(data, ("2", "3"), "beatmap")
    _check_timing(data)
    for event in _objects(data, "_events" if version == "2" else "basicBeatmapEvents", True):
        if event.get("_type" if version == "2" else "et", 0) == 100:
            raise ValueError("Unsupported tempo change lighting event type 100")
    keys = ("_time", "_type", "_lineIndex", "_lineLayer", "_cutDirection") if version == "2" else ("b", "c", "x", "y", "d")
    rows, records = [], []
    for index, note in enumerate(_objects(data, "_notes" if version == "2" else "colorNotes", version == "3")):
        values = [_number(note.get(key, 0) if version == "3" else note.get(key), key) for key in keys]
        beat, color, x, y, direction = values
        if beat < 0 or not color.is_integer() or color not in ((0, 1, 3) if version == "2" else (0, 1)):
            raise ValueError(f"Unsupported note time/color at source index {index}")
        if color == 3:
            continue
        encode_note(x, y, direction)
        angle = _number(note.get("a", 0), "Note angle") if version == "3" else 0.0
        if angle != 0:
            raise ValueError(f"Nonzero note angle at source index {index} cannot fit 109 classes")
        custom = note.get("customData", note.get("_customData", {}))
        if not isinstance(custom, dict):
            raise ValueError("Note custom data must be an object")
        if any(key.lstrip("_").lower() in {"position", "rotation", "localrotation", "cutdirection", "scale"} for key in custom):
            raise ValueError(f"Unsupported extended note geometry at source index {index}")
        seconds = beat * 60 / bpm
        if not np.isfinite(seconds):
            raise ValueError("Converted note timestamp is not finite")
        rows.append([index, beat, seconds, color, x, y, direction, angle])
        records.append(deepcopy(note))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8), records


def read_notes(path, bpm):
    """Return normalized colored records while preserving the public array API."""
    return _note_records(path, bpm)[0]


def reduce_notes(original, event_indices=None):
    """Keep the first cut per synchronized event and hand, with source-order ties."""
    original = np.asarray(original, dtype=np.float64).reshape(-1, 8)
    if event_indices is None:
        event_indices, _ = synchronize_events(original[:, 2], config.min_time_diff)
    directions = ((0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1))
    groups = {}
    for row, note in enumerate(original):
        groups.setdefault((int(event_indices[row]), int(note[3])), []).append(row)
    retained, reductions = [], []
    for (event, color), rows in sorted(groups.items()):
        selected, reason = rows[0], "single"
        cuts = original[rows, 6].astype(int)
        if len(rows) > 1:
            if np.all(cuts == cuts[0]) and cuts[0] != 8:
                projections = original[rows, 4:6] @ np.asarray(directions[cuts[0]])
                candidates = np.flatnonzero(projections == projections.min())
                selected = rows[int(candidates[0])]
                reason = "first_projection" if len(candidates) == 1 else "projection_tie_source_order"
            else:
                reason = "dots_source_order" if np.all(cuts == 8) else "conflicting_directions_source_order"
            reductions.append({"source_beat": float(original[rows, 1].min()), "color": color,
                               "event_index": event,
                               "kept_source_index": int(original[selected, 0]),
                               "discarded_source_indices": [int(original[row, 0]) for row in rows if row != selected],
                               "reason": reason})
        retained.append(selected)
    return np.asarray(retained, dtype=np.int64), reductions


def build_song(song_id, original, feature_arrays, metadata):
    """Synchronize nearby cuts before per-hand reduction and 100 Hz labeling."""
    arrays = dict(feature_arrays)
    meta = deepcopy(metadata)
    original = np.asarray(original, dtype=np.float64).reshape(-1, 8)
    if "original_note_records" not in meta:
        meta["original_note_records"] = [{"b": float(row[1]), "c": int(row[3]), "x": int(row[4]),
                                         "y": int(row[5]), "d": int(row[6]), "a": float(row[7])}
                                        for row in original]
        meta["original_note_records_provenance"] = "synthesized_from_normalized_columns"
    meta["original_note_record_source_indices"] = original[:, 0].astype(int).tolist()
    if len(meta["original_note_records"]) != len(original):
        raise ValueError("Original JSON records must align with normalized source rows")
    duration = meta["duration"]
    if len(original) and (np.any(original[:, 2] < 0) or np.any(original[:, 2] >= duration)):
        raise ValueError("Colored note lies outside the decoded audio duration")
    threshold = config.min_time_diff
    event_indices, event_times = synchronize_events(original[:, 2], threshold)
    retained, reductions = reduce_notes(original, event_indices)
    events = [[] for _ in event_times]
    for row in retained:
        events[event_indices[row]].append(row)
    # Half-frame ties go to the later frame. Clamp only to the final real frame.
    frames = np.minimum(np.floor(event_times / FRAME_SECONDS + 0.5).astype(np.int64), len(arrays["frame_times"]) - 1)
    if len(np.unique(frames)) != len(frames):
        first = int(np.flatnonzero(np.diff(frames) == 0)[0])
        raise ValueError("Distinct source onsets collide on the nearest 100 Hz frame "
                         f"{int(frames[first])}: "
                         f"seconds {event_times[first]:.17g}/{event_times[first + 1]:.17g}; map was not merged")
    classes = np.zeros((len(events), 2), dtype=np.int32)
    for event_index, rows in enumerate(events):
        for row in rows:
            note = original[row]
            classes[event_index, int(note[3])] = encode_note(*note[4:7])
    timing = np.zeros(len(arrays["frame_times"]), dtype=np.int32)
    timing[frames] = np.count_nonzero(classes, axis=1)
    intensity = np.zeros(len(arrays["bin_durations"]), dtype=np.float32)
    if len(retained):
        bins = np.floor(event_times).astype(int)
        np.add.at(intensity, bins, np.count_nonzero(classes, axis=1))
        intensity /= len(retained)
    arrays.update({"original_notes": original, "retained_indices": retained,
                   "original_event_indices": event_indices,
                   "event_times": event_times, "event_frames": frames,
                   "note_classes": classes, "timing": timing, "intensity": intensity})
    source_onsets = len(np.unique(original[:, 2]))
    meta.update({"intensity_target_valid": bool(len(events)),
                 "original_colored_note_count": len(original), "retained_colored_note_count": len(retained),
                 "unique_onset_count": len(events), "difficulty_nps": len(retained) / duration,
                 "reductions": reductions, "reduction_policy": "synchronized_event_per_hand_first_cut",
                 "reduction_time_epsilon_seconds": float(threshold),
                 "synchronization_anchor": "first_onset_inclusive",
                 "source_unique_onset_count": source_onsets,
                 "synchronized_onset_merge_count": source_onsets - len(events),
                 "reduction_fallback": "original_source_order", "frame_assignment": "nearest_half_up_final_frame_clamp"})
    song = Song(song_id, meta, arrays)
    validate_song(song)
    return song


def _record(path, role):
    return {"role": role, "path": str(path), "sha256": hash_file(path)}


def _load_song(info_path, beatmap_path, audio_path, identity, provenance, audio_cache):
    info, bpm = _info(info_path)
    key = str(audio_path)
    if key not in audio_cache:
        arrays, meta = extract_features(decode_audio(audio_path))
        audio_cache[key] = (arrays, meta, _record(audio_path, "audio"))
    features, meta, audio_record = audio_cache[key]
    meta = deepcopy(meta)
    meta.update({"bpm": bpm, "song_name": info.get("_songName", identity),
                 "song_author": info.get("_songAuthorName", ""), "mapper": info.get("_levelAuthorName", ""),
                 "provenance": provenance,
                 "sources": [_record(info_path, "info"), _record(beatmap_path, "beatmap"), audio_record],
                 "audio_sha256": audio_record["sha256"]})
    song_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    notes, records = _note_records(beatmap_path, bpm)
    meta.update({"original_note_records": records, "original_note_records_provenance": "verbatim_source_json"})
    return build_song(song_id, notes, features, meta)


def _failure(report, path, error, **extra):
    report["skipped"].append({"path": str(path), "reason": f"{type(error).__name__}: {error}", **extra})


def _raw_songs(root, report, audio_cache, difficulty=None):
    packages = [root] if any(path.name.lower() == "info.dat" for path in root.iterdir()) else sorted(path for path in root.iterdir() if path.is_dir())
    result = []
    for package in packages:
        try:
            if not package.resolve().is_relative_to(root):
                raise ValueError("Package resolves outside the input root")
            infos = [path for path in package.iterdir() if path.name.lower() == "info.dat" and path.is_file()]
            if len(infos) != 1:
                raise ValueError(f"Expected one Info.dat; found {len(infos)}")
            info_path = infos[0]
            info, _ = _info(info_path)
            audio_path = _package_file(package.resolve(), info.get("_songFilename"))
            groups = _objects(info, "_difficultyBeatmapSets")
            if not groups:
                raise ValueError("No difficulty beatmap sets")
            selected = 0
            for group in groups:
                characteristic = group.get("_beatmapCharacteristicName")
                for item in _objects(group, "_difficultyBeatmaps"):
                    if difficulty is not None and item.get("_difficulty") != difficulty:
                        continue
                    selected += 1
                    filename = item.get("_beatmapFilename")
                    try:
                        if characteristic != "Standard":
                            raise ValueError(f"Unsupported gameplay characteristic: {characteristic!r}")
                        beatmap_path = _package_file(package.resolve(), filename)
                        provenance = {"adapter": "raw_package", "metadata_verified": True,
                                      "package_path": str(package), "difficulty": item.get("_difficulty"),
                                      "characteristic": characteristic}
                        result.append(_load_song(info_path, beatmap_path, audio_path,
                                                 str(beatmap_path), provenance, audio_cache))
                    except (ValueError, OSError, RuntimeError) as error:
                        _failure(report, package, error, beatmap_filename=filename)
            if selected == 0:
                raise ValueError(f"No beatmap with requested difficulty: {difficulty!r}")
        except (ValueError, OSError, RuntimeError) as error:
            _failure(report, package, error)
    return result


def _prepared_songs(root, report, audio_cache):
    maps = root / "maps" if (root / "maps").is_dir() else root
    audio = root / "songs_egg" if (root / "songs_egg").is_dir() else root
    result = []
    for path in sorted(maps.glob("*.dat")):
        if path.stem.endswith("_info"):
            continue
        info_path, audio_path = maps / f"{path.stem}_info.dat", audio / f"{path.stem}.egg"
        try:
            for source in (path, info_path, audio_path):
                if not source.is_file() or not source.resolve().is_relative_to(root):
                    raise ValueError(f"Missing or external exact-basename prepared source: {source}")
            provenance = {"adapter": "prepared_basename_triple", "metadata_verified": False,
                          "difficulty": None, "characteristic": None,
                          "limitation": "Flattened copies do not verify original difficulty/characteristic"}
            result.append(_load_song(info_path, path, audio_path, str(path), provenance, audio_cache))
        except (ValueError, OSError, RuntimeError) as error:
            _failure(report, path, error)
    return result


def _tokens(song):
    return {song.song_id, song.metadata["audio_sha256"], song.metadata["decoded_sha256"],
            *(source["path"] for source in song.metadata.get("sources", []) if source["role"] == "audio")}


def assign_splits(songs, reserved=(), aliases=None, seed=SEED):
    """Group byte/decoded duplicates and explicit aliases before 80/10/10 splits.

    Alias groups are lists of source audio paths, hashes, or song IDs. A mapping
    alias->canonical identifier is accepted as a convenient equivalent.
    """
    parents = {}
    def find(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]
    def union(values):
        values = sorted(set(values))
        if values:
            first = find(values[0])
            for value in values[1:]:
                parents[find(value)] = first
    for song in songs:
        union(_tokens(song))
    for record in reserved:
        union(record["tokens"])
    groups = [[str(key), str(value)] for key, value in aliases.items()] if isinstance(aliases, dict) else (aliases or [])
    for group in groups:
        if isinstance(group, str) or not group or any(not isinstance(value, str) for value in group):
            raise ValueError("Each explicit alias group must contain string identifiers")
        union(group)
    reserved_groups = {find(token) for record in reserved for token in record["tokens"]}
    grouped = {}
    for song in songs:
        grouped.setdefault(find(song.metadata["audio_sha256"]), []).append(song)
    identifiers = {group: hashlib.sha256("\n".join(sorted({token for song in members for token in
                   (song.metadata["audio_sha256"], song.metadata["decoded_sha256"])})).encode()).hexdigest()
                   for group, members in grouped.items()}
    available = sorted((group for group in grouped if group not in reserved_groups), key=lambda group: identifiers[group])
    rng = np.random.default_rng(seed)
    if available:
        available = [available[index] for index in rng.permutation(len(available))]
    count = len(available)
    test_count = max(1, int(round(count * .1))) if count >= 3 else 0
    validation_count = max(1, int(round(count * .1))) if count >= 3 else 0
    splits = {}
    for index, group in enumerate(available):
        splits[group] = "test" if index < test_count else "validation" if index < test_count + validation_count else "train"
    for group, members in grouped.items():
        for song in members:
            song.metadata.update({"group_id": identifiers[group], "split": splits.get(group, "reserved")})
    return songs


def _reserved_audio(root, report, audio_cache):
    result = []
    if root is None or not root.is_dir():
        report["warnings"].append("Curated benchmark root unavailable; overlap reservation is unverified")
        return result
    for package in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            infos = [path for path in package.iterdir() if path.name.lower() == "info.dat" and path.is_file()]
            if len(infos) != 1:
                raise ValueError(f"Expected one Info.dat; found {len(infos)}")
            info = _read_json(infos[0])
            audio = _package_file(package.resolve(), info.get("_songFilename"))
            from .features import decoded_hash
            cached = audio_cache.get(str(audio))
            pcm_hash = cached[1]["decoded_sha256"] if cached else decoded_hash(decode_audio(audio))
            byte_hash = cached[2]["sha256"] if cached else hash_file(audio)
            result.append({"path": str(package), "tokens": [str(audio), byte_hash, pcm_hash]})
        except (ValueError, OSError, RuntimeError) as error:
            report["reserved_skipped"].append({"path": str(package), "reason": str(error)})
    return result


def prepare_songs(source_root=None, prepared_root=None, dummy=False, reserved_root=None, aliases=None, seed=SEED,
                  evaluation_only=False, difficulty=None):
    """Prepare raw packages or explicitly selected flattened sources, read-only."""
    report = {"schema_version": 1, "songs": [], "skipped": [], "reserved": [], "reserved_skipped": [],
              "warnings": [], "split_seed": seed, "split_ratio": [0.8, 0.1, 0.1],
              "overlap_policy": "audio_byte_hash_and_decoded_float32_hash_plus_explicit_aliases",
              "role": "evaluation" if evaluation_only else "training", "difficulty": difficulty}
    if prepared_root is not None and difficulty is not None:
        raise ValueError("Prepared triples have unverified difficulty; a difficulty filter requires raw packages")
    if dummy:
        songs = make_dummy_songs(seed=seed)
        report["adapter"] = "dummy"
    else:
        if source_root is not None and prepared_root is not None:
            raise ValueError("Choose source_root or prepared_root")
        if source_root is None and prepared_root is None:
            from tools.config import paths
            source_root = paths.bs_input_path or str(Path(paths.train_path) / "custom_bs_input")
        root = Path(prepared_root if prepared_root is not None else source_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Input root does not exist: {root}")
        report.update({"source_root": str(root), "adapter": "prepared" if prepared_root is not None else "raw"})
        audio_cache = {}
        songs = (_prepared_songs(root, report, audio_cache) if prepared_root is not None else
                 _raw_songs(root, report, audio_cache, difficulty))
        if reserved_root is None:
            candidates = [root / "evaluate_beats" / "bs_map_input", root.parent / "evaluate_beats" / "bs_map_input"]
            from tools.config import paths
            candidates.extend([Path(paths.train_path) / "evaluate_beats" / "bs_map_input",
                               Path(__file__).resolve().parents[3] / "Data" / "training" / "evaluate_beats" / "bs_map_input"])
            reserved_root = next((candidate for candidate in candidates if candidate.is_dir()), None)
        reserved = _reserved_audio(Path(reserved_root).expanduser().resolve() if reserved_root is not None else None, report, audio_cache)
        assign_splits(songs, reserved, aliases, seed)
        kept = []
        for song in songs:
            if song.metadata["split"] == "reserved":
                report["reserved"].append({"song_id": song.song_id, "sources": song.metadata["sources"],
                                           "reason": "Recording overlaps curated benchmark audio or an explicit alias"})
            if evaluation_only or song.metadata["split"] != "reserved":
                kept.append(song)
        songs = kept
    if evaluation_only:
        for song in songs:
            song.metadata.update({"split": "test", "dataset_role": "evaluation"})
    report["songs"] = [{"song_id": song.song_id, "group_id": song.metadata["group_id"],
                        "split": song.metadata["split"], "duration": song.metadata["duration"],
                        "original_notes": song.metadata["original_colored_note_count"],
                        "retained_notes": song.metadata["retained_colored_note_count"],
                        "unique_onsets": song.metadata["unique_onset_count"]} for song in songs]
    report.update({"success_count": len(songs), "skip_count": len(report["skipped"]),
                   "reserved_count": len(report["reserved"]),
                   "group_count": len({song.metadata["group_id"] for song in songs})})
    return songs, report


def make_dummy_songs(seed=SEED):
    """Small seeded recordings exercise doubles, silence, tails, and reductions."""
    rng = np.random.default_rng(seed)
    songs = []
    durations = (6.257, 0.073, 2.011, 3.5, 1.0, 4.127, 2.5, 5.003, 1.051, 3.21, 2.875, 4.01)
    for index, duration in enumerate(durations):
        count = round(duration * SAMPLE_RATE)
        time = np.arange(count) / SAMPLE_RATE
        audio = (0.15 * np.sin(2 * np.pi * (180 + index * 23) * time) * (.5 + .5 * np.sin(2 * np.pi * 2 * time))
                 + rng.normal(0, .001, count)).astype(np.float32)
        if index == 2:
            audio[:] = 0
        features, metadata = extract_features(audio)
        bpm = 90.0 + index * 5
        rows = []
        if index != 2:
            for onset, seconds in enumerate(np.arange(.02, duration - .005, .37)):
                colors = (0, 1) if onset % 3 == 0 else (onset % 2,)
                for color in colors:
                    rows.append([len(rows), seconds * bpm / 60, seconds, color,
                                 (onset + color) % 4, onset % 3, (onset + color) % 9, 0])
            if index == 0:
                rows.append([len(rows), rows[0][1], rows[0][2], 0, 3, 2, rows[0][6], 0])
        metadata.update({"bpm": bpm, "audio_sha256": metadata["decoded_sha256"],
                         "sources": [], "provenance": {"adapter": "deterministic_dummy", "metadata_verified": True}})
        songs.append(build_song(f"dummy-{index:02d}", rows, features, metadata))
    return assign_splits(songs, seed=seed)
