"""Prepare the dedicated evaluation dataset without changing source packages."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from uuid import uuid4

import numpy as np

from evaluation.human_maps import load_reference


DERIVED_FOLDERS = ("maps", "songs_egg", "maps_dict_all", "songs_diff")
REPORT_NAME = "preprocessing_report.json"


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(item, key, *, integer=False, minimum=0, maximum=None):
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{key} must be finite") from error
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{key} is outside the supported finite range")
    if integer and not value.is_integer():
        raise ValueError(f"{key} must be an integer")
    return value


def _objects(data, key):
    items = data.get(key, [])
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError(f"{key} must be an array of objects")
    return items


def _array(rows, width):
    if not rows:
        return np.zeros(0, dtype=float)
    result = np.asarray(rows, dtype=float).reshape(-1, width)
    if not np.isfinite(result).all():
        raise ValueError("Converted dictionary values must be finite")
    return result[np.argsort(result[:, 0], kind="stable")].T


def _legacy_arrays(reference):
    data = json.loads(reference["beatmap_path"].read_text(encoding="utf-8-sig"))
    is_v2 = "_notes" in data
    seconds_per_beat = 60 / reference["bpm"]
    notes, events, obstacles = [], [], []
    for item in _objects(data, "_notes" if is_v2 else "colorNotes"):
        keys = ("_time", "_lineIndex", "_lineLayer", "_type", "_cutDirection") if is_v2 else ("b", "x", "y", "c", "d")
        beat, x, y, color, direction = [_number(item, key, integer=index > 0,
                                              maximum=(None, 3, 2, 3 if is_v2 else 1, 8)[index])
                                       for index, key in enumerate(keys)]
        if color not in (0, 1, 3):
            raise ValueError(f"Unsupported note type: {color}")
        notes.append([beat * seconds_per_beat, x, y, color, direction])
    if not is_v2:
        for item in _objects(data, "bombNotes"):
            notes.append([_number(item, "b") * seconds_per_beat,
                          _number(item, "x", integer=True, maximum=3),
                          _number(item, "y", integer=True, maximum=2), 3, 8])
    if not notes:
        raise ValueError("No legacy notes to prepare")
    for item in _objects(data, "_events" if is_v2 else "basicBeatmapEvents"):
        time_key, type_key, value_key = ("_time", "_type", "_value") if is_v2 else ("b", "et", "i")
        event_type = _number(item, type_key, integer=True)
        if event_type == 100:
            raise ValueError("Unsupported tempo change lighting event type 100")
        events.append([_number(item, time_key) * seconds_per_beat, event_type,
                       _number(item, value_key, integer=True)])
    for index, item in enumerate(_objects(data, "_obstacles" if is_v2 else "obstacles")):
        try:
            if is_v2:
                beat = _number(item, "_time")
                x = _number(item, "_lineIndex", integer=True, minimum=None)
                obstacle_type = _number(item, "_type", integer=True, maximum=1)
                duration = _number(item, "_duration")
                width = _number(item, "_width", integer=True, minimum=1)
            else:
                beat = _number(item, "b")
                x = _number(item, "x", integer=True, minimum=None)
                geometry = (_number(item, "y", integer=True), _number(item, "h", integer=True))
                if geometry not in ((0, 5), (2, 3)):
                    raise ValueError(f"V3 obstacle geometry cannot fit the legacy dictionary: {geometry}")
                obstacle_type = 0 if geometry == (0, 5) else 1
                duration = _number(item, "d")
                width = _number(item, "w", integer=True, minimum=1)
        except ValueError as error:
            raise ValueError(f"Obstacle {index}: {error}") from error
        # Keep the historical dictionary's mixed units: start seconds, duration beats.
        obstacles.append([beat * seconds_per_beat, x, obstacle_type, duration, width])
    return _array(notes, 5), _array(events, 3), _array(obstacles, 5)


def _decode_audio(path):
    """Check the Ogg stream that will be copied byte-for-byte with an .egg suffix."""
    import aubio

    with path.open("rb") as handle:
        if handle.read(4) != b"OggS":
            raise ValueError("Legacy song copies require Ogg audio; source audio was not changed")
    source = aubio.source(str(path), samplerate=14800, channels=1)
    count = 0
    try:
        while True:
            samples, read = source()
            if not np.isfinite(samples[:read]).all():
                raise ValueError("Decoded audio contains nonfinite samples")
            count += read
            if read < source.hop_size:
                break
        if count == 0:
            raise ValueError("Decoded audio is empty")
        return count / source.samplerate
    finally:
        del source


def _prepared_name(title, package, used):
    title = unicodedata.normalize("NFC", str(title))
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip().rstrip(". ")[:80]
    if not title:
        title = "song"
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", title.split(".")[0], re.IGNORECASE):
        title = "_" + title
    package_id = re.sub(r"[^A-Za-z0-9_-]", "", package.name.split(" ")[0])[:24]
    package_id = package_id or hashlib.sha256(package.name.encode("utf-8")).hexdigest()[:8]
    name, counter = title, 1
    while name.casefold() in used or (name + "_info").casefold() in used:
        name = f"{title}__{package_id}" + (f"_{counter}" if counter > 1 else "")
        counter += 1
    return name


def _write_song(staging, name, reference, arrays):
    assets = [(reference["beatmap_path"], staging / "maps" / f"{name}.dat"),
              (reference["info_path"], staging / "maps" / f"{name}_info.dat"),
              (reference["audio_path"], staging / "songs_egg" / f"{name}.egg")]
    targets = [destination for _, destination in assets]
    targets += [staging / "maps_dict_all" / f"{name}_{kind}.dat" for kind in ("notes", "events", "obstacles")]
    try:
        for source, destination in assets:
            shutil.copy2(source, destination)
        for array, destination in zip(arrays, targets[3:]):
            array.dump(str(destination))
    except Exception:
        for target in targets:
            target.unlink(missing_ok=True)
        raise


def _remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _promote(staging, root, manifest):
    names = (*DERIVED_FOLDERS, REPORT_NAME)
    old = [name for name in names if (root / name).exists() or (root / name).is_symlink()]
    backup = root / "archive" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8])
    manifest["backup_path"] = str(backup) if old else None
    (staging / REPORT_NAME).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    moved, promoted = [], []
    try:
        if old:
            backup.mkdir(parents=True)
        for name in old:
            (root / name).rename(backup / name)
            moved.append(name)
        for name in names:
            (staging / name).rename(root / name)
            promoted.append(name)
    except Exception:
        for name in reversed(promoted):
            _remove(root / name)
        for name in reversed(moved):
            (backup / name).rename(root / name)
        if backup.is_dir() and not any(backup.iterdir()):
            backup.rmdir()
        raise


def prepare_evaluation(root, difficulty="ExpertPlus", characteristic="Standard"):
    """Refresh only the four derived siblings of root/bs_map_input, with backups."""
    root = Path(root).expanduser().resolve()
    source_root = root / "bs_map_input"
    if not source_root.is_dir():
        raise ValueError(f"Evaluation input folder does not exist: {source_root}")
    if source_root.is_symlink():
        raise ValueError("Evaluation bs_map_input must be a real directory, not an external link")
    packages = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not packages:
        raise ValueError(f"No input package directories in {source_root}")
    manifest = {"schema_version": 1, "evaluation_root": str(root), "source_root": str(source_root),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "difficulty": difficulty, "characteristic": characteristic, "songs": [], "skipped": [],
                "legacy_density_rule": "all notes including bombs and simultaneous notes / last-first note seconds; rounded to 2 decimals",
                "legacy_obstacle_units": "start in seconds; duration in beats"}
    with tempfile.TemporaryDirectory(prefix=".prepare-evaluation-", dir=root) as directory:
        staging = Path(directory)
        for folder in DERIVED_FOLDERS:
            (staging / folder).mkdir()
        used = set()
        for package in packages:
            try:
                if not package.resolve().is_relative_to(source_root):
                    raise ValueError("Input package resolves outside bs_map_input; external sources were not read")
                reference = load_reference(package, difficulty, characteristic)
                arrays = _legacy_arrays(reference)
                duration = _decode_audio(reference["audio_path"])
                notes = arrays[0]
                if notes[0, -1] > duration:
                    raise ValueError("Note lies beyond the decoded audio duration")
                span = notes[0, -1] - notes[0, 0]
                if span <= 0:
                    raise ValueError("Legacy density requires notes at distinct times")
                name = _prepared_name(reference["song_name"], package, used)
                song = {key: str(value) if isinstance(value, Path) else value
                        for key, value in reference.items() if key != "times"}
                song.update({"package_path": str(package), "prepared_name": name,
                             "audio_duration_seconds": duration, "note_count": notes.shape[1],
                             "reference_onset_count": len(reference["times"]),
                             "legacy_density_bps": round(notes.shape[1] / span, 2),
                             "input_sha256": {key: _hash_file(reference[key]) for key in
                                              ("audio_path", "info_path", "beatmap_path")}})
                _write_song(staging, name, reference, arrays)
                manifest["songs"].append(song)
                used.update((name.casefold(), (name + "_info").casefold()))
            except Exception as error:
                manifest["skipped"].append({"package_path": str(package),
                                            "error": f"{type(error).__name__}: {error}"})
        manifest["songs"].sort(key=lambda song: song["prepared_name"].casefold())
        np.save(staging / "songs_diff" / "name_ar.npy", np.asarray(
            [song["prepared_name"] for song in manifest["songs"]], dtype=str))
        np.save(staging / "songs_diff" / "diff_ar.npy", np.asarray(
            [song["legacy_density_bps"] for song in manifest["songs"]], dtype=float))
        manifest.update({"input_count": len(packages), "success_count": len(manifest["songs"]),
                         "skip_count": len(manifest["skipped"])})
        _promote(staging, root, manifest)
    return manifest
