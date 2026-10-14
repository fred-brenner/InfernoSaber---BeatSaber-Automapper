"""Read human note onsets without mutating the training/import pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _version(data: dict, supported: tuple[str, ...], label: str) -> str:
    version = data.get("_version", data.get("version", ""))
    parts = version.split(".") if isinstance(version, str) else []
    if len(parts) != 3 or not all(part.isdigit() for part in parts) or parts[0] not in supported:
        raise ValueError(f"Unsupported {label} version: {version!r}")
    return parts[0]


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _objects(data: dict, key: str) -> list:
    values = data.get(key)
    if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
        raise ValueError(f"Expected {key} array of objects")
    return values


def _check_timing(data) -> None:
    """Reject tempo/offset metadata wherever standard or custom JSON stores it."""
    if isinstance(data, dict):
        for key, value in data.items():
            normalized = key.lstrip("_").lower()
            if normalized in {"bpmchanges", "bpmevents"} and value:
                raise ValueError(f"Unsupported tempo changes in {key}")
            if normalized in {"songtimeoffset", "shuffle"} and _number(value, key) != 0:
                raise ValueError(f"Unsupported nonzero {key}")
            _check_timing(value)
    elif isinstance(data, list):
        for value in data:
            _check_timing(value)


def _package_file(package: Path, filename) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("Expected a nonempty filename in Info.dat")
    candidate = (package / filename.replace("\\", "/")).resolve()
    if not candidate.is_relative_to(package):
        raise ValueError(f"Referenced file must be inside the map package: {filename}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing map package file: {candidate}")
    return candidate


def load_reference(package: Path, difficulty="ExpertPlus", characteristic="Standard") -> dict:
    """Load v2 Info with v2/v3 notes, using a constant BPM and zero audio offset.

    Colored notes at precisely the same time count as one onset. Bombs, arcs,
    obstacles, and lighting do not contribute reference onsets. Tempo changes,
    shuffled timing, and audio offsets require alignment support and are rejected.
    """
    package = Path(package).expanduser().resolve()
    info_files = [path for path in package.iterdir() if path.name.lower() == "info.dat" and path.is_file()]
    if not info_files:
        raise FileNotFoundError(f"Missing Info.dat in {package}")
    if len(info_files) != 1:
        raise ValueError(f"Ambiguous Info.dat files in {package}")
    info_path = info_files[0]
    info = _read_json(info_path)
    _version(info, ("2",), "Info.dat")
    _check_timing(info)
    bpm = _number(info.get("_beatsPerMinute"), "BPM")
    if bpm <= 0:
        raise ValueError("BPM must be positive")

    matches = [beatmap for group in _objects(info, "_difficultyBeatmapSets")
               if group.get("_beatmapCharacteristicName") == characteristic
               for beatmap in _objects(group, "_difficultyBeatmaps")
               if beatmap.get("_difficulty") == difficulty]
    if len(matches) != 1:
        raise ValueError(f"Expected one {characteristic}/{difficulty} beatmap; found {len(matches)}")
    beatmap_path = _package_file(package, matches[0].get("_beatmapFilename"))
    audio_path = _package_file(package, info.get("_songFilename"))
    beatmap = _read_json(beatmap_path)
    version = _version(beatmap, ("2", "3"), "beatmap")
    _check_timing(beatmap)
    notes_key, time_key, color_key = ("_notes", "_time", "_type") if version == "2" else ("colorNotes", "b", "c")
    times = set()
    for note in _objects(beatmap, notes_key):
        beat = _number(note.get(time_key), "Note beat time")
        if beat < 0:
            raise ValueError("Note beat time must not be negative")
        color = note.get(color_key)
        if isinstance(color, bool) or color not in ((0, 1, 3) if version == "2" else (0, 1)):
            raise ValueError(f"Unsupported note color/type: {color!r}")
        seconds = beat * 60 / bpm
        if not math.isfinite(seconds):
            raise ValueError("Note elapsed time must be finite")
        if color in (0, 1):
            times.add(seconds)

    return {
        "times": sorted(times), "audio_path": audio_path, "info_path": info_path,
        "beatmap_path": beatmap_path, "song_name": info.get("_songName", package.name),
        "song_author": info.get("_songAuthorName", ""), "mapper": info.get("_levelAuthorName", ""),
        "bpm": bpm, "difficulty": difficulty, "characteristic": characteristic,
        "song_offset_seconds": 0,
    }
