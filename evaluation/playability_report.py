"""Decode Beat Saber map packages into playability note tuples.

Foundation for the playability corpus report: reads a v2 ``Info.dat``
package, selects Standard-characteristic difficulties, and converts v2
``_notes`` or v3 ``colorNotes`` entries into ``(t_seconds, hand, x, y,
d)`` tuples for ``playability_metrics.prepare_notes``. Validation reuses
``evaluation.human_maps`` internals to mirror the reference loader's
Info/timing rules (v2 Info.dat, positive constant BPM, no offsets or
shuffle, in-package paths), then adds its own checks: grid coordinates,
canonical difficulty names, and per-difficulty decode. Unlike the
reference loader it does not require the audio file, and skipped v2
bombs never reach the seconds check. Any failure raises and the caller
records the package as skipped.
"""

from __future__ import annotations

import math
from pathlib import Path

from evaluation.human_maps import (
    _check_timing,
    _number,
    _objects,
    _package_file,
    _read_json,
    _version,
)

DIFFICULTY_ORDER = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
_CHARACTERISTIC = "Standard"

# Note field names per beatmap version; v2 ``_type`` doubles as the color
# (0 red/left, 1 blue/right, 3 bomb — bombs are skipped, not notes).
_V2_FIELDS = {"beat": "_time", "x": "_lineIndex", "y": "_lineLayer",
              "color": "_type", "direction": "_cutDirection"}
_V3_FIELDS = {"beat": "b", "x": "x", "y": "y", "color": "c", "direction": "d"}


def _load_info(package):
    """Resolve and validate a package's v2 Info.dat; return its data and BPM.

    Same acceptance rules as ``human_maps.load_reference``: one
    case-insensitive Info.dat, v2 version, positive constant BPM."""
    package = Path(package).expanduser().resolve()
    info_files = [path for path in package.iterdir()
                  if path.name.lower() == "info.dat" and path.is_file()]
    if not info_files:
        raise FileNotFoundError(f"Missing Info.dat in {package}")
    if len(info_files) != 1:
        raise ValueError(f"Ambiguous Info.dat files in {package}")
    info = _read_json(info_files[0])
    _version(info, ("2",), "Info.dat")
    bpm = _number(info.get("_beatsPerMinute"), "BPM")
    if bpm <= 0:
        raise ValueError("BPM must be positive")
    _check_timing(info, bpm)
    return package, info, bpm


def _standard_beatmaps(info):
    """``{difficulty: filename}`` for beatmaps in the Standard group.

    Unknown or duplicated difficulty names are validation failures — a
    Standard beatmap outside the canonical five cannot be ranked."""
    entries = {}
    for group in _objects(info, "_difficultyBeatmapSets"):
        if group.get("_beatmapCharacteristicName") != _CHARACTERISTIC:
            continue
        for beatmap in _objects(group, "_difficultyBeatmaps"):
            difficulty = beatmap.get("_difficulty")
            if difficulty not in DIFFICULTY_ORDER:
                raise ValueError(
                    f"Unsupported {_CHARACTERISTIC} difficulty: {difficulty!r}")
            if difficulty in entries:
                raise ValueError(f"Duplicate {_CHARACTERISTIC}/{difficulty} beatmap")
            filename = beatmap.get("_beatmapFilename")
            if not isinstance(filename, str) or not filename:
                raise ValueError("Expected a nonempty filename in Info.dat")
            entries[difficulty] = filename
    return entries


def list_standard_difficulties(package):
    """Difficulties present in the package's Standard group, canonical order."""
    _, info, _ = _load_info(package)
    present = _standard_beatmaps(info)
    return [name for name in DIFFICULTY_ORDER if name in present]


def _cell(value, label, limit):
    """Grid coordinate: a finite integral number in ``0..limit-1``."""
    number = _number(value, label)
    if not number.is_integer() or not 0 <= number < limit:
        raise ValueError(f"{label} must be an integer in 0..{limit - 1}")
    return int(number)


def _decode_notes(beatmap, version, bpm):
    """Sorted ``(t, hand, x, y, d)`` tuples from the beatmap's note array.

    v2 ``_type`` 3 bombs drop out; every kept note emits one tuple —
    same-hand stacks are collapsed inside ``prepare_notes``, not here."""
    key, fields = ("_notes", _V2_FIELDS) if version == "2" else ("colorNotes", _V3_FIELDS)
    valid_colors = (0, 1, 3) if version == "2" else (0, 1)
    notes = []
    for entry in _objects(beatmap, key):
        beat = _number(entry.get(fields["beat"]), "Note beat time")
        if beat < 0:
            raise ValueError("Note beat time must not be negative")
        color = entry.get(fields["color"])
        if isinstance(color, bool) or color not in valid_colors:
            raise ValueError(f"Unsupported note color/type: {color!r}")
        if color == 3:  # v2 bomb — not a playable note
            continue
        seconds = beat * 60 / bpm
        if not math.isfinite(seconds):
            raise ValueError("Note elapsed time must be finite")
        notes.append((seconds, int(color),
                      _cell(entry.get(fields["x"]), "Note column", 4),
                      _cell(entry.get(fields["y"]), "Note row", 3),
                      _cell(entry.get(fields["direction"]), "Note cut direction", 9)))
    notes.sort(key=lambda note: note[0])
    return notes


def decode_package_notes(package, difficulty=None):
    """Decode one Standard difficulty of a map package into note tuples.

    ``difficulty=None`` selects the highest present in canonical order.
    Returns ``{"notes": [(t_seconds, hand, x, y, d), ...], "bpm": float,
    "difficulty": str}`` sorted by time; raises on any validation failure."""
    package, info, bpm = _load_info(package)
    entries = _standard_beatmaps(info)
    if difficulty is None:
        if not entries:
            raise ValueError(f"Expected one {_CHARACTERISTIC} beatmap; found 0")
        difficulty = next(name for name in reversed(DIFFICULTY_ORDER) if name in entries)
    filename = entries.get(difficulty)
    if filename is None:
        raise ValueError(f"Expected one {_CHARACTERISTIC}/{difficulty} beatmap; found 0")
    beatmap = _read_json(_package_file(package, filename))
    version = _version(beatmap, ("2", "3"), "beatmap")
    _check_timing(beatmap, bpm)
    return {"notes": _decode_notes(beatmap, version, bpm),
            "bpm": bpm, "difficulty": difficulty}
