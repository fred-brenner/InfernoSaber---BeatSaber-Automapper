import json
from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.playability_report import (
    DIFFICULTY_ORDER,
    decode_package_notes,
    list_standard_difficulties,
)


def write_package(tmp_path, *, info_changes=None, beatmaps=None):
    """Write a synthetic package; ``beatmaps`` maps difficulty -> beatmap dict."""
    beatmaps = beatmaps or {"ExpertPlus": {
        "version": "3.2.0", "bpmEvents": [],
        "colorNotes": [{"b": 2, "x": 1, "y": 1, "c": 0, "d": 0}],
    }}
    info = {
        "_version": "2.0.0", "_songName": "Human Song", "_songAuthorName": "Artist",
        "_levelAuthorName": "Mapper", "_beatsPerMinute": 120, "_songFilename": "song.ogg",
        "_songTimeOffset": 0, "_shuffle": 0, "_shufflePeriod": 0.5,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [
                {"_difficulty": name, "_beatmapFilename": f"{name}.dat"}
                for name in beatmaps],
        }],
    }
    info.update(info_changes or {})
    (tmp_path / "INFO.DAT").write_text(json.dumps(info), encoding="utf-8-sig")
    for name, beatmap in beatmaps.items():
        (tmp_path / f"{name}.dat").write_text(json.dumps(beatmap), encoding="utf-8-sig")
    (tmp_path / "song.ogg").write_bytes(b"audio")
    return tmp_path


def test_decode_v3_colored_notes(tmp_path):
    write_package(tmp_path, beatmaps={"ExpertPlus": {
        "version": "3.2.0", "bpmEvents": [],
        "colorNotes": [
            {"b": 4, "x": 2, "y": 1, "c": 1, "d": 1},
            {"b": 2, "x": 1, "y": 0, "c": 0, "d": 0},
            {"b": 3, "x": 0, "y": 2, "c": 0, "d": 8},
        ],
        "bombNotes": [{"b": 5, "x": 3, "y": 1}],
        "sliders": [{"b": 6, "x": 0, "y": 0, "c": 0, "d": 0}],
        "obstacles": [{"b": 7, "x": 0, "y": 0, "d": 1, "w": 1, "h": 1}],
    }})
    decoded = decode_package_notes(tmp_path)
    assert decoded["bpm"] == 120
    assert decoded["difficulty"] == "ExpertPlus"
    assert decoded["notes"] == [
        (1.0, 0, 1, 0, 0),
        (1.5, 0, 0, 2, 8),
        (2.0, 1, 2, 1, 1),
    ]


def test_decode_v2_notes(tmp_path):
    write_package(tmp_path, beatmaps={"ExpertPlus": {
        "_version": "2.2.0", "_BPMChanges": [],
        "_notes": [
            {"_time": 4, "_lineIndex": 2, "_lineLayer": 1, "_type": 1, "_cutDirection": 3},
            {"_time": 2, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},
            {"_time": 3, "_lineIndex": 0, "_lineLayer": 2, "_type": 3, "_cutDirection": 0},
        ],
        "_obstacles": [{"_time": 5, "_type": 0, "_lineIndex": 0}],
        "_events": [{"_time": 6, "_type": 1, "_value": 1}],
    }})
    decoded = decode_package_notes(tmp_path)
    assert decoded["notes"] == [
        (1.0, 0, 1, 0, 1),
        (2.0, 1, 2, 1, 3),
    ]


def test_decode_highest_difficulty_default(tmp_path):
    write_package(tmp_path, beatmaps={
        "Hard": {"version": "3.2.0", "colorNotes": [{"b": 2, "x": 0, "y": 0, "c": 0, "d": 0}]},
        "ExpertPlus": {"version": "3.2.0", "colorNotes": [{"b": 4, "x": 1, "y": 1, "c": 1, "d": 1}]},
    })
    decoded = decode_package_notes(tmp_path)
    assert decoded["difficulty"] == "ExpertPlus"
    assert decoded["notes"] == [(2.0, 1, 1, 1, 1)]


def test_decode_explicit_difficulty(tmp_path):
    write_package(tmp_path, beatmaps={
        "Hard": {"version": "3.2.0", "colorNotes": [{"b": 2, "x": 0, "y": 0, "c": 0, "d": 0}]},
        "ExpertPlus": {"version": "3.2.0", "colorNotes": [{"b": 4, "x": 1, "y": 1, "c": 1, "d": 1}]},
    })
    decoded = decode_package_notes(tmp_path, difficulty="Hard")
    assert decoded["difficulty"] == "Hard"
    assert decoded["notes"] == [(1.0, 0, 0, 0, 0)]
    with pytest.raises(ValueError, match="Standard/Easy"):
        decode_package_notes(tmp_path, difficulty="Easy")


def test_list_standard_difficulties_canonical_order(tmp_path):
    assert DIFFICULTY_ORDER == ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
    write_package(tmp_path, beatmaps={
        "ExpertPlus": {"version": "3.2.0", "colorNotes": []},
        "Easy": {"version": "3.2.0", "colorNotes": []},
        "Hard": {"version": "3.2.0", "colorNotes": []},
    })
    assert list_standard_difficulties(tmp_path) == ["Easy", "Hard", "ExpertPlus"]


def test_decode_nonstandard_skipped(tmp_path):
    groups = [{"_beatmapCharacteristicName": "OneSaber", "_difficultyBeatmaps": [
        {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlus.dat"},
    ]}]
    write_package(tmp_path, info_changes={"_difficultyBeatmapSets": groups})
    assert list_standard_difficulties(tmp_path) == []
    with pytest.raises(ValueError, match="Standard"):
        decode_package_notes(tmp_path)


def test_decode_tempo_change_rejected(tmp_path):
    write_package(tmp_path, beatmaps={"ExpertPlus": {
        "version": "3.2.0", "bpmEvents": [{"b": 4, "m": 150}],
        "colorNotes": [{"b": 2, "x": 0, "y": 0, "c": 0, "d": 0}],
    }})
    with pytest.raises(ValueError, match="tempo changes"):
        decode_package_notes(tmp_path)
