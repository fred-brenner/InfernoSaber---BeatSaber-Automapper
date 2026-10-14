import json
from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.human_maps import load_reference


def write_package(tmp_path, *, info_changes=None, beatmap=None):
    info = {
        "_version": "2.0.0", "_songName": "Human Song", "_songAuthorName": "Artist",
        "_levelAuthorName": "Mapper", "_beatsPerMinute": 120, "_songFilename": "song.ogg",
        "_songTimeOffset": 0, "_shuffle": 0, "_shufflePeriod": 0.5,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlus.dat"}],
        }],
    }
    info.update(info_changes or {})
    (tmp_path / "INFO.DAT").write_text(json.dumps(info), encoding="utf-8-sig")
    (tmp_path / "ExpertPlus.dat").write_text(json.dumps(beatmap or {
        "version": "3.2.0", "bpmEvents": [], "colorNotes": [{"b": 2, "c": 0}],
    }), encoding="utf-8-sig")
    (tmp_path / "song.ogg").write_bytes(b"audio")
    return tmp_path


@pytest.mark.parametrize("version", ["2", "3"])
def test_reference_counts_unique_colored_note_onsets(tmp_path, version):
    if version == "2":
        beatmap = {"_version": "2.2.0", "_BPMChanges": [], "_notes": [
            {"_time": 4, "_type": 1}, {"_time": 2, "_type": 0},
            {"_time": 2, "_type": 1}, {"_time": 2.005, "_type": 0}, {"_time": 3, "_type": 3},
        ]}
    else:
        beatmap = {"version": "3.2.0", "bpmEvents": [], "colorNotes": [
            {"b": 4, "c": 1}, {"b": 2, "c": 0}, {"b": 2, "c": 1}, {"b": 2.005, "c": 0},
        ], "bombNotes": [{"b": 3}], "sliders": [{"b": 5}], "obstacles": [{"b": 6}]}
    reference = load_reference(write_package(tmp_path, beatmap=beatmap))
    assert reference["times"] == [1.0, 1.0025, 2.0]
    assert reference["audio_path"] == tmp_path / "song.ogg"
    assert reference["info_path"] == tmp_path / "INFO.DAT"
    assert reference["beatmap_path"] == tmp_path / "ExpertPlus.dat"
    assert (reference["song_name"], reference["song_author"], reference["mapper"]) == (
        "Human Song", "Artist", "Mapper")
    assert reference["bpm"] == 120
    assert reference["song_offset_seconds"] == 0


def test_selects_explicit_difficulty_and_characteristic(tmp_path):
    groups = [{"_beatmapCharacteristicName": "OneSaber", "_difficultyBeatmaps": [
        {"_difficulty": "Expert", "_beatmapFilename": "Expert.dat"},
        {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlus.dat"},
    ]}]
    write_package(tmp_path, info_changes={"_difficultyBeatmapSets": groups})
    (tmp_path / "Expert.dat").write_text(json.dumps({
        "_version": "2.0.0", "_notes": [{"_time": 8, "_type": 0}],
    }))
    reference = load_reference(tmp_path, difficulty="Expert", characteristic="OneSaber")
    assert reference["times"] == [4]
    assert reference["difficulty"] == "Expert"
    assert reference["characteristic"] == "OneSaber"
    with pytest.raises(ValueError, match="Standard/ExpertPlus"):
        load_reference(tmp_path)


@pytest.mark.parametrize("bpm", [0, -120, float("nan"), float("inf"), "120", True, 10 ** 400])
def test_rejects_invalid_bpm(tmp_path, bpm):
    with pytest.raises(ValueError, match="BPM"):
        load_reference(write_package(tmp_path, info_changes={"_beatsPerMinute": bpm}))


@pytest.mark.parametrize("version", ["2", "3"])
@pytest.mark.parametrize("time", [-1, float("nan"), float("inf"), "2", True, None])
def test_rejects_invalid_note_times(tmp_path, version, time):
    beatmap = ({"_version": "2.0.0", "_notes": [{"_time": time, "_type": 0}]}
               if version == "2" else {"version": "3.2.0", "colorNotes": [{"b": time, "c": 0}]})
    with pytest.raises(ValueError, match="time"):
        load_reference(write_package(tmp_path, beatmap=beatmap))


@pytest.mark.parametrize("changes", [
    {"_songTimeOffset": 0.1}, {"_songTimeOffset": float("nan")}, {"_shuffle": 0.5},
    {"_customData": {"_BPMChanges": [{"_time": 4, "_BPM": 150}]}},
    {"_difficultyBeatmapSets": [{"_customData": {"bpmEvents": [{"b": 4, "m": 150}]}}]},
])
def test_rejects_unsupported_info_timing(tmp_path, changes):
    with pytest.raises(ValueError, match="songTimeOffset|shuffle|tempo changes"):
        load_reference(write_package(tmp_path, info_changes=changes))


@pytest.mark.parametrize("changes", [
    {"bpmEvents": [{"b": 4, "m": 150}]}, {"_BPMChanges": [{"_time": 4, "_BPM": 150}]},
    {"customData": {"bpmChanges": [{"b": 4, "m": 150}]}},
    {"_customData": {"_BPMChanges": [{"_time": 4, "_BPM": 150}]}},
])
def test_rejects_standard_and_custom_beatmap_tempo_changes(tmp_path, changes):
    beatmap = {"version": "3.2.0", "colorNotes": []}
    beatmap.update(changes)
    with pytest.raises(ValueError, match="tempo changes"):
        load_reference(write_package(tmp_path, beatmap=beatmap))


@pytest.mark.parametrize("filename", ["INFO.DAT", "ExpertPlus.dat", "song.ogg"])
def test_reports_missing_package_files(tmp_path, filename):
    write_package(tmp_path)
    (tmp_path / filename).unlink()
    with pytest.raises(FileNotFoundError):
        load_reference(tmp_path)


@pytest.mark.parametrize("changes,beatmap", [
    ({"_version": "4.0.0"}, None), ({"_version": None}, None),
    ({}, {"version": "4.0.0", "colorNotes": []}),
    ({}, {"version": "3.bad.0", "colorNotes": []}),
    ({}, {"colorNotes": []}),
])
def test_rejects_unknown_versions(tmp_path, changes, beatmap):
    with pytest.raises(ValueError, match="version"):
        load_reference(write_package(tmp_path, info_changes=changes, beatmap=beatmap))


def test_accepts_empty_colored_reference(tmp_path):
    reference = load_reference(write_package(tmp_path, beatmap={
        "version": "3.2.0", "colorNotes": [], "bombNotes": [{"b": 2}],
    }))
    assert reference["times"] == []


def test_rejects_malformed_note_array(tmp_path):
    with pytest.raises(ValueError, match="colorNotes array"):
        load_reference(write_package(tmp_path, beatmap={"version": "3.2.0"}))


def test_rejects_ambiguous_info_files(tmp_path):
    write_package(tmp_path)
    (tmp_path / "Info.dat").write_text("{}")
    with pytest.raises(ValueError, match="Ambiguous"):
        load_reference(tmp_path)


def test_rejects_files_outside_package(tmp_path):
    write_package(tmp_path, info_changes={"_songFilename": "../song.ogg"})
    with pytest.raises(ValueError, match="inside the map package"):
        load_reference(tmp_path)


@pytest.mark.parametrize("groups", [None, {}, [None], [{"_beatmapCharacteristicName": "Standard"}]])
def test_rejects_malformed_difficulty_metadata(tmp_path, groups):
    with pytest.raises(ValueError, match="array of objects"):
        load_reference(write_package(tmp_path, info_changes={"_difficultyBeatmapSets": groups}))
