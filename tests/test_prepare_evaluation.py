import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation import prepare_evaluation as preparation


def write_package(root, folder="1abc (Song)", title="Human Song", version="2", *, changes=None):
    package = root / "bs_map_input" / folder
    package.mkdir(parents=True)
    info = {
        "_version": "2.0.0", "_songName": title, "_songAuthorName": "Artist",
        "_levelAuthorName": "Mapper", "_beatsPerMinute": 120, "_songFilename": "source.egg",
        "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "ExpertPlus", "_beatmapFilename": "source.dat"},
        ]}],
    }
    if version == "2":
        beatmap = {"_version": "2.2.0", "_notes": [
            {"_time": 4, "_lineIndex": 2, "_lineLayer": 1, "_type": 1, "_cutDirection": 2},
            {"_time": 2, "_lineIndex": 0, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},
            {"_time": 2, "_lineIndex": 1, "_lineLayer": 2, "_type": 1, "_cutDirection": 3},
            {"_time": 3, "_lineIndex": 3, "_lineLayer": 1, "_type": 3, "_cutDirection": 8},
        ], "_events": [{"_time": 3, "_type": 1, "_value": 5}], "_obstacles": [
            {"_time": 1, "_lineIndex": 1, "_type": 0, "_duration": 2, "_width": 1},
        ]}
    else:
        beatmap = {"version": "3.2.0", "bpmEvents": [], "colorNotes": [
            {"b": 4, "x": 2, "y": 1, "c": 1, "d": 2},
            {"b": 2, "x": 0, "y": 0, "c": 0, "d": 1},
            {"b": 2, "x": 1, "y": 2, "c": 1, "d": 3},
        ], "bombNotes": [{"b": 3, "x": 3, "y": 1}],
            "basicBeatmapEvents": [{"b": 3, "et": 1, "i": 5, "f": 1}],
            "obstacles": [{"b": 1, "x": 1, "y": 0, "h": 5, "d": 2, "w": 1}]}
    beatmap.update(changes or {})
    (package / "Info.dat").write_text(json.dumps(info), encoding="utf-8-sig")
    (package / "source.dat").write_text(json.dumps(beatmap), encoding="utf-8-sig")
    (package / "source.egg").write_bytes(b"OggS fixture")
    return package


@pytest.fixture
def decoded(monkeypatch):
    monkeypatch.setattr(preparation, "_decode_audio", lambda path: 10.0)


def contents(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("version", ["2", "3"])
def test_prepares_legacy_contracts_and_keeps_sources(tmp_path, decoded, version):
    package = write_package(tmp_path, version=version)
    before = contents(package)
    manifest = preparation.prepare_evaluation(tmp_path)
    assert manifest["success_count"] == 1
    assert manifest["skip_count"] == 0
    assert manifest["backup_path"] is None
    assert manifest["evaluation_root"] == str(tmp_path)
    assert manifest["source_root"] == str(tmp_path / "bs_map_input")
    song = manifest["songs"][0]
    assert song["prepared_name"] == "Human Song"
    assert song["package_path"] == str(package)
    assert song["reference_onset_count"] == 2
    assert song["note_count"] == 4
    assert song["legacy_density_bps"] == 4.0
    assert song["input_sha256"]["audio_path"] == hashlib.sha256(before["source.egg"]).hexdigest()
    notes = np.load(tmp_path / "maps_dict_all/Human Song_notes.dat", allow_pickle=True)
    assert notes.shape == (5, 4)
    np.testing.assert_array_equal(notes[0], [1, 1, 1.5, 2])
    np.testing.assert_array_equal(notes[3], [0, 1, 3, 1])
    np.testing.assert_array_equal(np.load(tmp_path / "maps_dict_all/Human Song_events.dat", allow_pickle=True),
                                  [[1.5], [1], [5]])
    np.testing.assert_array_equal(np.load(tmp_path / "maps_dict_all/Human Song_obstacles.dat", allow_pickle=True),
                                  [[0.5], [1], [0], [2], [1]])
    assert np.load(tmp_path / "songs_diff/name_ar.npy").tolist() == ["Human Song"]
    assert np.load(tmp_path / "songs_diff/diff_ar.npy").tolist() == [4]
    assert (tmp_path / "maps/Human Song.dat").read_bytes() == before["source.dat"]
    assert (tmp_path / "maps/Human Song_info.dat").read_bytes() == before["Info.dat"]
    assert (tmp_path / "songs_egg/Human Song.egg").read_bytes() == before["source.egg"]
    assert contents(package) == before
    assert json.loads((tmp_path / preparation.REPORT_NAME).read_text()) == manifest


def test_archives_old_derived_data_and_preserves_other_folders(tmp_path, decoded):
    write_package(tmp_path)
    for folder in preparation.DERIVED_FOLDERS:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "old.asset").write_bytes(b"old data")
    old_report = b'{"old": true}'
    (tmp_path / preparation.REPORT_NAME).write_bytes(old_report)
    for folder in ("temp", "favorites_bs_input", "custom"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "user.asset").write_bytes(b"preserve")
    manifest = preparation.prepare_evaluation(tmp_path)
    backup = Path(manifest["backup_path"])
    assert backup.parent == tmp_path / "archive"
    for folder in preparation.DERIVED_FOLDERS:
        assert (backup / folder / "old.asset").read_bytes() == b"old data"
        assert not (tmp_path / folder / "old.asset").exists()
    assert (backup / preparation.REPORT_NAME).read_bytes() == old_report
    for folder in ("temp", "favorites_bs_input", "custom"):
        assert (tmp_path / folder / "user.asset").read_bytes() == b"preserve"
    assert not list(tmp_path.glob(".prepare-evaluation-*"))


def test_skips_each_bad_package_and_keeps_only_successful_indices(tmp_path, decoded):
    write_package(tmp_path, folder="good")
    tempo = write_package(tmp_path, folder="tempo", changes={"_BPMChanges": [{"_time": 2, "_BPM": 150}]})
    missing = write_package(tmp_path, folder="missing")
    (missing / "source.egg").unlink()
    no_expert_plus = write_package(tmp_path, folder="expert")
    info_path = no_expert_plus / "Info.dat"
    info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]["_difficulty"] = "Expert"
    info_path.write_text(json.dumps(info))
    no_info = tmp_path / "bs_map_input/no-info"
    no_info.mkdir()
    manifest = preparation.prepare_evaluation(tmp_path)
    assert manifest["input_count"] == 5
    assert manifest["success_count"] == 1
    assert manifest["skip_count"] == 4
    assert {Path(row["package_path"]).name for row in manifest["skipped"]} == {"tempo", "missing", "expert", "no-info"}
    assert any("tempo" in row["error"] for row in manifest["skipped"] if row["package_path"] == str(tempo))
    assert np.load(tmp_path / "songs_diff/name_ar.npy").tolist() == ["Human Song"]
    assert len(list((tmp_path / "maps_dict_all").iterdir())) == 3


def test_canonical_names_cover_windows_collisions_and_map_info_suffix(tmp_path, decoded):
    write_package(tmp_path, folder="1abc (A)", title='A/B:*?')
    write_package(tmp_path, folder="2def (B)", title='a\\b:*?')
    write_package(tmp_path, folder="3aaa (C)", title="Song")
    write_package(tmp_path, folder="4bbb (D)", title="Song_info")
    write_package(tmp_path, folder="5ccc (E)", title="CON")
    manifest = preparation.prepare_evaluation(tmp_path)
    names = [song["prepared_name"] for song in manifest["songs"]]
    assert len({name.casefold() for name in names}) == 5
    assert "A_B___" in names
    assert "a_b_____2def" in names
    assert "Song_info__4bbb" in names
    assert "_CON" in names
    assert len(list((tmp_path / "maps").iterdir())) == 10
    assert np.load(tmp_path / "songs_diff/name_ar.npy").tolist() == names


def test_empty_events_and_obstacles_keep_legacy_empty_shape(tmp_path, decoded):
    write_package(tmp_path, changes={"_events": [], "_obstacles": []})
    preparation.prepare_evaluation(tmp_path)
    for kind in ("events", "obstacles"):
        array = np.load(tmp_path / "maps_dict_all" / f"Human Song_{kind}.dat", allow_pickle=True)
        assert array.shape == (0,)


@pytest.mark.parametrize("version", ["2", "3"])
def test_preserves_outside_lane_obstacles_and_large_widths(tmp_path, decoded, version):
    if version == "2":
        changes = {"_obstacles": [
            {"_time": 1, "_lineIndex": -1, "_type": 0, "_duration": 2, "_width": 6},
            {"_time": 2, "_lineIndex": 4, "_type": 1, "_duration": 3, "_width": 8},
        ]}
    else:
        changes = {"obstacles": [
            {"b": 1, "x": -1, "y": 0, "h": 5, "d": 2, "w": 6},
            {"b": 2, "x": 4, "y": 2, "h": 3, "d": 3, "w": 8},
        ]}
    write_package(tmp_path, version=version, changes=changes)
    manifest = preparation.prepare_evaluation(tmp_path)
    assert manifest["success_count"] == 1
    assert manifest["skip_count"] == 0
    np.testing.assert_array_equal(
        np.load(tmp_path / "maps_dict_all/Human Song_obstacles.dat", allow_pickle=True),
        [[0.5, 1.0], [-1, 4], [0, 1], [2, 3], [6, 8]],
    )


@pytest.mark.parametrize("value", [0, -1, 1.5, float("nan")])
def test_invalid_obstacle_width_keeps_object_context(tmp_path, decoded, value):
    write_package(tmp_path, changes={"_obstacles": [
        {"_time": 1, "_lineIndex": -1, "_type": 0, "_duration": 2, "_width": value},
    ]})
    manifest = preparation.prepare_evaluation(tmp_path)
    assert manifest["skip_count"] == 1
    assert "Obstacle 0: _width" in manifest["skipped"][0]["error"]


@pytest.mark.parametrize("changes,error", [
    ({"_notes": []}, "No legacy notes"),
    ({"_notes": [{"_time": 1, "_lineIndex": 4, "_lineLayer": 0, "_type": 0, "_cutDirection": 1}]}, "_lineIndex"),
    ({"_notes": [{"_time": 1, "_lineIndex": 1.2, "_lineLayer": 0, "_type": 0, "_cutDirection": 1}]}, "integer"),
    ({"_notes": [{"_time": 1, "_lineIndex": 0, "_lineLayer": 0, "_type": 0, "_cutDirection": 1}]}, "distinct times"),
    ({"_events": [{"_time": 1, "_type": 100, "_value": 120}]}, "tempo change"),
    ({"_obstacles": [{"_time": 1, "_lineIndex": 0, "_type": 0, "_duration": -1, "_width": 1}]}, "_duration"),
    ({"_events": [{"_time": 1, "_type": 0, "_value": float("nan")}]}, "_value"),
])
def test_per_map_numeric_failures_are_reported_without_partial_outputs(tmp_path, decoded, changes, error):
    write_package(tmp_path, changes=changes)
    manifest = preparation.prepare_evaluation(tmp_path)
    assert manifest["success_count"] == 0
    assert manifest["skip_count"] == 1
    assert error in manifest["skipped"][0]["error"]
    assert not list((tmp_path / "maps").iterdir())
    assert not list((tmp_path / "songs_egg").iterdir())
    assert not list((tmp_path / "maps_dict_all").iterdir())
    assert np.load(tmp_path / "songs_diff/name_ar.npy").size == 0


def test_rejects_unrepresentable_v3_obstacles_per_map(tmp_path, decoded):
    write_package(tmp_path, version="3", changes={"obstacles": [{"b": 1, "x": 0, "y": 1, "h": 2, "d": 1, "w": 1}]})
    manifest = preparation.prepare_evaluation(tmp_path)
    assert manifest["success_count"] == 0
    assert "geometry cannot fit" in manifest["skipped"][0]["error"]


def test_full_ogg_stream_is_decode_checked_without_rewriting(tmp_path):
    import soundfile as sf

    package = write_package(tmp_path)
    samples = np.sin(2 * np.pi * 220 * np.arange(14800 * 3) / 14800).astype(np.float32) * 0.1
    sf.write(package / "source.egg", samples, 14800, format="OGG", subtype="VORBIS")
    before = (package / "source.egg").read_bytes()
    manifest = preparation.prepare_evaluation(tmp_path)
    assert manifest["success_count"] == 1
    # Aubio's decoded Ogg stream can include a small amount of codec padding.
    assert manifest["songs"][0]["audio_duration_seconds"] == pytest.approx(3.0, abs=0.04)
    assert (tmp_path / "songs_egg/Human Song.egg").read_bytes() == before


def test_audio_decode_errors_skip_only_that_map(tmp_path, monkeypatch):
    write_package(tmp_path, folder="bad")
    write_package(tmp_path, folder="good", title="Good Song")
    def decode(path):
        if path.parent.name == "bad":
            raise RuntimeError("decoder rejected source")
        return 10
    monkeypatch.setattr(preparation, "_decode_audio", decode)
    manifest = preparation.prepare_evaluation(tmp_path)
    assert manifest["success_count"] == manifest["skip_count"] == 1
    assert "decoder rejected source" in manifest["skipped"][0]["error"]
    assert np.load(tmp_path / "songs_diff/name_ar.npy").tolist() == ["Good Song"]


def test_non_ogg_source_is_reported_and_preserved(tmp_path):
    package = write_package(tmp_path)
    audio = package / "source.egg"
    audio.write_bytes(b"RIFF original source")
    manifest = preparation.prepare_evaluation(tmp_path)
    assert "require Ogg audio" in manifest["skipped"][0]["error"]
    assert audio.read_bytes() == b"RIFF original source"


def test_failed_promotion_restores_all_old_assets_and_report(tmp_path, decoded, monkeypatch):
    write_package(tmp_path)
    for folder in preparation.DERIVED_FOLDERS:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "old.asset").write_bytes(b"old data")
    (tmp_path / preparation.REPORT_NAME).write_bytes(b"old report")
    before = contents(tmp_path)
    rename = Path.rename
    failed = False
    def fail_once(path, target):
        nonlocal failed
        if path.parent.name.startswith(".prepare-evaluation-") and path.name == "songs_egg" and not failed:
            failed = True
            raise OSError("promotion failure")
        return rename(path, target)
    monkeypatch.setattr(Path, "rename", fail_once)
    with pytest.raises(OSError, match="promotion failure"):
        preparation.prepare_evaluation(tmp_path)
    assert contents(tmp_path) == before
    assert not list(tmp_path.glob(".prepare-evaluation-*"))


def test_missing_or_empty_source_is_rejected_without_changing_existing_data(tmp_path):
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps/old.dat").write_bytes(b"keep")
    with pytest.raises(ValueError, match="input folder"):
        preparation.prepare_evaluation(tmp_path)
    (tmp_path / "bs_map_input").mkdir()
    with pytest.raises(ValueError, match="No input package"):
        preparation.prepare_evaluation(tmp_path)
    assert (tmp_path / "maps/old.dat").read_bytes() == b"keep"


def test_external_package_symlink_is_skipped_before_reading_source(tmp_path, decoded, monkeypatch):
    root = tmp_path / "evaluation"
    write_package(root, folder="good")
    external = write_package(tmp_path / "external", folder="forbidden")
    link = root / "bs_map_input/external-link"
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlinks unavailable: {error}")
    load_reference = preparation.load_reference
    def selected_only(package, *args):
        assert package != link, "External package must be rejected before source loading"
        return load_reference(package, *args)
    monkeypatch.setattr(preparation, "load_reference", selected_only)
    manifest = preparation.prepare_evaluation(root)
    assert manifest["success_count"] == manifest["skip_count"] == 1
    assert manifest["skipped"][0]["package_path"] == str(link)
    assert "outside bs_map_input" in manifest["skipped"][0]["error"]


def test_symlinked_source_root_is_rejected_before_directory_scan(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "evaluation"
    root.mkdir()
    try:
        (root / "bs_map_input").symlink_to(external, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="real directory"):
        preparation.prepare_evaluation(root)
