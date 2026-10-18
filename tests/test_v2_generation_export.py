"""Structured generation exports and complete-package publication contracts."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from map_creation import v2_export as export
from tools.config import config, paths


@pytest.fixture
def state():
    notes = [{"id": 0, "time": 1., "hand": 0, "x": 0, "y": 0, "d": 1, "kind": "retained"},
             {"id": 1, "time": 2., "hand": 0, "x": 2, "y": 1, "d": 4, "kind": "retained"},
             {"id": 2, "time": 1., "hand": 1, "x": 3, "y": 2, "d": 8, "kind": "copy"}]
    return {"notes": notes, "bombs": [{"time": .75, "x": 1, "y": 1}],
            "walls": [{"time": 3., "end_time": 4., "x": 0, "y": 0, "width": 1, "height": 5},
                      {"time": 4., "end_time": 4.5, "x": 0, "y": 2, "width": 4, "height": 3},
                      {"time": 5., "end_time": 6., "x": 2, "y": 1, "width": 1, "height": 2}],
            "chains": [{**notes[0], "anchor_id": 0, "time": 1., "end_time": 1.5,
                        "tx": 1, "ty": 1, "slices": 4, "squish": .75}],
            "arcs": [{"time": 1., "end_time": 2., "hand": 0, "endpoint_ids": [0, 1],
                      "mu": .5, "tmu": 1.5, "anchor_mode": 2}],
            "lighting": [{"time": .25, "type": 1, "value": 5},
                         {"time": .25, "type": 0, "value": 3},
                         {"time": 2.5, "type": 2, "value": 0}]}


def _read_back(document, bpm=120.):
    from training.pipeline_v2.data import read_objects
    v2 = "_notes" in document
    notes = document["_notes"] if v2 else document["colorNotes"]
    indices = [i for i, note in enumerate(notes) if note.get("_type", note.get("c")) in (0, 1)]
    original = np.zeros((len(indices), 6))
    original[:, 0] = indices
    base = SimpleNamespace(metadata={"raw_beatmap": document, "bpm": bpm, "duration": 10.},
                           arrays={"original_notes": original, "retained_indices": np.arange(len(indices)),
                                   "original_event_indices": np.arange(len(indices))})
    return read_objects(base)


def test_native_v3_serializes_each_family_once_with_independent_lighting(state):
    before = deepcopy(state)
    document, omissions = export.serialize_beatmap(state, 120)
    assert document["version"] == "3.2.0"
    assert document["colorNotes"][0] == {"b": 2., "x": 0, "y": 0, "c": 0, "d": 1, "a": 0}
    assert document["bombNotes"] == [{"b": 1.5, "x": 1, "y": 1}]
    assert document["obstacles"][0] == {"b": 6., "d": 2., "x": 0, "y": 0, "w": 1, "h": 5}
    assert document["sliders"] == [{"b": 2., "x": 0, "y": 0, "c": 0, "d": 1, "tb": 4.,
                                    "tx": 2, "ty": 1, "tc": 4, "mu": .5, "tmu": 1.5, "m": 2}]
    assert document["burstSliders"] == [{"b": 2., "x": 0, "y": 0, "c": 0, "d": 1,
                                         "tb": 3., "tx": 1, "ty": 1, "sc": 4, "s": .75}]
    assert [event["b"] for event in document["basicBeatmapEvents"]] == [.5, .5, 5.]
    assert {event["et"] for event in document["basicBeatmapEvents"][:2]} == {0, 1}
    assert omissions == {"arcs": 0, "chain_bodies": 0, "walls": 0}
    assert state == before
    objects, diagnostic = _read_back(document)
    assert diagnostic["ignored_records"] == 0
    assert {family: len(objects[family]) for family in export.FAMILIES} == {
        "notes": 3, "bombs": 1, "walls": 3, "chains": 1, "arcs": 1, "lighting": 3}
    assert objects["arcs"][0]["time"] == 1.
    assert objects["chains"][0]["end_time"] == 1.5
    assert objects["walls"][0]["duration"] == 1.
    assert [event["time"] for event in objects["lighting"]] == [.25, .25, 2.5]


def test_legacy_retains_chain_heads_and_reports_unrepresentable_families(state):
    document, omissions = export.serialize_beatmap(state, 120, "v2")
    assert document["_version"] == "2.0.0"
    assert len(document["_notes"]) == 4
    assert document["_notes"][0]["_type"] == 3
    assert len(document["_obstacles"]) == 2
    assert [wall["_type"] for wall in document["_obstacles"]] == [0, 1]
    assert document["_obstacles"][0]["_duration"] == 2.
    assert len(document["_events"]) == 3
    assert omissions == {"arcs": 1, "chain_bodies": 1, "walls": 1}
    objects, diagnostic = _read_back(document)
    assert diagnostic["ignored_records"] == 0
    assert len(objects["notes"]) == 3
    assert len(objects["bombs"]) == 1
    assert len(objects["arcs"]) == len(objects["chains"]) == 0


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_empty_families_are_valid_and_lighting_does_not_need_notes(version):
    document, _ = export.serialize_beatmap({}, 100, version)
    objects, diagnostic = _read_back(document, 100)
    assert not any(objects.values())
    assert diagnostic["ignored_records"] == 0
    document, _ = export.serialize_beatmap({"lighting": [{"time": 1., "type": 0, "value": 1}]}, 60, version)
    objects, _ = _read_back(document, 60)
    assert len(objects["lighting"]) == 1
    assert not objects["notes"]


@pytest.mark.parametrize("bpm", [0, -1, float("nan"), float("inf"), True])
def test_bpm_must_be_finite_positive(bpm):
    with pytest.raises(ValueError, match="BPM"):
        export.serialize_beatmap({}, bpm)


@pytest.mark.parametrize("family,index,key,value", [
    ("notes", 0, "time", float("nan")), ("notes", 0, "x", 4),
    ("notes", 0, "d", 9), ("notes", 0, "id", 1),
    ("bombs", 0, "y", -1), ("walls", 0, "end_time", 3.),
    ("walls", 0, "width", 5), ("chains", 0, "anchor_id", 999),
    ("chains", 0, "slices", 1), ("chains", 0, "squish", 0),
    ("arcs", 0, "endpoint_ids", [0, 999]), ("arcs", 0, "hand", 1),
    ("arcs", 0, "mu", float("inf")), ("arcs", 0, "anchor_mode", 3),
    ("lighting", 0, "type", 100), ("lighting", 0, "time", -1),
])
def test_invalid_geometry_and_orphan_references_are_rejected(state, family, index, key, value):
    state[family][index][key] = value
    with pytest.raises(ValueError):
        export.serialize_beatmap(state, 120)


@pytest.fixture
def assets(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    audio = inputs / "song.egg"
    audio.write_bytes(b"unchanged audio source")
    (inputs / "song.json").write_text(json.dumps({"title": 'Title "quoted"', "artist": "Artist", "album": "Album"}))
    cover = inputs / "cover.jpg"
    cover.write_bytes(b"unchanged cover source")
    monkeypatch.setattr(paths, "songs_pred", inputs.as_posix() + "/")
    monkeypatch.setattr(paths, "new_map_path", (tmp_path / "output").as_posix() + "/")
    return audio, cover


def _variant(state, difficulty="ExpertPlus", internal=20.):
    return {"difficulty": difficulty, "internal_difficulty": internal, "state": state,
            "diagnostics": {"retained_notes": np.int64(2)}}


def test_package_metadata_each_variant_njs_and_assets_are_preserved(tmp_path, assets, state):
    from map_creation.map_creator import get_info_map_string
    audio, cover = assets
    before = config.max_speed
    target = tmp_path / "package"
    report = export.write_package(target, "song", 120, [_variant(state, "Easy", 8.), _variant(state)], audio, cover, "v3")
    assert report["path"] == str(target)
    info = json.loads((target / "info.dat").read_text())
    assert info["_songName"] == 'Title "quoted"'
    assert info["_songAuthorName"] == "Artist"
    assert info["_songSubName"] == "Album"
    assert info["_songFilename"] == "song.egg"
    assert (target / "song.egg").read_bytes() == audio.read_bytes()
    assert (target / "cover.jpg").read_bytes() == cover.read_bytes()
    entries = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
    assert [entry["_difficultyRank"] for entry in entries] == [1, 9]
    for entry, speed in zip(entries, (8., 20.)):
        expected = json.loads(get_info_map_string("song", 120, "ExpertPlus", max_speed=speed))
        expected_entry = expected["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
        assert entry["_noteJumpMovementSpeed"] == expected_entry["_noteJumpMovementSpeed"]
        assert entry["_noteJumpStartBeatOffset"] == expected_entry["_noteJumpStartBeatOffset"]
        assert (target / entry["_beatmapFilename"]).is_file()
    assert config.max_speed == before
    assert report["variants"][0]["diagnostics"] == {"retained_notes": 2}
    assert not (target / "generation_report.json").exists()
    assert len(list(audio.parent.iterdir())) == 3


def test_failed_variant_never_changes_an_existing_package(tmp_path, assets, state):
    audio, cover = assets
    target = tmp_path / "package"
    target.mkdir()
    (target / "keep.txt").write_text("old complete package")
    bad = _variant(state, "Hard")
    bad["state"] = {"notes": [{"bad": "invalid"}]}
    with pytest.raises((ValueError, KeyError)):
        export.write_package(target, "song", 120, [_variant(state), bad], audio, cover)
    assert list(target.iterdir()) == [target / "keep.txt"]
    assert not list(tmp_path.glob(".*.staging-*"))


def test_publication_rename_failure_restores_the_previous_complete_package(tmp_path, assets, state, monkeypatch):
    audio, cover = assets
    target = tmp_path / "package"
    target.mkdir()
    (target / "keep.txt").write_text("old complete package")
    original = export.os.replace
    def fail_staging(source, destination):
        if ".staging-" in Path(source).name:
            raise OSError("simulated publication failure")
        return original(source, destination)
    monkeypatch.setattr(export.os, "replace", fail_staging)
    with pytest.raises(OSError, match="publication"):
        export.write_package(target, "song", 120, [_variant(state)], audio, cover)
    assert (target / "keep.txt").read_text() == "old complete package"
    assert not list(tmp_path.glob(".*.backup-*"))
    assert not list(tmp_path.glob(".*.staging-*"))


def test_replacement_and_zip_publish_complete_packages(tmp_path, assets, state):
    audio, cover = assets
    target = tmp_path / "package"
    target.mkdir()
    (target / "stale.dat").write_text("old")
    export.write_package(target, "song", 100, [_variant(state)], audio, cover, "v2")
    assert not (target / "stale.dat").exists()
    destination = tmp_path / "package.zip"
    destination.write_bytes(b"old archive")
    assert export.write_zip(target, destination) == str(destination)
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {"ExpertPlus.dat", "info.dat", "cover.jpg", "song.egg"}
        info = json.loads(archive.read("info.dat"))
        assert archive.read(info["_songFilename"]) == audio.read_bytes()
        assert json.loads(archive.read("ExpertPlus.dat"))["_version"] == "2.0.0"
    assert not list(tmp_path.glob(".*.tmp"))


def test_zip_write_failure_preserves_previous_archive(tmp_path, assets, state, monkeypatch):
    audio, cover = assets
    target, destination = tmp_path / "package", tmp_path / "package.zip"
    export.write_package(target, "song", 100, [_variant(state)], audio, cover)
    destination.write_bytes(b"old archive")
    monkeypatch.setattr(export.zipfile.ZipFile, "write", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("archive failure")))
    with pytest.raises(OSError, match="archive"):
        export.write_zip(target, destination)
    assert destination.read_bytes() == b"old archive"
    assert not list(tmp_path.glob(".*.tmp"))


def test_combine_uses_explicit_completed_files_preserves_variant_njs_and_sources(tmp_path, assets, state):
    audio, cover = assets
    packages = []
    for difficulty, rank, speed in (("Easy", 1, 8.), ("Hard", 5, 20.)):
        path = tmp_path / difficulty
        export.write_package(path, "song", 120, [_variant(state, internal=speed)], audio, cover)
        packages.append({"path": path, "difficulty": difficulty, "rank": rank, "filename": "ExpertPlus.dat"})
    target, destination = tmp_path / "combined", tmp_path / "combined.zip"
    report = export.combine_packages("song", packages, target, destination)
    assert report["zip_path"] == str(destination)
    info = json.loads((target / "info.dat").read_text())
    entries = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
    assert [entry["_beatmapFilename"] for entry in entries] == ["Easy.dat", "Hard.dat"]
    assert info["_songName"] == 'Title "quoted"'
    assert (target / info["_songFilename"]).read_bytes() == audio.read_bytes()
    for package, entry in zip(packages, entries):
        original = json.loads((package["path"] / "info.dat").read_text())["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
        assert entry["_noteJumpMovementSpeed"] == original["_noteJumpMovementSpeed"]
        assert entry["_noteJumpStartBeatOffset"] == original["_noteJumpStartBeatOffset"]
        assert (target / entry["_beatmapFilename"]).read_bytes() == (package["path"] / "ExpertPlus.dat").read_bytes()
    assert all((package["path"] / "ExpertPlus.dat").is_file() for package in packages)


def test_cover_fallback_uses_repository_asset(tmp_path, assets):
    audio, _ = assets
    target = tmp_path / "package"
    export.write_package(target, "song", 120, [_variant({})], audio)
    original = Path(export.__file__).resolve().parents[1] / "app_helper" / "cover.jpg"
    assert (target / "cover.jpg").read_bytes() == original.read_bytes()


def test_variant_jump_speed_offsets_survive_restored_configuration(tmp_path, assets, state):
    audio, cover = assets
    original = config.jump_speed_offset
    variants = [_variant(state, "Easy"), _variant(state)]
    variants[0]["jump_speed_offset"] = -1.
    variants[1]["jump_speed_offset"] = 1.
    target = tmp_path / "package"
    export.write_package(target, "song", 100, variants, audio, cover)
    info = json.loads((target / "info.dat").read_text())
    entries = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
    assert entries[1]["_noteJumpMovementSpeed"] - entries[0]["_noteJumpMovementSpeed"] == pytest.approx(2.)
    assert config.jump_speed_offset == original


def test_zero_requested_density_keeps_finite_metadata_and_empty_gameplay(tmp_path, assets):
    audio, cover = assets
    target = tmp_path / "package"
    export.write_package(target, "song", 120, [_variant({}, internal=0.)], audio, cover)
    info = json.loads((target / "info.dat").read_text())
    entry = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
    assert np.isfinite(entry["_noteJumpMovementSpeed"])
    assert np.isfinite(entry["_noteJumpStartBeatOffset"])
    objects, diagnostic = _read_back(json.loads((target / "ExpertPlus.dat").read_text()))
    assert not any(objects.values())
    assert diagnostic["ignored_records"] == 0


def test_automatic_expert_keeps_legacy_expert_jump_rules(tmp_path, assets):
    from map_creation.map_creator import get_info_map_string

    audio, cover = assets
    reduced = 40. * config.expert_fact
    variants = [_variant({}, internal=40.), _variant({}, "Expert", reduced)]
    variants[1]["automatic_expert"] = True
    target = tmp_path / "automatic-expert"
    export.write_package(target, "song", 120, variants, audio, cover)
    info = json.loads((target / "info.dat").read_text())
    actual = next(entry for entry in info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
                  if entry["_difficulty"] == "Expert")
    expected_info = json.loads(get_info_map_string("song", 120, "Expert", max_speed=reduced))
    expected = next(entry for entry in expected_info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
                    if entry["_difficulty"] == "Expert")
    for key in ("_noteJumpMovementSpeed", "_noteJumpStartBeatOffset"):
        assert actual[key] == expected[key]

    requested = tmp_path / "requested-expert"
    export.write_package(requested, "song", 120, [_variant({}, "Expert", reduced)], audio, cover)
    requested_entry = json.loads((requested / "info.dat").read_text())["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
    ordinary_info = json.loads(get_info_map_string("song", 120, "ExpertPlus", max_speed=reduced))
    ordinary = ordinary_info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
    for key in ("_noteJumpMovementSpeed", "_noteJumpStartBeatOffset"):
        assert requested_entry[key] == ordinary[key]


@pytest.mark.parametrize("failure", ["missing_variant", "different_audio", "different_bpm"])
def test_combination_failure_preserves_existing_output(tmp_path, assets, state, failure):
    audio, cover = assets
    packages = []
    for difficulty, rank in (("Easy", 1), ("Hard", 5)):
        source = tmp_path / difficulty
        export.write_package(source, "song", 120, [_variant(state)], audio, cover)
        packages.append({"path": source, "difficulty": difficulty, "rank": rank, "filename": "ExpertPlus.dat"})
    second = packages[1]["path"]
    if failure == "missing_variant":
        (second / "ExpertPlus.dat").unlink()
    elif failure == "different_audio":
        (second / "song.egg").write_bytes(b"another recording")
    else:
        info = json.loads((second / "info.dat").read_text())
        info["_beatsPerMinute"] = 100
        (second / "info.dat").write_text(json.dumps(info))
    target = tmp_path / "combined"
    target.mkdir()
    (target / "keep.txt").write_text("previous completed output")
    with pytest.raises((FileNotFoundError, ValueError)):
        export.combine_packages("song", packages, target)
    assert (target / "keep.txt").read_text() == "previous completed output"
    assert list(target.iterdir()) == [target / "keep.txt"]
    assert not list(tmp_path.glob(".*.staging-*"))
