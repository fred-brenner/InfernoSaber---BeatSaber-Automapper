"""New model data contracts, using CPU features and disposable source files."""

from copy import deepcopy
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from training.beat_models.contracts import SAMPLE_RATE, Song, decode_note, validate_song
from training.beat_models.data import (
    assign_splits, build_song, hash_file, make_dummy_songs, prepare_songs,
    read_notes, reduce_notes,
)
from training.beat_models.features import decode_audio, extract_features, fit_normalization, normalize_songs
from tools.config import config


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _audio(path, samples=32003, frequency=220):
    data = (np.sin(np.arange(samples) * 2 * np.pi * frequency / SAMPLE_RATE) * 10000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(data.tobytes())


def _v2_note(beat=0.2, color=0, x=1, y=2, direction=1):
    return {"_time": beat, "_type": color, "_lineIndex": x, "_lineLayer": y, "_cutDirection": direction}


def _package(root, name="one", beatmaps=None, frequency=220):
    package = root / name
    package.mkdir(parents=True)
    _audio(package / "song.wav", frequency=frequency)
    beatmaps = beatmaps or {"Expert.dat": {"_version": "2.2.0", "_notes": [_v2_note()], "_events": []}}
    for filename, data in beatmaps.items():
        _json(package / filename, data)
    _json(package / "Info.dat", {"_version": "2.1.0", "_beatsPerMinute": 120,
          "_songFilename": "song.wav", "_songTimeOffset": 0, "_shuffle": 0,
          "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
              {"_beatmapFilename": filename, "_difficulty": filename.split(".")[0]} for filename in beatmaps]}]})
    return package


def _arrays(duration=2.1):
    return extract_features(np.zeros(round(duration * SAMPLE_RATE), dtype=np.float32))


def test_decoder_trims_last_block_and_features_cover_partial_tail(tmp_path):
    path = tmp_path / "short.wav"
    _audio(path, samples=16003)
    samples = decode_audio(path)
    assert len(samples) == 16003
    arrays, meta = extract_features(samples)
    assert meta["duration"] == 16003 / SAMPLE_RATE
    assert arrays["features"].shape == (101, 82)
    assert arrays["coarse_features"].shape == (2, 82)
    np.testing.assert_allclose(arrays["bin_durations"], [1, 3 / SAMPLE_RATE])
    assert arrays["frame_times"][-1] < meta["duration"]
    np.testing.assert_allclose(np.expm1(arrays["coarse_features"][0]),
                               np.expm1(arrays["features"][:100]).mean(axis=0), rtol=2e-5, atol=1e-7)
    tiny, tiny_meta = extract_features(np.zeros(1, dtype=np.float32))
    assert tiny["features"].shape == (1, 82)
    assert tiny_meta["duration"] == 1 / SAMPLE_RATE
    assert not tiny["features"].any()


def test_first_cut_projection_and_source_order_fallback_preserve_provenance():
    # Down keeps the highest cell; right keeps the leftmost. Conflicts/dots and
    # projection ties preserve the original input order, including arbitrary IDs.
    rows = np.asarray([
        [9, 1, .5, 0, 1, 0, 1, 0], [4, 1, .5, 0, 1, 2, 1, 0],
        [7, 1, .5, 1, 3, 1, 3, 0], [2, 1, .5, 1, 0, 1, 3, 0],
        [8, 2, 1, 0, 3, 2, 8, 0], [1, 2, 1, 0, 0, 0, 8, 0],
        [6, 3, 1.5, 0, 3, 2, 0, 0], [0, 3, 1.5, 0, 0, 0, 1, 0],
        [5, 3, 1.5, 1, 0, 2, 1, 0], [3, 3, 1.5, 1, 3, 2, 1, 0],
    ], dtype=float)
    kept, reductions = reduce_notes(rows)
    assert kept.tolist() == [1, 3, 4, 6, 8]
    assert [item["reason"] for item in reductions] == ["first_projection", "first_projection", "dots_source_order",
                                                         "conflicting_directions_source_order", "projection_tie_source_order"]
    arrays, meta = _arrays()
    meta["bpm"] = 120
    song = build_song("projection", rows, arrays, meta)
    assert song.metadata["retained_colored_note_count"] == 5
    assert song.metadata["unique_onset_count"] == 3
    np.testing.assert_array_equal(song.arrays["original_notes"], rows)
    assert decode_note(song.arrays["note_classes"][0, 0]) == (1, 2, 1)
    np.testing.assert_allclose(song.arrays["intensity"], [.4, .6, 0])


def test_near_onsets_synchronize_silently_and_preserve_original_records(capsys):
    arrays, meta = _arrays()
    meta["bpm"] = 120
    # The highest downward note occurs later and first in arbitrary source order.
    rows = np.array([[0, .2000143, .10000715, 0, 1, 2, 1, 0],
                     [1, .2, .100, 0, 1, 0, 1, 0],
                     [2, .201, .1005, 1, 0, 0, 0, 0]])
    before = rows.copy()
    song = build_song("collision", rows, arrays, meta)
    np.testing.assert_array_equal(rows, before)
    np.testing.assert_array_equal(song.arrays["original_notes"], before)
    np.testing.assert_array_equal(song.arrays["original_event_indices"], [0, 0, 0])
    np.testing.assert_array_equal(song.arrays["retained_indices"], [0, 2])
    np.testing.assert_array_equal(song.arrays["event_times"], [.100])
    assert song.arrays["event_frames"].tolist() == [10]
    assert song.arrays["timing"][10] == 2
    assert song.metadata["reduction_time_epsilon_seconds"] == config.min_time_diff
    assert song.metadata["synchronized_onset_merge_count"] == 2
    assert capsys.readouterr().out == ""
    assert validate_song(song)
    song = build_song("nearest", [[0, .21, .105, 0, 0, 0, 0, 0]], arrays, meta)
    assert song.arrays["event_frames"].tolist() == [11]


def test_synchronization_uses_config_and_first_event_anchor(monkeypatch):
    arrays, meta = _arrays()
    meta["bpm"] = 120
    monkeypatch.setattr(config, "min_time_diff", .02)
    # Inclusive boundary, with no chaining through the preceding merged note.
    rows = [[i, when * 2, when, 0, i % 4, 0, 8, 0]
            for i, when in enumerate([.1, .115, .12, .13])]
    song = build_song("anchored", rows, arrays, meta)
    np.testing.assert_array_equal(song.arrays["event_times"], [.1, .13])
    np.testing.assert_array_equal(song.arrays["original_event_indices"], [0, 0, 0, 1])
    np.testing.assert_array_equal(song.arrays["retained_indices"], [0, 3])
    monkeypatch.setattr(config, "min_time_diff", .005)
    song = build_song("configured", rows, arrays, meta)
    assert song.metadata["reduction_time_epsilon_seconds"] == .005
    assert song.metadata["unique_onset_count"] == 3
    monkeypatch.setattr(config, "min_time_diff", 0)
    with pytest.raises(ValueError, match="Distinct source onsets collide"):
        build_song("disabled", [[0, .2, .1, 0, 0, 0, 8, 0],
                                [1, .201, .1005, 1, 0, 0, 8, 0]], arrays, meta)


def test_synchronized_intensity_uses_event_time_across_bin_boundary():
    arrays, meta = _arrays()
    meta["bpm"] = 120
    rows = [[0, 1.998, .999, 0, 0, 0, 8, 0], [1, 2.01, 1.005, 1, 0, 0, 8, 0]]
    song = build_song("bin-boundary", rows, arrays, meta)
    np.testing.assert_array_equal(song.arrays["event_times"], [.999])
    np.testing.assert_array_equal(song.arrays["intensity"], [1, 0, 0])
    assert validate_song(song)
    song.arrays["original_event_indices"][1] = 1
    with pytest.raises(ValueError, match="synchroniz"):
        validate_song(song)


@pytest.mark.parametrize("modification,reason", [
    ({"bpmEvents": [{"b": 1, "m": 130}]}, "tempo changes"),
    ({"_customData": {"_BPMChanges": [{"_time": 1}]}}, "tempo changes"),
    ({"_events": [{"_type": 100}]}, "event type 100"),
    ({"_songTimeOffset": .1}, "nonzero"),
    ({"_shuffle": .5}, "nonzero"),
])
def test_unsupported_timing_skips_individually_without_mutation(tmp_path, modification, reason):
    good = {"_version": "2.2.0", "_notes": [_v2_note()], "_events": []}
    bad = {**good, **modification}
    package = _package(tmp_path, beatmaps={"Good.dat": good, "Bad.dat": bad})
    before = {path.name: hash_file(path) for path in package.iterdir()}
    songs, report = prepare_songs(tmp_path, reserved_root=tmp_path / "absent")
    assert len(songs) == 1
    assert report["skip_count"] == 1
    assert reason in report["skipped"][0]["reason"]
    assert {path.name: hash_file(path) for path in package.iterdir()} == before


def test_v3_defaults_bombs_angles_and_extended_geometry(tmp_path):
    path = tmp_path / "map.dat"
    _json(path, {"version": "3.3.0", "colorNotes": [{}, {"b": 1, "c": 1, "x": 3, "d": 8}],
                 "bombNotes": [{"b": .5, "x": 2, "y": 2}]})
    rows = read_notes(path, 120)
    assert rows.shape == (2, 8)
    np.testing.assert_allclose(rows[0], np.zeros(8))
    for note in ({"a": 10}, {"x": 4}, {"y": .5}, {"customData": {"coordinates": [1, 2], "position": [0, 0]}}):
        _json(path, {"version": "3.3.0", "colorNotes": [note]})
        with pytest.raises(ValueError):
            read_notes(path, 120)


def test_original_json_records_preserve_defaults_custom_metadata_and_source_order(tmp_path):
    records = [{"b": .2, "customData": {"label": "first", "extra": [1, {"value": True}]}},
               {"b": .2, "x": 3, "c": 1, "d": 8, "customData": {"label": "second"}}]
    package = _package(tmp_path, beatmaps={"Expert.dat": {"version": "3.3.0", "colorNotes": records}})
    source = package / "Expert.dat"
    before = source.read_bytes()
    songs, _ = prepare_songs(tmp_path, reserved_root=tmp_path / "absent")
    song = songs[0]
    assert song.metadata["original_note_records"] == records
    assert song.metadata["original_note_records_provenance"] == "verbatim_source_json"
    assert song.metadata["original_note_record_source_indices"] == [0, 1]
    assert "x" not in song.metadata["original_note_records"][0]
    song.metadata["original_note_records"][0]["customData"]["extra"][1]["value"] = False
    assert source.read_bytes() == before
    assert records[0]["customData"]["extra"][1]["value"] is True
    assert make_dummy_songs()[0].metadata["original_note_records_provenance"] == "synthesized_from_normalized_columns"


def test_prepared_adapter_requires_exact_triples_and_marks_metadata_unverified(tmp_path):
    maps, audio = tmp_path / "maps", tmp_path / "songs_egg"
    maps.mkdir()
    audio.mkdir()
    _audio(audio / "copy.egg")
    _json(maps / "copy.dat", {"_version": "2.0.0", "_notes": [_v2_note(color=1)]})
    _json(maps / "copy_info.dat", {"_version": "2.1.0", "_beatsPerMinute": 120, "_songFilename": "original-not-present.egg"})
    _json(maps / "orphan.dat", {"_version": "2.0.0", "_notes": []})
    songs, report = prepare_songs(prepared_root=tmp_path, reserved_root=tmp_path / "absent")
    assert len(songs) == 1
    assert report["skip_count"] == 1
    assert songs[0].metadata["provenance"]["metadata_verified"] is False
    assert songs[0].metadata["provenance"]["difficulty"] is None


@pytest.mark.parametrize("declared,actual", [
    ("ExpertPlusStandard.dat", "ExpertPlus.dat"),
    ("ExpertPlus.dat", "ExpertPlusStandard.dat"),
    ("ExpertPlusStandard.dat", "expertplus.dat"),
    ("ExpertPlus.dat", "EXPERTPLUS.DAT"),
    ("HardStandard.dat", "Hard.dat"),
])
def test_raw_import_resolves_standard_aliases_without_changing_sources(tmp_path, declared, actual):
    package = _package(tmp_path, beatmaps={actual: {"_version": "2.2.0", "_notes": [_v2_note()]}})
    path = package / "Info.dat"
    info = json.loads(path.read_text())
    item = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
    item.update({"_difficulty": "Hard" if declared.startswith("Hard") else "ExpertPlus",
                 "_beatmapFilename": declared})
    _json(path, info)
    before = {p.name: hash_file(p) for p in package.iterdir()}
    songs, report = prepare_songs(tmp_path, reserved_root=tmp_path / "absent")
    assert len(songs) == 1 and report["skip_count"] == 0
    provenance = songs[0].metadata["provenance"]
    assert provenance["beatmap_filename_declared"] == declared
    assert provenance["beatmap_filename_resolved"] == actual
    assert provenance["beatmap_filename_resolution"] in ("standard_filename_alias", "case_insensitive")
    assert Path(songs[0].metadata["sources"][1]["path"]).name == actual
    assert {p.name: hash_file(p) for p in package.iterdir()} == before


def test_exact_declared_beatmap_wins_over_legacy_alias(tmp_path):
    exact = {"_version": "2.2.0", "_notes": [_v2_note(color=1)]}
    other = {"_version": "2.2.0", "_notes": [_v2_note(color=0)]}
    package = _package(tmp_path, beatmaps={"ExpertPlusStandard.dat": exact, "ExpertPlus.dat": other})
    path = package / "Info.dat"
    info = json.loads(path.read_text())
    info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"] = [
        {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlusStandard.dat"}]
    _json(path, info)
    songs, report = prepare_songs(tmp_path, reserved_root=tmp_path / "absent")
    assert len(songs) == 1 and report["skip_count"] == 0
    assert songs[0].metadata["provenance"]["beatmap_filename_resolution"] == "declared"
    assert songs[0].arrays["note_classes"][0, 0] == 0


@pytest.mark.parametrize("declared", ["CustomExpertPlus.dat", "../ExpertPlusStandard.dat"])
def test_filename_fallback_never_guesses_arbitrary_or_external_references(tmp_path, declared):
    package = _package(tmp_path, beatmaps={"ExpertPlus.dat": {"_version": "2.2.0", "_notes": [_v2_note()]}})
    path = package / "Info.dat"
    info = json.loads(path.read_text())
    info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]["_beatmapFilename"] = declared
    _json(path, info)
    songs, report = prepare_songs(tmp_path, reserved_root=tmp_path / "absent")
    assert not songs and report["skip_count"] == 1
    assert "Missing map package file" in report["skipped"][0]["reason"] or "inside the map package" in report["skipped"][0]["reason"]


def test_alias_referenced_by_another_characteristic_is_not_reused(tmp_path):
    package = _package(tmp_path, beatmaps={"ExpertPlus.dat": {"_version": "2.2.0", "_notes": [_v2_note()]}})
    path = package / "Info.dat"
    info = json.loads(path.read_text())
    info["_difficultyBeatmapSets"] = [
        {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlusStandard.dat"}]},
        {"_beatmapCharacteristicName": "OneSaber", "_difficultyBeatmaps": [
            {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlus.dat"}]},
    ]
    _json(path, info)
    songs, report = prepare_songs(tmp_path, reserved_root=tmp_path / "absent")
    assert not songs
    assert any("Ambiguous" in row["reason"] for row in report["skipped"])


def test_case_ambiguous_fallback_is_reported_without_selecting_a_map(tmp_path):
    good = {"_version": "2.2.0", "_notes": [_v2_note()]}
    package = _package(tmp_path, beatmaps={"ExpertPlus.dat": good, "expertplus.dat": good})
    matches = [p for p in package.iterdir() if p.name.casefold() == "expertplus.dat"]
    if len(matches) != 2:
        pytest.skip("Filesystem does not support names differing only by case")
    path = package / "Info.dat"
    info = json.loads(path.read_text())
    info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"] = [
        {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlusStandard.dat"}]
    _json(path, info)
    songs, report = prepare_songs(tmp_path, reserved_root=tmp_path / "absent")
    assert not songs and report["skip_count"] == 1
    assert "Ambiguous" in report["skipped"][0]["reason"]


def test_malformed_sibling_reference_does_not_block_valid_alias(tmp_path):
    package = _package(tmp_path, beatmaps={"ExpertPlus.dat": {"_version": "2.2.0", "_notes": [_v2_note()]}})
    path = package / "Info.dat"
    info = json.loads(path.read_text())
    info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"] = [
        {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlusStandard.dat"},
        {"_difficulty": "Hard", "_beatmapFilename": "invalid\0.dat"},
    ]
    _json(path, info)
    songs, report = prepare_songs(tmp_path, reserved_root=tmp_path / "absent")
    assert len(songs) == 1 and report["skip_count"] == 1
    assert report["skipped"][0]["beatmap_filename"] == "invalid\0.dat"
    assert songs[0].metadata["provenance"]["beatmap_filename_resolved"] == "ExpertPlus.dat"


def test_alias_collection_does_not_preempt_valid_group_with_malformed_sibling(tmp_path):
    package = _package(tmp_path, beatmaps={"ExpertPlus.dat": {"_version": "2.2.0", "_notes": [_v2_note()]}})
    path = package / "Info.dat"
    info = json.loads(path.read_text())
    info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]["_beatmapFilename"] = "ExpertPlusStandard.dat"
    info["_difficultyBeatmapSets"].append({"_beatmapCharacteristicName": "OneSaber", "_difficultyBeatmaps": None})
    _json(path, info)
    songs, report = prepare_songs(tmp_path, reserved_root=tmp_path / "absent")
    assert len(songs) == 1 and report["skip_count"] == 1
    assert "_difficultyBeatmaps" in report["skipped"][0]["reason"]


def test_curated_recording_and_aliases_are_reserved(tmp_path):
    raw, curated = tmp_path / "raw", tmp_path / "benchmark"
    first = _package(raw, "copy")
    _package(raw, "other", frequency=330)
    benchmark = _package(curated, "human")
    # Same source bytes identify overlap regardless of package/beatmap name.
    assert hash_file(first / "song.wav") == hash_file(benchmark / "song.wav")
    songs, report = prepare_songs(raw, reserved_root=curated)
    assert len(songs) == 1 and report["reserved_count"] == 1
    assert "other" in songs[0].metadata["sources"][0]["path"]
    songs, report = prepare_songs(raw, reserved_root=curated,
                                 aliases=[[str(raw / "other" / "song.wav"), str(benchmark / "song.wav")]])
    assert not songs and report["reserved_count"] == 2


def test_recording_splits_and_train_only_normalization():
    songs = make_dummy_songs()
    assert len({song.metadata["group_id"] for song in songs}) >= 10
    assert {song.metadata["split"] for song in songs} == {"train", "validation", "test"}
    assert any(not song.metadata["intensity_target_valid"] for song in songs)
    duplicate = Song("another-difficulty", deepcopy(songs[0].metadata), songs[0].arrays)
    duplicate.metadata["decoded_sha256"] = "alternate-decode-id"
    grouped = assign_splits([*songs, duplicate])
    assert grouped[0].metadata["split"] == duplicate.metadata["split"]
    assert grouped[0].metadata["group_id"] == duplicate.metadata["group_id"]
    stats = fit_normalization(songs)
    held_out = next(song for song in songs if song.metadata["split"] == "test")
    held_out.arrays["features"][:] = 1e6
    held_out.arrays["coarse_features"][:] = 1e7
    assert fit_normalization(songs) == stats
    normalized = normalize_songs(songs, stats)
    train = [song for song in normalized if song.metadata["split"] == "train"]
    for key in ("features", "coarse_features"):
        np.testing.assert_allclose(np.concatenate([song.arrays[key] for song in train]).mean(axis=0), 0, atol=2e-6)
    assert held_out.arrays["features"][0, 0] == 1e6
    assert stats["features"]["mean"] != stats["coarse_features"]["mean"]
    assert all(validate_song(song) for song in normalized)
    from training.beat_models.artifacts import statistics_hash
    assert {song.metadata["normalization_sha256"] for song in normalized} == {statistics_hash(stats)}
    with pytest.raises(ValueError, match="already normalized"):
        normalize_songs(normalized, stats)
    with pytest.raises(ValueError, match="raw training features"):
        fit_normalization(normalized)


def test_evaluation_only_retains_curated_without_any_training_split(tmp_path):
    curated = tmp_path / "curated"
    good = {"_version": "2.2.0", "_notes": [_v2_note()]}
    _package(curated, beatmaps={"ExpertPlus.dat": good, "Easy.dat": good})
    songs, report = prepare_songs(curated, reserved_root=curated,
                                 evaluation_only=True, difficulty="ExpertPlus")
    assert len(songs) == 1 and report["reserved_count"] == 1
    assert report["role"] == "evaluation"
    assert songs[0].metadata["provenance"]["difficulty"] == "ExpertPlus"
    assert {song.metadata["split"] for song in songs} == {"test"}
    with pytest.raises(ValueError, match="training song"):
        fit_normalization(songs)
    training, training_report = prepare_songs(curated, reserved_root=curated, difficulty="ExpertPlus")
    assert not training and training_report["reserved_count"] == 1
    missing, missing_report = prepare_songs(curated, reserved_root=curated,
                                           evaluation_only=True, difficulty="Hard")
    assert not missing and missing_report["skip_count"] == 1
    assert "requested difficulty" in missing_report["skipped"][0]["reason"]
    with pytest.raises(ValueError, match="unverified difficulty"):
        prepare_songs(prepared_root=tmp_path, difficulty="ExpertPlus")
