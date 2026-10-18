"""Single-process commands report packaging failures and continue other songs."""

from pathlib import Path
import runpy

import pytest

import main
from tools.config import config, paths


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "songs_pred", str(tmp_path) + "/")
    monkeypatch.setattr(paths, "new_map_path", str(tmp_path) + "/")
    monkeypatch.setattr(main, "validate_bundle", lambda: {})
    monkeypatch.setattr(main, "check_music_files", lambda files, directory: ["first.egg", "second.egg"])
    generated = []

    def generate(name, logger):
        generated.append(name)
        folder = tmp_path / name
        folder.mkdir()
        (folder / "ExpertPlus.dat").write_text("completed gameplay")
        return {"song": name, "path": str(folder)}

    monkeypatch.setattr(main, "generate_song", generate)
    return tmp_path, generated


@pytest.mark.parametrize("failure", ["zip", "library"])
def test_packaging_failure_continues_remaining_songs_without_false_success(inputs, monkeypatch, failure):
    root, generated = inputs
    archived, exported = [], []

    def archive(folder, destination):
        archived.append(Path(folder).name)
        if failure == "zip" and Path(folder).name == "first":
            raise OSError("ZIP destination unavailable")
        return str(destination)

    def copy(name):
        exported.append(name)
        if failure == "library" and name.endswith("_first"):
            raise OSError("Library destination unavailable")
        return True

    monkeypatch.setattr(main, "write_zip", archive)
    monkeypatch.setattr(main, "shutil_copy_maps", copy)
    report = main.main(diff=20., export_results_to_bs=True)
    assert report["status"] == "failed"
    assert generated == archived == ["first", "second"]
    assert [result["song"] for result in report["results"]] == ["second"]
    assert [failure["song"] for failure in report["failures"]] == ["first"]
    assert "destination unavailable" in report["failures"][0]["error"]
    assert (root / "first/ExpertPlus.dat").read_text() == "completed gameplay"
    assert exported == (["20.0_first", "20.0_second"] if failure == "library" else ["20.0_second"])


@pytest.mark.parametrize("fails,exit_code", [(True, 1), (False, 0)])
def test_cli_exit_status_matches_generation_failures(tmp_path, monkeypatch, fails, exit_code):
    from bs_shift import export_map
    from map_creation import v2_generation, v2_runtime

    monkeypatch.setattr(paths, "songs_pred", str(tmp_path) + "/")
    monkeypatch.setattr(paths, "new_map_path", str(tmp_path) + "/")
    monkeypatch.setattr(v2_runtime, "validate_bundle", lambda: {})
    monkeypatch.setattr(export_map, "check_music_files", lambda files, directory: ["song.egg"])

    def generate(name, logger):
        if fails:
            raise ValueError("injected generation failure")
        return {"path": str(tmp_path)}

    monkeypatch.setattr(v2_generation, "generate_song", generate)
    from map_creation import v2_export
    monkeypatch.setattr(v2_export, "write_zip", lambda *args: str(tmp_path / "song.zip"))
    for key in ("max_speed", "jump_speed_offset", "inferno_model"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(Path(main.__file__)), run_name="__main__")
    assert result.value.code == exit_code


def test_parent_combination_uses_run_snapshot_without_overwriting_live_edits(tmp_path, monkeypatch):
    import main_multi
    from map_creation.v2_generation import settings_snapshot

    original_output, current_output = tmp_path / "original-output", tmp_path / "current-output"
    original_input, current_input = tmp_path / "original-input", tmp_path / "current-input"
    original_library, current_library = tmp_path / "original-library", tmp_path / "current-library"
    for folder in (original_output, current_output, original_input, current_input, original_library, current_library):
        folder.mkdir()
    (original_input / "song.egg").write_bytes(b"original recording")
    monkeypatch.setattr(paths, "new_map_path", str(original_output) + "/")
    monkeypatch.setattr(paths, "songs_pred", str(original_input) + "/")
    monkeypatch.setattr(paths, "bs_song_path", str(original_library) + "/")
    monkeypatch.setattr(config, "general_diff", "ExpertPlus")
    monkeypatch.setattr(config, "auto_move_song_afterwards", True)
    snapshot = settings_snapshot()
    paths.new_map_path, paths.songs_pred, paths.bs_song_path = map(str, (current_output, current_input, current_library))
    config.general_diff, config.auto_move_song_afterwards = "Hard", False
    source = original_output / "1234_20.0_song"
    source.mkdir()
    records = [{"song": "song", "status": "ok", "internal_difficulty": 20., "path": str(source)}]
    observed = []

    def combine(name, variants, folder, zip_path):
        observed.append(variants)
        folder.mkdir()
        (folder / "complete.dat").write_text("published gameplay")
        zip_path.write_bytes(b"published ZIP")
        return {"path": str(folder), "zip_path": str(zip_path)}

    monkeypatch.setattr(main_multi, "combine_packages", combine)
    report = main_multi._combine_maps(["song.egg"], records, [20.], True, snapshot)[0]
    assert observed[0][0]["filename"] == "ExpertPlus.dat"
    assert report["path"] == str(original_output / "12345_song")
    assert (original_library / "12345_song/complete.dat").read_text() == "published gameplay"
    assert not list(current_output.iterdir()) and not list(current_library.iterdir())
    assert (original_input / "y_done/song.egg").read_bytes() == b"original recording"
    assert not source.exists()
    assert paths.new_map_path == str(current_output)
    assert paths.songs_pred == str(current_input)
    assert paths.bs_song_path == str(current_library)
    assert config.general_diff == "Hard" and not config.auto_move_song_afterwards


def test_run_freezes_preparation_paths_bundle_and_report_destination(tmp_path, monkeypatch):
    import main_multi

    original_input, current_input = tmp_path / "original-input", tmp_path / "current-input"
    original_input.mkdir()
    current_input.mkdir()
    original_temp, current_temp = tmp_path / "original-temp", tmp_path / "current-temp"
    monkeypatch.setattr(paths, "songs_pred", str(original_input) + "/")
    monkeypatch.setattr(paths, "temp_path", str(original_temp) + "/")
    monkeypatch.setattr(config, "v2_model_bundle_path", "original-bundle")
    monkeypatch.setattr(config, "general_diff", "ExpertPlus")
    monkeypatch.setattr(main_multi, "configure_inference", lambda: None)
    bundles, prepared, captured, reports = [], [], [], []
    monkeypatch.setattr(main_multi, "validate_bundle", lambda path: bundles.append(path))

    def prepare(files, input_path):
        prepared.append(input_path)
        paths.songs_pred, paths.temp_path = str(current_input) + "/", str(current_temp) + "/"
        config.general_diff, config.v2_model_bundle_path = "Hard", "current-bundle"
        return ["song.egg"]

    monkeypatch.setattr(main_multi, "check_music_files", prepare)
    monkeypatch.setattr(main_multi, "process_song", lambda *args: [])
    monkeypatch.setattr(main_multi, "_combine_maps", lambda *args: captured.append(args[-1]) or [])
    monkeypatch.setattr(main_multi, "save_report", lambda report, root: reports.append(root) or "saved.json")
    main_multi.main_multi([5.], False)
    assert bundles == ["original-bundle"]
    assert prepared == [str(original_input) + "/"]
    assert captured[0]["paths"]["songs_pred"] == str(original_input) + "/"
    assert captured[0]["config"]["general_diff"] == "ExpertPlus"
    assert reports == [str(original_temp) + "/"]
    assert paths.songs_pred == str(current_input) + "/"
    assert paths.temp_path == str(current_temp) + "/"
    assert config.general_diff == "Hard" and config.v2_model_bundle_path == "current-bundle"
