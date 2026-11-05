"""The app and CLI run paths honor the mapper selection."""

import os

import pytest

import main
import main_app
from map_creation.v1_generation import V1_MAPPER_SELECTIONS
from tools.config import config, paths


def _stub_run(monkeypatch, captured):
    """Bypass asset checks and generation; record how run_process routes them."""
    monkeypatch.setattr(config, "use_mapper_selection", "v2_expert")
    monkeypatch.setattr(main_app, "get_summary", lambda *args: "Ready")
    monkeypatch.setattr(main_app, "update_model_file_paths",
                        lambda *args, **kwargs: captured.setdefault("path_calls", []).append(kwargs))
    monkeypatch.setattr(main_app, "validate_production_bundle",
                        lambda *args, **kwargs: captured.__setitem__("v2_checked", True))
    monkeypatch.setattr(main_app, "ensure_v1_bundle",
                        lambda selection: captured.setdefault("v1_checked", []).append(selection))
    monkeypatch.setattr(main_app, "main_multi_par",
                        lambda *args, **kwargs: captured.update(kwargs) or {"status": "ok"})


def test_mapper_choices_lists_v2_first_then_all_v1_presets():
    assert main_app.MAPPER_CHOICES == ["v2_expert", *V1_MAPPER_SELECTIONS]


def test_parameter_group_visibility():
    assert main_app.parameter_group_visibility("v2_expert") == {"v1_only": False, "v2_only": True}
    assert main_app.parameter_group_visibility("fav_15") == {"v1_only": True, "v2_only": False}


def test_run_process_passes_selected_model(monkeypatch):
    captured = {}
    _stub_run(monkeypatch, captured)
    output = list(main_app.run_process(3, "fav_15", 2, 4, 6, 0, 0))
    assert captured["model_selection"] == "fav_15"
    assert captured["v1_checked"] == ["fav_15"]
    assert "v2_checked" not in captured  # v1 selection must skip v2 bundle validation
    assert config.use_mapper_selection == "fav_15"
    assert "Downloading/validating model: fav_15" in output[0]


def test_run_process_defaults_to_v2_expert(monkeypatch):
    captured = {}
    _stub_run(monkeypatch, captured)
    list(main_app.run_process(3, "v2_expert", 2, 4, 6, 0, 0))
    assert captured["model_selection"] == "v2_expert"
    assert captured["v2_checked"] is True
    # Repoint paths.model_path without the legacy tf_* existence check;
    # validate_production_bundle is the real v2 verification.
    assert captured["path_calls"] == [{"check_model_exists": False}]
    assert "v1_checked" not in captured
    assert config.use_mapper_selection == "v2_expert"


def test_run_process_reports_bundle_errors_before_generating(monkeypatch):
    captured = {}
    _stub_run(monkeypatch, captured)
    def reject(selection):
        raise ValueError(f"Unknown v1 mapper selection: {selection}")
    monkeypatch.setattr(main_app, "ensure_v1_bundle", reject)
    output = list(main_app.run_process(3, "bogus", 2, 4, 6, 0, 0))
    assert output[-1].startswith("Error: ValueError")
    assert "model_selection" not in captured  # generation never started


def _stub_main_run(monkeypatch, tmp_path):
    """Bypass asset checks and song IO; main() still routes and packages per song."""
    monkeypatch.setattr(paths, "songs_pred", str(tmp_path) + "/")
    monkeypatch.setattr(paths, "new_map_path", str(tmp_path) + "/")
    monkeypatch.setattr(main, "check_music_files", lambda files, directory: ["song.egg"])


def test_main_uses_v1_wrapper_for_legacy_env(monkeypatch, tmp_path):
    monkeypatch.setenv("inferno_model", "fav_15")
    use_model = os.environ.get("inferno_model")  # same read as main.py's __main__ block
    _stub_main_run(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "validate_production_bundle",
                        lambda *args, **kwargs: pytest.fail("v1 skips v2 bundle validation"))
    monkeypatch.setattr(main, "generate_song",
                        lambda *args, **kwargs: pytest.fail("v1 selection must not run v2"))
    ensured, generated, zipped = [], [], []
    monkeypatch.setattr(main, "ensure_v1_bundle", lambda selection: ensured.append(selection))

    def fake_v1(name, difficulty, logger=None):
        generated.append((config.use_mapper_selection, name, difficulty))
        return {"status": "ok", "song": name, "internal_difficulty": difficulty,
                "path": str(tmp_path), "package": {}}

    monkeypatch.setattr(main, "generate_song_v1", fake_v1)
    monkeypatch.setattr(main, "write_zip", lambda *args: zipped.append(args) or args[-1])
    result = main.main(use_model, 20, export_results_to_bs=False)
    assert result["status"] == "ok"
    assert ensured == ["fav_15"]
    assert generated == [("fav_15", "song", 20)]
    # v1 folder zips under the shared <speed>_<name>.zip convention
    assert zipped == [(str(tmp_path), f"{tmp_path}/20.0_song.zip")]


def test_main_validates_v2_bundle_by_default(monkeypatch, tmp_path):
    _stub_main_run(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "ensure_v1_bundle",
                        lambda selection: pytest.fail("v2 must not touch v1 assets"))
    seen, generated = [], []
    monkeypatch.setattr(main, "validate_production_bundle", lambda: seen.append("v2_bundle"))

    def fake_v2(name, logger=None):
        generated.append((config.use_mapper_selection, name))
        return {"status": "ok", "song": name, "path": str(tmp_path), "package": {}}

    monkeypatch.setattr(main, "generate_song", fake_v2)
    monkeypatch.setattr(main, "generate_song_v1",
                        lambda *args, **kwargs: pytest.fail("v2 selection must not run v1"))
    monkeypatch.setattr(main, "write_zip", lambda *args: args[-1])
    result = main.main(None, 20, export_results_to_bs=False)
    assert result["status"] == "ok"
    assert seen == ["v2_bundle"]
    assert generated == [("v2_expert", "song")]
