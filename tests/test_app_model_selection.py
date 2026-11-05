"""The app run path honors the mapper dropdown selection."""

import main_app
from map_creation.v1_generation import V1_MAPPER_SELECTIONS
from tools.config import config


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
