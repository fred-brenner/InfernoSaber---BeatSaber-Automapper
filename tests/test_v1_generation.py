"""Worker-safe v1 wrapper: legacy routing, scoped settings, bundle checks."""

import sys
from types import SimpleNamespace

import pytest

import map_creation.v1_generation as v1_generation
from map_creation.v1_generation import V1_MAPPER_SELECTIONS, ensure_v1_bundle, generate_song_v1
from tools.config import config, paths


def test_generate_song_v1_calls_legacy_main_with_single_name(monkeypatch, tmp_path):
    calls = []
    fake = SimpleNamespace(main=lambda names: calls.append(names) or False)
    monkeypatch.setitem(sys.modules, "evaluation.legacy_generator", fake)
    monkeypatch.setattr(paths, "new_map_path", str(tmp_path) + "/")
    (tmp_path / "1234_20.0_song").mkdir()
    result = generate_song_v1("song", 20.0)
    assert calls == [["song"]]
    assert result["status"] == "ok" and result["internal_difficulty"] == 20.0
    assert result["path"].endswith("1234_20.0_song")


def test_generate_song_v1_failure_becomes_error(monkeypatch, tmp_path):
    fake = SimpleNamespace(main=lambda names: 1)  # legacy "not enough beats"
    monkeypatch.setitem(sys.modules, "evaluation.legacy_generator", fake)
    monkeypatch.setattr(paths, "new_map_path", str(tmp_path) + "/")
    with pytest.raises(RuntimeError):
        generate_song_v1("song", 20.0)


def test_generate_song_v1_restores_config(monkeypatch, tmp_path):
    original = config.max_speed_orig
    fake = SimpleNamespace(main=lambda names: setattr(config, "max_speed", -1.) or False)
    monkeypatch.setitem(sys.modules, "evaluation.legacy_generator", fake)
    monkeypatch.setattr(paths, "new_map_path", str(tmp_path) + "/")
    (tmp_path / "1234_20.0_song").mkdir()
    generate_song_v1("song", 20.0)
    assert config.max_speed_orig == original and config.max_speed == original


def test_ensure_v1_bundle_downloads_missing_assets(monkeypatch, tmp_path):
    calls = {"paths": 0, "download": []}
    monkeypatch.setattr(config, "use_mapper_selection", "v2_expert")
    monkeypatch.setattr(v1_generation, "update_model_file_paths",
                        lambda check_model_exists=True: calls.__setitem__("paths", calls["paths"] + 1))
    state = {"missing": True}

    def full_path(prefix, full_path=True):
        if state["missing"] and prefix == config.event_gen_version:
            raise FileNotFoundError(prefix)
        return str(tmp_path / prefix)

    monkeypatch.setattr(v1_generation, "get_full_model_path", full_path)
    monkeypatch.setattr(v1_generation, "model_download",
                        lambda model_branch=None: calls["download"].append(model_branch)
                        or state.__setitem__("missing", False))
    pkl = tmp_path / "encoder.pkl"
    pkl.write_bytes(b"1")
    for attr in ("notes_classify_dict_file", "beats_classify_encoder_file",
                 "events_classify_encoder_file"):
        monkeypatch.setattr(paths, attr, str(pkl))
    ensure_v1_bundle("fav_15")
    assert calls["download"] == ["fav_15"]
    assert calls["paths"] == 2
    assert config.use_mapper_selection == "fav_15"


def test_ensure_v1_bundle_still_missing_after_download_names_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "use_mapper_selection", "v2_expert")
    monkeypatch.setattr(v1_generation, "update_model_file_paths",
                        lambda check_model_exists=True: None)
    monkeypatch.setattr(v1_generation, "model_download", lambda model_branch=None: None)

    def full_path(prefix, full_path=True):
        if prefix == config.enc_version:
            raise FileNotFoundError(prefix)
        return str(tmp_path / prefix)

    monkeypatch.setattr(v1_generation, "get_full_model_path", full_path)
    pkl = tmp_path / "encoder.pkl"
    pkl.write_bytes(b"1")
    for attr in ("notes_classify_dict_file", "beats_classify_encoder_file",
                 "events_classify_encoder_file"):
        monkeypatch.setattr(paths, attr, str(pkl))
    with pytest.raises(FileNotFoundError, match="fav_15.*tf_model_enc_"):
        ensure_v1_bundle("fav_15")


def test_ensure_v1_bundle_rejects_unknown_selection():
    assert "v2_expert" not in V1_MAPPER_SELECTIONS
    with pytest.raises(ValueError):
        ensure_v1_bundle("v2_expert")
