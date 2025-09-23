import os

import numpy as np
import pytest
from keras.layers import Flatten

from training import helpers as helpers_module
from training import tensorflow_models as tf_models
from tools.config import config


def test_filter_by_bps_uses_bpm_selection(monkeypatch):
    monkeypatch.setattr(config, "use_bpm_selection", True, raising=False)

    def fake_load_npy(path):
        if path == helpers_module.paths.diff_ar_file:
            return np.array([1.0, 3.0, 5.0])
        if path == helpers_module.paths.name_ar_file:
            return np.array(["song1", "song2", "song3"])
        raise AssertionError(f"Unexpected path requested: {path}")

    monkeypatch.setattr(helpers_module, "load_npy", fake_load_npy)

    names, diffs = helpers_module.filter_by_bps(min_limit=2.0, max_limit=4.0)

    assert names == ["song2"]
    assert diffs == [3.0]


def test_filter_by_bps_uses_mapper_selection(monkeypatch):
    monkeypatch.setattr(config, "use_bpm_selection", False, raising=False)
    monkeypatch.setattr(config, "use_mapper_selection", "example", raising=False)
    monkeypatch.setattr(config, "min_bps_limit", 1.5, raising=False)

    monkeypatch.setattr(helpers_module, "return_mapper_list", lambda selection: [selection])
    monkeypatch.setattr(
        helpers_module,
        "get_maps_from_mapper",
        lambda mapper_name: np.array(["MapA", "MapB"]),
    )

    names, diffs = helpers_module.filter_by_bps()

    assert names == ["MapA", "MapB"]
    assert diffs == [1.5, 1.5]


def test_load_keras_model_selects_latest_file(monkeypatch, tmp_path):
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    monkeypatch.setattr(
        helpers_module.paths, "model_path", str(model_directory) + os.sep, raising=False
    )

    first = model_directory / "first.h5"
    second = model_directory / "second.h5"
    first.write_text("first")
    second.write_text("second")

    file_ctimes = {str(first): 1, str(second): 2}

    monkeypatch.setattr(
        helpers_module.glob, "glob", lambda pattern: [str(first), str(second)]
    )
    monkeypatch.setattr(
        helpers_module.os.path, "getctime", lambda path: file_ctimes[path]
    )

    loaded_paths = []

    def fake_load_model(path):
        loaded_paths.append(path)
        return "loaded-model"

    monkeypatch.setattr(helpers_module, "load_model", fake_load_model)

    model, latest_name = helpers_module.load_keras_model("old")

    assert model == "loaded-model"
    assert latest_name == "second.h5"
    assert loaded_paths == [str(second)]


def test_load_keras_model_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(
        helpers_module.paths, "model_path", str(tmp_path) + os.sep, raising=False
    )

    def fail_if_called(path):
        raise AssertionError("load_model should not be invoked when the file is missing")

    monkeypatch.setattr(helpers_module, "load_model", fail_if_called)

    model, latest_name = helpers_module.load_keras_model("missing_model")

    assert model is None
    assert latest_name == "missing_model"


def test_ai_encode_song_runs_inference(monkeypatch):
    test_song = np.array([[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(config, "enc_version", "enc_test", raising=False)

    requested_versions = []

    def fake_get_full_model_path(version):
        requested_versions.append(version)
        return "/tmp/enc_model.h5"

    monkeypatch.setattr(helpers_module, "get_full_model_path", fake_get_full_model_path)

    class DummyModel:
        def __init__(self):
            self.calls = []

        def predict(self, song_input, verbose=0):
            self.calls.append((song_input, verbose))
            return song_input * 2

    dummy_model = DummyModel()

    def fake_load_model(path):
        assert path == "/tmp/enc_model.h5"
        return dummy_model

    monkeypatch.setattr(helpers_module, "load_model", fake_load_model)

    encoded = helpers_module.ai_encode_song(test_song)

    assert requested_versions == ["enc_test"]
    assert len(dummy_model.calls) == 1
    np.testing.assert_array_equal(dummy_model.calls[0][0], test_song)
    assert dummy_model.calls[0][1] == 0
    np.testing.assert_array_equal(encoded, test_song * 2)


@pytest.mark.parametrize(
    "dim_in, dim_out",
    [([(16,), (8, 4), (8, 6)], 5), ([(4,), (2, 3), (2, 2)], 3)],
)
def test_create_keras_model_lstm1_output_shape(dim_in, dim_out):
    model = tf_models.create_keras_model("lstm1", dim_in=dim_in, dim_out=dim_out)

    assert len(model.inputs) == 3
    assert model.inputs[0].shape == (None, *dim_in[0])
    assert model.inputs[1].shape == (None, *dim_in[1])
    assert model.inputs[2].shape == (None, *dim_in[2])
    assert model.output_shape == (None, dim_out)


def test_create_music_model_tcn(monkeypatch):
    monkeypatch.setattr(tf_models, "TCN", lambda *args, **kwargs: Flatten())

    model = tf_models.create_music_model("tcn", dim_in=3, tcn_len=4)

    assert len(model.inputs) == 3
    assert model.inputs[0].shape == (None, 4, 3)
    assert model.inputs[1].shape == (None, 4, 1)
    assert model.inputs[2].shape == (None, 4, 1)
    assert model.output_shape == (None, 1)
