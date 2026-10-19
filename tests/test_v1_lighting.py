"""Retained lighting parity and export contracts without pretrained assets."""

from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from lighting_prediction.v1_lighting import V1Lighting, lighting_paths, validate_assets
from map_creation.v2_export import serialize_beatmap
from tools.config import config, paths


CATEGORIES = np.array([[0, 1], [1, 2000000804], [13, 2016777215]], np.int64)


class Model:
    input_shape = [(None, 16, 16, 1), (None, 16, 1), (None, 16, 3)]
    output_shape = (None, 16, 3)

    def __init__(self):
        self.inputs = []

    def __call__(self, inputs, training=False):
        index = len(self.inputs)
        self.inputs.append([np.asarray(value).copy() for value in inputs])
        winners = (np.arange(16) + index) % 3
        return (.1 + .8 * np.eye(3, dtype=np.float32)[winners])[None]

    def predict(self, x, verbose=0):
        return self(x)


@pytest.mark.parametrize("length", [32, 33, 47, 48, 49, 80])
def test_decoder_matches_legacy_blocks_feedback_and_categories(monkeypatch, length):
    from lighting_prediction import generate_lighting as legacy

    monkeypatch.setattr(config, "random_note_map_factor", 0)
    old_model, model = Model(), Model()
    times = np.arange(length, dtype=np.float64) * .25 + .1
    latents = np.arange(length * 16, dtype=np.float32).reshape(length, 16)
    monkeypatch.setattr(legacy, "load_keras_model", lambda path: (old_model, path))
    monkeypatch.setattr(legacy, "get_class_size", lambda path: len(CATEGORIES))

    def decode(onehot, path):
        return np.array([[f"{kind};{value}"] for kind, value in CATEGORIES[onehot.argmax(axis=1)]])

    monkeypatch.setattr(legacy, "decode_onehot_class", decode)
    expected = legacy.generate(latents, times, "fixture", 16, paths.events_classify_encoder_file)
    queried = []

    def encode(selected):
        queried.append(selected.copy())
        return latents[:len(selected)]

    events, report = V1Lighting(model, CATEGORIES)(times, encode)
    assert len(events) == max(length // 16 - 1, 0) * 16
    assert report["events"] == len(events) and report["backend"] == "v1"
    np.testing.assert_array_equal([[item["type"], item["value"]] for item in events], expected)
    np.testing.assert_array_equal([item["time"] for item in events], times[:len(events)])
    np.testing.assert_array_equal(queried[0], times[:len(events)])
    for old, current in zip(old_model.inputs, model.inputs):
        for old_input, current_input in zip(old, current):
            np.testing.assert_array_equal(current_input, old_input)
    assert len(old_model.inputs) == len(model.inputs) == report["blocks"]
    assert model.inputs[0][1][0, 0, 0] == 1


@pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 31])
def test_short_sequences_do_not_call_model_or_encoder(monkeypatch, length):
    model = Model()
    events, report = V1Lighting(model, CATEGORIES)(np.arange(length) / 4,
                                                  lambda times: pytest.fail("No complete output block"))
    assert not events and not model.inputs
    assert report["blocks"] == report["events"] == 0


def test_large_integer_lighting_values_survive_both_export_formats(monkeypatch):
    monkeypatch.setattr(config, "random_note_map_factor", 0)
    times = np.arange(32, dtype=np.float64) / 4
    events, _ = V1Lighting(Model(), CATEGORIES)(times, lambda selected: np.zeros((len(selected), 16)))
    modern, _ = serialize_beatmap({"lighting": events}, 120, "v3")
    legacy, _ = serialize_beatmap({"lighting": events}, 120, "v2")
    assert [item["i"] for item in modern["basicBeatmapEvents"]] == [item["value"] for item in events]
    assert [item["_value"] for item in legacy["_events"]] == [item["value"] for item in events]
    assert 2000000804 in [item["i"] for item in modern["basicBeatmapEvents"]]
    assert [item["b"] for item in modern["basicBeatmapEvents"]] == [time * 2 for time in times[:16]]


def test_asset_paths_and_exact_vocabulary_are_independent_of_mapper(monkeypatch, tmp_path):
    model, vocabulary = tmp_path / "light.h5", tmp_path / "events.pkl"
    model.write_bytes(b"non-executable lighting checkpoint")
    joblib.dump(SimpleNamespace(categories_=[np.array(["0;1", "1;2000000804"])]), vocabulary)
    monkeypatch.setattr(config, "v1_lighting_model_path", str(model))
    monkeypatch.setattr(config, "v1_lighting_vocabulary_path", str(vocabulary))
    monkeypatch.setattr(config, "use_mapper_selection", "unrelated-v2-style")
    assert lighting_paths() == (model, vocabulary)
    assets = validate_assets()
    assert assets["categories"].dtype == np.int64
    np.testing.assert_array_equal(assets["categories"], [[0, 1], [1, 2000000804]])
    vocabulary.unlink()
    with pytest.raises(ValueError, match="Missing v1 lighting asset"):
        validate_assets()


@pytest.mark.parametrize("labels", [["0;no"], ["0;1;2"], ["100;120"], ["0;-1"], []])
def test_invalid_event_vocabularies_fail_explicitly(monkeypatch, tmp_path, labels):
    model, vocabulary = tmp_path / "light.h5", tmp_path / "events.pkl"
    model.write_bytes(b"checkpoint")
    joblib.dump(SimpleNamespace(categories_=[np.array(labels)]), vocabulary)
    monkeypatch.setattr(config, "v1_lighting_model_path", str(model))
    monkeypatch.setattr(config, "v1_lighting_vocabulary_path", str(vocabulary))
    with pytest.raises(ValueError, match="Invalid v1 lighting vocabulary"):
        validate_assets()


def test_model_vocabulary_mismatch_is_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        V1Lighting(Model(), CATEGORIES[:2])


@pytest.mark.parametrize("times", [[1, 1], [1, .5], [-1], [float("nan")]])
def test_invalid_timestamps_fail_before_prediction(times):
    with pytest.raises(ValueError, match="timestamps"):
        V1Lighting(Model(), CATEGORIES)(times, lambda selected: pytest.fail("Invalid timeline"))
