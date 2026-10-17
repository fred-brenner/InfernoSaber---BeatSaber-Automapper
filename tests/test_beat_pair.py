"""Beat-only CPU adapter checks; fixtures never need note checkpoints or training."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
import pytest

from evaluation.beat_pair import BeatPair, _graph_signature, _load_cpu_model
from training.beat_models.artifacts import (statistics_hash, write_model_manifest)
from training.beat_models.contracts import (CORE, FEATURES, FRAME_SECONDS, HALO,
                                           REPRESENTATION, SCHEMA_VERSION, Song)
from training.beat_models.settings import architecture_settings


def statistics():
    return {"version": 1, "fit_split": "train", "song_ids": ["training-only"],
            **{key: {"mean": [0.] * FEATURES, "scale": [1.] * FEATURES, "count": 200}
               for key in ("features", "coarse_features")}}


def artifacts(tmp_path):
    dataset = {"schema_version": SCHEMA_VERSION, "representation": REPRESENTATION,
               "normalization": statistics(), "normalization_sha256": statistics_hash(statistics()),
               "sources": [], "splits": {"train": ["training-only"], "validation": [], "test": []}}
    result = {}
    for stage in ("intensity", "timing"):
        directory = tmp_path / stage
        directory.mkdir()
        checkpoint = directory / "best.keras"
        checkpoint.write_bytes(b"fixture checkpoint, not a model")
        write_model_manifest(directory, stage, checkpoint, dataset,
                             {"status": "trained", "architecture": architecture_settings(stage)})
        result[stage] = directory
    return result


def song_fixture(duration=10.301):
    frames = np.arange(int(np.ceil(duration / FRAME_SECONDS))) * FRAME_SECONDS
    durations = np.minimum(1., duration - np.arange(int(np.ceil(duration)))).astype(np.float32)
    features = np.zeros((len(frames), FEATURES), dtype=np.float32)
    features[:, 0] = np.arange(len(frames))
    return Song("audio-only", {"duration": duration, "bpm": 120., "normalization_version": 1,
                               "normalization_sha256": statistics_hash(statistics())},
                {"features": features, "coarse_features": np.zeros((len(durations), FEATURES), np.float32),
                 "frame_times": frames, "bin_durations": durations})


class Intensity:
    def __init__(self):
        self.calls = []

    def predict(self, inputs, verbose=0):
        self.calls.append(inputs)
        mass = inputs["durations"] * np.arange(1, inputs["durations"].shape[1] + 1)
        return (mass / mass.sum()).astype(np.float32)


class Timing:
    def __init__(self):
        self.calls = []
        self.events = {0: 1, 1: 2, 511: 2, 512: 1, 1023: 1, 1030: 2}

    def predict(self, inputs, verbose=0):
        self.calls.append(inputs)
        values = inputs["audio"][0, HALO:HALO + CORE, 0].astype(int)
        valid = inputs["mask"][0, HALO:HALO + CORE]
        # Deliberately predict events on padded output frames. They must vanish.
        counts = [self.events.get(int(frame), 0) if mask else 2 for frame, mask in zip(values, valid)]
        return np.eye(3, dtype=np.float32)[np.asarray(counts)][None]


def pair_fixture(tmp_path):
    paths = artifacts(tmp_path)
    predictors = {"intensity": Intensity(), "timing": Timing()}
    stages = []

    def loader(checkpoint, stage, manifest):
        stages.append(stage)
        assert checkpoint.name == "best.keras" and manifest["status"] == "trained"
        return predictors[stage]

    pair = BeatPair(paths["intensity"], paths["timing"], model_loader=loader)
    assert stages == ["intensity", "timing"]
    return pair, predictors


def test_import_does_not_load_tensorflow_runner_or_generation():
    code = "import sys; import evaluation.beat_pair; " \
           "assert not any(name in sys.modules for name in " \
           "('tensorflow','training.beat_models.models','training.beat_models.runner','training.beat_models.pipeline'))"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


def test_stitches_exact_core_halo_tail_and_preserves_dense_counts(tmp_path):
    pair, predictors = pair_fixture(tmp_path)
    song = song_fixture()
    result = pair.predict(song, 3)
    expected = sorted(predictors["timing"].events)
    np.testing.assert_array_equal(result["event_frames"], expected)
    np.testing.assert_array_equal(result["event_counts"], [predictors["timing"].events[i] for i in expected])
    np.testing.assert_array_equal(result["times_seconds"], song.arrays["frame_times"][expected])
    assert result["times_seconds"][1] == .01  # No reference-target spacing filter at inference.
    assert result["counts"].sum() == 9  # The requested NPS is conditioning, not a forced budget.
    assert result["frame_probabilities"].shape == (1031, 3)
    np.testing.assert_allclose(result["frame_probabilities"].sum(axis=-1), 1)
    assert len(predictors["timing"].calls) == 3
    first, middle, tail = predictors["timing"].calls
    assert first["audio"].shape == (1, 1020, FEATURES)
    assert not first["mask"][0, :HALO].any()
    assert not first["audio"][0, :HALO].any() and not first["conditions"][0, :HALO].any()
    assert middle["audio"][0, HALO, 0] == 512
    assert tail["audio"][0, HALO, 0] == 1024
    assert tail["mask"][0, HALO:HALO + CORE].sum() == 7
    assert not tail["mask"][0, HALO + 7:].any()
    assert result["local_nps"][-1] > 0  # The final partial bin carries real density.
    assert np.dot(result["local_nps"], song.arrays["bin_durations"]) == pytest.approx(3 * 10.301)
    assert pair.metadata["checkpoints"].keys() == {"intensity", "timing"}
    json.dumps(pair.metadata, allow_nan=False)


def test_zero_density_keeps_valid_probabilities_and_skips_timing(tmp_path):
    pair, predictors = pair_fixture(tmp_path)
    result = pair.predict(song_fixture(), 0)
    assert not predictors["timing"].calls
    assert not result["counts"].any() and not result["local_nps"].any()
    assert result["times_seconds"].shape == result["event_counts"].shape == (0,)
    np.testing.assert_array_equal(result["frame_probabilities"][:, 0], 1)
    assert result["intensity_shares"].sum() == pytest.approx(1.)


def test_only_audio_predicted_intensity_nps_and_bpm_condition_timing(tmp_path):
    pair, predictors = pair_fixture(tmp_path)
    song = song_fixture()
    # Targets can be absent or unreadable. The model receives none of them.
    class ForbiddenTarget:
        def __array__(self, *args, **kwargs):
            pytest.fail("Inference tried to inspect a reference target")

    for key in ("timing", "intensity", "note_classes", "event_times", "event_frames"):
        song.arrays[key] = ForbiddenTarget()
    first = pair.predict(song, 4)
    first_conditions = predictors["timing"].calls[0]["conditions"].copy()
    second = pair.predict(song, 4, bpm=240)
    second_conditions = predictors["timing"].calls[3]["conditions"]
    valid = predictors["timing"].calls[0]["mask"]
    np.testing.assert_array_equal(first["local_nps"], second["local_nps"])
    np.testing.assert_array_equal(first_conditions[..., :3], second_conditions[..., :3])
    np.testing.assert_array_equal(first_conditions[..., 3][valid], 1)
    np.testing.assert_array_equal(second_conditions[..., 3][valid], 2)
    third = pair.predict(song, 2)
    third_conditions = predictors["timing"].calls[6]["conditions"]
    np.testing.assert_allclose(first["local_nps"], 2 * third["local_nps"])
    np.testing.assert_array_equal(first_conditions[..., 0], third_conditions[..., 0])
    np.testing.assert_array_equal(first_conditions[..., 3], third_conditions[..., 3])
    assert song.metadata["bpm"] == 120
    assert len(predictors["intensity"].calls) == 1  # One audio profile across difficulty passes.


@pytest.mark.parametrize("density", [-1, float("nan"), float("inf"), True, "3"])
def test_rejects_invalid_density(tmp_path, density):
    pair, _ = pair_fixture(tmp_path)
    with pytest.raises(ValueError, match="Requested NPS"):
        pair.predict(song_fixture(), density)


@pytest.mark.parametrize("bpm", [0, -1, float("nan"), float("inf"), True, "120"])
def test_rejects_invalid_bpm(tmp_path, bpm):
    pair, _ = pair_fixture(tmp_path)
    with pytest.raises(ValueError, match="BPM"):
        pair.predict(song_fixture(), 3, bpm=bpm)


@pytest.mark.parametrize("stage", ["intensity", "timing"])
@pytest.mark.parametrize("damage", ["checksum", "status", "architecture", "representation", "normalization"])
def test_rejects_incompatible_artifacts_before_loading_models(tmp_path, stage, damage):
    paths = artifacts(tmp_path)
    manifest_path = paths[stage] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "checksum":
        (paths[stage] / "best.keras").write_bytes(b"changed")
    elif damage == "status":
        manifest["training"]["status"] = "untrained"
    elif damage == "architecture":
        manifest["training"]["architecture"]["channels"] += 1
    elif damage == "representation":
        manifest["representation"]["hop"] += 1
    else:
        manifest["normalization"]["features"]["mean"][0] = 1.
        manifest["normalization_sha256"] = statistics_hash(manifest["normalization"])
    manifest_path.write_text(json.dumps(manifest))

    def forbidden(*args):
        pytest.fail("An incompatible artifact reached the model loader")

    with pytest.raises(ValueError):
        BeatPair(paths["intensity"], paths["timing"], model_loader=forbidden)


@pytest.mark.parametrize("bad", [np.zeros((1, 11)), np.full((1, 11), np.nan), np.full((1, 11), -1.)])
def test_rejects_invalid_intensity_probabilities(tmp_path, bad):
    pair, predictors = pair_fixture(tmp_path)
    predictors["intensity"].predict = lambda *args, **kwargs: bad
    with pytest.raises(ValueError, match="intensity probabilities"):
        pair.predict(song_fixture(), 3)
    assert not predictors["timing"].calls


def test_rejects_invalid_timing_probabilities(tmp_path):
    pair, predictors = pair_fixture(tmp_path)
    predictors["timing"].predict = lambda *args, **kwargs: np.zeros((1, CORE, 3), np.float32)
    with pytest.raises(ValueError, match="timing probabilities"):
        pair.predict(song_fixture(), 3)


def test_prepare_uses_shared_synchronization_and_normalization_without_writes(tmp_path):
    from evaluation.human_maps import load_reference

    pair, _ = pair_fixture(tmp_path)
    package = tmp_path / "package"
    package.mkdir()
    info = {"_version": "2.0.0", "_songName": "fixture", "_beatsPerMinute": 120,
            "_songFilename": "song.wav", "_difficultyBeatmapSets": [
                {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
                    {"_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlus.dat"}]}]}
    notes = {"version": "3.0.0", "colorNotes": [
        {"b": .4, "c": 0, "x": 0, "y": 2, "d": 1},
        {"b": .42, "c": 0, "x": 0, "y": 0, "d": 1},
        {"b": .4, "c": 1, "x": 1, "y": 1, "d": 0},
        {"b": 2., "c": 1, "x": 1, "y": 0, "d": 1}]}
    (package / "Info.dat").write_text(json.dumps(info))
    (package / "ExpertPlus.dat").write_text(json.dumps(notes))
    with wave.open(str(package / "song.wav"), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.zeros(16048, dtype="<i2").tobytes())
    before = {path.name: path.read_bytes() for path in package.iterdir()}
    song = pair.prepare(load_reference(package))
    assert song.metadata["duration"] == pytest.approx(1.003)
    assert song.metadata["retained_colored_note_count"] == 3
    assert song.metadata["original_colored_note_count"] == 4
    np.testing.assert_array_equal(song.arrays["event_times"], [.2, 1.])
    np.testing.assert_array_equal(song.arrays["timing"][[20, 100]], [2, 1])
    assert song.arrays["bin_durations"][-1] == pytest.approx(.003)
    assert song.arrays["frame_times"][-1] == 1.
    assert song.metadata["normalization_sha256"] == pair.normalization_sha256
    assert before == {path.name: path.read_bytes() for path in package.iterdir()}


def test_graph_comparison_tolerates_only_known_serialization_difference():
    graph = {"class_name": "Functional", "config": {"name": "intensity", "layers": [
        {"name": "audio", "class_name": "InputLayer", "config": {"name": "audio", "dtype": "float32"},
         "inbound_nodes": []},
        {"name": "auto_mask", "class_name": "ApplyMask", "config": {"name": "auto_mask"},
         "inbound_nodes": [[["audio", 0, 0, {}]]]}],
        "input_layers": [["audio", 0, 0]], "output_layers": [["auto_mask", 0, 0]]}}
    compatible = deepcopy(graph)
    compatible["config"]["layers"][0]["config"]["optional"] = False
    compatible["config"]["layers"][1]["name"] = "auto_mask_99"
    compatible["config"]["layers"][1]["config"]["name"] = "auto_mask_99"
    compatible["config"]["output_layers"][0][0] = "auto_mask_99"
    assert _graph_signature(compatible) == _graph_signature(graph)
    compatible["config"]["layers"][0]["config"]["optional"] = True
    assert _graph_signature(compatible) != _graph_signature(graph)


@pytest.mark.parametrize("stage", ["intensity", "timing"])
def test_cpu_loader_restores_matching_weights_without_notes_or_training(tmp_path, monkeypatch, stage):
    tf = pytest.importorskip("tensorflow")
    from training.beat_models import models
    from training.beat_models.artifacts import file_hash

    def forbidden(*args, **kwargs):
        pytest.fail("Beat-only restoration attempted note construction or training")

    monkeypatch.setitem(models.BUILDERS, "notes", forbidden)
    monkeypatch.setattr(tf.keras.Model, "fit", forbidden)
    monkeypatch.setattr(tf.keras.optimizers, "Adam", forbidden)
    with tf.device("/CPU:0"):
        model = models.build_model(stage)
    song = song_fixture()
    from training.beat_models.contracts import frame_conditions, intensity_inputs, timing_inputs
    if stage == "intensity":
        inputs = intensity_inputs(song)
    else:
        song.arrays["timing"] = np.zeros(len(song.arrays["frame_times"]), np.int32)
        shares = song.arrays["bin_durations"] / song.metadata["duration"]
        conditions = frame_conditions(song, shares, 3.)
        inputs = timing_inputs(song, conditions, 0)[0]
    expected = model(inputs, training=False).numpy()
    path = tmp_path / (stage + ".keras")
    model.save(path)
    before = file_hash(path)
    restored = _load_cpu_model(path, stage, {"training": {"parameters": model.count_params()}})
    np.testing.assert_allclose(restored.predict(inputs), expected, rtol=1e-6, atol=1e-6)
    assert file_hash(path) == before
