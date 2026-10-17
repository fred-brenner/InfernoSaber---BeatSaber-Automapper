"""Configured training previews and CLI inheritance, with no optimizer or fit."""

import json

import numpy as np
import pytest
from tools.config import config

from training.beat_models import __main__ as command
from training.beat_models import runner
from training.beat_models.artifacts import json_value
from training.beat_models.contracts import STAGES, frame_conditions


@pytest.fixture(scope="module")
def prepared_data():
    return command._dummy_data()


@pytest.fixture
def no_training(monkeypatch):
    from training.beat_models import models

    def forbidden(*args, **kwargs):
        pytest.fail("Configuration checks must not create optimizers or train")

    monkeypatch.setattr(models.keras.optimizers, "get", forbidden)
    monkeypatch.setattr(models.keras.optimizers, "Adam", forbidden)
    monkeypatch.setattr(models.keras.Model, "fit", forbidden)
    yield
    models.keras.backend.clear_session()


@pytest.mark.parametrize("stage", STAGES)
def test_dry_run_uses_stage_settings_and_explicit_overrides(stage, prepared_data, tmp_path, monkeypatch, no_training):
    overrides = {"optimizer": "SGD", "learning_rate": .02, "epochs": 137, "batch_size": 2,
                 "clipnorm": .75, "early_stopping_patience": 4, "early_stopping_min_delta": .003,
                 "monitor_mode": "auto",
                 "seed": 17 + STAGES.index(stage), "shuffle": False, "verbose": 0}
    if stage != "intensity":
        overrides.update({"conditioning": "noisy-reference", "intensity_noise_std": .35})
    for key, value in overrides.items():
        field = {"epochs": "n_epochs", "seed": "random_seed"}.get(key, key)
        monkeypatch.setattr(config, f"{stage}_{field}", value)
    selected = runner.training_settings(stage)
    original_profiles = runner._profiles
    profile_calls = []

    def profiles(songs, conditioning="reference", intensity_model=None, rng=None, noise_std=None):
        profile_calls.append((conditioning, noise_std, rng.bit_generator.state))
        return original_profiles(songs, conditioning, intensity_model, rng, noise_std)

    monkeypatch.setattr(runner, "_profiles", profiles)
    songs, manifest = prepared_data
    report = runner.train_stage(songs, manifest, stage, tmp_path / "configured", dry_run=True)
    for key, value in selected.items():
        assert report[key] == (None if key == "conditioning" and stage == "intensity" else value)
    assert report["architecture"] == runner.architecture_settings(stage)
    assert report["input_shapes"]["audio"][0] == 2
    assert report["status"] == "untrained"
    assert report["optimizer_steps"] == 0
    assert np.isfinite(report["preview_loss"])
    assert profile_calls == [(selected["conditioning"], selected["intensity_noise_std"],
                              np.random.default_rng(selected["seed"]).bit_generator.state)]
    assert json.loads((tmp_path / "configured" / "report.json").read_text()) == json_value(report)

    override = runner.train_stage(songs, manifest, stage, tmp_path / "overridden", dry_run=True,
                                  epochs=151, batch_size=1, conditioning="reference")
    assert override["epochs"] == 151
    assert override["batch_size"] == 1
    assert override["conditioning"] == (None if stage == "intensity" else "reference")
    assert override["input_shapes"]["audio"][0] == 1
    assert profile_calls[-1][0] == "reference"
    assert override["seed"] == selected["seed"]
    assert [path.name for path in (tmp_path / "overridden").iterdir()] == ["report.json"]


def test_noisy_profiles_use_selected_noise_and_reproducible_seed(prepared_data, monkeypatch):
    songs, _ = prepared_data
    song = next(song for song in songs if len(song.arrays["intensity"]) > 3 and song.arrays["intensity"].sum())
    seed, noise_std = 29, .6
    shares = song.arrays["intensity"].astype(np.float32)
    shares = shares * np.exp(np.random.default_rng(seed).normal(0, noise_std, len(shares)))
    shares = (shares / shares.sum()).astype(np.float32)
    density = np.count_nonzero(song.arrays["note_classes"]) / song.metadata["duration"]
    expected = frame_conditions(song, shares, density)
    actual = runner._profiles([song], "noisy-reference", rng=np.random.default_rng(seed), noise_std=noise_std)
    np.testing.assert_array_equal(actual[song.song_id], expected)
    monkeypatch.setattr(config, "timing_intensity_noise_std", noise_std)
    inherited = runner._profiles([song], "noisy-reference", rng=np.random.default_rng(seed))
    np.testing.assert_array_equal(inherited[song.song_id], expected)
    zero_noise = runner._profiles([song], "noisy-reference", rng=np.random.default_rng(seed), noise_std=0)
    reference = runner._profiles([song])
    np.testing.assert_array_equal(zero_noise[song.song_id], reference[song.song_id])
    assert not np.array_equal(actual[song.song_id], reference[song.song_id])


def test_smoke_and_evaluation_use_stage_batches_and_record_loss_settings(prepared_data, tmp_path, monkeypatch, no_training):
    batches = {"intensity": 2, "timing": 3, "notes": 2}
    for stage, size in batches.items():
        monkeypatch.setattr(config, f"{stage}_batch_size", size)
    monkeypatch.setattr(config, "timing_count_loss_weight", .4)
    monkeypatch.setattr(config, "notes_red_loss_weight", .25)
    monkeypatch.setattr(config, "notes_blue_loss_weight", .75)
    observed = {stage: [] for stage in STAGES}
    original_batch = runner._batch

    def batch(stage, references, profiles):
        observed[stage].append(len(references))
        return original_batch(stage, references, profiles)

    monkeypatch.setattr(runner, "_batch", batch)
    songs, manifest = prepared_data
    songs = songs[:4]
    smoke = runner.smoke_models(songs, manifest, tmp_path / "smoke", max_events=1)
    for stage, size in batches.items():
        assert smoke["stages"][stage]["batch_size"] == size
        assert observed[stage][0] == size
        assert max(observed[stage]) <= size
        assert sum(observed[stage]) == smoke["stages"][stage]["examples"]
        assert smoke["stages"][stage]["batches"] == len(observed[stage])
        metadata = json.loads((tmp_path / "smoke" / stage / "manifest.json").read_text())
        assert metadata["training"]["batch_size"] == size
        assert metadata["training"]["architecture"] == json_value(runner.architecture_settings(stage))
        observed[stage].clear()

    artifacts = {stage: tmp_path / "smoke" / stage for stage in STAGES}
    monkeypatch.setattr(config, "timing_dropout", .2)
    monkeypatch.setattr(config, "timing_count_loss_weight", .7)
    evaluation = runner.evaluate_models(songs, manifest, artifacts, tmp_path / "evaluation", max_events=1)
    for stage, size in batches.items():
        result = evaluation["teacher_forced"][stage]
        assert result["batch_size"] == size
        assert observed[stage][0] == size
        assert max(observed[stage]) <= size
        assert sum(observed[stage]) == result["examples"]
        assert result["configured_architecture"] == runner.architecture_settings(stage)
    assert evaluation["checkpoints"]["timing"]["architecture"]["dropout"] == .1
    assert evaluation["checkpoints"]["timing"]["architecture"]["count_loss_weight"] == .4
    assert evaluation["teacher_forced"]["timing"]["configured_architecture"]["dropout"] == .2
    timing = evaluation["teacher_forced"]["timing"]
    assert timing["loss"] == pytest.approx(timing["cross_entropy"] + .7 * timing["normalized_count_huber"])


def test_train_cli_inherits_configuration_and_forwards_explicit_overrides(prepared_data, monkeypatch):
    calls = []
    monkeypatch.setattr(command, "load_dataset", lambda path: prepared_data)
    monkeypatch.setattr(runner, "train_stage", lambda *args, **kwargs: calls.append((args, kwargs)))
    required = ["train", "--dataset", "prepared", "--stage", "timing", "--run-dir", "run", "--dry-run"]
    command.main(required)
    assert calls[-1][1]["epochs"] is None
    assert calls[-1][1]["batch_size"] is None
    assert calls[-1][1]["conditioning"] is None
    command.main(required + ["--epochs", "173", "--batch-size", "3", "--conditioning", "reference"])
    assert calls[-1][1]["epochs"] == 173
    assert calls[-1][1]["batch_size"] == 3
    assert calls[-1][1]["conditioning"] == "reference"
