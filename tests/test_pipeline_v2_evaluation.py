"""Comparison tests are deliberately opt-in, including synthetic comparisons."""

import os

import numpy as np
import pytest


pytestmark = [pytest.mark.comparison,
              pytest.mark.skipif(os.environ.get("INFERNOSABER_RUN_V2_COMPARISONS") != "1",
                                 reason="V2 comparisons require an explicitly requested run")]


def test_comparison_cli_defaults_to_a_plan_without_loading_files(capsys, monkeypatch):
    from pathlib import Path
    from evaluation.evaluate_pipeline_v2 import main
    def forbidden(*args, **kwargs):
        pytest.fail("A comparison plan must not traverse files")
    monkeypatch.setattr(Path, "open", forbidden)
    plan = main(["--stage", "arrows", "--dataset", "missing", "--evaluation-dataset", "missing-curated",
                 "--artifact", "missing", "--output", "missing"])
    assert plan["status"] == "planned"
    assert plan["execution_requested"] is False
    assert "planned" in capsys.readouterr().out


@pytest.mark.parametrize("stage", ("positions", "arrows", "multiplier", "hazards", "lighting", "arcs"))
def test_independent_stage_baselines_use_training_examples_only(stage):
    from training.pipeline_v2.data import make_dummy_dataset
    from training.pipeline_v2.evaluation import ReferenceBaseline, evaluate_stage, fit_reference_baselines
    dataset = make_dummy_dataset()
    fitted = fit_reference_baselines(dataset, stage)
    train_ids = {song.base.song_id for song in dataset.songs if song.base.metadata["split"] == "train"}
    assert set(fitted["training_song_ids"]) <= train_ids
    predictor = ReferenceBaseline(stage, dataset.vocabularies, fitted)
    report = evaluate_stage(dataset, stage, predictor, batch_size=2, allow_synthetic=True)
    assert report["stage"] == stage
    assert report["evaluated_split"] == "test"
    assert report["optimizer_steps"] == 0
    assert report["scores"]["model"] == report["scores"]["frequency"]


def test_encoder_parity_and_reconstruction_are_distinct():
    from training.pipeline_v2.evaluation import evaluate_encoder
    class Encoder:
        metadata = {"normalization": "fixed_song"}
        def patches(self, times):
            return np.ones((len(times), 24, 20, 1), np.float32)
        def __call__(self, times):
            return np.ones((len(times), 16), np.float32)
    report = evaluate_encoder(Encoder(), [0., 1.],
                              reference_predict=lambda patches: np.ones((len(patches), 16), np.float32),
                              autoencoder=lambda patches: patches)
    assert report["scores"]["latent_max_abs_error"] == 0
    assert report["scores"]["reconstruction_mse"] == 0
    assert report["scores"]["zero_patch_reference_mse"] == 1


def test_real_reference_sources_must_be_curated_and_exclude_favorites():
    from training.pipeline_v2.data import make_dummy_dataset
    from training.pipeline_v2.evaluation import validate_curated_dataset
    dataset = make_dummy_dataset()
    song = next(song for song in dataset.songs if song.base.metadata["split"] == "test")
    dataset.songs = [song]
    song.base.metadata["sources"] = [
        {"role": "audio", "path": "/tmp/evaluate_beats/bs_map_input/map/song.egg", "sha256": "a" * 64},
        {"role": "beatmap", "path": "/tmp/evaluate_beats/bs_map_input/map/ExpertPlus.dat", "sha256": "b" * 64}]
    assert validate_curated_dataset(dataset)[0]["synthetic"] is False
    song.base.metadata["sources"][0]["path"] = "/tmp/evaluate_beats/favorites_bs_input/map/song.egg"
    with pytest.raises(ValueError, match="outside the curated"):
        validate_curated_dataset(dataset, allow_synthetic=True)


def test_augmented_hazard_scores_keep_scheduler_coverage_separate():
    from training.pipeline_v2.data import make_dummy_dataset
    from training.pipeline_v2.evaluation import ReferenceBaseline, evaluate_stage, fit_reference_baselines
    dataset = make_dummy_dataset(settings={})
    fitted = fit_reference_baselines(dataset, "hazards")
    report = evaluate_stage(dataset, "hazards", ReferenceBaseline("hazards", dataset.vocabularies, fitted),
                            batch_size=2, allow_synthetic=True)
    assert sum(report["hazard_query_counts"].values()) == report["examples"]
    assert set(report["hazard_scheduler_coverage_before_augmentation"]) == {"walls", "bombs"}
    assert set(report["scheduler_only_scores"]) == {"model", "frequency", "none"}
    assert "human" in report["hazard_query_conditioning"]
