"""Scoring and orchestration regression tests, with model inference substituted."""

import copy
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation import evaluate_beat_algorithms as benchmark
from tools.config import config, paths


def test_import_and_help_do_not_load_runtime_or_change_config():
    subprocess.run([sys.executable, "-c", """
import sys
from evaluation.evaluate_beat_algorithms import calculate_beat_accuracy
assert 'tensorflow' not in sys.modules
assert 'tools.config.config' not in sys.modules
assert calculate_beat_accuracy([1, 1], [1], 0) == (0.5, 1, 2/3)
"""], cwd=ROOT_DIR, check=True)
    subprocess.run([sys.executable, str(ROOT_DIR / "evaluation/evaluate_beat_algorithms.py"),
                    "--help"], cwd=ROOT_DIR, check=True, capture_output=True)


def test_one_to_one_matching_penalizes_duplicate_predictions():
    result = benchmark.score_beats([2, 1.01, 1, 1], [1, 2], 0.02)
    assert result["matched_count"] == 2
    assert result["false_positives"] == 2
    assert result["false_negatives"] == 0
    assert (result["precision"], result["recall"], result["f1"]) == (0.5, 1, 2/3)


def test_earliest_eligible_matching_preserves_maximum_match_count():
    result = benchmark.score_beats([0.11, 0.09], [0.10, 0], 0.1)
    assert result["matched_count"] == 2
    assert result["mean_absolute_error_ms"] == pytest.approx(50)


@pytest.mark.parametrize("pred,real", [([], []), ([1], []), ([], [1]), ([1], [2])])
def test_empty_or_unmatched_inputs_have_defined_scores(pred, real):
    result = benchmark.score_beats(pred, real)
    assert result["precision"] == result["recall"] == result["f1"] == 0
    assert result["mean_absolute_error_ms"] is None
    assert result["false_positives"] == len(pred)
    assert result["false_negatives"] == len(real)


def test_inclusive_tolerance_and_exact_zero_tolerance():
    assert benchmark.score_beats([1.02], [1], 0.02)["matched_count"] == 1
    assert benchmark.score_beats([1.020001], [1], 0.02)["matched_count"] == 0
    assert benchmark.score_beats([1 + 1e-13], [1], 0)["matched_count"] == 0
    assert benchmark.score_beats([1], [1], 0)["matched_count"] == 1


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf")])
def test_invalid_tolerance_and_timestamps_are_rejected(value):
    with pytest.raises(ValueError):
        benchmark.score_beats([], [], value)
    with pytest.raises(ValueError):
        benchmark.score_beats([value], [1])
    with pytest.raises(ValueError):
        benchmark.score_beats([1], [value])


@pytest.fixture
def input_packages(tmp_path, monkeypatch):
    packages = []
    for index in range(2):
        package = tmp_path / f"human_{index}"
        package.mkdir()
        (package / "Info.dat").write_text(json.dumps({
            "_version": "2.0.0", "_songName": f"Song {index}", "_beatsPerMinute": 60,
            "_songFilename": "audio.ogg", "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [{
                    "_difficulty": "ExpertPlus", "_beatmapFilename": "ExpertPlus.dat"}]}],
        }))
        (package / "ExpertPlus.dat").write_text(json.dumps({
            "version": "3.2.0", "colorNotes": [{"b": 1, "c": 0}, {"b": 2, "c": 1}],
        }))
        (package / "audio.ogg").write_bytes(b"source audio")
        packages.append(package)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "tf_beat_gen_test.h5").write_bytes(b"model identity")
    monkeypatch.setattr(benchmark, "audio_duration_seconds", lambda path: 10.0)
    return packages, model_dir


def test_multiple_songs_and_bps_restore_settings_and_use_audio_copies(input_packages):
    packages, model_dir = input_packages
    original = copy.deepcopy((config.max_speed, config.max_speed_orig, config.thresh_beat,
                              config.obstacle_time_gap, config.add_beat_intensity))
    path_state = paths.songs_pred, paths.model_path
    numpy_state, random_state = np.random.get_state(), random.getstate()
    calls = []
    draws = []

    def generate(names, debug_beats):
        assert debug_beats is True
        assert config.max_speed == config.max_speed_orig
        assert config.thresh_beat == original[2]
        assert config.add_beat_intensity == original[4]
        np.testing.assert_array_equal(config.obstacle_time_gap, original[3])
        staged = Path(paths.songs_pred) / (names[0] + ".egg")
        assert staged.read_bytes() == b"source audio"
        assert not any(staged.is_relative_to(package) for package in packages)
        calls.append((names[0], config.max_speed, staged))
        draws.append((np.random.random(), random.random()))
        config.thresh_beat = -1
        config.add_beat_intensity = -1
        config.obstacle_time_gap[:] = -1
        staged.write_bytes(b"modified copy")
        return np.array([1, 2, 3])

    # The debug path is read-only, but even an unexpected write cannot touch originals.
    report = benchmark.run_benchmark(packages, [3, 7], model_dir=model_dir, generate_beats=generate)
    assert [(name, speed) for name, speed, _ in calls] == [
        ("song_0000", 12), ("song_0000", 28), ("song_0001", 12), ("song_0001", 28)]
    assert all(not staged.exists() for _, _, staged in calls)
    assert all((package / "audio.ogg").read_bytes() == b"source audio" for package in packages)
    assert all(draw == draws[0] for draw in draws)
    assert (paths.songs_pred, paths.model_path) == path_state
    assert (config.max_speed, config.max_speed_orig, config.thresh_beat) == original[:3]
    assert config.add_beat_intensity == original[4]
    np.testing.assert_array_equal(config.obstacle_time_gap, original[3])
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == random_state
    assert len(report["songs"]) == 2
    assert len(report["results"]) == 4
    assert report["model"]["sha256"] == benchmark.file_sha256(model_dir / "tf_beat_gen_test.h5")
    for row in report["summary_by_bps"]:
        assert row["successful_runs"] == 2
        assert row["predicted_count"] == 6
        assert row["reference_count"] == row["matched_count"] == 4
        assert row["precision"] == pytest.approx(2/3)
        assert row["predicted_density_bps"] == 0.3
        assert row["reference_density_bps"] == 0.2
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", [1, None, RuntimeError("broken"), SystemExit(0), np.array([[1]])])
def test_generation_failures_are_recorded_and_settings_restored(input_packages, failure):
    packages, model_dir = input_packages
    original_speed, original_threshold = config.max_speed, config.thresh_beat
    path_state = paths.songs_pred, paths.model_path

    def generate(names, debug_beats):
        config.thresh_beat = -1
        if isinstance(failure, BaseException):
            raise failure
        return failure

    report = benchmark.run_benchmark(packages, [5], model_dir=model_dir, generate_beats=generate)
    assert all(row["status"] == "failed" for row in report["results"])
    assert report["summary_by_bps"][0]["failed_runs"] == 2
    assert report["summary_by_bps"][0]["matched_count"] == 0
    assert (config.max_speed, config.thresh_beat) == (original_speed, original_threshold)
    assert (paths.songs_pred, paths.model_path) == path_state


def test_bad_reference_is_recorded_without_skipping_valid_songs(input_packages):
    packages, model_dir = input_packages
    (packages[0] / "Info.dat").write_text("bad json")
    report = benchmark.run_benchmark(packages, [3, 5], model_dir=model_dir,
                                     generate_beats=lambda names, debug_beats: [1, 2])
    assert [row["status"] for row in report["results"]] == ["failed", "failed", "ok", "ok"]
    assert all(row["failed_runs"] == row["successful_runs"] == 1 for row in report["summary_by_bps"])


def test_corrupt_audio_metadata_becomes_input_failure(tmp_path, monkeypatch):
    from mutagen import MutagenError
    import mutagen

    def corrupt(path):
        raise MutagenError("truncated stream")

    monkeypatch.setattr(mutagen, "File", corrupt)
    with pytest.raises(ValueError, match="decode audio metadata"):
        benchmark.audio_duration_seconds(tmp_path / "broken.ogg")


def test_cli_writes_report_and_refuses_overwrite(input_packages, tmp_path, monkeypatch):
    packages, model_dir = input_packages
    run_benchmark = benchmark.run_benchmark

    def run(*args, **kwargs):
        return run_benchmark(*args, **kwargs, generate_beats=lambda names, debug_beats: [1, 2])

    monkeypatch.setattr(benchmark, "run_benchmark", run)
    output = tmp_path / "results" / "benchmark.json"
    args = ["--maps-dir", str(packages[0]), str(packages[1]), "--bps", "3", "5", "3",
            "--model-dir", str(model_dir), "--output", str(output)]
    assert benchmark.main(args) == 0
    report = json.loads(output.read_text())
    assert report["stage"] == "processed_timing_proposals"
    assert len(report["results"]) == 4
    assert len(report["summary_by_bps"]) == 2
    assert report["runtime"]["python"]
    assert report["code"]["source_sha256"]["evaluation/human_maps.py"]
    assert report["training_overlap"].startswith("unknown")
    with pytest.raises(SystemExit) as exc:
        benchmark.main(args)
    assert exc.value.code == 2


def test_discovery_deduplicates_and_sorts_packages(input_packages, tmp_path):
    packages, _ = input_packages
    assert benchmark.discover_packages([tmp_path, packages[0]]) == packages
    with pytest.raises(ValueError, match="No map packages"):
        benchmark.discover_packages([tmp_path / "model"])


@pytest.mark.parametrize("fail_import", [False, True])
def test_lazy_generator_import_preserves_callers_state(input_packages, monkeypatch, fail_import):
    import builtins
    from types import SimpleNamespace

    packages, model_dir = input_packages
    original_import = builtins.__import__
    numpy_state = np.random.get_state()
    original_threshold = config.thresh_beat

    def importing(name, *args, **kwargs):
        if name == "map_creation.gen_beats":
            np.random.seed(42)  # Existing preprocessing modules seed at import.
            config.thresh_beat = -1
            if fail_import:
                raise ImportError("missing runtime")
            return SimpleNamespace(main=lambda names, debug_beats: [1, 2])
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    if fail_import:
        with pytest.raises(ImportError, match="missing runtime"):
            benchmark.run_benchmark(packages, [5], model_dir=model_dir)
    else:
        report = benchmark.run_benchmark(packages, [5], model_dir=model_dir)
        assert report["summary_by_bps"][0]["successful_runs"] == 2
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert config.thresh_beat == original_threshold


def test_aggregate_scores_and_error_weight_matches_instead_of_songs():
    rows = []
    for predicted, reference, duration in [([1.01, 3], [1], 10), (list(range(10)), list(range(10)), 20)]:
        row = benchmark.score_beats(predicted, reference)
        row.update(status="ok", bps_setting=5, audio_duration_seconds=duration)
        rows.append(row)
    summary = benchmark.summarize_results(rows, [5])[0]
    assert summary["precision"] == pytest.approx(11/12)
    assert summary["recall"] == 1
    assert summary["mean_absolute_error_ms"] == pytest.approx(10/11)
    assert summary["predicted_density_bps"] == 12/30


def test_cli_saves_failed_runs_and_returns_nonzero(input_packages, tmp_path, monkeypatch):
    packages, model_dir = input_packages
    run_benchmark = benchmark.run_benchmark

    def run(*args, **kwargs):
        return run_benchmark(*args, **kwargs, generate_beats=lambda names, debug_beats: 1)

    monkeypatch.setattr(benchmark, "run_benchmark", run)
    output = tmp_path / "failed.json"
    assert benchmark.main(["--maps-dir", str(packages[0]), "--bps", "5",
                           "--model-dir", str(model_dir), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["results"][0]["status"] == "failed"
    assert "failure status 1" in report["results"][0]["error"]
    assert report["summary_by_bps"][0]["failed_runs"] == 1
