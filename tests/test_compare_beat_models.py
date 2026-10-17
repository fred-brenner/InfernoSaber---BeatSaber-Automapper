"""Comparable beat stages, explicit density units and CLI isolation from notes."""

from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import compare_beat_models as comparison
from evaluation import evaluate_beat_algorithms as benchmark
from training.beat_models.contracts import Song


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    packages, references = [], {}
    for index in range(2):
        package = tmp_path / f"map-{index}"
        package.mkdir()
        for name in ("Info.dat", "ExpertPlus.dat", "song.ogg"):
            (package / name).write_bytes(name.encode())
        packages.append(package)
        references[package] = {"times": [.1, .100007, .5], "song_name": package.name,
            "bpm": 120, "difficulty": "ExpertPlus", "characteristic": "Standard",
            "info_path": package / "Info.dat", "beatmap_path": package / "ExpertPlus.dat",
            "audio_path": package / "song.ogg"}
    monkeypatch.setattr(comparison, "load_reference", lambda package, *args: references[package])
    return packages


class FakePair:
    metadata = {"checkpoints": {"intensity": "trained", "timing": "trained"}}
    manifests = {}

    def __init__(self, broken=None):
        self.calls, self.broken = [], broken

    def prepare(self, reference):
        if reference["song_name"] == self.broken:
            raise ValueError("Unsupported fixture")
        return Song(reference["song_name"], {"duration": 1, "decoded_sha256": "pcm"}, {
            "event_times": np.array([.1, .5]), "original_notes": np.zeros((3, 8)),
            "retained_indices": np.arange(3), "intensity": np.array([1], dtype=np.float32)})

    def predict(self, song, density):
        self.calls.append((song.song_id, density))
        counts = np.zeros(100, dtype=np.int32)
        counts[[10, 50]] = [2, 1]
        return {"times_seconds": np.array([.1, .5]), "intensity_shares": np.array([1], dtype=np.float32),
                "counts": counts, "frame_probabilities": np.eye(3, dtype=np.float32)[counts], "bpm": 120}


def old_runner(packages, bps_values, *args, generate_beats):
    rows = []
    for index, package in enumerate(packages):
        for bps in bps_values:
            rows.append({"song_id": str(package.resolve()), "bps_setting": bps, "status": "ok",
                         "prediction_times_seconds": [.1, .5], "audio_duration_seconds": 1})
            generate_beats.records.append({"name": f"song_{index:04d}", "bps_setting": bps,
                "times_seconds": [.1, .3], "probabilities": [.1, .9], "effective_threshold": .5,
                "beat_spacing": 100, "window": .0, "captured": True})
    return {"results": rows, "settings": {}, "model": {"sha256": "legacy"}}


def test_comparison_scores_same_sync_and_exact_references_and_saves_arrays(inputs, tmp_path):
    pair, observer = FakePair(), SimpleNamespace(records=[])
    report = comparison.run_comparison(inputs, [3, 5], beat_pair=pair, legacy_runner=old_runner,
                                      legacy_observer=observer, prediction_dir=tmp_path / "arrays")
    assert len(report["songs"]) == 2
    assert len(report["results"]) == 14  # old raw+processed, new nominal+reference density
    new = [row for row in report["results"] if row["backend"] == "intensity_timing"]
    assert all(row["reference_count"] == 2 and row["f1"] == 1 for row in new)
    assert all(row["exact_reference_metrics"]["reference_count"] == 3 for row in new)
    assert [density for _, density in pair.calls] == [3, 5, 3, 3, 5, 3]
    for row in new:
        assert row["colored_note_count"] == 3
        assert row["predicted_count"] == 2
        assert row["class_argmax_counts"] == [98, 1, 1]
        with np.load(row["arrays"]["file"], allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["counts"][[10, 50]], [2, 1])
    pairs = {row["comparator"]: row for row in report["paired_comparisons"] if row["bps_setting"] == 3}
    assert pairs["legacy_raw"]["f1_difference"] == .5
    assert pairs["legacy_processed"]["f1_difference"] == 0
    assert pairs["legacy_raw"]["paired_songs"] == 2
    assert report["reference_density_summary"]["f1"] == 1


def test_pair_only_never_invokes_legacy_or_notes(inputs):
    def forbidden(*args, **kwargs):
        pytest.fail("Legacy generation was invoked")
    report = comparison.run_comparison(inputs, [5], backend="intensity-timing", beat_pair=FakePair(),
                                      legacy_runner=forbidden)
    assert {row["backend"] for row in report["results"]} == {"intensity_timing"}
    assert report["paired_comparisons"] == []


def test_failed_shared_input_is_reported_and_paired_scores_use_common_songs(inputs):
    report = comparison.run_comparison(inputs, [5], beat_pair=FakePair(broken="map-1"),
        legacy_runner=old_runner, legacy_observer=SimpleNamespace(records=[]))
    failed = [row for row in report["results"] if row["status"] == "failed"]
    assert len(failed) == 2
    assert all("Unsupported fixture" in row["error"] for row in failed)
    assert all(row["paired_songs"] == 1 for row in report["paired_comparisons"])


def test_reference_density_failure_is_counted_in_its_summary(inputs):
    class BrokenReferencePair(FakePair):
        def predict(self, song, density):
            if song.song_id == "map-1" and density == 3:
                raise ValueError("Failed reference density")
            return super().predict(song, density)
    report = comparison.run_comparison(inputs, [5], backend="intensity-timing", beat_pair=BrokenReferencePair())
    assert report["reference_density_summary"]["successful_runs"] == 1
    assert report["reference_density_summary"]["failed_runs"] == 1


def test_cli_dispatches_only_explicit_pair_artifacts(inputs, tmp_path, monkeypatch):
    from training.beat_models import artifacts
    monkeypatch.setattr(artifacts, "load_model_manifest", lambda *args: (None, {"status": "trained"}))
    calls = []

    def run(packages, *args, **kwargs):
        calls.append((packages, kwargs))
        return {"results": [{"status": "ok"}], "summary_by_bps": []}

    monkeypatch.setattr(comparison, "run_comparison", run)
    run_root = tmp_path / "trained"
    output = tmp_path / "pair.json"
    assert benchmark.main(["--maps-dir", *map(str, inputs), "--backend", "intensity-timing",
                           "--trained-run", str(run_root), "--output", str(output)]) == 0
    assert len(calls) == 1
    assert calls[0][1]["intensity_artifact"] == run_root / "intensity"
    assert calls[0][1]["timing_artifact"] == run_root / "timing"
    assert "notes" not in calls[0][1]


@pytest.mark.parametrize("flag_args", [
    ["--backend", "compare"],
    ["--trained-run", "unused"],
    ["--backend", "compare", "--trained-run", "unused", "--timing-artifact", "other"],
])
def test_cli_rejects_missing_or_conflicting_artifacts_before_preparation(tmp_path, monkeypatch, flag_args):
    from evaluation import prepare_evaluation
    monkeypatch.setattr(prepare_evaluation, "prepare_evaluation", lambda *args: pytest.fail("Preparation ran"))
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--evaluation-root", str(tmp_path), *flag_args])
    assert error.value.code == 2


def test_new_backend_prepare_only_does_not_load_checkpoints(tmp_path, monkeypatch):
    from evaluation import prepare_evaluation
    from training.beat_models import artifacts
    monkeypatch.setattr(artifacts, "load_model_manifest", lambda *args: pytest.fail("Checkpoint was loaded"))
    monkeypatch.setattr(prepare_evaluation, "prepare_evaluation", lambda *args: {"songs": [], "skipped": []})
    assert benchmark.main(["--evaluation-root", str(tmp_path), "--backend", "intensity-timing", "--prepare-only"]) == 1
