"""The curated evaluation command uses only successfully prepared input packages."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import evaluate_beat_algorithms as benchmark


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "evaluate_beats"
    source = root / "bs_map_input"
    accepted = source / "valid song"
    rejected = source / "unsupported song"
    accepted.mkdir(parents=True)
    rejected.mkdir()
    (tmp_path / "favorites_bs_input").mkdir()
    manifest = {"evaluation_root": str(root), "source_root": str(source),
                "songs": [{"package_path": str(accepted), "prepared_name": "valid song"}],
                "skipped": [{"package_path": str(rejected), "error": "Unsupported tempo changes"}]}
    calls = []

    def prepare(folder, difficulty, characteristic):
        assert folder == root
        assert (difficulty, characteristic) == ("ExpertPlus", "Standard")
        calls.append(folder)
        return manifest

    monkeypatch.setitem(sys.modules, "evaluation.prepare_evaluation", SimpleNamespace(prepare_evaluation=prepare))
    return root, accepted, manifest, calls


def test_evaluation_root_benchmarks_only_accepted_packages(prepared, monkeypatch):
    root, accepted, manifest, calls = prepared
    seen = []

    def run(packages, bps, *args):
        seen.extend(packages)
        assert bps == [3, 5, 7]
        return {"results": [{"status": "ok"}], "summary_by_bps": []}

    monkeypatch.setattr(benchmark, "run_benchmark", run)
    assert benchmark.main(["--evaluation-root", str(root)]) == 1  # A skip is visible to automation.
    assert seen == [accepted]
    assert calls == [root]
    outputs = list((root / "results").glob("*.json"))
    assert len(outputs) == 1
    assert json.loads(outputs[0].read_text())["preprocessing"] == manifest


def test_prepare_only_never_runs_model(prepared, monkeypatch):
    root, _, _, calls = prepared

    def forbidden(*args, **kwargs):
        raise AssertionError("Model execution must not happen")

    monkeypatch.setattr(benchmark, "run_benchmark", forbidden)
    assert benchmark.main(["--evaluation-root", str(root), "--prepare-only"]) == 1
    assert calls == [root]
    assert not (root / "results").exists()


def test_empty_preparation_reports_skips_without_model(prepared, monkeypatch):
    root, _, manifest, _ = prepared
    manifest["songs"] = []
    monkeypatch.setattr(benchmark, "run_benchmark", lambda *args: pytest.fail("No model for empty input"))
    assert benchmark.main(["--evaluation-root", str(root)]) == 1


@pytest.mark.parametrize("destination", ["bs_map_input/report.json", "maps/report.json", "archive/report.json"])
def test_output_cannot_pollute_sources_or_derived_folders(prepared, destination):
    root, _, _, calls = prepared
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--evaluation-root", str(root), "--output", str(root / destination)])
    assert error.value.code == 2
    assert calls == []


def test_invalid_bps_rejected_before_preparation(prepared):
    root, _, _, calls = prepared
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--evaluation-root", str(root), "--bps", "nan"])
    assert error.value.code == 2
    assert calls == []
