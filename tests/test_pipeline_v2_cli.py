"""CPU command contracts with synthetic data; no training/comparison execution."""

import json
import subprocess
import sys

from training.pipeline_v2.__main__ import main


def test_help_is_inert_without_tensorflow_import():
    code = "from training.pipeline_v2.__main__ import parser; import sys; parser().format_help(); assert 'tensorflow' not in sys.modules; assert 'main_training' not in sys.modules; assert 'main_app' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_dummy_prepare_validate_and_concept_stop(tmp_path):
    dataset = str(tmp_path / "dataset")
    assert main(["prepare", "--dummy", "--output", dataset]) == 0
    assert main(["validate", "--dataset", dataset]) == 0
    output = tmp_path / "concept"
    assert main(["concept", "--dataset", dataset, "--requested-nps", "2", "--output", str(output)]) == 3
    result = json.loads((output / "concept.json").read_text())
    assert result["review_stops"][0]["boundary"] == "bpm_alignment"
    assert not result["review_stops"][0]["bypassed"]
    assert not list(output.glob("*.dat"))


def test_dummy_smoke_runs_all_stages_without_training(tmp_path):
    assert main(["smoke", "--backend", "dummy", "--output", str(tmp_path / "smoke")]) == 0
    report = json.loads((tmp_path / "smoke/report.json").read_text())
    assert report["optimizer_steps"] == 0
    assert report["status"] == "dummy"
    assert len(report["result"]["review_stops"]) >= 5
    assert not report["result"]["diagnostics"]["safety_reviewed"]
