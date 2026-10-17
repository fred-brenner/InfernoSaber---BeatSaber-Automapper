"""Development queue contracts; subprocesses are mocked and never train models."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import main_training


@pytest.mark.parametrize("selection, expected", [
    ("positions arrows", ("positions", "arrows")),
    ("1, 2 3 8", ("intensity", "timing", "positions", "arcs")),
    ("ARCS, positions 3 arcs", ("positions", "arcs")),
    ("all", main_training.STAGES),
    ("alle", main_training.STAGES),
    ("   ", ()),
])
def test_menu_selection(selection, expected):
    assert main_training.selected_models(selection) == expected


@pytest.mark.parametrize("selection", ["0", "9", "legacy", "all unknown"])
def test_menu_rejects_unknown_models(selection):
    with pytest.raises(ValueError, match="Unknown model"):
        main_training.selected_models(selection)


@pytest.fixture
def queue_paths(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    beat_dataset = tmp_path / "beat-dataset"
    beat_dataset.mkdir()
    (beat_dataset / "manifest.json").write_text("{}")
    return dataset, beat_dataset, tmp_path / "queue"


@pytest.mark.parametrize("dry_run", [False, True])
def test_all_eight_commands_are_sequential_and_use_the_active_python(queue_paths, monkeypatch, dry_run):
    dataset, beat_dataset, root = queue_paths
    calls = []

    def run(command, *, cwd):
        # Only the queue root exists; each trainer owns its stage output directory.
        assert root.is_dir()
        assert not (root / command[5]).exists()
        calls.append((command, cwd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(main_training.subprocess, "run", run)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("--models must skip the menu"))
    args = ["--dataset", str(dataset), "--beat-dataset", str(beat_dataset),
            "--run-dir", str(root), "--models", "all"]
    if dry_run:
        args.append("--dry-run")
    assert main_training.main(args) == 0
    assert len(calls) == 8
    for stage, (command, cwd) in zip(main_training.STAGES, calls):
        beat = stage in ("intensity", "timing")
        module = "training.beat_models" if beat else "training.pipeline_v2"
        selected_dataset = beat_dataset if beat else dataset
        expected = [sys.executable, "-m", module, "train", "--stage", stage,
                    "--dataset", str(selected_dataset.resolve()), "--run-dir", str(root.resolve() / stage)]
        if dry_run:
            expected.append("--dry-run")
        assert command == expected
        assert cwd == Path(main_training.__file__).resolve().parent


def test_interactive_menu_and_cli_share_canonical_order(queue_paths, monkeypatch, capsys):
    dataset, _, root = queue_paths
    calls = []
    monkeypatch.setattr("builtins.input", lambda _: "8, 3 positions")
    monkeypatch.setattr(main_training.subprocess, "run",
                        lambda command, **_: calls.append(command) or SimpleNamespace(returncode=0))
    assert main_training.main(["--dataset", str(dataset), "--run-dir", str(root)]) == 0
    assert [command[5] for command in calls] == ["positions", "arcs"]
    output = capsys.readouterr().out
    for number, stage in enumerate(main_training.STAGES, 1):
        assert f"{number}. {stage}" in output
    assert "[1/2] positions" in output
    assert "[2/2] arcs" in output


@pytest.mark.parametrize("stage, relative, module", [
    ("positions", "pipeline_v2/full/dataset", "training.pipeline_v2"),
    ("intensity", "beat_models_v2/full-2026-10-17/dataset", "training.beat_models"),
    ("timing", "beat_models_v2/full-2026-10-17/dataset", "training.beat_models"),
])
def test_default_dataset_and_timestamped_queue_root(tmp_path, monkeypatch, stage, relative, module):
    work = tmp_path / "repo"
    work.mkdir()
    dataset = tmp_path / "Data/training" / relative
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}")
    monkeypatch.chdir(work)
    calls = []
    monkeypatch.setattr(main_training.subprocess, "run",
                        lambda command, **_: calls.append(command) or SimpleNamespace(returncode=0))
    assert main_training.main(["--models", stage]) == 0
    assert calls[0][2] == module
    assert calls[0][7] == str(dataset.resolve())
    stage_output = Path(calls[0][9])
    assert stage_output.name == stage
    assert stage_output.parent.parent == tmp_path / "Data/training/pipeline_v2/runs"
    assert stage_output.parent.is_dir()


@pytest.mark.parametrize("selection", [None, ["unknown"]])
def test_cancel_or_invalid_selection_never_creates_outputs(tmp_path, monkeypatch, selection):
    root = tmp_path / "queue"
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(main_training.subprocess, "run", lambda *_, **__: pytest.fail("No training expected"))
    args = ["--dataset", str(tmp_path / "missing"), "--run-dir", str(root)]
    if selection is None:
        assert main_training.main(args) == 0
    else:
        with pytest.raises(SystemExit) as error:
            main_training.main(args + ["--models"] + selection)
        assert error.value.code == 2
    assert not root.exists()


@pytest.mark.parametrize("stage, flag", [("positions", "--dataset"), ("intensity", "--beat-dataset")])
def test_missing_manifest_is_rejected_before_output_creation(tmp_path, monkeypatch, stage, flag):
    root = tmp_path / "new-parent" / "queue"
    monkeypatch.setattr(main_training.subprocess, "run", lambda *_, **__: pytest.fail("No training expected"))
    with pytest.raises(SystemExit) as error:
        main_training.main([flag, str(tmp_path), "--run-dir", str(root), "--models", stage])
    assert error.value.code == 2
    assert not root.parent.exists()


def test_existing_output_is_refused(queue_paths, monkeypatch):
    dataset, beat_dataset, root = queue_paths
    root.mkdir()
    marker = root / "keep.txt"
    marker.write_text("keep")
    monkeypatch.setattr(main_training.subprocess, "run", lambda *_, **__: pytest.fail("No training expected"))
    with pytest.raises(SystemExit) as error:
        main_training.main(["--beat-dataset", str(beat_dataset), "--run-dir", str(root), "--models", "1"])
    assert error.value.code == 2
    assert marker.read_text() == "keep"
    assert list(root.iterdir()) == [marker]


@pytest.mark.parametrize("returncode, expected", [(9, 9), (-2, 130)])
def test_stage_failure_stops_the_queue(queue_paths, monkeypatch, returncode, expected):
    dataset, beat_dataset, root = queue_paths
    calls = []

    def run(command, **_):
        calls.append(command)
        return SimpleNamespace(returncode=returncode if command[5] == "arrows" else 0)

    monkeypatch.setattr(main_training.subprocess, "run", run)
    assert main_training.main(["--dataset", str(dataset), "--beat-dataset", str(beat_dataset),
                               "--run-dir", str(root), "--models", "all"]) == expected
    assert [command[5] for command in calls] == ["intensity", "timing", "positions", "arrows"]


def test_keyboard_interrupt_returns_130(queue_paths, monkeypatch):
    dataset, _, root = queue_paths

    def interrupted(*_, **__):
        raise KeyboardInterrupt

    monkeypatch.setattr(main_training.subprocess, "run", interrupted)
    assert main_training.main(["--dataset", str(dataset), "--run-dir", str(root), "--models", "positions"]) == 130


@pytest.mark.parametrize("beat", [False, True])
def test_only_selected_dataset_family_is_required(queue_paths, monkeypatch, beat):
    dataset, beat_dataset, root = queue_paths
    missing = dataset.parent / "unused-missing"
    calls = []
    monkeypatch.setattr(main_training.subprocess, "run",
                        lambda command, **_: calls.append(command) or SimpleNamespace(returncode=0))
    models = ["intensity", "timing"] if beat else ["positions"]
    args = ["--dataset", str(missing if beat else dataset),
            "--beat-dataset", str(beat_dataset if beat else missing),
            "--run-dir", str(root), "--models", *models]
    assert main_training.main(args) == 0
    assert [command[5] for command in calls] == models


def test_mixed_queue_checks_both_datasets_before_starting_anything(queue_paths, monkeypatch):
    dataset, _, root = queue_paths
    missing = dataset.parent / "missing-beat-dataset"
    monkeypatch.setattr(main_training.subprocess, "run", lambda *_, **__: pytest.fail("No training expected"))
    with pytest.raises(SystemExit) as error:
        main_training.main(["--dataset", str(dataset), "--beat-dataset", str(missing),
                            "--run-dir", str(root), "--models", "positions", "timing"])
    assert error.value.code == 2
    assert not root.exists()


def test_import_and_help_are_inert(tmp_path):
    repo = Path(main_training.__file__).resolve().parent
    code = """
import builtins
from pathlib import Path
import subprocess
import sys

def unexpected(*args, **kwargs):
    raise AssertionError('Import/help must not prompt, create output or start subprocesses')

builtins.input = unexpected
Path.mkdir = unexpected
subprocess.run = unexpected
sys.path.insert(0, sys.argv[1])
import main_training
try:
    main_training.main(['--help'])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('Help must exit')
for name in ('tensorflow', 'tools.config.config', 'training.helpers', 'main_app'):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(repo)], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    assert "--models" in result.stdout
    assert "--parallel" in result.stdout
    assert list(tmp_path.iterdir()) == []


class SimulatedProcess:
    def __init__(self, command):
        self.command = command
        self.returncode = None
        self.terminated = self.killed = False
        self.waits = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.returncode


@pytest.mark.parametrize("parallel", [2, 3, 12])
@pytest.mark.parametrize("dry_run", [False, True])
def test_parallel_queue_refills_free_slots_and_preserves_commands(queue_paths, monkeypatch, parallel, dry_run):
    dataset, beat_dataset, root = queue_paths
    processes, overlaps = [], []

    def active():
        return [process for process in processes if process.returncode is None]

    def popen(command, *, cwd):
        assert cwd == Path(main_training.__file__).resolve().parent
        assert len(active()) < parallel
        process = SimulatedProcess(command)
        processes.append(process)
        return process

    def advance(seconds):
        assert seconds == .1
        live = active()
        overlaps.append(len(live))
        # Complete the most recently launched job first. The first job remains
        # busy while subsequent jobs use the other available slot(s).
        live[-1].returncode = 0

    monkeypatch.setattr(main_training.subprocess, "Popen", popen)
    monkeypatch.setattr(main_training.subprocess, "run", lambda *_, **__: pytest.fail("Parallel queue used run"))
    monkeypatch.setattr(main_training.time, "sleep", advance)
    args = ["--dataset", str(dataset), "--beat-dataset", str(beat_dataset),
            "--run-dir", str(root), "--models", "all", "--parallel", str(parallel)]
    if dry_run:
        args.append("--dry-run")
    assert main_training.main(args) == 0
    assert [process.command[5] for process in processes] == list(main_training.STAGES)
    assert max(overlaps) == min(parallel, len(main_training.STAGES))
    for process in processes:
        stage = process.command[5]
        beat = stage in main_training.BEAT_STAGES
        expected = [sys.executable, "-m", "training.beat_models" if beat else "training.pipeline_v2",
                    "train", "--stage", stage, "--dataset", str((beat_dataset if beat else dataset).resolve()),
                    "--run-dir", str(root.resolve() / stage)]
        assert process.command == expected + (["--dry-run"] if dry_run else [])
        assert not process.terminated and not process.killed
    if parallel == 2:
        assert overlaps[:-1] == [2] * (len(main_training.STAGES) - 1)


@pytest.mark.parametrize("returncode,expected", [(9, 9), (-2, 130)])
def test_parallel_failure_stops_active_children_and_pending_jobs(queue_paths, monkeypatch, returncode, expected):
    dataset, _, root = queue_paths
    processes = []

    def popen(command, **kwargs):
        process = SimulatedProcess(command)
        processes.append(process)
        return process

    def fail_one(seconds):
        assert len(processes) == 2
        processes[1].returncode = returncode

    monkeypatch.setattr(main_training.subprocess, "Popen", popen)
    monkeypatch.setattr(main_training.time, "sleep", fail_one)
    assert main_training.main(["--dataset", str(dataset), "--run-dir", str(root),
                               "--models", "positions", "arrows", "multiplier", "hazards",
                               "--parallel", "2"]) == expected
    assert [process.command[5] for process in processes] == ["positions", "arrows"]
    assert processes[0].terminated and processes[0].waits == [5]
    assert not processes[1].terminated


@pytest.mark.parametrize("unresponsive", [False, True])
def test_parallel_interrupt_cleans_up_children_and_returns_130(queue_paths, monkeypatch, unresponsive):
    dataset, _, root = queue_paths
    processes = []

    def popen(command, **kwargs):
        process = SimulatedProcess(command)
        if unresponsive and not processes:
            original_wait = process.wait

            def wait(timeout=None):
                if timeout is not None:
                    raise subprocess.TimeoutExpired(command, timeout)
                return original_wait(timeout)

            process.wait = wait
        processes.append(process)
        return process

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(main_training.subprocess, "Popen", popen)
    monkeypatch.setattr(main_training.time, "sleep", interrupt)
    assert main_training.main(["--dataset", str(dataset), "--run-dir", str(root),
                               "--models", "positions", "arrows", "multiplier", "--parallel", "2"]) == 130
    assert len(processes) == 2 and all(process.terminated for process in processes)
    assert processes[0].killed == unresponsive


def test_parallel_launch_error_cleans_up_already_started_child(queue_paths, monkeypatch):
    dataset, _, root = queue_paths
    processes = []

    def popen(command, **kwargs):
        if processes:
            raise OSError("Cannot start next process")
        process = SimulatedProcess(command)
        processes.append(process)
        return process

    monkeypatch.setattr(main_training.subprocess, "Popen", popen)
    with pytest.raises(SystemExit) as error:
        main_training.main(["--dataset", str(dataset), "--run-dir", str(root),
                            "--models", "positions", "arrows", "--parallel", "2"])
    assert error.value.code == 2
    assert processes[0].terminated and processes[0].waits == [5]


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_invalid_parallelism_is_rejected_before_any_work(queue_paths, monkeypatch, value):
    dataset, _, root = queue_paths
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("No prompt expected"))
    monkeypatch.setattr(main_training.subprocess, "Popen", lambda *_, **__: pytest.fail("No processes expected"))
    with pytest.raises(SystemExit) as error:
        main_training.main(["--dataset", str(dataset), "--run-dir", str(root), "--parallel", value])
    assert error.value.code == 2
    assert not root.exists()


def test_parallelism_defaults_to_one():
    assert main_training.parser().parse_args([]).parallel == 1
