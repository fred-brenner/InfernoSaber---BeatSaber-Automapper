"""Choose v2 models to train sequentially or with bounded parallelism."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


BEAT_STAGES = ("intensity", "timing")
MENU_STAGES = {1: "intensity", 2: "timing", 3: "positions", 4: "arrows",
               5: "multiplier", 6: "hazards", 8: "arcs"}
STAGES = tuple(MENU_STAGES.values())


def selected_models(value):
    """Accept menu numbers/names and return unique models in pipeline order."""
    selected = set()
    for token in value.lower().replace(",", " ").split():
        if token == "lighting" or (token.isdigit() and int(token) == 7):
            raise ValueError("V2 lighting is not yet implemented for production and is deferred from this selector. "
                             "For experimental training, use: python -m training.pipeline_v2 train --stage lighting")
        if token in ("all", "alle"):
            selected.update(STAGES)
        elif token in STAGES:
            selected.add(token)
        elif token.isdigit() and int(token) in MENU_STAGES:
            selected.add(MENU_STAGES[int(token)])
        else:
            raise ValueError(f"Unknown model: {token}. Choose 1-6 or 8, model names, or all.")
    return tuple(stage for stage in STAGES if stage in selected)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dataset", type=Path, default=Path("../Data/training/pipeline_v2/full/dataset"),
                         help="Prepared downstream v2 dataset for positions through arcs")
    command.add_argument("--beat-dataset", type=Path,
                         default=Path("../Data/training/beat_models_v2/full-2026-10-17/dataset"),
                         help="Prepared beat dataset for intensity and timing")
    command.add_argument("--run-dir", type=Path,
                         help="New queue output directory (default: ../Data/training/pipeline_v2/runs/<timestamp>)")
    command.add_argument("--models", nargs="+", metavar="MODEL",
                         help="Model names or menu numbers; all selects all seven supported v2 stages and skips the menu")
    command.add_argument("--parallel", type=int, default=1, metavar="N",
                         help="Maximum simultaneous model processes (default: 1, sequential)")
    command.add_argument("--dry-run", action="store_true", help="Pass --dry-run to each stage training command")
    return command


def run_parallel(jobs, limit, cwd):
    """Start up to limit processes; stop this queue's children on failure/Ctrl+C."""
    pending = iter(enumerate(jobs, 1))
    running = []
    try:
        while True:
            for stage, process in running[:]:
                code = process.poll()
                if code is None:
                    continue
                running.remove((stage, process))
                if code:
                    print(f"{stage} failed (exit code {code}); stopping remaining model processes.",
                          file=sys.stderr)
                    return code if code > 0 else 128 - code
            while len(running) < limit:
                job = next(pending, None)
                if job is None:
                    break
                number, (stage, training) = job
                print(f"[{number}/{len(jobs)}] {stage}", flush=True)
                running.append((stage, subprocess.Popen(training, cwd=cwd)))
            if not running:
                return 0
            time.sleep(0.1)
    finally:
        for _, process in running:
            if process.poll() is None:
                process.terminate()
        for _, process in running:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.parallel < 1:
        command.error("--parallel must be at least 1")
    try:
        if args.models is None:
            print("Choose v2 models to train (spaces/commas; all/alle selects all; blank cancels):")
            for number, stage in MENU_STAGES.items():
                print(f"  {number}. {stage}")
            selection = input("Models: ")
        else:
            selection = " ".join(args.models)
        models = selected_models(selection)
        if not models:
            print("Training cancelled.")
            return 0

        datasets = {stage: (args.beat_dataset if stage in BEAT_STAGES else args.dataset).expanduser().resolve()
                    for stage in models}
        for dataset in dict.fromkeys(datasets.values()):
            if not (dataset / "manifest.json").is_file():
                command.error(f"Prepared dataset manifest is missing: {dataset / 'manifest.json'}")
        root = args.run_dir or Path("../Data/training/pipeline_v2/runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        root = root.expanduser().absolute()
        if root.exists() or root.is_symlink():
            command.error(f"Queue output directory already exists: {root}. Choose a new --run-dir.")
        root.mkdir(parents=True)
        root = root.resolve()
        for dataset in dict.fromkeys(datasets.values()):
            print(f"Dataset: {dataset}")
        print(f"Queue output: {root}")
        repo = Path(__file__).resolve().parent
        jobs = []
        for stage in models:
            stage_output = root / stage
            module = "training.beat_models" if stage in BEAT_STAGES else "training.pipeline_v2"
            training = [sys.executable, "-m", module, "train", "--stage", stage,
                        "--dataset", str(datasets[stage]), "--run-dir", str(stage_output)]
            if args.dry_run:
                training.append("--dry-run")
            jobs.append((stage, training))
        if args.parallel > 1:
            code = run_parallel(jobs, args.parallel, repo)
            if code:
                return code
        else:
            for number, (stage, training) in enumerate(jobs, 1):
                print(f"[{number}/{len(models)}] {stage}: {root / stage}", flush=True)
                result = subprocess.run(training, cwd=repo)
                if result.returncode:
                    print(f"{stage} failed (exit code {result.returncode}); remaining models were not started.",
                          file=sys.stderr)
                    return result.returncode if result.returncode > 0 else 128 - result.returncode
        print(f"Completed {len(models)} model(s): {root}")
        return 0
    except (ValueError, OSError) as error:
        command.error(str(error))
    except KeyboardInterrupt:
        print("\nTraining interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
