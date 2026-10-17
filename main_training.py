"""Choose and train all eight v2 models sequentially."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


BEAT_STAGES = ("intensity", "timing")
STAGES = (*BEAT_STAGES, "positions", "arrows", "multiplier", "hazards", "lighting", "arcs")


def selected_models(value):
    """Accept menu numbers/names and return unique models in pipeline order."""
    selected = set()
    for token in value.lower().replace(",", " ").split():
        if token in ("all", "alle"):
            selected.update(STAGES)
        elif token in STAGES:
            selected.add(token)
        elif token.isdigit() and 1 <= int(token) <= len(STAGES):
            selected.add(STAGES[int(token) - 1])
        else:
            raise ValueError(f"Unknown model: {token}. Choose 1-{len(STAGES)}, model names, or all.")
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
                         help="Model names or menu numbers; all selects all eight and skips the menu")
    command.add_argument("--dry-run", action="store_true", help="Pass --dry-run to each stage training command")
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        if args.models is None:
            print("Choose v2 models to train (spaces/commas; all/alle selects all; blank cancels):")
            for number, stage in enumerate(STAGES, 1):
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
        for number, stage in enumerate(models, 1):
            stage_output = root / stage
            print(f"[{number}/{len(models)}] {stage}: {stage_output}", flush=True)
            module = "training.beat_models" if stage in BEAT_STAGES else "training.pipeline_v2"
            training = [sys.executable, "-m", module, "train", "--stage", stage,
                        "--dataset", str(datasets[stage]), "--run-dir", str(stage_output)]
            if args.dry_run:
                training.append("--dry-run")
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
