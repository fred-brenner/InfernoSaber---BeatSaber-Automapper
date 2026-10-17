"""Explicit preparation, diagnostics and opt-in training for beat model artifacts."""

import argparse

import numpy as np

from .artifacts import (execution_identity, file_hash, load_dataset, load_statistics, new_run_directory, read_json, write_dataset,
                        write_json)
from .contracts import SCHEMA_VERSION, SEED, STAGES
from .training_data import load_training_dataset


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Extract and normalize an independent dataset")
    source = prepare.add_mutually_exclusive_group()
    source.add_argument("--dummy", action="store_true")
    source.add_argument("--source-root")
    source.add_argument("--prepared-root")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--reserved-root", help="Curated evaluation packages excluded from learning")
    prepare.add_argument("--normalization-from", help="Existing dataset/checkpoint manifest; never refit evaluation songs")
    prepare.add_argument("--evaluation-only", action="store_true", help="Keep curated recordings as test-only examples; requires existing normalization")
    prepare.add_argument("--difficulty", help="Select a raw Standard difficulty, for example ExpertPlus")
    prepare.add_argument("--aliases", help="JSON recording alias groups or alias-to-canonical mapping")
    prepare.add_argument("--seed", type=int, default=SEED)
    validate = sub.add_parser("validate", help="Check persisted arrays, hashes and representation")
    validate.add_argument("--dataset", required=True)
    smoke = sub.add_parser("smoke", help="Inference checks; keras also checks gradients and serialization without updates")
    smoke.add_argument("--backend", choices=("dummy", "keras"), default="dummy")
    smoke.add_argument("--dataset", help="Omit to use normalized deterministic dummy songs")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--max-events", type=int, default=8)
    train = sub.add_parser("train", help="Train only the explicitly selected new model stage")
    train.add_argument("--dataset", required=True)
    train.add_argument("--stage", choices=STAGES, required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--dry-run", action="store_true", help="Save configuration and a forward-check report; no optimizer steps")
    train.add_argument("--conditioning", choices=("reference", "noisy-reference", "predicted"),
                       help="Override the selected stage's tools/config/config.py default")
    train.add_argument("--intensity-artifact", "--intensity-checkpoint", dest="intensity_artifact")
    train.add_argument("--epochs", type=int, help="Override the selected stage's configured epoch count")
    train.add_argument("--batch-size", type=int, help="Override the selected stage's configured batch size")
    evaluate = sub.add_parser("evaluate", help="Compare new predictions with original and retained colored notes")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--backend", choices=("dummy", "keras"), default="dummy")
    evaluate.add_argument("--split", choices=("test", "validation", "train", "all"), default="test")
    for stage in STAGES:
        evaluate.add_argument(f"--{stage}-artifact", f"--{stage}-checkpoint", dest=f"{stage}_artifact")
    evaluate.add_argument("--requested-nps", type=float)
    evaluate.add_argument("--max-events", type=int)
    return command


def _dummy_data():
    from .data import make_dummy_songs
    from .features import fit_normalization, normalize_songs
    songs = make_dummy_songs()
    stats = fit_normalization(songs)
    normalized = normalize_songs(songs, stats)
    from .artifacts import statistics_hash
    from .contracts import REPRESENTATION, SCHEMA_VERSION
    manifest = {"schema_version": SCHEMA_VERSION, "representation": REPRESENTATION,
                "normalization": stats, "normalization_sha256": statistics_hash(stats),
                "sources": [], "splits": {split: [song.song_id for song in normalized if song.metadata["split"] == split]
                                           for split in ("train", "validation", "test")}}
    return normalized, manifest


def evaluate_dummy(songs, manifest, output, requested_nps=None, max_events=None):
    from .metrics import summarize
    from .pipeline import generate
    root = new_run_directory(output)
    reports, predictions = [], []
    for index, song in enumerate(songs):
        nps = requested_nps if requested_nps is not None else len(song.arrays["retained_indices"]) / song.metadata["duration"]
        generated = generate(song, nps, max_events=max_events)
        filename = f"prediction_{index:05d}.npz"
        np.savez_compressed(root / filename, intensity=generated.intensity,
                            local_nps=generated.local_nps, frame_probabilities=generated.frame_probabilities,
                            counts=generated.counts, event_frames=generated.event_frames,
                            event_times=generated.event_times, note_classes=generated.note_classes)
        predictions.append({"song_id": song.song_id, "file": filename, "sha256": file_hash(root / filename)})
        reports.append(generated.metrics)
    report = {"kind": "evaluation_report", "schema_version": SCHEMA_VERSION,
              "backend": "dummy", "status": "dummy", "seed": SEED, "representation": manifest["representation"],
              "normalization": manifest["normalization"],
              "normalization_sha256": manifest["normalization_sha256"],
              "sources": manifest.get("sources", []), "splits": manifest.get("splits", {}),
              "summary": summarize(reports), "songs": reports, "predictions": predictions,
              "evaluated_split": manifest.get("evaluation_split", "all"),
              "partial_diagnostic": any(report["partial_diagnostic"] for report in reports),
              "interpretation": "Deterministic interface diagnostics; untrained output is not a quality baseline.",
              **execution_identity()}
    write_json(root / "report.json", report)
    return report


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_songs
        from .features import fit_normalization, normalize_songs
        if args.evaluation_only and not args.normalization_from:
            command.error("--evaluation-only requires --normalization-from; evaluation songs cannot fit statistics")
        aliases = read_json(args.aliases) if args.aliases else None
        songs, report = prepare_songs(source_root=args.source_root, prepared_root=args.prepared_root,
                                     dummy=args.dummy, reserved_root=args.reserved_root, seed=args.seed,
                                     aliases=aliases, evaluation_only=args.evaluation_only, difficulty=args.difficulty)
        for key in ("skipped", "reserved_skipped"):
            for skip in report.get(key, []):
                print(f"Skipped: {skip}")
        if not songs:
            destination = new_run_directory(args.output)
            write_json(destination / "preparation-report.json", report)
            raise SystemExit(f"No accepted songs; full skip report saved in {destination / 'preparation-report.json'}")
        stats = load_statistics(args.normalization_from) if args.normalization_from else fit_normalization(songs)
        report["normalization_from"] = args.normalization_from
        manifest = write_dataset(args.output, normalize_songs(songs, stats), stats, report)
        print(f"Prepared {len(songs)} songs in {args.output}; splits { {k: len(v) for k, v in manifest['splits'].items()} }")
    elif args.command == "validate":
        songs, _ = load_dataset(args.dataset)
        print(f"Validated {len(songs)} songs, arrays, checksums, normalization and representation")
    elif args.command == "smoke":
        songs, manifest = load_dataset(args.dataset) if args.dataset else _dummy_data()
        if args.max_events <= 0:
            command.error("--max-events must be positive")
        if args.backend == "dummy":
            evaluate_dummy(songs, manifest, args.output, max_events=args.max_events)
        else:
            from .runner import smoke_models
            smoke_models(songs, manifest, args.output, max_events=args.max_events)
        print(f"{args.backend} smoke artifacts saved in {args.output}")
    elif args.command == "train":
        if (args.epochs is not None and args.epochs <= 0) or (args.batch_size is not None and args.batch_size <= 0):
            command.error("Epochs and batch size must be positive")
        if args.conditioning == "predicted" and not args.intensity_artifact:
            command.error("Predicted conditioning requires --intensity-artifact")
        if args.stage in ("intensity", "timing"):
            songs, manifest = load_training_dataset(args.dataset, args.stage)
        else:
            songs, manifest = load_dataset(args.dataset)
        from .runner import train_stage
        train_stage(songs, manifest, args.stage, args.run_dir, dry_run=args.dry_run,
                    conditioning=args.conditioning, intensity_artifact=args.intensity_artifact,
                    epochs=args.epochs, batch_size=args.batch_size)
        print(f"Stage {args.stage} {'dry run' if args.dry_run else 'training'} saved in {args.run_dir}")
    elif args.command == "evaluate":
        if args.requested_nps is not None and (not np.isfinite(args.requested_nps) or args.requested_nps < 0):
            command.error("--requested-nps must be finite and nonnegative")
        if args.max_events is not None and args.max_events <= 0:
            command.error("--max-events must be positive")
        songs, manifest = load_dataset(args.dataset)
        songs = [song for song in songs if args.split == "all" or song.metadata["split"] == args.split]
        if not songs:
            command.error(f"Dataset has no songs in the requested {args.split} split")
        manifest = dict(manifest, evaluation_split=args.split)
        if args.backend == "dummy":
            evaluate_dummy(songs, manifest, args.output, args.requested_nps, args.max_events)
        else:
            artifacts = {stage: getattr(args, f"{stage}_artifact") for stage in STAGES}
            if not all(artifacts.values()):
                command.error("Keras evaluation requires all three explicit stage artifacts")
            from .runner import evaluate_models
            evaluate_models(songs, manifest, artifacts, args.output, requested_nps=args.requested_nps,
                            max_events=args.max_events)
        print(f"Evaluation saved in {args.output}")


if __name__ == "__main__":
    main()
