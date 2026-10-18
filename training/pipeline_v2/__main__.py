"""Prepare, validate and connect v2 stages; training requires its own command."""

import argparse
from pathlib import Path

from .settings import STAGES


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Build immutable downstream targets and cached retained-encoder inputs")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--dummy", action="store_true")
    source.add_argument("--beat-dataset", help="Existing training.beat_models dataset; preserves splits/normalization")
    source.add_argument("--source-root", help="Intact raw map packages")
    source.add_argument("--prepared-root", help="Explicit existing beatmap/Info/audio triples adapter")
    prepare.add_argument("--encoder-checkpoint", help="Explicit retained 24x20-to-16 encoder checkpoint")
    prepare.add_argument("--structure-only", action="store_true", help="Zero audio latents; marked dummy and prohibited for training")
    prepare.add_argument("--normalization-from", help="Existing beat dataset/run used when reading raw inputs")
    prepare.add_argument("--reserved-root", help="Curated benchmark packages excluded from learning")
    prepare.add_argument("--aliases", help="Existing recording alias JSON for new raw preparation")
    prepare.add_argument("--evaluation-only", action="store_true", help="Prepare curated test-only packages using frozen training vocabularies/statistics")
    prepare.add_argument("--training-dataset", help="V2 training dataset supplying frozen evaluation contracts")
    prepare.add_argument("--output", required=True)
    validate = sub.add_parser("validate", help="Validate dataset hashes, stages, masks and split identities")
    validate.add_argument("--dataset", required=True)
    smoke = sub.add_parser("smoke", help="Synthetic interface or untrained forward/gradient/serialization checks")
    smoke.add_argument("--backend", choices=("dummy", "keras"), default="dummy")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--stage", choices=STAGES)
    smoke.add_argument("--max-examples", type=int, default=8)
    train = sub.add_parser("train", help="Explicit future training of one downstream stage")
    train.add_argument("--dataset", required=True)
    train.add_argument("--stage", choices=STAGES, required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    calibrate = sub.add_parser("calibrate", help="Calibrate one trained sparse stage on validation data only")
    calibrate.add_argument("--dataset", required=True)
    calibrate.add_argument("--stage", choices=("hazards", "lighting", "arcs"), required=True)
    calibrate.add_argument("--artifact", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--batch-size", type=int, default=128)
    concept = sub.add_parser("concept", help="Intermediate concept objects, with explicit review stops")
    concept.add_argument("--dataset", required=True)
    concept.add_argument("--song-id", help="Defaults to the first prepared song")
    concept.add_argument("--backend", choices=("dummy", "keras"), default="dummy")
    concept.add_argument("--requested-nps", type=float, required=True)
    concept.add_argument("--beat-run", help="Trained run containing intensity/ and timing/")
    concept.add_argument("--encoder-checkpoint")
    for stage in STAGES:
        concept.add_argument(f"--{stage}-artifact")
    concept.add_argument("--diagnostic", action="store_true", help="Explicitly bypass review stops; output remains unreviewed, never a map package")
    concept.add_argument("--max-events", type=int, help="Explicit partial diagnostic limit")
    concept.add_argument("--output", required=True)
    return command


def _prepare(args, command):
    from training.beat_models.artifacts import file_hash, load_statistics, read_json
    from .artifacts import BeatSongs, DatasetWriter, load_dataset, write_dataset
    from .data import make_dummy_dataset, prepare_dataset
    from .settings import preparation_settings
    settings = preparation_settings()
    frozen, normalization = None, None
    if args.evaluation_only:
        if not args.training_dataset or not args.source_root or args.dummy or args.prepared_root or args.beat_dataset:
            command.error("--evaluation-only requires --training-dataset and curated --source-root packages")
        training = load_dataset(args.training_dataset)
        settings = training.manifest["data_settings"]
        frozen = training.vocabularies
        normalization = training.manifest.get("beat_normalization")
        if args.normalization_from:
            normalization = load_statistics(args.normalization_from)
        if normalization is None:
            command.error("Selected training dataset has no beat statistics; provide --normalization-from")
        from training.beat_models.artifacts import statistics_hash
        if statistics_hash(normalization) != training.manifest["normalization_sha256"]:
            command.error("Evaluation normalization differs from the selected training dataset")
        source = Path(args.source_root).resolve()
        if "favorites_bs_input" in source.parts or source.name != "bs_map_input" or source.parent.name != "evaluate_beats":
            command.error("Evaluation source must be the curated evaluate_beats/bs_map_input directory")
    elif args.training_dataset:
        command.error("--training-dataset is used with --evaluation-only")
    if args.dummy:
        if args.encoder_checkpoint:
            command.error("Dummy preparation does not load checkpoints")
        return write_dataset(args.output, make_dummy_dataset(settings))
    if args.structure_only and args.encoder_checkpoint:
        command.error("Choose structural dummy latents or the explicit encoder, not both")
    if not args.structure_only and not args.encoder_checkpoint:
        command.error("Real training data requires --encoder-checkpoint; use --structure-only for zero-latent diagnostics")
    base_report = {}
    if args.beat_dataset:
        songs = BeatSongs(args.beat_dataset)
        normalization = songs.manifest["normalization"]
        base_report = {**songs.manifest.get("preparation", {}),
                       "base_dataset": str(Path(args.beat_dataset).resolve()),
                       "base_manifest_sha256": file_hash(Path(args.beat_dataset) / "manifest.json")}
    else:
        from training.beat_models.data import prepare_songs
        from training.beat_models.features import fit_normalization, normalize_songs
        from tools.config import paths
        reserved = (args.source_root if args.evaluation_only else args.reserved_root or
                    str(Path(paths.train_path) / "evaluate_beats/bs_map_input"))
        if not Path(reserved).is_dir():
            command.error("Curated reservation folder is unavailable; supply --reserved-root explicitly")
        songs, base_report = prepare_songs(source_root=args.source_root, prepared_root=args.prepared_root,
                                          reserved_root=reserved, aliases=read_json(args.aliases) if args.aliases else None,
                                          evaluation_only=args.evaluation_only, difficulty="ExpertPlus" if args.evaluation_only else None)
        if not songs:
            command.error("No accepted base songs: " + str(base_report.get("skipped", [])))
        if normalization is None:
            normalization = load_statistics(args.normalization_from) if args.normalization_from else fit_normalization(songs)
        songs = normalize_songs(songs, normalization)
    factory = None
    if args.encoder_checkpoint:
        from .runtime import EncoderFactory
        factory = EncoderFactory(args.encoder_checkpoint, settings)
        if args.evaluation_only and factory.identity != training.manifest["encoder_identity"]:
            command.error("Evaluation encoder differs from the training dataset")
    if args.evaluation_only and args.structure_only != (training.manifest.get("status") == "structural_dummy"):
        command.error("Evaluation and training datasets must use the same real or structural encoder mode")
    writer = DatasetWriter(args.output)
    try:
        dataset = prepare_dataset(songs, settings=settings, encoder=factory, emit=writer.add, collect=False, vocabularies=frozen)
    except ValueError as error:
        if hasattr(error, "preparation_audit"):
            from training.beat_models.artifacts import write_json
            write_json(writer.root / "preparation-audit.json", error.preparation_audit)
        raise
    dataset.manifest["base_preparation"] = base_report
    dataset.manifest["beat_normalization"] = normalization
    dataset.manifest["evaluation_only"] = args.evaluation_only
    if args.evaluation_only:
        dataset.manifest["training_dataset_identity"] = {
            "path": str(Path(args.training_dataset).resolve()),
            "manifest_sha256": file_hash(Path(args.training_dataset) / "manifest.json")}
    dataset.manifest["preparation_songs"] = dataset.manifest.pop("songs", [])
    report = writer.finish(dataset.vocabularies, dataset.manifest)
    for origin, rows in (("base preparation", base_report.get("skipped", [])),
                         ("reservation scan", base_report.get("reserved_skipped", [])),
                         ("downstream preparation", dataset.manifest.get("skipped", []))):
        for skipped in rows:
            print(f"Skipped ({origin}): {skipped}")
    return report


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.command == "prepare":
        report = _prepare(args, command)
        print(f"Prepared {len(report['songs'])} songs: {args.output}; status={report['status']}")
        return 0
    if args.command == "validate":
        from .artifacts import load_dataset
        from .contracts import validate
        dataset = load_dataset(args.dataset)
        for song in dataset.songs:
            validate(song)
        print(f"Validated {len(dataset.songs)} songs and six downstream stage contracts")
        return 0
    if args.command == "calibrate":
        from .artifacts import load_dataset
        from .calibration import calibrate_stage
        calibrate_stage(load_dataset(args.dataset), args.stage, args.artifact, args.output, batch_size=args.batch_size)
        print(f"Validation calibration saved: {args.output}; no training or bundle changes")
        return 0
    if args.command == "smoke":
        from .data import make_dummy_dataset
        dataset = make_dummy_dataset()
        if args.max_examples <= 0:
            command.error("--max-examples must be positive")
        if args.backend == "keras":
            from .runner import smoke_models
            smoke_models(dataset, args.output, stages=[args.stage] if args.stage else None, max_examples=args.max_examples)
        else:
            from .artifacts import execution_identity, new_run_directory, write_json
            from .pipeline import generate
            root = new_run_directory(args.output)
            result = generate(dataset.songs[0], 2., vocabularies=dataset.vocabularies,
                              settings=dataset.manifest["data_settings"], contract_mode=True)
            write_json(root / "report.json", {"kind": "v2_dummy_smoke", "status": "dummy", "optimizer_steps": 0,
                       "result": result.to_dict(), **execution_identity()})
        print(f"{args.backend} framework smoke saved: {args.output}; no training")
        return 0
    if args.command == "train":
        from .artifacts import load_dataset
        from .runner import train_stage
        train_stage(load_dataset(args.dataset), args.stage, args.run_dir, dry_run=args.dry_run,
                    epochs=args.epochs, batch_size=args.batch_size)
        print(f"{args.stage} {'dry run' if args.dry_run else 'training'} saved: {args.run_dir}")
        return 0
    from .artifacts import load_dataset, new_run_directory, write_json
    from .pipeline import ReviewRequired, generate
    dataset = load_dataset(args.dataset)
    song = next((s for s in dataset.songs if args.song_id is None or s.base.song_id == args.song_id), None)
    if song is None:
        command.error("Requested song ID is absent from the dataset")
    predictors, encoder, checkpoint_manifests = None, None, {}
    if args.backend == "keras":
        if not args.beat_run or not args.encoder_checkpoint:
            command.error("Keras concept inference needs --beat-run and --encoder-checkpoint")
        from .runtime import EncoderFactory, trained_predictors
        predictors, checkpoint_manifests = trained_predictors(dataset, args.beat_run,
            {stage: getattr(args, f"{stage}_artifact") for stage in STAGES})
        factory = EncoderFactory(args.encoder_checkpoint, dataset.manifest["data_settings"])
        if factory.identity != dataset.manifest.get("encoder_identity"):
            command.error("Selected retained encoder differs from the prepared dataset")
        encoder = factory(song)
    root = new_run_directory(args.output)
    stopped = False
    try:
        result = generate(song, args.requested_nps, predictors=predictors, encoder=encoder,
                          settings=dataset.manifest["data_settings"], vocabularies=dataset.vocabularies,
                          diagnostic=args.diagnostic, max_events=args.max_events)
    except ReviewRequired as error:
        stopped, result = True, error.partial_result
        print(str(error))
    write_json(root / "concept.json", {"backend": args.backend, "song_id": song.base.song_id,
               "checkpoint_manifests": checkpoint_manifests, **result.to_dict()})
    print(f"Concept intermediate output: {root / 'concept.json'}")
    return 3 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
