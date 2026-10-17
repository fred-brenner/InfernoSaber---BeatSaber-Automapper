"""Plan or explicitly execute an independent v2 model comparison on CPU.

Default invocation only displays a plan and does not traverse files or load
models. Add --execute when the user explicitly requests a comparison run.
"""

import argparse
import json


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--stage", required=True,
                         choices=("encoder", "positions", "arrows", "multiplier", "hazards", "lighting", "arcs"))
    command.add_argument("--dataset", help="Original prepared training dataset for frozen compatibility and training-only baseline statistics")
    command.add_argument("--evaluation-dataset", required=True, help="Curated test-only v2 dataset prepared with the original frozen vocabulary/normalization")
    command.add_argument("--evaluation-root", help="Explicit relocated copy of the curated bs_map_input folder; favorites are always refused")
    command.add_argument("--artifact", required=True, help="Explicit stage artifact or retained encoder checkpoint")
    command.add_argument("--output", required=True, help="A new comparison report directory")
    command.add_argument("--split", choices=("test", "validation"), default="test")
    command.add_argument("--batch-size", type=int, default=128)
    command.add_argument("--threshold", type=float, default=.5)
    command.add_argument("--audio", help="Explicit source audio for encoder comparison")
    command.add_argument("--autoencoder-artifact", help="Optional retained reconstruction checkpoint")
    command.add_argument("--encoder-grid-seconds", type=float, default=.5)
    action = command.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="Display the plan without reading files (the default)")
    action.add_argument("--execute", action="store_true", help="Explicitly run inference and create a report")
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.batch_size <= 0 or not 0 < args.threshold < 1 or args.encoder_grid_seconds <= 0:
        command.error("Batch size and encoder grid must be positive; threshold must be between zero and one")
    if args.stage == "encoder" and not args.audio:
        command.error("Encoder comparisons require --audio")
    if args.stage != "encoder" and not args.dataset:
        command.error("Stage comparisons require --dataset")
    if not args.execute:
        plan = {"status": "planned", "stage": args.stage, "dataset": args.dataset,
                "evaluation_dataset": args.evaluation_dataset, "artifact": args.artifact,
                "output": args.output, "execution_requested": False,
                "instruction": "Add --execute only when an independent comparison run is requested."}
        print(json.dumps(plan, indent=2))
        return plan

    from pathlib import Path
    from training.beat_models.artifacts import file_hash, new_run_directory, write_json
    from training.pipeline_v2.evaluation import (evaluate_encoder, evaluate_stage, preparation_reports,
                                                validate_curated_dataset)
    from training.pipeline_v2.artifacts import load_dataset
    from training.pipeline_v2 import models

    try:
        models.tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        if models.tf.config.get_visible_devices("GPU"):
            raise RuntimeError("Independent v2 comparisons require CPU execution") from error
    with models.tf.device("/CPU:0"):
        evaluation_dataset = load_dataset(args.evaluation_dataset, lazy=True)
        curated = validate_curated_dataset(evaluation_dataset, split=args.split, evaluation_root=args.evaluation_root)
        if args.stage == "encoder":
            from training.pipeline_v2.encoder import FixedEncoder, decode_encoder_audio
            checkpoint = Path(args.artifact)
            selected_audio = str(Path(args.audio).resolve())
            audio_sources = [source for song in curated for source in song["sources"]
                             if source.get("role") == "audio" and str(Path(source["path"]).resolve()) == selected_audio]
            if not audio_sources:
                raise ValueError("--audio must be an exact original source member of the curated evaluation dataset")
            if any(source["sha256"] != file_hash(args.audio) for source in audio_sources):
                raise ValueError("Curated encoder audio changed since evaluation preparation")
            durations = {float(song.base.metadata["duration"]) for song in evaluation_dataset.songs
                         if any(source.get("role") == "audio" and
                                str(Path(source["path"]).resolve()) == selected_audio
                                for source in song.base.metadata.get("sources", []))}
            if len(durations) != 1:
                raise ValueError("Curated encoder audio needs one canonical decoded song duration")
            model = models.keras.models.load_model(checkpoint, compile=False)
            predict = lambda patches: model(patches, training=False).numpy()
            encoder = FixedEncoder(decode_encoder_audio(args.audio), predict, file_hash(checkpoint),
                                   args.encoder_grid_seconds, args.batch_size, duration=durations.pop())
            autoencoder = None
            if args.autoencoder_artifact:
                reconstruction = models.keras.models.load_model(args.autoencoder_artifact, compile=False)
                autoencoder = lambda patches: reconstruction(patches, training=False).numpy()
            report = evaluate_encoder(encoder, encoder.grid, reference_predict=predict, autoencoder=autoencoder, status="trained")
            report.update(audio_sha256=file_hash(args.audio), checkpoint_sha256=file_hash(checkpoint), sources=curated,
                          **preparation_reports(evaluation_dataset))
        else:
            from training.pipeline_v2.runner import load_stage_artifact
            dataset = load_dataset(args.dataset, lazy=True)
            model, manifest = load_stage_artifact(args.artifact, args.stage, dataset)
            predictor = lambda inputs: {key: value.numpy() for key, value in model(inputs, training=False).items()}
            report = evaluate_stage(evaluation_dataset, args.stage, predictor, split=args.split, batch_size=args.batch_size,
                                    threshold=args.threshold, artifact_metadata=manifest, baseline_dataset=dataset,
                                    evaluation_root=args.evaluation_root)
    destination = new_run_directory(args.output)
    write_json(destination / "report.json", report)
    print(f"Independent {args.stage} report saved in {destination}")
    return report


if __name__ == "__main__":
    main()
