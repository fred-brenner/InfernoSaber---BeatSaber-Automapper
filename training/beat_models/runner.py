"""Explicit stage commands. TensorFlow is imported only when a model is needed.

Smoke commands differentiate losses without applying gradients. Only an explicit
non-dry-run train command constructs an optimizer and calls fit.
"""

from pathlib import Path

import numpy as np

from .artifacts import (execution_identity, file_hash, load_model_manifest, new_run_directory, validate_compatibility,
                        write_json, write_model_manifest)
from .contracts import (CORE, FEATURES, FRAME_SECONDS, HISTORY, SCHEMA_VERSION, SEED, STAGES, frame_conditions,
                        intensity_inputs, note_inputs, timing_inputs)
from .features import normalize_songs
from .metrics import summarize, timing_scores
from .pipeline import generate
from .settings import architecture_settings, training_settings
from .training_metrics import (INTENSITY_METRIC_DEFINITIONS, add_intensity_metrics,
                              intensity_metrics_callback, intensity_reference)

CONDITIONING = ("reference", "noisy-reference", "predicted")


def _report_context(manifest):
    return {"schema_version": SCHEMA_VERSION, "representation": manifest["representation"],
            "normalization_sha256": manifest["normalization_sha256"], "seed": SEED,
            "sources": manifest.get("sources", []), "splits": manifest.get("splits", {}),
            **execution_identity()}


class Predictor:
    """Use eager model calls during event decoding, avoiding per-event datasets."""

    def __init__(self, model):
        self.model = model

    def predict(self, inputs, verbose=0):
        result = self.model(inputs, training=False)
        if isinstance(result, dict):
            return {key: value.numpy() for key, value in result.items()}
        return result.numpy()


def _normalized(songs, manifest):
    validate_compatibility(manifest)
    songs = list(songs)
    if not songs:
        raise ValueError("A model command needs at least one song")
    normalized = [song.metadata.get("normalization_version") is not None for song in songs]
    if any(normalized):
        if not all(normalized):
            raise ValueError("Mixed raw and normalized songs")
        for song in songs:
            if (song.metadata.get("normalization_version") != manifest["normalization"]["version"]
                    or song.metadata.get("normalization_sha256") != manifest["normalization_sha256"]):
                raise ValueError("Song normalization does not match dataset statistics")
        return songs
    return normalize_songs(songs, manifest["normalization"])


def _load_model(path, stage, manifest, require_trained=False):
    from . import models

    checkpoint, metadata = load_model_manifest(path, stage, manifest)
    if require_trained and metadata.get("training", {}).get("status") != "trained":
        raise ValueError("Predicted training conditioning requires a trained intensity checkpoint")
    return models.keras.models.load_model(checkpoint, compile=False), metadata


def _profiles(songs, conditioning="reference", intensity_model=None, rng=None, noise_std=None):
    if conditioning not in CONDITIONING:
        raise ValueError("Unknown intensity conditioning mode")
    if conditioning == "predicted" and intensity_model is None:
        raise ValueError("Predicted conditioning requires an explicit intensity artifact")
    if conditioning == "noisy-reference" and noise_std is None:
        noise_std = training_settings("timing")["intensity_noise_std"]
    rng = np.random.default_rng(SEED) if rng is None else rng
    profiles = {}
    for song in songs:
        shares = song.arrays["intensity"].astype(np.float32).copy()
        if conditioning == "predicted":
            shares = np.asarray(intensity_model(intensity_inputs(song), training=False))[0]
        elif not song.metadata.get("intensity_target_valid", bool(shares.sum())):
            shares = song.arrays["bin_durations"] / song.metadata["duration"]
        if conditioning == "noisy-reference":
            shares = shares * np.exp(rng.normal(0, noise_std, len(shares)))
        if not np.isfinite(shares).all() or np.any(shares < 0) or shares.sum() <= 0:
            raise ValueError("Invalid conditioning profile")
        shares = (shares / shares.sum()).astype(np.float32)
        density = np.count_nonzero(song.arrays["note_classes"]) / song.metadata["duration"]
        profiles[song.song_id] = frame_conditions(song, shares, density)
    return profiles


def _references(songs, stage, max_events=None, bounded=False):
    refs = []
    for song in songs:
        if stage == "intensity":
            if song.metadata.get("intensity_target_valid", bool(song.arrays["intensity"].sum())):
                refs.append((song, None))
        elif stage == "timing":
            starts = list(range(0, len(song.arrays["frame_times"]), CORE))
            if bounded:
                starts = sorted({starts[0], starts[-1]})
            refs.extend((song, start) for start in starts)
        else:
            length = len(song.arrays["event_frames"])
            limit = length if max_events is None else min(length, max_events)
            indices = np.unique(np.linspace(0, length - 1, limit, dtype=int)) if limit else []
            refs.extend((song, int(index)) for index in indices)
    return refs


def _sample(stage, song, index, profiles):
    if stage == "intensity":
        return intensity_inputs(song), song.arrays["intensity"][None], np.ones(1, dtype=bool)
    conditions = profiles[song.song_id]
    if stage == "timing":
        return timing_inputs(song, conditions, index)
    frames = song.arrays["event_frames"]
    frame = int(frames[index])
    classes = song.arrays["note_classes"]
    count = int(np.count_nonzero(classes[index]))
    # Frame timestamps keep teacher history causal under nearest-frame rounding.
    beginning = max(0, index - HISTORY)
    history_times = song.arrays["frame_times"][frames[beginning:index]]
    inputs = note_inputs(song, frame, count, conditions, history_times, classes[beginning:index])
    return inputs, classes[index:index + 1], np.ones(1, dtype=bool)


def _batch(stage, references, profiles):
    samples = [_sample(stage, song, index, profiles) for song, index in references]
    if stage == "intensity":
        width = max(len(sample[1][0]) for sample in samples)
        inputs = {"audio": np.zeros((len(samples), width, FEATURES), dtype=np.float32),
                  "durations": np.zeros((len(samples), width), dtype=np.float32),
                  "mask": np.zeros((len(samples), width), dtype=bool)}
        target = np.zeros((len(samples), width), dtype=np.float32)
        for index, (item, labels, _) in enumerate(samples):
            size = labels.shape[1]
            for key in inputs:
                inputs[key][index, :size] = item[key][0]
            target[index, :size] = labels[0]
        return inputs, target, np.ones(len(samples), dtype=bool)
    inputs = {key: np.concatenate([sample[0][key] for sample in samples]) for key in samples[0][0]}
    return inputs, np.concatenate([sample[1] for sample in samples]), np.concatenate([sample[2] for sample in samples])


def _loss(stage, target, predicted, mask, inputs):
    from . import models

    if stage == "intensity":
        return models.intensity_loss(target, predicted, inputs["mask"], mask)
    if stage == "timing":
        return models.timing_loss(target, predicted, mask)
    return models.notes_loss(target, predicted, mask)


def _numpy_outputs(value):
    return {key: np.asarray(item) for key, item in value.items()} if isinstance(value, dict) else {"output": np.asarray(value)}


def _hand_scores(target, predicted):
    """Independent head accuracy at reference events; absent hands stay valid."""
    target = np.asarray(target, dtype=np.int32).reshape(-1, 2)
    predicted = np.asarray(predicted, dtype=np.int32).reshape(-1, 2)
    result = {}
    for hand, name in enumerate(("red", "blue")):
        truth, output = target[:, hand], predicted[:, hand]
        both = (truth > 0) & (output > 0)
        result[name] = {"class_accuracy": float(np.mean(truth == output)) if len(truth) else None,
                        "presence_accuracy": float(np.mean((truth > 0) == (output > 0))) if len(truth) else None,
                        "position_accuracy": float(np.mean((truth[both] - 1) // 9 == (output[both] - 1) // 9)) if both.any() else None,
                        "direction_accuracy": float(np.mean((truth[both] - 1) % 9 == (output[both] - 1) % 9)) if both.any() else None,
                        "compared_colored_notes": int(both.sum()), "evaluated_events": len(truth)}
    return result


def _intensity_scores(song, shares):
    count = np.count_nonzero(song.arrays["note_classes"])
    error = np.asarray(shares) - song.arrays["intensity"]
    return {"song_id": song.song_id, "split": song.metadata.get("split"),
            "reference_colored_notes": int(count), "share_mae": float(np.abs(error).mean()),
            "share_total_variation": float(np.abs(error).sum() / 2),
            "bin_count_mae": float(np.abs(error * count).mean()),
            "bin_count_absolute_error_sum": float(np.abs(error * count).sum()),
            "total_count_error": float(error.sum() * count)}


def _timing_stage_scores(song, counts, expected, valid):
    arrays = song.arrays
    frames = np.flatnonzero((counts > 0) & valid)
    reference = arrays["event_times"][valid[arrays["event_frames"]]]
    reference_count = int(arrays["timing"][valid].sum())
    predicted_count = int(counts[valid].sum())
    expected_count = float(expected[valid].sum())
    duration = float(np.minimum(FRAME_SECONDS, song.metadata["duration"] - arrays["frame_times"][valid]).sum())
    return {"song_id": song.song_id, "split": song.metadata.get("split"),
            "valid_frames": int(valid.sum()), "evaluated_duration_seconds": duration,
            "reference_unique_onsets": len(reference), "predicted_unique_onsets": len(frames),
            "reference_colored_notes": reference_count, "predicted_colored_notes": predicted_count,
            "expected_colored_notes": expected_count, "colored_note_count_error": predicted_count - reference_count,
            "absolute_colored_note_count_error": abs(predicted_count - reference_count),
            "reference_nps": reference_count / duration, "predicted_nps": predicted_count / duration,
            "density_error_nps": (predicted_count - reference_count) / duration,
            "expected_density_error_nps": (expected_count - reference_count) / duration,
            "frame_count_accuracy": float(np.mean(counts[valid] == arrays["timing"][valid])),
            "timing": timing_scores(reference, arrays["frame_times"][frames])}


def _timing_stage_summary(reports):
    reference = sum(item["reference_unique_onsets"] for item in reports)
    predicted = sum(item["predicted_unique_onsets"] for item in reports)
    matches = sum(item["timing"]["matched_onsets"] for item in reports)
    precision = matches / predicted if predicted else float(not reference)
    recall = matches / reference if reference else float(not predicted)
    duration = sum(item["evaluated_duration_seconds"] for item in reports)
    count_error = sum(item["colored_note_count_error"] for item in reports)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "colored_note_count_error": count_error,
            "absolute_colored_note_count_error": sum(item["absolute_colored_note_count_error"] for item in reports),
            "density_error_nps": count_error / duration,
            "valid_frames": sum(item["valid_frames"] for item in reports)}


def _gradient_check(model, stage, batch):
    from . import models

    inputs, target, mask = batch
    with models.tf.GradientTape() as tape:
        predicted = model(inputs, training=False)
        loss = _loss(stage, target, predicted, mask, inputs)
    gradients = tape.gradient(loss, model.trainable_variables)
    if not np.isfinite(float(loss)):
        raise ValueError(f"Nonfinite {stage} loss")
    for variable, gradient in zip(model.trainable_variables, gradients):
        if gradient is None:
            raise ValueError(f"Disconnected {stage} variable: {variable.name}")
        values = gradient.values if isinstance(gradient, models.tf.IndexedSlices) else gradient
        if not bool(models.tf.reduce_all(models.tf.math.is_finite(values))):
            raise ValueError(f"Nonfinite {stage} gradient: {variable.name}")
    return {"loss": float(loss), "gradient_variables": len(gradients), "finite_gradients": True}


def smoke_models(songs, manifest, output, max_events=8):
    """Forward/loss/gradient/save checks on bounded real batches, with zero updates."""
    from . import models

    if max_events < 1 or max_events > 8:
        raise ValueError("Smoke note batches require between one and eight events per song")
    songs = _normalized(songs, manifest)
    root = new_run_directory(output)
    profiles = _profiles(songs)
    report = {"kind": "model_smoke", "status": "untrained", "optimizer_steps": 0,
              "device": "CPU", "songs": len(songs), "max_note_events_per_song": max_events,
              **_report_context(manifest),
              "stages": {}}
    with models.tf.device("/CPU:0"):
        for stage in STAGES:
            settings = training_settings(stage)
            batch_size = settings["batch_size"]
            models.keras.utils.set_random_seed(settings["seed"])
            configuration = {"batch_size": batch_size, "seed": settings["seed"],
                             "architecture": architecture_settings(stage), "conditioning": "reference"}
            model = models.build_model(stage)
            before = [value.copy() for value in model.get_weights()]
            refs = _references(songs, stage, max_events, bounded=True)
            if not refs:
                raise ValueError(f"No valid {stage} smoke targets")
            checks = []
            first = None
            for offset in range(0, len(refs), batch_size):
                batch = _batch(stage, refs[offset:offset + batch_size], profiles)
                first = batch if first is None else first
                checks.append(_gradient_check(model, stage, batch))
            if not all(np.array_equal(old, new) for old, new in zip(before, model.get_weights())):
                raise ValueError("Smoke gradient calculation changed model weights")
            stage_dir = new_run_directory(root / stage)
            checkpoint = stage_dir / "model.keras"
            model.save(checkpoint)
            write_model_manifest(stage_dir, stage, checkpoint, manifest,
                                 {"status": "untrained", "optimizer_steps": 0, "purpose": "smoke", **configuration})
            restored, _ = _load_model(stage_dir, stage, manifest)
            expected = _numpy_outputs(model(first[0], training=False))
            actual = _numpy_outputs(restored(first[0], training=False))
            max_error = max(float(np.max(np.abs(expected[key] - actual[key]))) for key in expected)
            if max_error > 1e-6 or not all(np.array_equal(old, new) for old, new in zip(before, restored.get_weights())):
                raise ValueError(f"{stage} checkpoint round trip changed predictions or weights")
            report["stages"][stage] = {"parameters": model.count_params(), "batches": len(checks),
                                       **configuration,
                                       "examples": len(refs), "checks": checks,
                                       "input_shapes": {key: list(value.shape) for key, value in first[0].items()},
                                       "output_shapes": {key: list(value.shape) for key, value in expected.items()},
                                       "weights_unchanged": True, "roundtrip_max_error": max_error,
                                       "artifact": str(stage_dir)}
    write_json(root / "report.json", report)
    return report


def train_stage(songs, manifest, stage, run_dir, dry_run=False, conditioning=None,
                intensity_artifact=None, epochs=None, batch_size=None):
    """Future explicit training entrypoint; dry runs never create an optimizer."""
    from . import models

    if stage not in STAGES:
        raise ValueError("Unknown model stage")
    settings = training_settings(stage)
    conditioning = settings["conditioning"] if conditioning is None else conditioning
    epochs = settings["epochs"] if epochs is None else epochs
    batch_size = settings["batch_size"] if batch_size is None else batch_size
    if conditioning not in CONDITIONING:
        raise ValueError("Unknown stage or conditioning mode")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Epoch count must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    songs = _normalized(songs, manifest)
    train = [song for song in songs if song.metadata.get("split") == "train"]
    validation = [song for song in songs if song.metadata.get("split") == "validation"]
    if not train or not validation:
        raise ValueError("Training requires separate nonempty train and validation splits")
    upstream = None
    if stage != "intensity" and conditioning == "predicted":
        if intensity_artifact is None:
            raise ValueError("Predicted conditioning requires --intensity-artifact")
        upstream, _ = _load_model(intensity_artifact, "intensity", manifest, require_trained=True)
    train_refs, validation_refs = _references(train, stage), _references(validation, stage)
    if not train_refs or not validation_refs:
        raise ValueError("Training and validation need valid stage targets")
    root = new_run_directory(run_dir)
    models.keras.utils.set_random_seed(settings["seed"])
    model = models.build_model(stage)
    config = {"stage": stage, "status": "untrained" if dry_run else "trained", "dry_run": dry_run,
              **_report_context(manifest),
              **settings, "epochs": epochs, "batch_size": batch_size,
              "architecture": architecture_settings(stage),
              "conditioning": conditioning if stage != "intensity" else None,
              "validation_conditioning": "predicted" if stage != "intensity" and conditioning == "predicted" else "reference",
              "intensity_artifact": str(intensity_artifact) if intensity_artifact else None,
              "train_examples": len(train_refs), "validation_examples": len(validation_refs),
              "parameters": model.count_params()}
    if stage == "intensity":
        config["intensity_references"] = {
            split: intensity_reference(selected, models.keras.backend.epsilon())
            for split, selected in (("train", train), ("validation", validation))}
        config["metric_definitions"] = INTENSITY_METRIC_DEFINITIONS
    preview_mode = conditioning if stage != "intensity" else "reference"
    preview = _batch(stage, train_refs[:batch_size],
                     _profiles(train, preview_mode, upstream, np.random.default_rng(settings["seed"]),
                               noise_std=settings["intensity_noise_std"]))
    config["input_shapes"] = {key: list(value.shape) for key, value in preview[0].items()}
    if dry_run:
        config["optimizer_steps"] = 0
        config["preview_loss"] = float(_loss(stage, preview[1], model(preview[0], training=False), preview[2], preview[0]))
        if stage == "intensity":
            # A preview covers one batch, so its entropy/reference must match that batch.
            reference = intensity_reference([song for song, _ in train_refs[:batch_size]], models.keras.backend.epsilon())
            config["preview_metrics"] = {"loss": config["preview_loss"]}
            add_intensity_metrics(config["preview_metrics"], {"train": reference})
        write_json(root / "report.json", config)
        return config

    class StageSequence(models.keras.utils.Sequence):
        """Retain indices; allocate only one batch of audio patches at a time."""

        def __init__(self, selected, references, mode, shuffle):
            self.songs, self.references, self.mode = selected, references, mode
            self.shuffle = shuffle
            self.rng = np.random.default_rng(settings["seed"])
            self.order = np.arange(len(references))
            self.on_epoch_end()

        def __len__(self):
            return (len(self.order) + batch_size - 1) // batch_size

        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.order)
            self.profiles = _profiles(self.songs, self.mode, upstream, self.rng,
                                      noise_std=settings["intensity_noise_std"])

        def __getitem__(self, index):
            chosen = self.order[index * batch_size:(index + 1) * batch_size]
            inputs, target, mask = _batch(stage, [self.references[item] for item in chosen], self.profiles)
            if stage == "timing":
                target = np.stack((target, mask.astype(np.int32)), axis=-1)
            elif stage == "notes":
                target = {"red": target[:, 0], "blue": target[:, 1]}
            return inputs, target

    optimizer = models.keras.optimizers.get({"class_name": settings["optimizer"],
                                            "config": {"learning_rate": settings["learning_rate"],
                                                       "clipnorm": settings["clipnorm"]}})
    if stage == "notes":
        loss = {hand: models.keras.losses.SparseCategoricalCrossentropy() for hand in ("red", "blue")}
        model.compile(optimizer=optimizer, loss=loss,
                      loss_weights={hand: config["architecture"][f"{hand}_loss_weight"] for hand in ("red", "blue")})
    else:
        model.compile(optimizer=optimizer, loss=models.intensity_objective if stage == "intensity" else models.timing_objective)
    checkpoint = root / "best.keras"
    callbacks = [models.keras.callbacks.EarlyStopping(monitor=settings["monitor"],
                                                      mode=settings["monitor_mode"],
                                                      patience=settings["early_stopping_patience"],
                                                      min_delta=settings["early_stopping_min_delta"],
                                                      restore_best_weights=True),
                 models.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor=settings["monitor"],
                                                        mode=settings["monitor_mode"], save_best_only=True)]
    if stage == "intensity":
        callbacks.insert(0, intensity_metrics_callback(config["intensity_references"]))
    train_mode = conditioning if stage != "intensity" else "reference"
    valid_mode = "predicted" if train_mode == "predicted" else "reference"
    history = model.fit(StageSequence(train, train_refs, train_mode, settings["shuffle"]),
                        validation_data=StageSequence(validation, validation_refs, valid_mode, False),
                        epochs=epochs, callbacks=callbacks, shuffle=False, verbose=settings["verbose"])
    config["optimizer_steps"] = int(optimizer.iterations.numpy())
    config["epochs_completed"] = len(history.history["loss"])
    config["best_validation_loss"] = float(min(history.history["val_loss"]))
    config["history"] = history.history
    write_model_manifest(root, stage, checkpoint, manifest, config)
    write_json(root / "report.json", config)
    return config


def evaluate_models(songs, manifest, artifacts, output, requested_nps=None, max_events=None):
    """Report reference-conditioned losses and separate composed predictions."""
    from . import models

    if set(artifacts) != set(STAGES):
        raise ValueError("Evaluation requires explicit artifacts for all three stages")
    songs = _normalized(songs, manifest)
    loaded, checkpoint_metadata = {}, {}
    for stage in STAGES:
        loaded[stage], checkpoint_metadata[stage] = _load_model(artifacts[stage], stage, manifest)
    root = new_run_directory(output)
    profiles = _profiles(songs)
    teacher_forced = {}
    timing_loss_totals = np.zeros(4, dtype=np.float64)
    intensity_reports = []
    timing_outputs = {song.song_id: (np.zeros(len(song.arrays["frame_times"]), dtype=np.int32),
                                     np.zeros(len(song.arrays["frame_times"]), dtype=np.float32),
                                     np.zeros(len(song.arrays["frame_times"]), dtype=bool)) for song in songs}
    note_outputs = {song.song_id: ([], []) for song in songs}
    for stage in STAGES:
        batch_size = training_settings(stage)["batch_size"]
        refs = _references(songs, stage, max_events)
        losses, weights = [], []
        for offset in range(0, len(refs), batch_size):
            selected = refs[offset:offset + batch_size]
            batch = _batch(stage, selected, profiles)
            inputs, target, mask = batch
            predicted = loaded[stage](inputs, training=False)
            losses.append(float(_loss(stage, target, predicted, mask, inputs)))
            weights.append(len(target))
            if stage == "intensity":
                for row, (song, _) in enumerate(selected):
                    intensity_reports.append(_intensity_scores(song, np.asarray(predicted)[row, :len(song.arrays["intensity"])]))
            elif stage == "timing":
                timing_loss_totals += np.asarray([float(value) for value in models.timing_loss_components(target, predicted, mask)])
                probabilities = np.asarray(predicted)
                for row, (song, start) in enumerate(selected):
                    local = np.flatnonzero(mask[row])
                    frames = start + local
                    counts, expected, valid = timing_outputs[song.song_id]
                    counts[frames] = probabilities[row, local].argmax(axis=-1)
                    expected[frames] = probabilities[row, local, 1] + 2 * probabilities[row, local, 2]
                    valid[frames] = True
            else:
                classes = np.column_stack([np.asarray(predicted[hand]).argmax(axis=-1) for hand in ("red", "blue")])
                for row, (song, _) in enumerate(selected):
                    note_outputs[song.song_id][0].append(target[row])
                    note_outputs[song.song_id][1].append(classes[row])
        teacher_forced[stage] = {"loss": float(np.average(losses, weights=weights)) if losses else None,
                                  "batch_size": batch_size, "configured_architecture": architecture_settings(stage),
                                  "examples": len(refs), "conditioning": "reference",
                                  "history": "reference" if stage == "notes" else None}
    teacher_forced["intensity"].update({"songs": intensity_reports,
                                      "empty_targets_skipped": len(songs) - len(intensity_reports),
                                      "mean_share_mae": float(np.mean([item["share_mae"] for item in intensity_reports])) if intensity_reports else None,
                                      "mean_bin_count_mae": float(np.mean([item["bin_count_mae"] for item in intensity_reports])) if intensity_reports else None})
    timing_reports = [_timing_stage_scores(song, *timing_outputs[song.song_id]) for song in songs]
    ce_sum, frames, huber_sum, chunks = timing_loss_totals
    ce, huber = ce_sum / frames if frames else 0, huber_sum / chunks if chunks else 0
    count_loss_weight = teacher_forced["timing"]["configured_architecture"]["count_loss_weight"]
    teacher_forced["timing"].update({"loss": float(ce + count_loss_weight * huber), "cross_entropy": float(ce),
                                    "normalized_count_huber": float(huber),
                                    "songs": timing_reports, "summary": _timing_stage_summary(timing_reports)})
    all_targets = [item for truth, _ in note_outputs.values() for item in truth]
    all_predicted = [item for _, predicted in note_outputs.values() for item in predicted]
    teacher_forced["notes"].update({"hands": _hand_scores(all_targets, all_predicted),
                                  "decoding": "independent_head_argmax", "max_events_per_song": max_events,
                                  "event_sampling": "evenly_spaced_reference_events",
                                  "partial_diagnostic": any(len(note_outputs[song.song_id][0]) < len(song.arrays["event_frames"]) for song in songs),
                                  "songs": [{"song_id": song.song_id, "split": song.metadata.get("split"),
                                             "reference_events": len(song.arrays["event_frames"]),
                                             "evaluated_events": len(note_outputs[song.song_id][0]),
                                             "hands": _hand_scores(*note_outputs[song.song_id])} for song in songs]})
    reports, predictions = [], []
    predictors = {stage: Predictor(model) for stage, model in loaded.items()}
    for index, song in enumerate(songs):
        density = np.count_nonzero(song.arrays["note_classes"]) / song.metadata["duration"]
        density = density if requested_nps is None else requested_nps
        generated = generate(song, density, predictors, max_events=max_events)
        arrays = {key: getattr(generated, key) for key in ("intensity", "local_nps", "frame_probabilities",
                                                         "counts", "event_frames", "event_times", "note_classes")}
        filename = f"prediction_{index:05d}.npz"
        np.savez_compressed(root / filename, **arrays)
        predictions.append({"song_id": song.song_id, "file": filename, "sha256": file_hash(root / filename)})
        reports.append(generated.metrics)
    statuses = {stage: checkpoint_metadata[stage].get("status", checkpoint_metadata[stage].get("training", {}).get("status", "unknown"))
                for stage in STAGES}
    trained = all(status == "trained" for status in statuses.values())
    interpretation = ("Trained checkpoints evaluated with separate reference-conditioned and composed predictions."
                      if trained else "Includes untrained or unknown-status checkpoints; outputs verify execution and contracts, not mapping quality.")
    report = {"kind": "model_evaluation", "status": "trained" if trained else "untrained_or_mixed", **_report_context(manifest),
              "evaluated_split": manifest.get("evaluation_split", "all"), "predictions": predictions,
              "songs": [{"song_id": song.song_id, "split": song.metadata.get("split")}
                                                       for song in songs],
              "teacher_forced": teacher_forced,
              "composed": {"conditioning": "predicted", "history": "autoregressive",
                           "max_events": max_events, "songs": reports, "summary": summarize(reports)},
              "partial_diagnostic": any(item["partial_diagnostic"] for item in reports) or teacher_forced["notes"]["partial_diagnostic"],
              "interpretation": interpretation,
              "checkpoints": {stage: {"path": str(artifacts[stage]),
                                      "status": statuses[stage],
                                      "architecture": checkpoint_metadata[stage].get("training", {}).get("architecture")}
                              for stage in STAGES}}
    write_json(root / "report.json", report)
    return report
