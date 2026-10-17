"""Explicit v2 stage commands; importing this module never imports TensorFlow.

Dry runs and smoke checks make no optimizer. Only an explicitly requested
``train_stage(..., dry_run=False)`` may update weights; each stage is trained
independently against its own strictly preceding reference history.
"""

import json
import zipfile

import numpy as np

from training.beat_models.artifacts import new_run_directory, write_json

from .artifacts import execution_identity

from .contracts import NEW_STAGES, StageExamples, validate_examples
from .settings import architecture_settings, training_settings


def _check_stage(stage):
    if stage not in NEW_STAGES:
        raise ValueError(f"Unknown v2 stage: {stage}")


def _slice(examples, indices):
    return StageExamples(*({key: value[indices] for key, value in family.items()}
                           for family in (examples.inputs, examples.targets, examples.weights)),
                         examples.times[indices])


def _merge(examples):
    return StageExamples(*({key: np.concatenate([getattr(item, name)[key] for item in examples])
                           for key in getattr(examples[0], name)}
                          for name in ("inputs", "targets", "weights")),
                         np.concatenate([item.times for item in examples]))


def _selected_songs(dataset, split):
    songs = dataset.songs
    records = getattr(songs, "records", None)
    if split is not None and records is not None:
        # Immutable lazy datasets expose the split index without reading NPZs.
        for index, record in enumerate(records):
            if record["split"] == split:
                yield songs[index]
        return
    for song in songs:
        if split is None or song.base.metadata.get("split") == split:
            yield song


def batches(dataset, stage, split=None, batch_size=128, shuffle=False, rng=None):
    """Stream one stage at a time, retaining at most one song and one batch.

    A lazy, reiterable ``dataset.songs`` is supported. Shuffle permutes examples
    within each song; complete recording groups keep their prepared split.
    """
    _check_stage(stage)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    rng = np.random.default_rng(3) if rng is None else rng
    pending, size = [], 0
    for song in _selected_songs(dataset, split):
        examples = song.examples.get(stage)
        if examples is None:
            continue
        validate_examples(examples, stage)
        order = np.arange(len(examples.times))
        if shuffle:
            rng.shuffle(order)
        offset = 0
        while offset < len(order):
            count = min(batch_size - size, len(order) - offset)
            pending.append(_slice(examples, order[offset:offset + count]))
            size += count
            offset += count
            if size == batch_size:
                yield _merge(pending)
                pending, size = [], 0
    if pending:
        yield _merge(pending)


def _split_counts(dataset, stage):
    result = {split: {"songs": 0, "examples": 0, "supervised_values": 0}
              for split in ("train", "validation", "test")}
    groups = {}
    for song in dataset.songs:
        split = song.base.metadata.get("split")
        group = song.base.metadata.get("group_id")
        if split not in result or not group:
            raise ValueError("V2 training requires recording groups and explicit dataset splits")
        if group in groups and groups[group] != split:
            raise ValueError("Recording group leaks across v2 training splits")
        groups[group] = split
        examples = song.examples.get(stage)
        if examples is not None:
            validate_examples(examples, stage)
            result[split]["songs"] += 1
            result[split]["examples"] += len(examples.times)
            result[split]["supervised_values"] += sum(int(np.count_nonzero(value)) for value in examples.weights.values())
    return result


def _outputs(value):
    return {key: np.asarray(item) for key, item in value.items()}


def gradient_check(stage, model, examples):
    """Differentiate a masked objective without constructing/applying an optimizer."""
    from . import models

    validate_examples(examples, stage)
    before = model.get_weights()
    with models.tf.GradientTape() as tape:
        predicted = model(examples.inputs, training=False)
        loss = models.stage_loss(stage, examples.targets, predicted, examples.weights)
    gradients = tape.gradient(loss, model.trainable_variables)
    if not np.isfinite(float(loss)):
        raise ValueError(f"Nonfinite {stage} loss")
    active = 0
    for variable, gradient in zip(model.trainable_variables, gradients):
        if gradient is None:
            raise ValueError(f"Disconnected {stage} variable: {variable.name}")
        value = gradient.values if isinstance(gradient, models.tf.IndexedSlices) else gradient
        if not bool(models.tf.reduce_all(models.tf.math.is_finite(value))):
            raise ValueError(f"Nonfinite {stage} gradient: {variable.name}")
        active += int(bool(models.tf.reduce_any(value != 0)))
    if any(np.any(value > 0) for value in examples.weights.values()) and not active:
        raise ValueError(f"No active {stage} gradients")
    unchanged = all(np.array_equal(old, new) for old, new in zip(before, model.get_weights()))
    if not unchanged:
        raise ValueError("Gradient inspection changed model weights")
    return {"loss": float(loss), "gradient_variables": len(gradients),
            "active_gradient_variables": active, "finite_gradients": True, "weights_unchanged": True}


def _checkpoint_signature(document):
    from evaluation.beat_pair import _graph_signature

    def stable(value):
        if isinstance(value, dict):
            # Keras saves RNN initializer object identities after the first call.
            # These process-local addresses describe sharing, not model behavior.
            return {key: stable(item) for key, item in value.items() if key != "shared_object_id"}
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    return _graph_signature(stable(document))


def load_stage_artifact(path, stage, dataset=None):
    """Load only an explicitly selected stage/encoder/vocabulary-compatible artifact."""
    from . import models
    from .artifacts import load_checkpoint_manifest

    root, manifest = load_checkpoint_manifest(path, stage, dataset)
    checkpoint = root / manifest["checkpoint"]
    # The graph check is shared with the retained beat-pair CPU adapter. Rebuild
    # only an identical graph before restoring weights across tf.keras versions.
    with zipfile.ZipFile(checkpoint) as archive:
        saved = json.loads(archive.read("config.json"))
    model = models.build_model(stage, manifest["vocabularies"])
    if _checkpoint_signature(saved) != _checkpoint_signature(json.loads(model.to_json())):
        raise ValueError(f"Unsupported {stage} checkpoint graph; the current model does not match")
    model.load_weights(checkpoint)
    return model, manifest


def predict_stage(path, stage, inputs, dataset=None):
    model, _ = load_stage_artifact(path, stage, dataset)
    return _outputs(model(inputs, training=False))


def smoke_models(dataset, output, stages=None, max_examples=8):
    """Bounded CPU forward/gradient/serialization checks, with zero weight updates."""
    from . import models
    from .artifacts import compatibility, save_checkpoint_manifest

    stages = tuple(NEW_STAGES if stages is None else stages)
    if not stages or len(set(stages)) != len(stages):
        raise ValueError("Smoke checks require distinct explicitly selected stages")
    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or not 1 <= max_examples <= 8:
        raise ValueError("Smoke checks require between one and eight examples")
    for stage in stages:
        _check_stage(stage)
        compatibility(dataset, stage)
    root = new_run_directory(output)
    report = {"kind": "v2_model_smoke", "status": "untrained", "optimizer_steps": 0,
              "device": "CPU", "stages": {}, **execution_identity()}
    with models.tf.device("/CPU:0"):
        for stage in stages:
            examples = next(batches(dataset, stage, batch_size=max_examples), None)
            if examples is None:
                raise ValueError(f"No {stage} smoke examples")
            models.keras.utils.set_random_seed(training_settings(stage)["seed"])
            model = models.build_model(stage, dataset.vocabularies)
            before = model.get_weights()
            check = gradient_check(stage, model, examples)
            expected = _outputs(model(examples.inputs, training=False))
            if not all(np.isfinite(value).all() for value in expected.values()):
                raise ValueError(f"Nonfinite {stage} prediction")
            stage_dir = new_run_directory(root / stage)
            model.save(stage_dir / "model.keras")
            save_checkpoint_manifest(stage_dir, stage, dataset, "untrained", checkpoint="model.keras",
                                     extra={"purpose": "smoke", "optimizer_steps": 0})
            restored, _ = load_stage_artifact(stage_dir, stage, dataset)
            actual = _outputs(restored(examples.inputs, training=False))
            error = max(float(np.max(np.abs(expected[key] - actual[key]))) for key in expected)
            if error > 1e-6 or not all(np.array_equal(old, new) for old, new in zip(before, restored.get_weights())):
                raise ValueError(f"{stage} checkpoint round trip changed predictions or weights")
            report["stages"][stage] = {**check, "parameters": model.count_params(),
                                       "examples": len(examples.times), "architecture": architecture_settings(stage),
                                       "input_shapes": {key: list(value.shape) for key, value in examples.inputs.items()},
                                       "output_shapes": {key: list(value.shape) for key, value in expected.items()},
                                       "roundtrip_max_error": error, "artifact": str(stage_dir)}
    write_json(root / "report.json", report)
    return report


def _epoch(dataset, stage, model, batch_size, optimizer=None, shuffle=False, rng=None):
    from . import models

    totals = {}
    for examples in batches(dataset, stage, "train" if optimizer is not None else "validation",
                            batch_size, shuffle, rng):
        if optimizer is None:
            predicted = model(examples.inputs, training=False)
            components = models.loss_components(stage, examples.targets, predicted, examples.weights)
        else:
            with models.tf.GradientTape() as tape:
                predicted = model(examples.inputs, training=True)
                components = models.loss_components(stage, examples.targets, predicted, examples.weights)
                loss = models.combine_loss_components(stage, components)
            if not np.isfinite(float(loss)):
                raise ValueError(f"Nonfinite {stage} training loss")
            gradients = tape.gradient(loss, model.trainable_variables)
            for variable, gradient in zip(model.trainable_variables, gradients):
                if gradient is None:
                    raise ValueError(f"Disconnected {stage} training variable: {variable.name}")
                value = gradient.values if isinstance(gradient, models.tf.IndexedSlices) else gradient
                models.tf.debugging.assert_all_finite(value, f"Nonfinite gradient for {variable.name}")
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for key, pair in components.items():
            totals.setdefault(key, np.zeros(2, np.float64))
            totals[key] += [float(value) for value in pair]
    result = float(models.combine_loss_components(stage, totals))
    if not totals or not np.isfinite(result):
        raise ValueError(f"No finite {stage} epoch result")
    return result


def train_stage(dataset, stage, output, dry_run=True, epochs=None, batch_size=None):
    """Train one selected stage only; the default dry run never creates an optimizer.

    Prepared examples carry fixed upstream references and teacher-forced history.
    This entrypoint does not retrain intensity, timing, or the retained encoder.
    """
    from . import models
    from .artifacts import compatibility, save_checkpoint_manifest

    _check_stage(stage)
    identity = compatibility(dataset, stage)
    if not dry_run:
        encoder = dataset.manifest.get("encoder_identity", {})
        if (dataset.manifest.get("status") != "prepared" or encoder.get("status") != "retained"
                or not encoder.get("checkpoint_fingerprint")):
            raise ValueError("Training requires real prepared examples from the explicit retained encoder")
        for song in dataset.songs:
            adapter = str(song.base.metadata.get("provenance", {}).get("adapter", ""))
            if "dummy" in adapter.lower() or adapter == "v2_synthetic_fixture":
                raise ValueError("Dummy examples cannot be used for training")
    settings = training_settings(stage)
    epochs = settings["epochs"] if epochs is None else epochs
    batch_size = settings["batch_size"] if batch_size is None else batch_size
    for name, value in (("Epoch count", epochs), ("Batch size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    counts = _split_counts(dataset, stage)
    if any(not counts[split]["supervised_values"] for split in ("train", "validation")):
        raise ValueError("Training requires separately supervised train and validation splits")
    if settings["monitor"] not in ("loss", "val_loss"):
        raise ValueError("V2 training currently monitors loss or val_loss")
    models.keras.utils.set_random_seed(settings["seed"])
    model = models.build_model(stage, dataset.vocabularies)
    preview = next(batches(dataset, stage, "train", batch_size))
    before = model.get_weights()
    preview_loss = float(models.stage_loss(stage, preview.targets, model(preview.inputs, training=False), preview.weights))
    if not np.isfinite(preview_loss):
        raise ValueError(f"Nonfinite {stage} preview loss")
    root = new_run_directory(output)
    report = {"kind": "v2_stage_training", "stage": stage, "status": "untrained", "dry_run": bool(dry_run),
              "optimizer_steps": 0, "conditioning": "reference", "history_conditioning": "strictly-preceding-teacher-forced",
              "compatibility": identity, "architecture": architecture_settings(stage), **settings,
              "epochs": epochs, "batch_size": batch_size, "splits": counts,
              "parameters": model.count_params(), "preview_loss": preview_loss,
              "input_shapes": {key: list(value.shape) for key, value in preview.inputs.items()}, **execution_identity()}
    if dry_run:
        report["weights_unchanged"] = all(np.array_equal(old, new) for old, new in zip(before, model.get_weights()))
        if not report["weights_unchanged"]:
            raise ValueError("Dry-run preview changed weights")
        write_json(root / "report.json", report)
        return report

    optimizer = models.keras.optimizers.get({"class_name": settings["optimizer"],
                                            "config": {"learning_rate": settings["learning_rate"],
                                                       "clipnorm": settings["clipnorm"]}})
    rng = np.random.default_rng(settings["seed"])
    history = {"loss": [], "val_loss": []}
    mode = settings["monitor_mode"]
    maximize = mode == "max"
    best, stale = (-np.inf if maximize else np.inf), 0
    for epoch in range(epochs):
        history["loss"].append(_epoch(dataset, stage, model, batch_size, optimizer, settings["shuffle"], rng))
        history["val_loss"].append(_epoch(dataset, stage, model, batch_size))
        if settings["verbose"]:
            print(f"{stage} epoch {epoch + 1}/{epochs}: loss={history['loss'][-1]:.6f} "
                  f"val_loss={history['val_loss'][-1]:.6f}", flush=True)
        current = history[settings["monitor"]][-1]
        delta = settings["early_stopping_min_delta"]
        improved = current > best + delta if maximize else current < best - delta
        if improved:
            best, stale = current, 0
            model.save(root / "best.keras")
        else:
            stale += 1
            if stale >= settings["early_stopping_patience"]:
                break
    report.update(status="trained", optimizer_steps=int(optimizer.iterations.numpy()),
                  epochs_completed=len(history["loss"]), history=history,
                  best_monitored_value=float(best), best_validation_loss=float(min(history["val_loss"])))
    save_checkpoint_manifest(root, stage, dataset, "trained", extra=report)
    write_json(root / "report.json", report)
    return report
