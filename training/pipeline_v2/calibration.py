"""Explicit validation-only sparse-stage calibration with decoded own histories.

No optimizer, training, test loading, or production-bundle promotion happens here.
The saved inputs supply fixed audio and upstream reference context for every
query; only the selected stage's decoded history changes during replay.
"""

from copy import copy, deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from training.beat_models.artifacts import (new_run_directory, read_json,
                                           statistics_hash, write_json)

from .arc_policy import arc_candidates, history_value, stage_split, validate_split_overlay
from .artifacts import compatibility, execution_identity, load_checkpoint_manifest
from .policies import SPARSE_STAGES, repaired, semantic_identity, split_identity


THRESHOLDS = tuple(value / 100 for value in range(1, 100))
OBJECTIVE_HEAD = {"hazards": "bombs", "arcs": "attach", "lighting": "events"}


def _arc_clearance_controls():
    from tools.config import config

    if not config.add_slider_flag:
        raise ValueError("Arc calibration requires add_slider_flag enabled")
    return {"add_slider_flag": bool(config.add_slider_flag),
            "slider_movement_minimum": float(config.slider_movement_minimum),
            "min_time_diff": float(config.min_time_diff), "slider_probability": 1.,
            "configured_slider_probability": float(config.slider_probability),
            "probability_filter": "disabled during calibration"}


def _predict(predictor, inputs):
    outputs = predictor.predict(inputs, verbose=0) if hasattr(predictor, "predict") else predictor(inputs)
    if not isinstance(outputs, dict):
        raise ValueError("Sparse-stage predictions require named output heads")
    return {head: np.asarray(value) for head, value in outputs.items()}


def _validate_outputs(stage, outputs, examples, vocab, options):
    if set(outputs) - ({"bomb_presence"} if stage == "hazards" else set()) != set(examples.targets):
        raise ValueError("Calibration prediction heads disagree with prepared targets")
    for head, value in outputs.items():
        if head == "bomb_presence":
            target = np.zeros((len(examples.times), 1), np.float32)
        else:
            target = examples.targets[head]
        classes = len(vocab["walls"]) if head == "wall" else 2 if head == "attach" else 3 if head == "anchor" else None
        expected = (*target.shape, classes) if classes is not None else target.shape
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"Invalid {stage}.{head} output shape or finite values")
        if head == "shape":
            lo, hi = options["arc_shape_bounds"]
            if np.any((value < lo) | (value > hi)):
                raise ValueError("Arc controls are outside the recorded shape bounds")
        elif np.any((value < 0) | (value > 1)) or (classes is not None and not np.allclose(value.sum(axis=-1), 1, atol=1e-5)):
            raise ValueError(f"Invalid {stage}.{head} probabilities")


def teacher_forced_outputs(song, stage, predictor, vocab, options, batch_size=128):
    """Predict immutable prepared inputs once, with bounded model batches."""
    examples = song.examples[stage]
    parts = []
    for start in range(0, len(examples.times), batch_size):
        inputs = {key: value[start:start + batch_size] for key, value in examples.inputs.items()}
        parts.append(_predict(predictor, inputs))
    if not parts:
        return None
    outputs = {head: np.concatenate([part[head] for part in parts]) for head in parts[0]}
    _validate_outputs(stage, outputs, examples, vocab, options)
    return outputs


def _history_inputs(examples, index, stage, history, options):
    """Replace only causal own-stage context in one saved input row/block."""
    inputs = {key: value[index:index + 1].copy() for key, value in examples.inputs.items()}
    timeline = np.asarray(examples.times[index]).reshape(-1)
    now = float(timeline[0])
    length = inputs["history"].shape[1]
    prior = []
    for item in reversed(history):
        if float(item["time"]) < now:
            prior.append(item)
            if len(prior) == length:
                break
    prior.reverse()
    inputs["history"].fill(0)
    inputs["history_mask"].fill(False)
    previous = prior
    values = [history_value(item) if stage == "arcs" else item["value"] for item in previous]
    if previous:
        offset = length - len(previous) if stage == "lighting" else 0
        inputs["history"][0, offset:offset + len(previous)] = values
        inputs["history_mask"][0, offset:offset + len(previous)] = True
    if stage == "lighting":
        valid = inputs["mask"][0].astype(bool)
        times = timeline[valid]
        inputs["gaps"].fill(0)
        if len(times):
            inputs["gaps"][0, np.flatnonzero(valid), 0] = np.diff(np.r_[prior[-1]["time"] if prior else 0., times])
    else:
        inputs["history_age"].fill(0)
        if previous:
            inputs["history_age"][0, :len(previous), 0] = [now - item["time"] for item in previous]
        inputs["conditions"][0, 0] = now - prior[-1]["time"] if prior else now
    return inputs


def _row_outputs(outputs, index):
    return {head: value[index:index + 1] for head, value in outputs.items()}


def _one_prediction(song, stage, index, predictor, vocab, options, history, fixed):
    if fixed is not None:
        return _row_outputs(fixed, index)
    from .contracts import StageExamples

    examples = song.examples[stage]
    inputs = _history_inputs(examples, index, stage, history, options)
    outputs = _predict(predictor, inputs)
    selected = StageExamples(inputs, {key: value[index:index + 1] for key, value in examples.targets.items()},
                             {key: value[index:index + 1] for key, value in examples.weights.items()}, examples.times[index:index + 1])
    _validate_outputs(stage, outputs, selected, vocab, options)
    return outputs


def _score(target, predicted, weights, *, categorical=False):
    target, predicted = np.asarray(target), np.asarray(predicted)
    weight = np.asarray(weights, np.float64)
    while weight.ndim < target.ndim:
        weight = weight[..., None]
    weight = np.broadcast_to(weight, target.shape)
    truth = target != 0 if categorical else target > .5
    positive = predicted != 0 if categorical else predicted > .5
    matched = positive & truth & (target == predicted) if categorical else positive & truth
    tp = float((matched * weight).sum())
    fp = float(((positive & ~matched) * weight).sum())
    fn = float(((truth & ~matched) * weight).sum())
    return _statistics(tp, fp, fn)


def _statistics(tp, fp, fn):
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "reference_positives": tp + fn, "predicted_positives": tp + fp,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


_SPARSE_CATEGORICAL_HEADS = {"wall", "attach", "anchor"}


def _tolerance_seconds(options):
    return float((options.get("schedule") or {}).get("tolerance_seconds", .25))


def _tolerant_match(reference, predicted, window, fuzzy=("time",), exact=()):
    """Greedy nearest one-to-one match; fuzzy fields need |delta| <= window."""
    remaining = list(range(len(reference)))
    tp = 0
    for pred in sorted(predicted, key=lambda item: item["time"]):
        best = best_gap = None
        for index in remaining:
            ref = reference[index]
            if any(abs(float(pred.get(key, 0.)) - float(ref.get(key, 0.))) > window for key in fuzzy):
                continue
            if any(pred.get(key) != ref.get(key) for key in exact):
                continue
            gap = abs(pred["time"] - ref["time"])
            if best_gap is None or gap < best_gap:
                best, best_gap = index, gap
        if best is not None:
            remaining.remove(best)
            tp += 1
    return _statistics(float(tp), float(len(predicted) - tp), float(len(reference) - tp))


def _history_regime(stage, head, examples):
    """Per-query flag: the prepared reference history already contains a positive."""
    history, mask = examples.inputs["history"], examples.inputs["history_mask"]
    if stage == "hazards":
        channel = history[..., 0] > 0 if head == "wall" else (history[..., 1:] > 0).any(axis=-1)
    elif stage == "arcs":
        channel = history[..., 0] > 0
    else:
        channel = (history > 0).any(axis=-1)
    return (channel & mask).any(axis=1)


def _regime_report(stage, examples, predicted):
    """Split head scores by own-history regime and flag feature-enabled songs.

    ``start`` covers queries whose prepared history has no positive decision —
    the cold-start regime that fails autoregressively. ``enabled`` marks songs
    where the mapper used the feature at least once, matching always-on
    generation semantics.
    """
    regimes, enabled = {"start": {}, "continuation": {}}, {}
    for head, values in predicted.items():
        target, weight = examples.targets[head], examples.weights[head]
        categorical = head in _SPARSE_CATEGORICAL_HEADS
        supervised = np.asarray(weight) > 0
        truth = (np.asarray(target) != 0) if categorical else (np.asarray(target) > .5)
        enabled[head] = bool((truth & supervised).any())
        regime = _history_regime(stage, head, examples)
        for name, selection in (("start", ~regime), ("continuation", regime)):
            regimes[name][head] = _score(target[selection], values[selection],
                                         weight[selection], categorical=categorical)
    return {"regimes": regimes, "enabled": enabled}


def _reference_state(song, vocab, options, *, hazards=False):
    from .targets import reference_expanded_state

    state = reference_expanded_state(song, vocab, options)
    state.update(walls=[], bombs=[], arcs=[], lighting=[])
    if hazards:
        if repaired(options, "hazards") and "hazards" in song.examples:
            from .hazard_sections import SectionDecoder, candidates

            decoder = SectionDecoder(state, song.base.metadata["duration"], vocab, options)
            examples = song.examples["hazards"]
            queries = candidates(song.base.metadata["duration"], song.base.metadata["bpm"], options)
            if len(queries) != len(examples.times):
                raise ValueError("Reference hazard schedule disagrees with prepared examples")
            for index, query in enumerate(queries):
                wall = int(examples.targets["wall"][index]) if np.any(examples.weights["wall"][index] > 0) else 0
                bombs = examples.targets["bombs"][index] * (np.asarray(examples.weights["bombs"][index]) > 0)
                decoder.step(query, wall, bombs)
            final = decoder.finish()
            state.update(walls=final["walls"], bombs=final["bombs"])
        else:
            state.update(walls=deepcopy(song.objects.get("walls", [])), bombs=deepcopy(song.objects.get("bombs", [])))
    return state


def _arc_validator(state, song, options):
    from map_creation.v2_postprocessing import validate_arcs

    return validate_arcs(state, song, settings=options, probability=1.)


def _arc_replay(song, predictor, vocab, options, threshold, fixed, upstream_state, retain_objects):
    state = copy(upstream_state) if upstream_state is not None else _reference_state(song, vocab, options, hazards=True)
    state["arcs"] = []
    examples = song.examples["arcs"]
    queries = arc_candidates(state["notes"], options)
    if len(queries) != len(examples.times) or not np.allclose([query["time"] for query in queries], examples.times, atol=1e-8, rtol=0):
        raise ValueError("Arc replay endpoints disagree with the prepared reference schedule")
    history, decisions, rejected = [], [], 0
    for index, query in enumerate(queries):
        output = _one_prediction(song, "arcs", index, predictor, vocab, options, history, fixed)
        attach = bool(output["attach"][0, 1] >= threshold)
        shape, anchor = output["shape"][0], int(output["anchor"][0].argmax())
        previous = len(state["arcs"])
        if attach:
            state["arcs"].append({**deepcopy(query), "mu": float(shape[0]), "tmu": float(shape[1]), "anchor_mode": anchor})
            _arc_validator(state, song, options)
        accepted = len(state["arcs"]) > previous
        rejected += int(attach and not accepted)
        decisions.append(int(accepted))
        history.append({"time": query["time"], "value": history_value({"attach": accepted, "shape": shape, "anchor": anchor})})
        del history[:-(options["history_length"] + 2)]
    decisions = np.asarray(decisions)
    heads = {"attach": _score(examples.targets["attach"], decisions, examples.weights["attach"])}
    regime = _regime_report("arcs", examples, {"attach": decisions})
    window = _tolerance_seconds(options)
    supervised = (np.asarray(examples.targets["attach"]) > 0) & (np.asarray(examples.weights["attach"]) > 0)
    reference = [queries[index] for index in np.flatnonzero(supervised)]
    tolerant = {"attach": _tolerant_match(reference, state["arcs"], window, ("time", "end_time"), ("hand",))}
    return heads, regime, {"arcs": state["arcs"]} if retain_objects else {}, {
        "queries": len(queries), "clearance_rejections": rejected,
        "slider_probability": 1., "probability_filter": "disabled for calibration"}, tolerant


def _hazard_replay(song, predictor, vocab, options, threshold, fixed, upstream_state, retain_objects):
    from .hazard_sections import SectionDecoder, candidates, _sample_probs, _sample_cells, _BombRateWindow

    state = upstream_state if upstream_state is not None else _reference_state(song, vocab, options)
    examples = song.examples["hazards"]
    queries = candidates(song.base.metadata["duration"], song.base.metadata["bpm"], options)
    if len(queries) != len(examples.times) or not np.allclose([query["time"] for query in queries], examples.times, atol=1e-8, rtol=0):
        raise ValueError("Hazard replay schedule disagrees with prepared examples")
    decoder = SectionDecoder(state, song.base.metadata["duration"], vocab, options)
    masked = bool(options.get("hazard_feasibility_decode"))
    temperature = float(options.get("hazard_sample_temperature", 0.) or 0.)
    sampling = masked and temperature > 0
    # With rate-matching on, the swept threshold acts as the bomb_rate_factor.
    rate = (_BombRateWindow({**options, "bomb_rate_factor": threshold})
            if options.get("hazard_bomb_rate_decode") and options.get("hazard_bombs_enabled", True) else None)
    rng = np.random.default_rng(int(options.get("hazard_sample_seed", 0)))
    mask_redirects, cells_masked = 0, 0
    for index, query in enumerate(queries):
        output = _one_prediction(song, "hazards", index, predictor, vocab, options, decoder.history, fixed)
        wall, bombs = np.asarray(output["wall"][0]).copy(), np.asarray(output["bombs"][0]).copy()
        presence = output.get("bomb_presence")
        if presence is not None:
            bombs *= float(np.asarray(presence).reshape(-1)[0])
        choice, fires = int(wall.argmax()), bombs >= threshold
        if not options.get("hazard_bombs_enabled", True):
            fires = np.zeros_like(fires)
        if query["kind"] == "wall":
            if masked:
                feasible = decoder.feasible_wall_classes(query)
                if sampling:
                    allowed = np.flatnonzero(feasible)
                    choice = int(allowed[_sample_probs(wall[feasible], temperature, rng)])
                elif choice and not feasible[choice]:
                    mask_redirects += 1
                    choice = 0
        else:
            eligible = decoder.eligible_bomb_cells(float(query["time"])) if (masked or rate is not None) else None
            if rate is not None:
                fires = rate.fires(float(query["time"]), bombs, eligible)
                cells_masked += int((~eligible & fires).sum())
            elif masked:
                cells_masked += int((~eligible & fires).sum())
                fires = (fires if not sampling else _sample_cells(bombs, temperature, rng)) & eligible
        # Wall geometry remains categorical argmax, including NONE.
        accepted = decoder.step(query, choice, fires)
        if rate is not None and query["kind"] == "bomb":
            rate.placed(float(query["time"]), accepted["bombs"])
    final = decoder.finish()
    walls = np.zeros_like(examples.targets["wall"])
    bombs = np.zeros_like(examples.targets["bombs"])
    bomb_cells = {}
    for bomb in final["bombs"]:
        bomb_cells.setdefault(round(bomb["time"], 9), []).append(4 * bomb["y"] + bomb["x"])
    wall_index = 0
    for index, query in enumerate(queries):
        if query["kind"] == "wall":
            while wall_index < len(final["walls"]) and final["walls"][wall_index]["end_time"] < query["end_time"] - 1e-8:
                wall_index += 1
            if wall_index < len(final["walls"]):
                wall = final["walls"][wall_index]
                if wall["time"] <= query["time"] + 1e-8 and wall["end_time"] >= query["end_time"] - 1e-8:
                    walls[index] = wall["template_id"]
        else:
            bombs[index, bomb_cells.get(round(query["time"], 9), [])] = 1
    heads = {"wall": _score(examples.targets["wall"], walls, examples.weights["wall"], categorical=True),
             "bombs": _score(examples.targets["bombs"], bombs, examples.weights["bombs"])}
    regime = _regime_report("hazards", examples, {"wall": walls, "bombs": bombs})
    window = _tolerance_seconds(options)
    wall_supervised = (np.asarray(examples.targets["wall"]) > 0) & (np.asarray(examples.weights["wall"]) > 0)
    reference_walls = [{"time": queries[i]["time"], "end_time": queries[i]["end_time"]}
                       for i in np.flatnonzero(wall_supervised)]
    bomb_positive = (np.asarray(examples.targets["bombs"]) > 0) & (np.asarray(examples.weights["bombs"]) > 0)
    reference_bombs = [{"time": queries[i]["time"], "cell": c} for i, c in np.argwhere(bomb_positive)]
    predicted_bombs = [{"time": b["time"], "cell": 4 * b["y"] + b["x"]} for b in final["bombs"]]
    tolerant = {"wall": _tolerant_match(reference_walls, final["walls"], window, ("time", "end_time")),
                "bombs": _tolerant_match(reference_bombs, predicted_bombs, window, ("time",), ("cell",))}
    diagnostics = {**final["diagnostics"], "feasibility_masked": masked,
                   "wall_mask_redirects": mask_redirects, "bomb_cells_masked": cells_masked,
                   "bomb_rate_matching": rate is not None,
                   **({"bomb_rate_seconds": rate.seconds, "bomb_rate_factor": rate.factor} if rate else {})}
    return heads, regime, {key: final[key] for key in ("walls", "bombs")} if retain_objects else {}, diagnostics, tolerant


def _lighting_replay(song, predictor, vocab, options, threshold, fixed, upstream_state, retain_objects):
    from .basic_lighting import decode

    examples = song.examples["lighting"]
    if fixed is not None and not retain_objects:
        decisions = decode(fixed["events"], vocab, threshold)
        decisions *= examples.inputs["mask"][..., None]
        heads = {"events": _score(examples.targets["events"], decisions, examples.weights["events"])}
        regime = _regime_report("lighting", examples, {"events": decisions})
        return heads, regime, {}, {"blocks": len(examples.times), "events": int(decisions.sum())}, {}
    history, events, event_count = [], [], 0
    decisions = np.zeros_like(examples.targets["events"])
    for index, timeline in enumerate(examples.times):
        output = _one_prediction(song, "lighting", index, predictor, vocab, options, history, fixed)
        projected = decode(output["events"][0], vocab, threshold)
        for offset, when in enumerate(timeline):
            if when < 0 or not examples.inputs["mask"][index, offset]:
                continue
            decisions[index, offset] = projected[offset]
            history.append({"time": float(when), "value": projected[offset].tolist()})
            event_count += int(projected[offset].sum())
            if retain_objects:
                events.extend({"time": float(when), **deepcopy(vocab["lighting"][category])} for category in np.flatnonzero(projected[offset]))
        del history[:-options["lighting_block_length"]]
    heads = {"events": _score(examples.targets["events"], decisions, examples.weights["events"])}
    regime = _regime_report("lighting", examples, {"events": decisions})
    return heads, regime, {"lighting": events} if retain_objects else {}, {"blocks": len(examples.times), "events": event_count}, {}


def replay_stage(song, stage, predictor, vocab, options, threshold=.5, *, conditioning="autoregressive", outputs=None,
                 upstream_state=None, retain_objects=True):
    """Replay one sparse stage against fixed upstream reference queries.

    Teacher forcing uses cached raw predictions but still applies the actual
    shared clearance/projection decoder. Autoregression records accepted own
    outputs, including explicit NONE decisions. The source song is untouched.
    """
    if stage not in SPARSE_STAGES or not repaired(options, stage):
        raise ValueError("Decoded replay requires one repaired sparse-stage policy")
    if conditioning not in ("autoregressive", "teacher_forced"):
        raise ValueError("Unknown sparse-stage replay conditioning")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Calibration threshold must be between zero and one")
    if stage not in song.examples:
        raise ValueError("Selected song lacks prepared stage examples")
    if conditioning == "teacher_forced":
        fixed = outputs if outputs is not None else teacher_forced_outputs(song, stage, predictor, vocab, options)
        if fixed is not None:
            _validate_outputs(stage, fixed, song.examples[stage], vocab, options)
    else:
        if outputs is not None:
            raise ValueError("Autoregressive replay cannot consume teacher-forced outputs")
        fixed = None
    replay = {"arcs": _arc_replay, "hazards": _hazard_replay, "lighting": _lighting_replay}[stage]
    heads, regime, objects, diagnostics, tolerant = replay(song, predictor, vocab, options, threshold, fixed, upstream_state, retain_objects)
    return {"song_id": song.base.song_id, "group_id": song.base.metadata["group_id"], "stage": stage,
            "conditioning": conditioning, "threshold": float(threshold), "heads": heads,
            "tolerant": tolerant, "tolerance_seconds": _tolerance_seconds(options) if tolerant else None,
            "regimes": regime["regimes"], "enabled": regime["enabled"],
            "objective": heads[OBJECTIVE_HEAD[stage]], "objects": objects, "diagnostics": diagnostics}


def _validation_songs(dataset, stage, shard=None):
    """Select lazy records before opening their arrays; TEST is never read."""
    records = getattr(dataset.songs, "records", None)
    if records is not None:
        identities = {}
        for record in records:
            split = stage_split(dataset.manifest, stage, record)
            for key in ("group_id", "audio_sha256", "decoded_sha256"):
                if record.get(key):
                    token = ("group" if key == "group_id" else "audio", record[key])
                    if identities.get(token, split) != split:
                        raise ValueError("Recording identity leaks across calibration splits")
                    identities[token] = split
        position = 0
        for index, record in enumerate(records):
            if stage_split(dataset.manifest, stage, record) == "validation":
                keep = shard is None or position % shard[1] == shard[0]
                position += 1
                if not keep:
                    continue
                song = dataset.songs[index]
                if (song.base.song_id != record["song_id"]
                        or any(song.base.metadata.get(key) != record.get(key) for key in ("group_id", "split"))
                        or any(record.get(key) and song.base.metadata.get(key) != record[key]
                               for key in ("audio_sha256", "decoded_sha256"))):
                    raise ValueError("Calibration song metadata disagrees with recording identity")
                yield song
        return
    position = 0
    for song in dataset.songs:
        if stage_split(dataset.manifest, stage, {"song_id": song.base.song_id, **song.base.metadata}) == "validation":
            if shard is None or position % shard[1] == shard[0]:
                yield song
            position += 1


def _aggregate(reports, stage):
    heads, enabled, enabled_songs, tolerant = {}, {}, {}, {}
    regimes = {"start": {}, "continuation": {}}
    for report in reports:
        for head, values in report["heads"].items():
            total = heads.setdefault(head, [0., 0., 0.])
            for index, key in enumerate(("true_positive", "false_positive", "false_negative")):
                total[index] += values[key]
            if report.get("enabled", {}).get(head):
                enabled_songs[head] = enabled_songs.get(head, 0) + 1
                total = enabled.setdefault(head, [0., 0., 0.])
                for index, key in enumerate(("true_positive", "false_positive", "false_negative")):
                    total[index] += values[key]
        for head, values in report.get("tolerant", {}).items():
            total = tolerant.setdefault(head, [0., 0., 0.])
            for index, key in enumerate(("true_positive", "false_positive", "false_negative")):
                total[index] += values[key]
        for name in ("start", "continuation"):
            for head, values in report.get("regimes", {}).get(name, {}).items():
                total = regimes[name].setdefault(head, [0., 0., 0.])
                for index, key in enumerate(("true_positive", "false_positive", "false_negative")):
                    total[index] += values[key]
    scores = {head: _statistics(*values) for head, values in heads.items()}
    objective = scores.get(OBJECTIVE_HEAD[stage], _statistics(0., 0., 0.))
    return {"heads": scores, "objective_head": OBJECTIVE_HEAD[stage], "objective": objective,
            "tolerant": {head: _statistics(*values) for head, values in tolerant.items()},
            "tolerance_seconds": next((report.get("tolerance_seconds") for report in reports if report.get("tolerant")), None),
            "regimes": {name: {head: _statistics(*values) for head, values in heads.items()}
                        for name, heads in regimes.items()},
            "enabled_only": {head: _statistics(*values) for head, values in enabled.items()},
            "enabled_songs": enabled_songs,
            "songs": [{key: value for key, value in report.items() if key != "objects"} for report in reports]}


def _calibration_setup(dataset, stage, artifact):
    """Validated sparse-stage predictor and provenance metadata for replays."""
    from .runner import load_stage_artifact

    if stage not in SPARSE_STAGES:
        raise ValueError("Calibration supports hazards, arcs and lighting")
    options = dataset.manifest.get("data_settings", {})
    if not repaired(options, stage) or dataset.manifest.get("status") != "prepared":
        raise ValueError("Calibration requires real prepared data with the repaired stage policy")
    arc_controls = _arc_clearance_controls() if stage == "arcs" else None
    validate_split_overlay(dataset.manifest)
    _, preflight = load_checkpoint_manifest(artifact, stage, dataset)
    if preflight.get("status") != "trained":
        raise ValueError("Calibration requires an explicitly trained stage artifact")
    indexed_split = split_identity(dataset.manifest, stage)
    if preflight.get("training", {}).get("split_identity") != indexed_split:
        raise ValueError("Calibration dataset split identity differs from the artifact training split")
    saved_overlay = preflight.get("training", {}).get("stage_split")
    overlay_fingerprint = dataset.manifest.get("stage_splits", {}).get(stage, {}).get("fingerprint")
    if isinstance(saved_overlay, dict) and saved_overlay.get("fingerprint") != overlay_fingerprint:
        raise ValueError("Calibration arc overlay differs from the artifact training overlay")
    model, artifact_metadata = load_stage_artifact(artifact, stage, dataset)
    if (artifact_metadata.get("status") != "trained" or artifact_metadata.get("checkpoint_sha256") != preflight.get("checkpoint_sha256")
            or artifact_metadata.get("training", {}).get("split_identity") != indexed_split):
        raise ValueError("Calibration artifact changed after split provenance validation")
    predictor = lambda inputs: model(inputs, training=False)
    compiled_calls = hasattr(model, "trainable_variables")
    if compiled_calls:
        import tensorflow as tf

        predictor = tf.function(predictor, reduce_retracing=True)
    return {"predictor": predictor, "options": options, "artifact_metadata": artifact_metadata,
            "compiled_calls": compiled_calls, "arc_controls": arc_controls,
            "indexed_split": indexed_split, "overlay_fingerprint": overlay_fingerprint}


def _calibration_shard(task):
    """Spawn-pool worker: replay one validation-song shard, return counts."""
    from .artifacts import load_dataset

    dataset = load_dataset(task["dataset"])
    stage = task["stage"]
    setup = _calibration_setup(dataset, stage, task["artifact"])
    predictor, options = setup["predictor"], setup["options"]
    shard = (task["shard"], task["shards"])
    if task["mode"] == "sweep":
        rankings, validation = {}, []
        for song in _validation_songs(dataset, stage, shard=shard):
            if stage not in song.examples or not len(song.examples[stage].times):
                continue
            validation.append({"song_id": song.base.song_id, "group_id": song.base.metadata["group_id"]})
            outputs = teacher_forced_outputs(song, stage, predictor, dataset.vocabularies, options, task["batch_size"])
            upstream = (_reference_state(song, dataset.vocabularies, options, hazards=stage == "arcs")
                        if stage != "lighting" else None)
            for threshold in THRESHOLDS:
                replay = replay_stage(song, stage, predictor, dataset.vocabularies, options, threshold,
                                      conditioning="teacher_forced", outputs=outputs, upstream_state=upstream,
                                      retain_objects=False)
                rankings.setdefault(threshold, []).append({"heads": replay["heads"]})
        return {"validation": validation, "rankings": rankings}
    reports = {threshold: [] for threshold in task["thresholds"]}
    for song in _validation_songs(dataset, stage, shard=shard):
        if stage not in song.examples or not len(song.examples[stage].times):
            continue
        for threshold in task["thresholds"]:
            reports[threshold].append(
                replay_stage(song, stage, predictor, dataset.vocabularies, options, threshold, retain_objects=False))
    return {"reports": reports}


def calibrate_stage(dataset, stage, artifact, output, *, batch_size=128, workers=1, dataset_path=None,
                    ar_grid=None):
    """Save one checkpoint-bound sidecar after a 99-point validation sweep."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Calibration batch_size must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("Calibration workers must be a positive integer")
    if ar_grid is not None and (isinstance(ar_grid, bool) or not isinstance(ar_grid, (int, float))
                                or not np.isfinite(ar_grid) or ar_grid <= 0 or ar_grid >= .5):
        raise ValueError("Calibration ar_grid must be a step inside (0, .5)")
    setup = _calibration_setup(dataset, stage, artifact)
    predictor, options = setup["predictor"], setup["options"]
    artifact_metadata = setup["artifact_metadata"]
    compiled_calls = setup["compiled_calls"]
    arc_controls, indexed_split = setup["arc_controls"], setup["indexed_split"]
    overlay_fingerprint = setup["overlay_fingerprint"]
    rankings = {threshold: [] for threshold in THRESHOLDS}
    validation = []
    if workers > 1:
        if dataset_path is None:
            raise ValueError("Parallel calibration requires the dataset path for worker loading")
        import multiprocessing as mp

        tasks = [{"dataset": str(dataset_path), "stage": stage, "artifact": str(artifact),
                  "shard": index, "shards": workers, "batch_size": batch_size, "mode": "sweep"}
                 for index in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            for partial in pool.map(_calibration_shard, tasks):
                validation.extend(partial["validation"])
                for threshold, rows in partial["rankings"].items():
                    rankings[threshold].extend(rows)
    else:
        for song in _validation_songs(dataset, stage):
            if stage not in song.examples or not len(song.examples[stage].times):
                continue
            validation.append({"song_id": song.base.song_id, "group_id": song.base.metadata["group_id"]})
            outputs = teacher_forced_outputs(song, stage, predictor, dataset.vocabularies, options, batch_size)
            upstream = _reference_state(song, dataset.vocabularies, options, hazards=stage == "arcs") if stage != "lighting" else None
            for threshold in THRESHOLDS:
                replay = replay_stage(song, stage, predictor, dataset.vocabularies, options, threshold,
                                      conditioning="teacher_forced", outputs=outputs, upstream_state=upstream, retain_objects=False)
                rankings[threshold].append({"heads": replay["heads"]})
    if not validation:
        raise ValueError("Calibration requires nonempty validation stage examples")
    teacher = [{"threshold": threshold, **_aggregate(reports, stage)} for threshold, reports in rankings.items()]
    for row in teacher:
        del row["songs"]
    teacher.sort(key=lambda row: (row["objective"]["f1"], row["threshold"]), reverse=True)
    # Teacher-forced ranking can pick a region far from the autoregressive
    # optimum (proven on hazards and arcs); a coarse AR grid widens coverage.
    shortlist = teacher[:3]
    ar_thresholds = sorted(
        {candidate["threshold"] for candidate in shortlist}
        | ({round(float(threshold), 2) for threshold in np.arange(ar_grid, 1., ar_grid)} if ar_grid else set()))
    reranked = []
    if workers > 1:
        import multiprocessing as mp

        tasks = [{"dataset": str(dataset_path), "stage": stage, "artifact": str(artifact),
                  "shard": index, "shards": workers, "batch_size": batch_size, "mode": "replay",
                  "thresholds": ar_thresholds}
                 for index in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            partials = pool.map(_calibration_shard, tasks)
        for threshold in ar_thresholds:
            reports = [report for partial in partials for report in partial["reports"][threshold]]
            reranked.append({"threshold": threshold, **_aggregate(reports, stage)})
    else:
        for threshold in ar_thresholds:
            reports = [replay_stage(song, stage, predictor, dataset.vocabularies, options, threshold, retain_objects=False)
                       for song in _validation_songs(dataset, stage) if stage in song.examples and len(song.examples[stage].times)]
            reranked.append({"threshold": threshold, **_aggregate(reports, stage)})
    reranked.sort(key=lambda row: (row["objective"]["f1"], row["threshold"]), reverse=True)
    selected = reranked[0]
    score = selected["objective"]
    report = {"kind": "pipeline_v2_stage_calibration", "version": 1, "stage": stage, "status": "calibrated",
              "created_at": datetime.now(timezone.utc).isoformat(), "fit_split": "validation",
              "conditioning": "reference_upstream_own_stage_autoregressive", "optimizer_steps": 0,
              "checkpoint_sha256": artifact_metadata["checkpoint_sha256"],
              "compatibility": compatibility(dataset, stage), "threshold": selected["threshold"],
              "thresholds": {OBJECTIVE_HEAD[stage]: selected["threshold"]},
              "validation_song_ids": sorted(item["song_id"] for item in validation),
              "validation_group_ids": sorted({item["group_id"] for item in validation}),
              "split_identity": indexed_split, "stage_split_fingerprint": overlay_fingerprint,
              "reproducibility": {"arc_clearance_controls": arc_controls} if stage == "arcs" else {},
              "teacher_forced_ranking": teacher, "autoregressive_shortlist": reranked,
              "selection": ("highest autoregressive decoded F1 among top three teacher-forced thresholds; "
                            "ties choose higher threshold") if not ar_grid else
                           (f"highest autoregressive decoded F1 over a {ar_grid} grid plus the top three "
                            "teacher-forced thresholds; ties choose higher threshold"),
              "positive_cases_detected": bool(score["reference_positives"] and score["true_positive"]),
              "gameplay_validated": False, **execution_identity()}
    report["runtime"]["compiled_calls"] = compiled_calls
    report["fingerprint"] = statistics_hash(report)
    root = new_run_directory(output)
    write_json(root / "calibration.json", report)
    return report


def load_calibration(path, stage, artifact_metadata):
    """Load only an intact validation sidecar for this exact trained checkpoint."""
    supplied = Path(path)
    report = read_json(supplied / "calibration.json" if supplied.is_dir() else supplied)
    if not isinstance(report, dict):
        raise ValueError("Invalid validation calibration sidecar")
    payload = deepcopy(report)
    fingerprint = payload.pop("fingerprint", None)
    if fingerprint != statistics_hash(payload):
        raise ValueError("Calibration sidecar fingerprint mismatch")
    if (report.get("kind") != "pipeline_v2_stage_calibration" or type(report.get("version")) is not int or report.get("version") != 1
            or report.get("stage") != stage or report.get("status") != "calibrated"
            or report.get("fit_split") != "validation" or report.get("optimizer_steps") != 0):
        raise ValueError("Invalid validation calibration sidecar")
    threshold = report.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid calibrated threshold")
    if stage not in OBJECTIVE_HEAD or report.get("thresholds") != {OBJECTIVE_HEAD[stage]: threshold}:
        raise ValueError("Calibration threshold heads disagree with the stage")
    checkpoint = report.get("checkpoint_sha256")
    if (artifact_metadata.get("status") != "trained" or not isinstance(checkpoint, str) or not checkpoint
            or checkpoint != artifact_metadata.get("checkpoint_sha256")):
        raise ValueError("Calibration sidecar belongs to another trained checkpoint")
    recorded, expected = report.get("compatibility"), artifact_metadata.get("compatibility")
    if not isinstance(recorded, dict) or not isinstance(expected, dict):
        raise ValueError("Calibration requires checkpoint compatibility provenance")
    if recorded.get("stage") != stage or expected.get("stage") != stage:
        raise ValueError("Calibration sidecar checkpoint compatibility stage mismatch")
    if statistics_hash(semantic_identity(recorded)) != statistics_hash(semantic_identity(expected)):
        raise ValueError("Calibration sidecar checkpoint compatibility mismatch")
    training_split = artifact_metadata.get("training", {}).get("split_identity")
    recorded_split = report.get("split_identity")
    if not isinstance(recorded_split, dict) or recorded_split != training_split:
        raise ValueError("Calibration sidecar differs from the checkpoint training split identity")
    split_payload = {key: value for key, value in recorded_split.items() if key != "sha256"}
    if (recorded_split.get("version") != 1 or recorded_split.get("stage") != stage
            or not recorded_split.get("records") or recorded_split.get("sha256") != statistics_hash(split_payload)):
        raise ValueError("Invalid calibration sidecar training split identity")
    validation = {row["song_id"]: row["group_id"] for row in recorded_split["records"] if row.get("split") == "validation"}
    identifiers = report.get("validation_song_ids")
    if (not isinstance(identifiers, list) or not identifiers or len(set(identifiers)) != len(identifiers)
            or not set(identifiers) <= set(validation)
            or report.get("validation_group_ids") != sorted({validation[song_id] for song_id in identifiers})):
        raise ValueError("Calibration sidecar names records outside its validation split")
    training_overlay = artifact_metadata.get("training", {}).get("stage_split")
    if isinstance(training_overlay, dict) and training_overlay.get("fingerprint") != report.get("stage_split_fingerprint"):
        raise ValueError("Calibration sidecar differs from the checkpoint arc overlay")
    return report
