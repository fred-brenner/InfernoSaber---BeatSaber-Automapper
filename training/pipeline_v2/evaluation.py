"""Opt-in stage comparisons against fixed reference-conditioned examples.

Importing this module performs no file traversal, checkpoint loading, inference,
or training. Repaired sparse stages also replay their own accepted decisions;
upstream stages remain reference conditioned and are never composed here.
"""

from datetime import datetime, timezone
from pathlib import Path
from copy import deepcopy

import numpy as np

from training.beat_models.artifacts import statistics_hash

from .artifacts import compatibility, execution_identity
from .contracts import NEW_STAGES
from .arc_policy import stage_split, validate_split_overlay
from .policies import repaired


COMPARISON_STAGES = ("encoder", *NEW_STAGES)


def _selected_songs(dataset, split, stage=None):
    """Select saved records before opening arrays; retain at most one song."""
    records = getattr(dataset.songs, "records", None)
    if records is not None:
        for index, record in enumerate(records):
            effective = record["split"] if stage is None else stage_split(dataset.manifest, stage, record)
            if effective != split:
                continue
            song = dataset.songs[index]
            if song.base.song_id != record["song_id"] or any(
                    key in record and song.base.metadata.get(key) != record[key]
                    for key in ("split", "group_id", "audio_sha256", "decoded_sha256")):
                raise ValueError("Evaluation song identity disagrees with its saved recording record")
            yield song
        return
    for song in dataset.songs:
        record = {**song.base.metadata, "song_id": song.base.song_id}
        effective = record["split"] if stage is None else stage_split(dataset.manifest, stage, record)
        if effective == split:
            yield song


def preparation_reports(dataset):
    """Copy every preparation skip into a report without running evaluation.

    Accept either a dataset or its manifest so provenance can be checked from
    metadata alone. Original records and the complete base report are retained.
    """
    manifest = getattr(dataset, "manifest", dataset)
    base = manifest.get("base_preparation", {})
    origins = {"base_preparation.skipped": base.get("skipped", []),
               "base_preparation.reserved_skipped": base.get("reserved_skipped", []),
               "skipped": manifest.get("skipped", [])}
    skipped = []
    for origin, records in origins.items():
        for record in records:
            item = deepcopy(record) if isinstance(record, dict) else {"record": deepcopy(record)}
            item["report_origin"] = origin
            skipped.append(item)
    return {"base_preparation": deepcopy(base), "preparation": deepcopy(manifest.get("preparation", {})),
            "skipped": skipped, "skip_counts": {origin: len(records) for origin, records in origins.items()}}


def validate_curated_dataset(dataset, *, split="test", evaluation_root=None, allow_synthetic=False, stage=None):
    """Require the curated package sources; favorites never enter comparisons.

    A relocated copy of the curated folder must be named explicitly. Synthetic
    fixtures require a separate programmatic switch and cannot admit real data.
    """
    root = str(Path(evaluation_root).resolve()).replace("\\", "/").rstrip("/").casefold() if evaluation_root else None
    if root and "favorites_bs_input" in root.split("/"):
        raise ValueError("favorites_bs_input cannot be a curated evaluation root")
    records = []
    for song in _selected_songs(dataset, split, stage):
        metadata = song.base.metadata
        sources = metadata.get("sources", [])
        fixture = metadata.get("provenance", {}).get("adapter") == "v2_synthetic_fixture"
        if allow_synthetic and fixture and not sources:
            records.append({"song_id": song.base.song_id, "sources": [], "synthetic": True})
            continue
        roles = {source.get("role") for source in sources}
        if not {"audio", "beatmap"} <= roles:
            raise ValueError(f"{song.base.song_id}: curated comparison requires original audio and beatmap source provenance")
        for source in sources:
            path = str(Path(source.get("path", "")).resolve()).replace("\\", "/").casefold()
            parts = path.split("/")
            canonical = any(parts[index:index + 2] == ["evaluate_beats", "bs_map_input"] for index in range(len(parts) - 1))
            explicit = root is not None and path.startswith(root + "/")
            if "favorites_bs_input" in parts or not (explicit if root is not None else canonical):
                raise ValueError(f"{song.base.song_id}: source is outside the curated bs_map_input packages: {source.get('path')}")
            digest = source.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{song.base.song_id}: curated source needs its recorded SHA256")
        records.append({"song_id": song.base.song_id, "sources": sources, "synthetic": False})
    if not records:
        raise ValueError(f"No curated reference songs in the requested {split} split")
    return records


def _classes(stage, head, vocabularies):
    if stage == "positions":
        return 13
    if stage == "arrows":
        return 9
    if stage == "multiplier":
        return len(vocabularies["multiplier"])
    if head == "wall":
        return len(vocabularies["walls"])
    if head == "attach":
        return 2
    if head == "anchor":
        return 3
    raise ValueError(f"No categorical class count for {stage}.{head}")


def _kind(stage, head):
    if head == "shape":
        return "regression"
    if (stage == "hazards" and head == "bombs") or stage == "lighting":
        return "multilabel"
    return "categorical"


def _weights(value, shape):
    weights = np.asarray(value, np.float64)
    while weights.ndim < len(shape):
        weights = weights[..., None]
    try:
        result = np.broadcast_to(weights, shape)
    except ValueError as error:
        raise ValueError("Reference target weights do not match target shape") from error
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Reference weights must be finite and nonnegative")
    return result


def _predict(predictor, inputs):
    if hasattr(predictor, "predict"):
        outputs = predictor.predict(inputs, verbose=0)
    else:
        outputs = predictor(inputs)
    if not isinstance(outputs, dict):
        raise ValueError("Stage predictors must return named output heads")
    return {key: np.asarray(value) for key, value in outputs.items()}


def fit_reference_baselines(dataset, stage):
    """Fit frequencies on training recording groups only; never fit test labels."""
    if stage not in NEW_STAGES:
        raise ValueError("Select one independently evaluated v2 stage")
    totals, counts, songs, recordings = {}, {}, [], []
    validate_split_overlay(dataset.manifest)
    for song in _selected_songs(dataset, "train", stage):
        if stage not in song.examples:
            continue
        songs.append(song.base.song_id)
        recordings.append({"song_id": song.base.song_id, "group_id": song.base.metadata.get("group_id"),
                           "decoded_audio_sha256": song.base.metadata.get("decoded_sha256"),
                           "source_sha256": [source.get("sha256") for source in song.base.metadata.get("sources", [])]})
        examples = song.examples[stage]
        for head, target in examples.targets.items():
            kind = _kind(stage, head)
            target = np.asarray(target)
            weights = _weights(examples.weights[head], target.shape)
            if kind == "categorical":
                classes = _classes(stage, head, dataset.vocabularies)
                if np.any((target < 0) | (target >= classes)):
                    raise ValueError("Reference category outside frozen vocabulary")
                sums = np.bincount(target.astype(int).ravel(), weights=weights.ravel(), minlength=classes)
                count = float(weights.sum())
            else:
                axes = tuple(range(target.ndim - 1))
                sums = (target * weights).sum(axis=axes)
                count = weights.sum(axis=axes)
            totals[head] = totals.get(head, np.zeros_like(sums, dtype=np.float64)) + sums
            counts[head] = counts.get(head, np.zeros_like(count, dtype=np.float64)) + count
    if not songs:
        raise ValueError("Frequency baselines require separate prepared training examples")
    means = {}
    for head, total in totals.items():
        kind = _kind(stage, head)
        count = counts[head]
        fallback = np.full_like(total, 1 / len(total)) if kind == "categorical" else np.zeros_like(total)
        means[head] = np.divide(total, count, out=fallback, where=np.asarray(count) > 0).astype(np.float32)
    return {"stage": stage, "kind": "training_reference_frequency", "fit_split": "train", "training_song_ids": songs,
            "training_recordings": recordings,
            "supervised_values": counts,
            "values": means, "identity": statistics_hash({"stage": stage, "songs": songs, "values": means})}


class ReferenceBaseline:
    def __init__(self, stage, vocabularies, fitted, name="frequency"):
        if name not in ("frequency", "none"):
            raise ValueError("Select frequency or none reference")
        if name == "none" and stage in ("positions", "arrows"):
            raise ValueError("Positions and arrows require occupied-note predictions; NONE is not a legal baseline")
        self.stage, self.vocabularies, self.fitted, self.name = stage, vocabularies, fitted, name

    def predict(self, inputs, verbose=0):
        batch = len(inputs["audio"])
        outputs = {}
        for head, mean in self.fitted["values"].items():
            value = np.asarray(mean, np.float32)
            if self.name == "none":
                value = np.zeros_like(value)
                if _kind(self.stage, head) == "categorical":
                    value[0] = 1
                elif head == "shape":
                    value.fill(1.)
            shape = (batch, inputs["audio"].shape[1], len(value)) if self.stage == "lighting" else (batch, len(value))
            outputs[head] = np.broadcast_to(value, shape).copy()
        return outputs


def _head_scores(stage, head, targets, probabilities, weights, threshold):
    target, output = np.asarray(targets), np.asarray(probabilities, np.float64)
    kind = _kind(stage, head)
    expected = (*target.shape, output.shape[-1]) if kind == "categorical" else target.shape
    if output.shape != expected or not np.isfinite(output).all():
        raise ValueError(f"Invalid {stage}.{head} output shape or finite values")
    weight = _weights(weights, target.shape)
    count = float(weight.sum())
    if kind != "regression" and np.any((output < 0) | (output > 1)):
        raise ValueError(f"Invalid {stage}.{head} probabilities")
    if kind == "categorical":
        if not np.allclose(output.sum(axis=-1), 1, atol=1e-5):
            raise ValueError("Categorical scores must sum to one")
        if np.any((target < 0) | (target >= output.shape[-1])):
            raise ValueError("Reference category outside predicted vocabulary")
        selected = np.take_along_axis(output, target.astype(int)[..., None], axis=-1)[..., 0]
        loss = -np.log(np.clip(selected, 1e-7, 1))
        return {"weight": count, "loss_sum": float((loss * weight).sum()),
                "correct": float(((output.argmax(axis=-1) == target) * weight).sum()), "kind": kind}
    if kind == "multilabel":
        predicted = output >= threshold
        truth = target > .5
        clipped = np.clip(output, 1e-7, 1 - 1e-7)
        loss = -target * np.log(clipped) - (1 - target) * np.log(1 - clipped)
        return {"weight": count, "loss_sum": float((loss * weight).sum()), "kind": kind,
                "true_positive": float((predicted * truth * weight).sum()),
                "false_positive": float((predicted * ~truth * weight).sum()),
                "false_negative": float((~predicted * truth * weight).sum())}
    return {"weight": count, "loss_sum": float(((output - target) ** 2 * weight).sum()),
            "absolute_error_sum": float((np.abs(output - target) * weight).sum()), "kind": kind}


def _summarize_heads(totals):
    result = {}
    for head, values in totals.items():
        weight = values["weight"]
        row = {"supervised_values": weight, "loss": values["loss_sum"] / weight if weight else None,
               "kind": values["kind"]}
        if "correct" in values:
            row["accuracy"] = values["correct"] / weight if weight else None
        if "true_positive" in values:
            tp, fp, fn = (values[key] for key in ("true_positive", "false_positive", "false_negative"))
            row.update(precision=tp / (tp + fp) if tp + fp else None,
                       recall=tp / (tp + fn) if tp + fn else None,
                       f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
        if "absolute_error_sum" in values:
            row["mae"] = values["absolute_error_sum"] / weight if weight else None
        result[head] = row
    return result


def _source_coverage(song, stage):
    """Keep unscheduled/unknown source objects visible outside conditional scores."""
    diagnostics = song.diagnostics.get("targets", {})
    if stage in ("positions", "arrows", "multiplier"):
        total = len(song.base.arrays["retained_indices"])
        missing = diagnostics.get("unsupported_expansions", []) if stage == "multiplier" else []
        return {"reference_objects": total, "unrepresented_objects": len(missing),
                "unrepresented_records": missing}
    families = {"hazards": ("walls", "bombs"), "lighting": ("lighting",), "arcs": ("arcs",)}[stage]
    missing = {}
    for family in families:
        for identity in diagnostics.get(f"unscheduled_{family}", []):
            missing[str(identity)] = {"id": identity, "family": family, "reason": "not represented by schedule/vocabulary"}
    if stage == "lighting":
        for identity in diagnostics.get("unknown_lighting", []):
            missing[str(identity)] = {"id": identity, "family": "lighting", "reason": "unknown vocabulary or unsupported float value"}
        for collision in diagnostics.get("lighting_collisions", []):
            for identity in collision.get("source_ids", []):
                missing[str(identity)] = {"id": identity, "family": "lighting", "reason": "same_type_collision"}
    total = sum(len(song.objects.get(family, [])) for family in families)
    if stage == "hazards":
        from .hazard_sections import source_family
        ignored = [{**record, "family": source_family(record)} for record in song.objects.get("ignored", [])
                   if source_family(record) in families]
    else:
        ignored = [record for record in song.objects.get("ignored", []) if record.get("family") in families]
    return {"reference_objects": total, "unrepresented_objects": len(missing),
            "unrepresented_records": list(missing.values()), "ignored_source_records": ignored}


def _add_scores(total, row):
    if not total:
        total.update({key: value if key == "kind" else 0. for key, value in row.items()})
    for key, value in row.items():
        if key != "kind":
            total[key] += value


def _hazard_origins(dataset, split):
    for song in dataset.songs:
        if song.base.metadata.get("split") != split or "hazards" not in song.examples:
            continue
        origins = song.diagnostics.get("hazard_query_origins", [])
        if len(origins) != len(song.examples["hazards"].times) or any(origin not in ("scheduler", "human_positive") for origin in origins):
            raise ValueError("Hazard comparisons require per-query scheduler/native-positive provenance")
        yield from origins


def _subset_weights(weights, row_mask):
    weights = np.asarray(weights)
    return weights * np.asarray(row_mask).reshape((len(row_mask),) + (1,) * (weights.ndim - 1))


def _scheduler_coverage(songs):
    result = {}
    for family in ("walls", "bombs"):
        source, matched, missed = 0, 0, []
        for song in songs:
            coverage = song["diagnostics"].get("hazard_scheduler_coverage", {}).get(family, {})
            source += coverage.get("source_objects", 0)
            matched += coverage.get("matched_source_objects", 0)
            missed.extend({"song_id": song["song_id"], "source_id": identity}
                          for identity in coverage.get("unmatched_source_ids", []))
        result[family] = {"source_objects": source, "matched_source_objects": matched,
                          "unmatched_source_records": missed,
                          "matched_fraction": matched / source if source else None}
    return result


def _positive_scores(targets, predicted, weights, *, categorical=False):
    """Exact positive geometry/classes count as matches; NONE is never a TP."""
    target, predicted = np.asarray(targets), np.asarray(predicted)
    weight = _weights(weights, target.shape)
    truth = target != 0 if categorical else target > .5
    positive = predicted != 0 if categorical else predicted > .5
    matched = truth & positive & (target == predicted) if categorical else truth & positive
    return {"true_positive": float((matched * weight).sum()),
            "false_positive": float(((positive & ~matched) * weight).sum()),
            "false_negative": float(((truth & ~matched) * weight).sum())}


def _add_positive(total, heads):
    for head, values in heads.items():
        target = total.setdefault(head, {key: 0. for key in ("true_positive", "false_positive", "false_negative")})
        for key in target:
            target[key] += values[key]


def _summarize_positive(heads):
    result = {}
    for head, values in heads.items():
        tp, fp, fn = (values[key] for key in ("true_positive", "false_positive", "false_negative"))
        reference, predicted = tp + fn, tp + fp
        result[head] = {**values, "reference_positives": reference, "predicted_positives": predicted,
                        "precision": tp / predicted if predicted else 0.,
                        "recall": tp / reference if reference else 0.,
                        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                        "positive_cases_detected": bool(reference and tp),
                        "all_empty_failure": bool(reference and not predicted)}
    return result


def _positive_status(heads):
    references = [values for values in heads.values() if values["reference_positives"] > 0]
    if not references:
        return "no_supported_positive_reference"
    if any(values["all_empty_failure"] for values in references):
        return "failed_all_empty"
    if all(values["positive_cases_detected"] for values in references):
        return "positive_matches_detected"
    return "positive_matches_missing"


def _sparse_diagnostics(song, stage):
    """Keep compact source accounting instead of retaining every query record."""
    targets = song.diagnostics.get("targets", {})
    if stage == "hazards":
        keys = ("hazard_raw_source_counts", "hazard_parsed_source_counts", "hazard_supported_reference",
                "hazard_exclusion_counts", "hazard_source_exclusions", "hazard_reference_rollout")
        result = {key: deepcopy(targets[key]) for key in keys if key in targets}
        errors = targets.get("hazard_quantization_errors", [])
        values = [abs(record[key]) for record in errors for key in ("start_error_seconds", "end_error_seconds") if key in record]
        result["quantization_error_seconds"] = {"endpoints": len(values), "maximum_absolute": max(values, default=0.),
                                                "mean_absolute": sum(values) / len(values) if values else 0.}
        return result
    if stage == "lighting":
        return {key: deepcopy(targets[key]) for key in ("lighting_scheduler_coverage", "lighting_collisions",
                                                       "unsupported_lighting", "unscheduled_lighting") if key in targets}
    return {"raw_source_arcs": len(song.objects.get("arcs", [])),
            "unscheduled_arcs": deepcopy(targets.get("unscheduled_arcs", [])),
            "ignored_source_arcs": [deepcopy(record) for record in song.objects.get("ignored", []) if record.get("family") == "arcs"]}


def _sparse_source_coverage(song, stage):
    coverage = _source_coverage(song, stage)
    missing = {(record["family"], record["id"]): record for record in coverage["unrepresented_records"]}
    ignored = coverage["ignored_source_records"]
    for record in ignored:
        missing[(record["family"], record["id"])] = {
            "id": record["id"], "family": record["family"], "reason": record.get("reason", "unsupported source record")}
    coverage.update(reference_objects=coverage["reference_objects"] + len(ignored),
                    unrepresented_objects=len(missing), unrepresented_records=list(missing.values()))
    if stage == "hazards":
        diagnostics = song.diagnostics.get("targets", {})
        supported = diagnostics.get("hazard_supported_reference", {})
        if "fully_supported_wall_source_objects" in supported:
            raw = diagnostics.get("hazard_raw_source_counts", {}).get("walls", len(song.objects.get("walls", [])))
            complete = supported["fully_supported_wall_source_objects"]
            partial = supported["wall_source_objects"] - complete
            coverage["wall_source_coverage"] = {"raw_objects": raw, "fully_represented_objects": complete,
                                                 "partially_represented_objects": partial,
                                                 "unrepresented_objects": raw - complete - partial}
    return coverage


def _evaluate_sparse(dataset, stage, predictors, fitted, *, split, threshold, batch_size,
                     status, artifact_metadata, sources):
    """Score raw reference inputs and two actual-decoder replays, one song at a time."""
    from .calibration import OBJECTIVE_HEAD, replay_stage, teacher_forced_outputs

    options = dataset.manifest["data_settings"]
    raw_totals, positives = {name: {} for name in predictors}, {name: {} for name in predictors}
    decoded = {conditioning: {name: {} for name in predictors} for conditioning in ("teacher_forced", "autoregressive")}
    training_hashes = {digest for recording in fitted["training_recordings"] for digest in recording["source_sha256"] if digest}
    training_decoded = {recording["decoded_audio_sha256"] for recording in fitted["training_recordings"] if recording["decoded_audio_sha256"]}
    overlay = dataset.manifest.get("stage_splits", {}).get(stage, {})
    selected_songs, examples_count = [], 0
    object_heads = {"hazards": ("wall", "bombs"), "arcs": ("attach",), "lighting": ("events",)}[stage]
    for song in _selected_songs(dataset, split, stage):
        examples = song.examples.get(stage)
        item = {"song_id": song.base.song_id, "group_id": song.base.metadata.get("group_id"),
                "source_split": song.base.metadata["split"], "evaluated_split": split,
                "diagnostics": _sparse_diagnostics(song, stage), "source_coverage": _sparse_source_coverage(song, stage),
                "upstream_training_exposure": song.base.metadata.get("group_id") in overlay.get("moved_group_ids", []),
                "training_overlap": {"source_hash_match": any(source.get("sha256") in training_hashes for source in song.base.metadata.get("sources", [])),
                                     "decoded_audio_hash_match": song.base.metadata.get("decoded_sha256") in training_decoded,
                                     "arbitrary_reencodes": "unknown unless recording aliases were supplied during preparation"},
                "decoded_scores": {}}
        selected_songs.append(item)
        if examples is None or not len(examples.times):
            item["skip_reason"] = "no prepared stage queries"
            continue
        examples_count += len(examples.times)
        for name, selected in predictors.items():
            outputs = teacher_forced_outputs(song, stage, selected, dataset.vocabularies, options, batch_size)
            for start in range(0, len(examples.times), batch_size):
                selection = slice(start, start + batch_size)
                for head, target in examples.targets.items():
                    row = _head_scores(stage, head, target[selection], outputs[head][selection],
                                       examples.weights[head][selection], threshold)
                    _add_scores(raw_totals[name].setdefault(head, {}), row)
                    if head in object_heads:
                        predicted = outputs[head][selection].argmax(axis=-1) if head in ("wall", "attach") else outputs[head][selection] >= threshold
                        _add_positive(positives[name], {head: _positive_scores(target[selection], predicted,
                                      examples.weights[head][selection], categorical=head in ("wall", "attach"))})
            item["decoded_scores"][name] = {}
            for conditioning in decoded:
                replay = replay_stage(song, stage, selected, dataset.vocabularies, options, threshold,
                                      conditioning=conditioning, outputs=outputs if conditioning == "teacher_forced" else None,
                                      retain_objects=False)
                _add_positive(decoded[conditioning][name], replay["heads"])
                item["decoded_scores"][name][conditioning] = {
                    "heads": _summarize_positive(replay["heads"]), "diagnostics": replay["diagnostics"]}
            del outputs
    if not examples_count:
        raise ValueError(f"No {stage} examples in the requested {split} split")
    positive_scores = {name: _summarize_positive(heads) for name, heads in positives.items()}
    decoded_scores = {}
    for conditioning, predictions in decoded.items():
        decoded_scores[conditioning] = {}
        for name, heads in predictions.items():
            summarized = _summarize_positive(heads)
            decoded_scores[conditioning][name] = {
                "heads": summarized, "objective_head": OBJECTIVE_HEAD[stage], "objective": summarized[OBJECTIVE_HEAD[stage]],
                "positive_case_status": _positive_status(summarized),
                "all_empty_failure": any(values["all_empty_failure"] for values in summarized.values())}
    source_total = sum(song["source_coverage"]["reference_objects"] for song in selected_songs)
    missing_total = sum(song["source_coverage"]["unrepresented_objects"] for song in selected_songs)
    source_coverage = {"reference_objects": source_total, "unrepresented_objects": missing_total,
                       "represented_fraction": (source_total - missing_total) / source_total if source_total else None,
                       "interpretation": "Decoded metrics use supported reference queries; raw-source exclusions remain separate misses."}
    wall_coverage = [song["source_coverage"]["wall_source_coverage"] for song in selected_songs
                     if "wall_source_coverage" in song["source_coverage"]]
    if wall_coverage:
        source_coverage["wall_source_coverage"] = {key: sum(row[key] for row in wall_coverage) for key in wall_coverage[0]}
        source_coverage["represented_fraction_basis"] = "Any supported source-object portion; complete and partial wall coverage are reported separately."
    return {"kind": "v2_independent_stage_comparison", "status": status, "stage": stage, "evaluated_split": split,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "examples": examples_count,
            "conditioning": "fixed upstream reference; raw teacher-forced scores and decoded own-stage replays",
            "threshold": threshold, "scores": {name: _summarize_heads(total) for name, total in raw_totals.items()},
            "positive_sensitive_scores": positive_scores, "decoded_scores": decoded_scores,
            "positive_case_status": decoded_scores["autoregressive"]["model"]["positive_case_status"],
            "all_empty_failure": decoded_scores["autoregressive"]["model"]["all_empty_failure"],
            "baseline": fitted, "artifact": artifact_metadata or {}, "songs": selected_songs,
            "sources": sources, "reference_policy": "curated_bs_map_input_only_excluding_favorites",
            "source_coverage": source_coverage,
            "stage_split_fingerprint": overlay.get("fingerprint"), "upstream_exposure": deepcopy(overlay.get("provenance", {}).get("previous_upstream_exposure")),
            **preparation_reports(dataset),
            "dataset_identity": {key: dataset.manifest.get(key) for key in
                                 ("status", "encoder_identity", "normalization_sha256", "vocabularies_fingerprint", "data_settings")},
            "limitations": ["Reference agreement does not establish composed generation or playability.",
                            "Raw scores use fixed own-stage reference histories; both decoded scores apply the actual stage decoder.",
                            "Autoregressive replay changes only this stage's own history; upstream notes remain reference conditioned.",
                            "Zero predicted positives fail positive-case evaluation when supported positive reference objects exist.",
                            "Hazard wall metrics count cleared positive sections after merged-wall validation; NONE contributes no positive matches.",
                            "Arc replay disables probability thinning; application user controls and full generation remain separate checks."],
            "optimizer_steps": 0, **execution_identity()}


def evaluate_stage(dataset, stage, predictor, *, split="test", batch_size=128,
                   threshold=.5, baseline_names=("frequency", "none"), artifact_metadata=None,
                   baseline_dataset=None, evaluation_root=None, allow_synthetic=False):
    """Evaluate one stage with fixed upstream/reference histories and audio.

    This deliberately does not call the concept generator or the beat-pair
    comparison. Source objects remain available in the prepared dataset.
    """
    from .runner import batches
    status = (artifact_metadata or {}).get("status", getattr(predictor, "status", "untrained"))
    if status not in ("dummy", "untrained", "trained"):
        raise ValueError("Comparison artifact status must be dummy, untrained or trained")
    if stage not in NEW_STAGES or split not in ("test", "validation"):
        raise ValueError("Independent comparisons require one v2 stage and a held-out split")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Comparison threshold must be between zero and one")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Comparison batch_size must be a positive integer")
    validate_split_overlay(dataset.manifest)
    sources = validate_curated_dataset(dataset, split=split, evaluation_root=evaluation_root,
                                       allow_synthetic=allow_synthetic, stage=stage)
    if baseline_dataset is None:
        if not allow_synthetic or any(not source["synthetic"] for source in sources):
            raise ValueError("An explicit original training dataset is required for reference baseline statistics")
        baseline_dataset = dataset
    if compatibility(baseline_dataset, stage) != compatibility(dataset, stage):
        raise ValueError("Curated and training datasets must share frozen vocabulary, encoder, normalization and stage contracts")
    fitted = fit_reference_baselines(baseline_dataset, stage)
    baseline_names = tuple(name for name in baseline_names if name != "none" or stage not in ("positions", "arrows"))
    predictors = {"model": predictor, **{name: ReferenceBaseline(stage, dataset.vocabularies, fitted, name)
                                          for name in baseline_names}}
    if stage in ("hazards", "arcs", "lighting") and repaired(dataset.manifest.get("data_settings", {}), stage):
        return _evaluate_sparse(dataset, stage, predictors, fitted, split=split, threshold=threshold,
                                batch_size=batch_size, status=status, artifact_metadata=artifact_metadata, sources=sources)
    totals = {name: {} for name in predictors}
    scheduler_totals = {name: {} for name in predictors}
    positive_totals = {name: {} for name in predictors}
    origins = _hazard_origins(dataset, split) if stage == "hazards" else None
    origin_counts = {"scheduler": 0, "human_positive": 0}
    examples_count = 0
    for examples in batches(dataset, stage, split=split, batch_size=batch_size):
        examples_count += len(examples.times)
        batch_origins = [next(origins) for _ in examples.times] if origins is not None else None
        if batch_origins is not None:
            for origin in batch_origins:
                origin_counts[origin] += 1
        for name, selected in predictors.items():
            outputs = _predict(selected, examples.inputs)
            if set(outputs) != set(examples.targets):
                raise ValueError("Prediction heads disagree with independent reference targets")
            for head, target in examples.targets.items():
                row = _head_scores(stage, head, target, outputs[head], examples.weights[head], threshold)
                _add_scores(totals[name].setdefault(head, {}), row)
                if batch_origins is not None:
                    for origin, subset in (("scheduler", scheduler_totals), ("human_positive", positive_totals)):
                        mask = np.asarray(batch_origins) == origin
                        scores = _head_scores(stage, head, target, outputs[head],
                                              _subset_weights(examples.weights[head], mask), threshold)
                        _add_scores(subset[name].setdefault(head, {}), scores)
    if not examples_count:
        raise ValueError(f"No {stage} examples in the requested {split} split")
    selected_songs = []
    training_hashes = {digest for recording in fitted["training_recordings"] for digest in recording["source_sha256"] if digest}
    training_decoded = {recording["decoded_audio_sha256"] for recording in fitted["training_recordings"] if recording["decoded_audio_sha256"]}
    for song in dataset.songs:
        if song.base.metadata.get("split") == split:
            selected_songs.append({"song_id": song.base.song_id, "group_id": song.base.metadata.get("group_id"),
                                   "diagnostics": song.diagnostics, "source_coverage": _source_coverage(song, stage),
                                   "training_overlap": {"source_hash_match": any(source.get("sha256") in training_hashes for source in song.base.metadata.get("sources", [])),
                                                        "decoded_audio_hash_match": song.base.metadata.get("decoded_sha256") in training_decoded,
                                                        "arbitrary_reencodes": "unknown unless recording aliases were supplied during preparation"}})
    source_total = sum(song["source_coverage"]["reference_objects"] for song in selected_songs)
    missing_total = sum(song["source_coverage"]["unrepresented_objects"] for song in selected_songs)
    report = {"kind": "v2_independent_stage_comparison", "status": status, "stage": stage, "evaluated_split": split,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "examples": examples_count,
            "conditioning": "fixed upstream reference and preceding own-stage reference history",
            "threshold": threshold, "scores": {name: _summarize_heads(total) for name, total in totals.items()},
            "baseline": fitted, "artifact": artifact_metadata or {}, "songs": selected_songs,
            "sources": sources, "reference_policy": "curated_bs_map_input_only_excluding_favorites",
            "source_coverage": {"reference_objects": source_total, "unrepresented_objects": missing_total,
                                "represented_fraction": (source_total - missing_total) / source_total if source_total else None,
                                "interpretation": "Unrepresented source objects are misses outside conditional model scores, never negative labels."},
            **preparation_reports(dataset),
            "dataset_identity": {key: dataset.manifest.get(key) for key in
                                 ("status", "encoder_identity", "normalization_sha256", "vocabularies_fingerprint", "data_settings")},
            "limitations": ["Reference agreement does not establish composed generation or playability.",
                            "Unknown or uncovered source targets are masked and reported by preparation.",
                            "Conditional scores must be read with source coverage; unscheduled source objects remain misses.",
                            "Training overlap must be assessed from the recorded source hashes and groups."],
            "optimizer_steps": 0, **execution_identity()}
    if stage == "hazards":
        report["hazard_query_conditioning"] = "union of scheduler queries and native human-positive intervals/times; conditional geometry prediction"
        report["hazard_query_counts"] = origin_counts
        report["scheduler_only_scores"] = {name: _summarize_heads(total) for name, total in scheduler_totals.items()}
        report["native_positive_only_scores"] = {name: _summarize_heads(total) for name, total in positive_totals.items()}
        report["hazard_scheduler_coverage_before_augmentation"] = _scheduler_coverage(selected_songs)
        report["limitations"].append("Native positive queries supply human timestamps/intervals; their scores do not measure scheduler recall or end-to-end hazard quality.")
    return report


def evaluate_encoder(encoder, times, *, reference_predict=None, autoencoder=None, status="untrained"):
    """Compare the retained representation/network, with explicit injected models.

    ``reference_predict`` sees the identical normalized patches, making latent
    parity distinct from the intentionally changed fixed-song normalization.
    An optional reconstruction model is scored against its input patches.
    """
    times = np.asarray(times, np.float64)
    if status not in ("dummy", "untrained", "trained"):
        raise ValueError("Encoder comparison status must be dummy, untrained or trained")
    if times.ndim != 1 or not len(times):
        raise ValueError("Select at least one explicit encoder query time")
    patches = np.asarray(encoder.patches(times), np.float32)
    latents = np.asarray(encoder(times), np.float32)
    if patches.shape != (len(times), 24, 20, 1) or latents.shape != (len(times), 16):
        raise ValueError("Encoder comparison received incompatible shapes")
    if not np.isfinite(patches).all() or not np.isfinite(latents).all():
        raise ValueError("Nonfinite encoder comparison values")
    scores = {"queries": len(times), "latent_nonzero_fraction": float(np.count_nonzero(latents) / latents.size)}
    if reference_predict is not None:
        reference = np.asarray(reference_predict(patches), np.float32)
        if reference.shape != latents.shape or not np.isfinite(reference).all():
            raise ValueError("Invalid retained reference encoder output")
        scores.update(latent_max_abs_error=float(np.max(np.abs(latents - reference))),
                      latent_mae=float(np.mean(np.abs(latents - reference))))
    if autoencoder is not None:
        reconstructed = np.asarray(autoencoder(patches), np.float32)
        if reconstructed.shape != patches.shape or not np.isfinite(reconstructed).all():
            raise ValueError("Invalid autoencoder reconstruction")
        scores.update(reconstruction_mse=float(np.mean((reconstructed - patches) ** 2)),
                      zero_patch_reference_mse=float(np.mean(patches ** 2)))
    return {"kind": "v2_encoder_comparison", "status": status, "stage": "encoder", "scores": scores,
            "times_seconds": times, "encoder": getattr(encoder, "metadata", {}), "optimizer_steps": 0,
            "limitations": ["Latent parity checks identical patches; fixed-song normalization is an approved representation change.",
                            "Reconstruction quality does not establish mapping quality."], **execution_identity()}
