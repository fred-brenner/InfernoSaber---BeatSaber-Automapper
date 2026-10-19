"""Experimental v2 lighting: not yet implemented for production (optional/deferred).

Versioned, basic-only lighting supervision and inference projection.

Source objects are never rewritten. All times here are elapsed seconds; the
selected grid's unit is recorded explicitly in its frozen stage policy.
"""

from collections import Counter
from copy import deepcopy
import math
import time

import numpy as np


PROFILE = "default_basic_v1"
SCHEDULES = tuple({"unit": "beats", "step": step} for step in (.5, .25, .125, .0625, .03125)) + (
    {"unit": "seconds", "step": .04}, {"unit": "seconds", "step": .02})
POLICY_DEFAULTS = {"version": 2, "profile": PROFILE, "max_candidates": 50000,
                   "match_tolerance": .02, "coverage_target": .95}


def _integer(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    return int(value) if math.isfinite(value) and value == int(value) else None


def _meaningful(value):
    if isinstance(value, dict):
        return any(_meaningful(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_meaningful(item) for item in value)
    # Zero and false can be significant custom parameters (e.g. light IDs).
    return value is not None and value != ""


def _fields(record):
    raw = record.get("raw", record)
    raw = raw if isinstance(raw, dict) else {}
    kind = record.get("type", raw.get("et", raw.get("_type")))
    value = record.get("value", raw.get("i", raw.get("_value")))
    brightness = record.get("float_value", raw.get("f", raw.get("_floatValue", 1)))
    return raw, _integer(kind), _integer(value), brightness


def classify(record):
    """Return (canonical {type,value}, None), or (None, rejection reason)."""
    raw, kind, value, brightness = _fields(record)
    if any(_meaningful(raw.get(key)) for key in ("customData", "_customData")):
        return None, "custom_data_dependent"
    if kind is None or value is None:
        return None, "invalid_type_or_value"
    if value >= 2000000000:
        return None, "chroma_encoded_value"
    if kind not in (*range(5), 8, 9, 12, 13):
        return None, "unsupported_event_type"
    if isinstance(brightness, (bool, np.bool_)) or not isinstance(brightness, (int, float, np.integer, np.floating)) or brightness != 1:
        return None, "unsupported_float_value"
    if kind in (8, 9):
        value = 0
    elif not 0 <= value <= (12 if kind < 5 else 15):
        return None, "unsupported_event_value"
    return {"type": kind, "value": value}, None


def policy_from(options):
    """Read a frozen v2 lighting policy; absence identifies legacy artifacts."""
    value = options.get("stage_policies", {}).get("lighting")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Lighting stage policy must be an object")
    policy = deepcopy(value)
    if policy.get("version") != 2 or policy.get("profile") != PROFILE:
        raise ValueError("Unsupported lighting stage policy version/profile")
    for key in ("max_candidates", "match_tolerance", "coverage_target"):
        item = policy.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0:
            raise ValueError(f"Invalid lighting policy {key}")
    if _integer(policy["max_candidates"]) is None or policy["coverage_target"] > 1:
        raise ValueError("Invalid lighting candidate bound or coverage target")
    policy["max_candidates"] = int(policy["max_candidates"])
    schedule = policy.get("schedule")
    if schedule is not None and schedule not in SCHEDULES:
        raise ValueError("Unsupported lighting schedule")
    return policy


def _onsets(base, options):
    audio = base.arrays.get("features", base.arrays.get("audio"))
    frames = base.arrays.get("frame_times")
    if audio is None or frames is None or len(audio) < 3:
        return np.empty(0, np.float64)
    onset = np.asarray(audio)[:, -1]
    threshold = options.get("schedule", {}).get("lighting_onset_threshold", 1.)
    keep = ((onset[1:-1] >= threshold) & (onset[1:-1] > onset[:-2]) & (onset[1:-1] >= onset[2:]))
    times = np.asarray(frames, np.float64)[1:-1][keep]
    return times[(times >= 0) & (times < base.metadata["duration"])]


def _timeline(duration, bpm, onsets, policy):
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("Lighting schedules require finite positive duration and BPM")
    schedule = policy.get("schedule")
    if schedule is None:
        raise ValueError("Lighting policy requires a fitted schedule")
    step = schedule["step"] * (60. / bpm if schedule["unit"] == "beats" else 1.)
    count = int(math.ceil(duration / step))
    maximum = policy["max_candidates"]
    if count > maximum:
        raise ValueError(f"Lighting grid has {count} queries; exceeds max_candidates={maximum}")
    grid = np.arange(count, dtype=np.float64) * step
    grid = grid[grid < duration]
    # The independent tail query covers events after the last regular grid point.
    tail = np.nextafter(float(duration), 0.)
    times = np.unique(np.round(np.r_[grid, onsets], 9))
    times = times[(times >= 0) & (times < duration)]
    if not len(times) or abs(times[-1] - tail) > 1e-9:
        times = np.r_[times, tail]
    if len(times) > maximum:
        raise ValueError(f"Lighting schedule has {len(times)} queries; exceeds max_candidates={maximum}")
    return times


def candidates(song, options):
    """Frozen grid plus existing audio onsets and one partial-tail query."""
    base = getattr(song, "base", song)
    policy = policy_from(options)
    if policy is None:
        raise ValueError("Basic lighting requires an explicit versioned stage policy")
    times = _timeline(base.metadata["duration"], base.metadata["bpm"], _onsets(base, options), policy)
    return [{"kind": "lighting", "time": float(time)} for time in times]


def _snapshot(song, options):
    base = getattr(song, "base", song)
    if hasattr(song, "objects"):
        objects, diagnostics = song.objects, song.diagnostics
    else:
        from .data import read_objects

        objects, diagnostics = read_objects(base)
    records = []
    for record in [*objects.get("lighting", []), *(item for item in objects.get("ignored", []) if item.get("family") == "lighting")]:
        raw, kind, _, _ = _fields(record)
        when = record.get("time")
        if when is None:
            beat = raw.get("b", raw.get("_time"))
            if isinstance(beat, (int, float)) and not isinstance(beat, bool) and math.isfinite(beat):
                when = beat * 60. / base.metadata["bpm"]
        if when is not None and (not isinstance(when, (int, float)) or not math.isfinite(when)):
            when = None
        template, reason = classify(record)
        if record.get("family") == "lighting":
            template, reason = None, reason or record.get("reason", "invalid_source_record")
        records.append({"id": record.get("id"), "time": when, "type": kind, "template": template, "reason": reason})
    return {"song_id": base.song_id, "duration": base.metadata["duration"], "bpm": base.metadata["bpm"],
            "onsets": _onsets(base, options), "records": records,
            "complete": bool(diagnostics.get("source_objects_complete", False))}


def _assign(snapshot, times, tolerance, vocabulary=None):
    """Assign once, then mask ambiguity only for the affected query and type."""
    assigned, blocked, unsupported, unscheduled = {}, {}, [], []
    known = None if vocabulary is None else {(item["type"], item["value"]) for item in vocabulary}
    eligible = 0
    for record in snapshot["records"]:
        template = record["template"]
        eligible += template is not None
        when = record["time"]
        index = None
        if when is not None and len(times):
            right = int(np.searchsorted(times, when))
            choices = [item for item in (right - 1, right) if 0 <= item < len(times)]
            nearest = min(choices, key=lambda item: (abs(times[item] - when), item))
            if abs(times[nearest] - when) <= tolerance + 1e-10:
                index = nearest
        reason = record["reason"]
        if template is not None and known is not None and (template["type"], template["value"]) not in known:
            reason = "class_outside_training_vocabulary"
        if reason:
            unsupported.append({**record, "reason": reason})
            # An untimed malformed source may hide any occurrence of its type.
            indices = range(len(times)) if when is None else (() if index is None else (index,))
            for query in indices:
                blocked.setdefault(query, set()).add(record["type"])
        elif index is None:
            unscheduled.append(record["id"])
        else:
            assigned.setdefault(index, {}).setdefault(template["type"], []).append(record)
    collisions = []
    for query, types in assigned.items():
        for kind, records in types.items():
            if len({record["template"]["value"] for record in records}) > 1:
                blocked.setdefault(query, set()).add(kind)
                collisions.append({"query": query, "time": float(times[query]), "type": kind,
                                   "source_ids": [record["id"] for record in records]})
    effective, error, counts = [], 0., Counter()
    for query, types in assigned.items():
        for kind, records in types.items():
            if not snapshot["complete"] or kind in blocked.get(query, ()) or None in blocked.get(query, ()):
                continue
            for record in records:
                effective.append(record["id"])
                error += abs(float(times[query]) - record["time"])
                counts[(kind, record["template"]["value"])] += 1
    return {"assigned": assigned, "blocked": blocked, "collisions": collisions, "unsupported": unsupported,
            "unscheduled": unscheduled, "effective_ids": effective, "eligible": eligible, "counts": counts,
            "effective": len(effective), "timing_error": error}


def preflight(songs, options, skipped=None, *, progress=None):
    """Inspect TRAIN once; evaluate seven schedules using compact metadata only.

    The returned report is JSON serializable and includes a policy/vocabulary
    only when coverage and effective-positive fitting both succeed.
    """
    configured = policy_from(options)
    policy = {**POLICY_DEFAULTS, **(configured or {})}
    snapshots = []
    started = time.monotonic()
    total = len(songs) if hasattr(songs, "__len__") else None
    for index, song in enumerate(songs, 1):
        base = getattr(song, "base", song)
        if base.metadata.get("split") != "train":
            if progress is not None:
                progress({"phase": "lighting_sources", "index": index, "total": total, "song_id": base.song_id,
                          "elapsed_seconds": time.monotonic() - started})
            continue
        try:
            snapshots.append(_snapshot(song, options))
        except (ValueError, OSError, RuntimeError) as error:
            if skipped is not None:
                skipped.append({"song_id": base.song_id, "sources": deepcopy(base.metadata.get("sources", [])),
                                "phase": "lighting_preflight", "reason": f"{type(error).__name__}: {error}"})
        if progress is not None:
            progress({"phase": "lighting_sources", "index": index, "total": total, "song_id": base.song_id,
                      "elapsed_seconds": time.monotonic() - started})
    if not snapshots:
        return {"status": "no_readable_train_songs", "fit_split": "train", "training_song_ids": [],
                "coverage_target": policy["coverage_target"], "candidate_schedules": [], "unsupported_records": [],
                "reason": "Basic lighting fitting requires at least one readable TRAIN song"}
    reports, best_result = [], None
    for order, schedule in enumerate(SCHEDULES):
        selected = {**policy, "schedule": deepcopy(schedule)}
        counts, query_count, eligible, effective, error = Counter(), 0, 0, 0, 0.
        per_song, collisions, bounded = [], [], True
        for index, snapshot in enumerate(snapshots, 1):
            try:
                times = _timeline(snapshot["duration"], snapshot["bpm"], snapshot["onsets"], selected)
            except ValueError as exception:
                bounded = False
                per_song.append({"song_id": snapshot["song_id"], "reason": str(exception)})
                continue
            assignment = _assign(snapshot, times, policy["match_tolerance"])
            counts.update(assignment["counts"])
            query_count += len(times)
            eligible += assignment["eligible"]
            effective += assignment["effective"]
            error += assignment["timing_error"]
            per_song.append({"song_id": snapshot["song_id"], "queries": len(times),
                             "eligible_events": assignment["eligible"], "effective_events": assignment["effective"],
                             "unscheduled_events": len(assignment["unscheduled"])})
            collisions.extend({"song_id": snapshot["song_id"], **item} for item in assignment["collisions"])
            if progress is not None:
                progress({"phase": "lighting_schedule", "index": index, "total": len(snapshots),
                          "song_id": snapshot["song_id"], "schedule_index": order + 1, "schedule_total": len(SCHEDULES),
                          "schedule": deepcopy(schedule), "elapsed_seconds": time.monotonic() - started})
        coverage = effective / eligible if eligible else 0.
        report = {"schedule": deepcopy(schedule), "bounded": bounded, "queries": query_count,
                  "eligible_events": eligible, "effective_events": effective, "coverage": coverage,
                  "timing_error_seconds": error, "songs": per_song,
                  "collision_queries": len(collisions),
                  "collision_source_events": sum(len(item["source_ids"]) for item in collisions)}
        reports.append(report)
        score = (query_count, error, order)
        if bounded and coverage >= policy["coverage_target"] and (best_result is None or score < best_result[0]):
            best_result = (score, selected, counts, {**report, "collisions": collisions})
    report = {"status": "coverage_unmet", "fit_split": "train", "training_song_ids": sorted(item["song_id"] for item in snapshots),
              "coverage_target": policy["coverage_target"], "candidate_schedules": reports,
              "unsupported_records": [{"song_id": snapshot["song_id"], **record} for snapshot in snapshots
                                      for record in snapshot["records"] if record["reason"]]}
    if best_result is None:
        report["reason"] = f"No bounded lighting schedule reaches {policy['coverage_target']:.0%} effective TRAIN coverage"
        return report
    _, selected, counts, best = best_result
    ordered = sorted(counts, key=lambda item: (-counts[item], item))
    maximum = options.get("max_light_classes", len(ordered))
    kept = ordered[:maximum]
    vocabulary = [{"type": kind, "value": value} for kind, value in kept]
    report.update({"status": "ready" if vocabulary else "no_effective_positives", "policy": selected,
                   "vocabulary": vocabulary, "selected": best,
                   "class_positive_counts": [{"type": kind, "value": value, "count": counts[(kind, value)]} for kind, value in kept],
                   "dropped_class_limit": len(ordered) - len(kept)})
    observed = {(item["template"]["type"], item["template"]["value"]) for snapshot in snapshots
                for item in snapshot["records"] if item["template"] is not None}
    report["dropped_zero_positive_classes"] = [{"type": kind, "value": value} for kind, value in sorted(observed - set(counts))]
    return report


def fit_policy(songs, options, skipped=None):
    """Return (frozen policy, effective TRAIN vocabulary, preflight report)."""
    report = preflight(songs, options, skipped)
    if report["status"] != "ready":
        raise ValueError(report.get("reason", "Basic lighting has no effective positive TRAIN classes"))
    return deepcopy(report["policy"]), deepcopy(report["vocabulary"]), report


def decode(probabilities, vocab, threshold=.5):
    """Project scores to at most one value per type, then apply the threshold."""
    vocabulary = vocab["lighting"] if isinstance(vocab, dict) else vocab
    scores = np.asarray(probabilities)
    if scores.ndim < 1 or scores.shape[-1] != len(vocabulary) or not np.isfinite(scores).all():
        raise ValueError("Lighting scores must be finite and end in the frozen class axis")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("lighting_threshold must be between zero and one")
    projected = np.zeros(scores.shape, np.float32)
    for kind in sorted({item["type"] for item in vocabulary}):
        indices = [index for index, item in enumerate(vocabulary) if item["type"] == kind]
        winning = np.argmax(scores[..., indices], axis=-1)
        for local, index in enumerate(indices):
            projected[..., index] = (winning == local) & (scores[..., index] >= threshold)
    return projected


def decode_events(scores, vocab, threshold=.5):
    vocabulary = vocab["lighting"] if isinstance(vocab, dict) else vocab
    selected = decode(scores, vocabulary, threshold)
    if selected.ndim != 1:
        raise ValueError("decode_events expects one lighting query")
    return [deepcopy(vocabulary[index]) for index in np.flatnonzero(selected)]


def build_examples(song, state, vocab, options, encoder=None):
    """Build class-masked blocks with one causal history record per real query."""
    from .contracts import StageExamples, build_inputs

    policy = policy_from(options)
    queries = candidates(song, options)
    query_times = np.asarray([query["time"] for query in queries], np.float64)
    vocabulary = vocab["lighting"] if isinstance(vocab, dict) else vocab
    indices = {(item["type"], item["value"]): index for index, item in enumerate(vocabulary)}
    snapshot = _snapshot(song, options)
    assignment = _assign(snapshot, query_times, policy["match_tolerance"], vocabulary)
    conflicting = {item["query"]: set() for item in assignment["collisions"]}
    for item in assignment["collisions"]:
        conflicting[item["query"]].add(item["type"])
    length, count = options["lighting_block_length"], len(vocabulary)
    inputs, targets, weights, times, history = [], [], [], [], []
    context = {**state, "light_classes": count}
    for start in range(0, len(queries), length):
        block = query_times[start:start + length]
        candidate = {"time": float(block[0]), "times": block.tolist()}
        inputs.append(build_inputs("lighting", song, context, candidate, history, encoder, options))
        target = np.zeros((length, count), np.float32)
        weight = np.zeros_like(target)
        timeline = np.full(length, -1., np.float64)
        for offset, when in enumerate(block):
            query = start + offset
            timeline[offset] = when
            weight[offset] = float(snapshot["complete"])
            blocked = assignment["blocked"].get(query, set())
            for index, template in enumerate(vocabulary):
                if None in blocked or template["type"] in blocked:
                    weight[offset, index] = 0.
            for records in assignment["assigned"].get(query, {}).values():
                for record in records:
                    template = record["template"]
                    category = indices.get((template["type"], template["value"]))
                    if category is not None:
                        target[offset, category] = 1.
            # A conflicting type has no deterministic supported projection.
            for index, template in enumerate(vocabulary):
                if template["type"] in conflicting.get(query, ()):
                    target[offset, index] = 0.
            history.append({"time": float(when), "value": target[offset].tolist()})
        targets.append(target)
        weights.append(weight)
        times.append(timeline)
    arrays = {key: np.concatenate([item[key] for item in inputs], axis=0) for key in inputs[0]}
    examples = StageExamples(arrays, {"events": np.asarray(targets, np.float32)},
                             {"events": np.asarray(weights, np.float32)}, np.asarray(times, np.float64))
    diagnostics = {"lighting_policy": deepcopy(policy), "lighting_query_provenance": deepcopy(queries),
                   "unsupported_lighting": assignment["unsupported"], "unknown_lighting": [item["id"] for item in assignment["unsupported"]],
                   "unscheduled_lighting": assignment["unscheduled"], "lighting_collisions": assignment["collisions"],
                   "lighting_scheduler_coverage": {"eligible_events": assignment["eligible"], "effective_events": assignment["effective"],
                                                    "coverage": assignment["effective"] / assignment["eligible"] if assignment["eligible"] else 0.},
                   "lighting_history_queries": len(history)}
    return examples, diagnostics
