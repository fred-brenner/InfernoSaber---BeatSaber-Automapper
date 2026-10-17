"""Bounded TRAIN-only vocabularies and masked reference stage supervision."""

from collections import Counter
from copy import deepcopy
import hashlib
import json

import numpy as np

from .contracts import StageExamples, build_inputs, expansion_occupancy_notes, index_state


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _quantize(value, quantum):
    return float(round(round(float(value) / quantum) * quantum, 9))


def expansion_targets(song, options):
    """Describe only same-event inherited-cut copies or an explicit burst chain.

    Later retained events are never removed or reassigned to an earlier anchor.
    Unsupported patterns remain records with a reason and receive zero weight.
    """
    notes = song.objects["notes"]
    groups = {}
    for note in notes:
        groups.setdefault((note["event_index"], note["hand"]), []).append(note)
    targets = {}
    for group, records in groups.items():
        anchors = [record for record in records if record["retained"]]
        if len(anchors) != 1:
            continue
        anchor = anchors[0]
        extras = [record for record in records if not record["retained"]]
        chains = [record for record in song.objects["chains"] if record["hand"] == anchor["hand"]
                  and record["x"] == anchor["x"] and record["y"] == anchor["y"]
                  and record["d"] == anchor["d"] and abs(record["time"] - anchor["time"]) <= 1e-7]
        related_chains = [chain for chain in song.objects["chains"] if any(
            chain["hand"] == record["hand"] and (chain["x"], chain["y"], chain["d"]) == (record["x"], record["y"], record["d"])
            and abs(chain["time"] - record["time"]) <= 1e-7 for record in records)]
        event_time = float(song.base.arrays["event_times"][group[0]])
        reason, template = None, {"kind": "none"}
        if len(related_chains) != len(chains):
            reason = "chain_head_not_retained_anchor"
        elif any(record["d"] != anchor["d"] for record in extras) or (extras and anchor["d"] == 8):
            reason = "conflicting_or_unrestricted_expansion_directions"
        elif len(extras) > options["max_copies"] or len(chains) > 1 or (extras and chains):
            reason = "expansion_exceeds_single_bounded_template"
        elif extras:
            copies = [{"dx": int(record["x"] - anchor["x"]), "dy": int(record["y"] - anchor["y"]),
                       "dt": _quantize(record["time"] - event_time, options["expansion_time_quantum"])} for record in extras]
            copies.sort(key=lambda item: (item["dt"], item["dx"], item["dy"]))
            if any(item["dx"] == item["dy"] == 0 and item["dt"] == 0 for item in copies):
                reason = "duplicate_anchor_geometry"
            else:
                template = {"kind": "copies", "copies": copies}
        elif chains:
            chain = chains[0]
            span = _quantize(chain["end_time"] - event_time, options["chain_span_quantum"])
            if (chain["slices"] > options["max_chain_slices"] or not 0 < span <= options["max_chain_span_seconds"]
                    or event_time + span >= song.base.metadata["duration"]):
                reason = "chain_exceeds_bounded_template"
            else:
                template = {"kind": "chain", "dx": chain["tx"] - anchor["x"], "dy": chain["ty"] - anchor["y"],
                            "span": span, "slices": chain["slices"], "squish": float(chain["squish"])}
                chain["anchor_id"] = anchor["id"]
        targets[group] = {"template": template, "reason": reason, "anchor_id": anchor["id"],
                          "source_ids": [record["id"] for record in records],
                          "chain_ids": [record["id"] for record in chains]}
    return targets


def wall_template(record):
    return {"kind": "wall", **{key: record[key] for key in ("x", "y", "width", "height")}}


def light_template(record):
    return {key: record[key] for key in ("type", "value")}


def fit_vocabularies(songs, options, skipped=None):
    from .contracts import V2Song
    from .data import read_objects

    counters = {key: Counter() for key in ("multiplier", "walls", "lighting")}
    training = []
    for base in songs:
        if base.metadata.get("split") != "train":
            continue
        try:
            objects, diagnostics = read_objects(base)
        except (ValueError, OSError, RuntimeError) as error:
            if skipped is not None:
                skipped.append({"song_id": base.song_id, "sources": deepcopy(base.metadata.get("sources", [])),
                                "phase": "vocabulary", "reason": f"{type(error).__name__}: {error}"})
            continue
        training.append(base.song_id)
        song = V2Song(base, objects, diagnostics=diagnostics)
        for record in expansion_targets(song, options).values():
            if record["reason"] is None and record["template"]["kind"] != "none":
                counters["multiplier"][_key(record["template"])] += 1
        counters["walls"].update(_key(wall_template(record)) for record in objects["walls"])
        counters["lighting"].update(_key(light_template(record)) for record in objects["lighting"] if record["float_value"] == 1)
    if not training:
        raise ValueError("V2 vocabularies require at least one TRAIN song")
    result = {"fit_split": "train", "training_song_ids": sorted(training), "counts": {}}
    for family, maximum in (("multiplier", options["max_multiplier_templates"]), ("walls", options["max_wall_templates"]),
                            ("lighting", options["max_light_classes"])):
        ordered = sorted(counters[family], key=lambda key: (-counters[family][key], key))
        prefix = [] if family == "lighting" else [{"kind": "none"}]
        selected = ordered[:max(0, maximum - len(prefix))]
        result[family] = prefix + [json.loads(key) for key in selected]
        result["counts"][family] = {"observed": len(ordered), "kept": len(selected), "dropped": len(ordered) - len(selected)}
    if not result["lighting"]:
        result["lighting"] = [{"type": 0, "value": 0, "structural_placeholder": True}]
    result["fingerprint"] = hashlib.sha256(_key(result).encode()).hexdigest()
    return result


def validate_vocabularies(vocabularies):
    """Validate an explicit frozen TRAIN vocabulary before evaluation reuse."""
    if not isinstance(vocabularies, dict) or vocabularies.get("fit_split") != "train":
        raise ValueError("Frozen vocabularies must identify their TRAIN fitting split")
    identifiers = vocabularies.get("training_song_ids")
    if not isinstance(identifiers, list) or not identifiers or any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("Frozen vocabularies require training-song provenance")
    for family in ("multiplier", "walls", "lighting"):
        records = vocabularies.get(family)
        if not isinstance(records, list) or not records or any(not isinstance(value, dict) for value in records):
            raise ValueError(f"Invalid frozen vocabulary: {family}")
        if len({_key(value) for value in records}) != len(records):
            raise ValueError(f"Duplicate frozen categories: {family}")
        if family != "lighting" and records[0] != {"kind": "none"}:
            raise ValueError("Template vocabularies must reserve NONE at index zero")
    payload = deepcopy(vocabularies)
    digest = payload.pop("fingerprint", None)
    if not isinstance(digest, str) or hashlib.sha256(_key(payload).encode()).hexdigest() != digest:
        raise ValueError("Frozen vocabulary fingerprint mismatch")
    return True


def _stack(stage, rows, inputs, targets, weights, template_inputs):
    if inputs:
        input_arrays = {key: np.concatenate([item[key] for item in inputs], axis=0) for key in inputs[0]}
    else:
        input_arrays = {key: np.empty((0, *array.shape[1:]), array.dtype) for key, array in template_inputs.items()}
    return StageExamples(input_arrays, targets, weights, np.asarray(rows, np.float64))


def reference_state(song, vocab):
    """Upstream human notes with canonical retained times and source IDs."""
    notes = []
    for record in song.objects["notes"]:
        item = {key: record[key] for key in ("id", "time", "hand", "x", "y", "d")}
        if record["retained"]:
            item["time"] = float(song.base.arrays["event_times"][record["event_index"]])
            item["kind"] = "retained"
            notes.append(item)
    return {"notes": notes, "chains": [], "requested_nps": song.base.metadata["difficulty_nps"],
            "shares": song.base.arrays["intensity"], "light_classes": len(vocab["lighting"]),
            "multiplier_classes": len(vocab["multiplier"])}


def build_examples(song, vocab, options, encoder=None):
    from .scheduling import arc_candidates, hazard_candidates, lighting_candidates

    state = index_state(reference_state(song, vocab))
    expansions = expansion_targets(song, options)
    indices = {family: {_key(template): index for index, template in enumerate(vocab[family])} for family in ("multiplier", "walls", "lighting")}
    diagnostics = {"unsupported_expansions": [], "unsupported_chains": [], "unscheduled_walls": [], "unscheduled_bombs": [],
                   "unscheduled_arcs": [], "unscheduled_lighting": [], "unknown_lighting": []}
    output = {}
    event_times = song.base.arrays["event_times"]
    classes = song.base.arrays["note_classes"]
    for stage in ("positions", "arrows", "multiplier"):
        history, inputs, values, masks = [], [], [], []
        for event, when in enumerate(event_times):
            candidate = {"time": float(when), "count": int(np.count_nonzero(classes[event]))}
            inputs.append(build_inputs(stage, song, state, candidate, history, encoder, options))
            occupied = classes[event] != 0
            if stage == "positions":
                pair = np.where(occupied, (classes[event] - 1) // 9 + 1, 0).astype(np.int32)
                valid = np.ones(2, np.float32)
            elif stage == "arrows":
                pair = np.where(occupied, (classes[event] - 1) % 9, 9).astype(np.int32)
                valid = occupied.astype(np.float32)
            else:
                pair, valid = np.zeros(2, np.int32), occupied.astype(np.float32)
                for hand in (0, 1):
                    target = expansions.get((event, hand))
                    if target is None:
                        valid[hand] = 0
                        continue
                    category = indices["multiplier"].get(_key(target["template"]))
                    reason = target["reason"] or ("template_outside_training_vocabulary" if category is None else None)
                    if reason:
                        valid[hand] = 0
                        diagnostics["unsupported_expansions"].append({**deepcopy(target), "event_index": event, "hand": hand, "reason": reason})
                    else:
                        pair[hand] = category
            values.append(pair)
            masks.append(valid)
            if stage != "multiplier" or np.all(valid[occupied] > 0):
                history.append({"time": float(when), "value": pair.tolist()})
        template = build_inputs(stage, song, state, {"time": 0., "count": 0}, [], encoder, options)
        values, masks = np.asarray(values, np.int32).reshape(-1, 2), np.asarray(masks, np.float32).reshape(-1, 2)
        targets = {hand: values[:, index] for index, hand in enumerate(("red", "blue"))}
        weights = {hand: masks[:, index] for index, hand in enumerate(("red", "blue"))}
        # A direction head is ignored for an absent hand, but its stored target
        # remains a valid 0..8 category, independent of loss implementation.
        if stage == "arrows":
            targets = {hand: np.where(weights[hand] > 0, value, 0).astype(np.int32) for hand, value in targets.items()}
        output[stage] = _stack(stage, event_times, inputs, targets, weights, template)
    # Expansion context includes only source objects representable by the
    # approved multiplier target policy. Unsupported originals remain preserved.
    allowed = {identity for target in expansions.values() if target["reason"] is None and
               _key(target["template"]) in indices["multiplier"] for identity in target["source_ids"]}
    retained_ids = {item["id"] for item in state["notes"]}
    for record in song.objects["notes"]:
        if record["id"] in allowed and record["id"] not in retained_ids:
            state["notes"].append({key: record[key] for key in ("id", "time", "hand", "x", "y", "d")})
    state["notes"].sort(key=lambda item: (item["time"], item["hand"], item["id"]))
    allowed_chains = {identity for target in expansions.values() if target["reason"] is None and
                      _key(target["template"]) in indices["multiplier"] for identity in target["chain_ids"]}
    state["chains"] = [record for record in song.objects["chains"] if record["id"] in allowed_chains]
    index_state(state)
    diagnostics["unsupported_chains"] = [{"id": record["id"], "reason": "unmatched_or_unsupported_expansion_anchor"}
                                         for record in song.objects["chains"] if record["id"] not in allowed_chains]
    duration, bpm = song.base.metadata["duration"], song.base.metadata["bpm"]
    tolerance = options["lighting_match_tolerance"]
    complete = song.diagnostics["source_objects_complete"]
    scheduled = hazard_candidates(expansion_occupancy_notes(state), duration, bpm, options["schedule"])
    queries = [{**query, "origin": "scheduler", "source_ids": []} for query in scheduled]
    scheduled_matches = {"walls": set(), "bombs": set()}
    for family, kind in (("walls", "wall"), ("bombs", "bomb")):
        for record in song.objects[family]:
            end = record.get("end_time", record["time"])
            matches = [query for query in queries if query["kind"] == kind and query["origin"] != "source"
                       and abs(query["time"] - record["time"]) <= tolerance
                       and abs(query["end_time"] - end) <= tolerance]
            if matches:
                query = min(matches, key=lambda item: abs(item["time"] - record["time"]) + abs(item["end_time"] - end))
                query["origin"] = "scheduler_and_source"
                query["source_ids"].append(record["id"])
                scheduled_matches[family].add(record["id"])
            else:
                matches = [query for query in queries if query["kind"] == kind and query["time"] == record["time"] and query["end_time"] == end]
                if matches:
                    matches[0]["source_ids"].append(record["id"])
                else:
                    queries.append({"kind": kind, "time": record["time"], "end_time": end,
                                    "origin": "source", "source_ids": [record["id"]]})
    queries.sort(key=lambda query: (query["time"], query["kind"], query["end_time"]))
    diagnostics["hazard_query_provenance"] = deepcopy(queries)
    song.diagnostics["hazard_query_origins"] = ["human_positive" if query["origin"] == "source" else "scheduler" for query in queries]
    song.diagnostics["hazard_query_provenance"] = deepcopy(queries)
    diagnostics["hazard_scheduler_coverage"] = {family: {"source_objects": len(song.objects[family]),
                                                         "matched_source_objects": len(scheduled_matches[family]),
                                                         "unmatched_source_ids": [record["id"] for record in song.objects[family]
                                                                                  if record["id"] not in scheduled_matches[family]]}
                                                for family in ("walls", "bombs")}
    diagnostics["hazard_query_policy"] = "source_positive_queries_plus_inference_scheduler_queries"
    song.diagnostics["hazard_scheduler_coverage"] = deepcopy(diagnostics["hazard_scheduler_coverage"])
    history, inputs, walls, bombs, wall_weights, bomb_weights, used_walls, used_bombs = [], [], [], [], [], [], set(), set()
    for candidate in queries:
        inputs.append(build_inputs("hazards", song, state, candidate, history, encoder, options))
        wall, cells, ww, bw = 0, np.zeros(12, np.float32), 0., 0.
        if candidate["kind"] == "wall":
            matches = [record for record in song.objects["walls"] if abs(record["time"] - candidate["time"]) <= tolerance
                       and abs(record["end_time"] - candidate["end_time"]) <= tolerance]
            ww = float(complete)
            overlaps = [record for record in song.objects["walls"] if record["time"] < candidate["end_time"]
                        and record["end_time"] > candidate["time"]]
            if overlaps and not matches:
                ww = 0.
            if len(matches) > 1:
                ww = 0.
            elif matches:
                record = matches[0]
                category = indices["walls"].get(_key(wall_template(record)))
                if category is None:
                    ww = 0.
                else:
                    wall = category
                    used_walls.add(record["id"])
        else:
            bw = float(complete)
            for record in song.objects["bombs"]:
                if abs(record["time"] - candidate["time"]) <= tolerance:
                    cells[4 * record["y"] + record["x"]] = 1
                    used_bombs.add(record["id"])
        for ignored in song.objects["ignored"]:
            if "time" not in ignored or abs(ignored["time"] - candidate["time"]) <= tolerance:
                if ignored.get("family") == "walls":
                    ww = 0.
                elif ignored.get("family") in ("bombs", "notes"):
                    bw = 0.
        walls.append(wall); bombs.append(cells); wall_weights.append(ww); bomb_weights.append(bw)
        if (candidate["kind"] == "wall" and ww > 0) or (candidate["kind"] == "bomb" and bw > 0):
            history.append({"time": candidate["time"], "value": [wall, *cells.tolist()]})
    template = build_inputs("hazards", song, state, {"time": 0., "end_time": 0.}, [], encoder, options)
    output["hazards"] = _stack("hazards", [query["time"] for query in queries], inputs,
                                 {"wall": np.asarray(walls, np.int32), "bombs": np.asarray(bombs, np.float32).reshape(-1, 12)},
                                 {"wall": np.asarray(wall_weights, np.float32), "bombs": np.asarray(bomb_weights, np.float32)}, template)
    diagnostics["unscheduled_walls"] = [record["id"] for record in song.objects["walls"] if record["id"] not in used_walls]
    diagnostics["unscheduled_bombs"] = [record["id"] for record in song.objects["bombs"] if record["id"] not in used_bombs]
    diagnostics["hazard_positive_query_coverage"] = {"walls": len(used_walls), "bombs": len(used_bombs),
                                                      "scheduler_queries": len(scheduled), "augmented_queries": len(queries),
                                                      "source_only_queries": sum(query["origin"] == "source" for query in queries)}
    queries = arc_candidates(state["notes"], options["schedule"])
    source_pairs = {}
    for arc in song.objects["arcs"]:
        pair = []
        for tail in (False, True):
            when, x, y, direction = ((arc["end_time"], arc["tx"], arc["ty"], arc["td"]) if tail else
                                     (arc["time"], arc["x"], arc["y"], arc["d"]))
            matches = [record for record in state["notes"] if record["hand"] == arc["hand"] and
                       (record["x"], record["y"], record["d"]) == (x, y, direction) and abs(record["time"] - when) <= tolerance]
            if len(matches) == 1:
                pair.append(matches[0]["id"])
        if len(pair) == 2:
            source_pairs.setdefault(tuple(pair), []).append(arc)
    history, inputs, labels, shapes, anchors, attach_weights, shape_weights, used_arcs = [], [], [], [], [], [], [], set()
    for candidate in queries:
        inputs.append(build_inputs("arcs", song, state, candidate, history, encoder, options))
        matches = source_pairs.get(tuple(candidate["endpoint_ids"]), [])
        attach, shape, anchor, weight, shape_weight = 0, [0., 0.], 0, float(complete), 0.
        if len(matches) > 1:
            weight = 0.
        elif matches:
            record = matches[0]
            lower, upper = options["arc_shape_bounds"]
            if lower <= record["mu"] <= upper and lower <= record["tmu"] <= upper:
                attach, shape, anchor, shape_weight = 1, [record["mu"], record["tmu"]], record["anchor"], 1.
                used_arcs.add(record["id"])
            else:
                weight = 0.
        if any(record.get("family") == "arcs" and ("time" not in record or abs(record["time"] - candidate["time"]) <= tolerance)
               for record in song.objects["ignored"]):
            weight, shape_weight = 0., 0.
        labels.append(attach); shapes.append(shape); anchors.append(anchor); attach_weights.append(weight); shape_weights.append(shape_weight)
        if weight:
            history.append({"time": candidate["time"], "value": [attach, *shape, anchor]})
    # No synthetic endpoints are invented for an empty arc schedule.
    if inputs:
        template = inputs[0]
    else:
        template = {"audio": np.empty((1, 16), np.float32), "history": np.empty((1, options["history_length"], 4), np.float32),
                    "history_age": np.empty((1, options["history_length"], 1), np.float32), "history_mask": np.zeros((1, options["history_length"]), bool),
                    "conditions": np.empty((1, 6), np.float32), "endpoints": np.empty((1, 2), np.int32),
                    "endpoint_features": np.empty((1, 2, 3), np.float32), "gap": np.empty((1, 1), np.float32)}
    output["arcs"] = _stack("arcs", [query["time"] for query in queries], inputs,
                              {"attach": np.asarray(labels, np.int32), "shape": np.asarray(shapes, np.float32).reshape(-1, 2), "anchor": np.asarray(anchors, np.int32)},
                              {"attach": np.asarray(attach_weights, np.float32), "shape": np.asarray(shape_weights, np.float32), "anchor": np.asarray(shape_weights, np.float32)}, template)
    diagnostics["unscheduled_arcs"] = [record["id"] for record in song.objects["arcs"] if record["id"] not in used_arcs]
    queries = lighting_candidates(song, settings=options["schedule"])
    assigned = {}
    ignored_lighting = set()
    mask_all_lighting = False
    query_times = np.asarray([query["time"] for query in queries])
    for record in song.objects["lighting"]:
        index = int(np.argmin(np.abs(query_times - record["time"]))) if len(query_times) else -1
        if index >= 0 and abs(query_times[index] - record["time"]) <= tolerance:
            assigned.setdefault(index, []).append(record)
        else:
            diagnostics["unscheduled_lighting"].append(record["id"])
    for record in song.objects["ignored"]:
        if record.get("family") != "lighting":
            continue
        if "time" not in record:
            mask_all_lighting = True
        elif len(query_times):
            index = int(np.argmin(np.abs(query_times - record["time"])))
            if abs(query_times[index] - record["time"]) <= tolerance:
                ignored_lighting.add(index)
    history, inputs, labels, masks, times = [], [], [], [], []
    length, count = options["lighting_block_length"], len(vocab["lighting"])
    for start in range(0, len(queries), length):
        block = queries[start:start + length]
        candidate = {"time": block[0]["time"], "times": [query["time"] for query in block]}
        inputs.append(build_inputs("lighting", song, state, candidate, history, encoder, options))
        target, weight, time = np.zeros((length, count), np.float32), np.zeros(length, np.float32), np.full(length, -1., np.float64)
        pending = []
        for offset, query in enumerate(block):
            time[offset] = query["time"]
            weight[offset] = float(complete and not vocab["lighting"][0].get("structural_placeholder", False))
            if mask_all_lighting or start + offset in ignored_lighting:
                weight[offset] = 0.
            for record in assigned.get(start + offset, []):
                category = indices["lighting"].get(_key(light_template(record)))
                if category is None or record["float_value"] != 1:
                    weight[offset] = 0.
                    diagnostics["unknown_lighting"].append(record["id"])
                else:
                    target[offset, category] = 1.
            if weight[offset]:
                pending.append({"time": query["time"], "value": target[offset].tolist()})
        history.extend(pending)
        labels.append(target); masks.append(weight); times.append(time)
    template = build_inputs("lighting", song, state, {"time": 0., "times": []}, [], encoder, options)
    output["lighting"] = _stack("lighting", np.asarray(times, np.float64).reshape(-1, length), inputs,
                                  {"events": np.asarray(labels, np.float32).reshape(-1, length, count)},
                                  {"events": np.asarray(masks, np.float32).reshape(-1, length)}, template)
    song.diagnostics["targets"] = diagnostics
    song.diagnostics["expansion_relationships"] = [{"event_index": event, "hand": hand, **deepcopy(target)} for (event, hand), target in expansions.items()]
    song.diagnostics["coverage"] = {stage: {"examples": len(examples.times),
                                           "supervised_heads": {key: int(np.count_nonzero(value)) for key, value in examples.weights.items()}}
                                    for stage, examples in output.items()}
    return output
