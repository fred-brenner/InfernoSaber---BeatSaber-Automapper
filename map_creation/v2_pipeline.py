"""Production v2 inference with chronological histories and reviewed cleanup.

Training targets and source beatmaps are never inputs. The concept command keeps
its independent review stops; this path uses concrete application postprocessors.
"""

from copy import deepcopy
from numbers import Real
from time import perf_counter

import numpy as np

from tools.config import config
from training.beat_models.contracts import CORE, FRAME_SECONDS, frame_conditions, intensity_inputs, timing_window
from training.pipeline_v2.contracts import build_inputs, expansion_occupancy_notes, index_state
from training.pipeline_v2.pipeline import (PipelineResult, _decode_positions, _distribution,
                                           _expand, _multilabel, _output, _predict)
from training.pipeline_v2.scheduling import hazard_candidates
from training.pipeline_v2.arc_policy import arc_candidates, history_value
from training.pipeline_v2.policies import repaired


def _log(logger, message):
    if logger is None:
        print(message, flush=True)
    else:
        logger(message)


def _progress(logger, stage, index, total):
    if index == 0 or index + 1 == total or (index + 1) % 250 == 0:
        _log(logger, f"V2 {stage}: {index + 1}/{total}")


def _postprocess_intensity(song, shares, requested_nps):
    """Shape normalized shares and separate their overall conditioning gain."""
    options = {}
    for key in ("low_boost", "high_reduction", "scale", "adjustment_strength"):
        name = f"v2_intensity_{key}"
        value = getattr(config, name)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0 or (key != "scale" and value > 1)):
            bounds = "nonnegative" if key == "scale" else "within [0, 1]"
            raise ValueError(f"{name} must be a finite numeric value {bounds}")
        options[key] = float(value)
    strength = options["adjustment_strength"]
    adjusted, density = shares.copy(), float(requested_nps)
    if strength:
        uniform = np.asarray(song.arrays["bin_durations"], np.float64) / song.metadata["duration"]
        target = (shares + options["low_boost"] * np.maximum(uniform - shares, 0)
                  - options["high_reduction"] * np.maximum(shares - uniform, 0))
        target /= target.sum()
        mass = (1 - strength) * shares + strength * options["scale"] * target
        total = float(mass.sum())
        adjusted = (mass / total).astype(np.float32) if total else shares.copy()
        density *= total
        if not np.isfinite(density):
            raise ValueError("V2 intensity adjustment produced non-finite conditioning NPS")
    return adjusted, density, {**options, "requested_nps": float(requested_nps), "conditioning_nps": density}


def infer_beats(song, requested_nps, predictors, batch_size=8, intensity=None):
    """Batch independent timing windows without reading any target arrays."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("V2 timing batch size must be a positive integer")
    bins, frames = len(song.arrays["bin_durations"]), len(song.arrays["frame_times"])
    started = perf_counter()
    if intensity is None:
        intensity = _output(_predict(predictors["intensity"], intensity_inputs(song)), "shares")
    raw_shares = _distribution(np.asarray(intensity).reshape(1, bins), (1, bins), "intensity")[0]
    shares, conditioning_nps, adjustment = _postprocess_intensity(song, raw_shares, requested_nps)
    intensity_seconds = perf_counter() - started
    started = perf_counter()
    conditions = frame_conditions(song, shares, conditioning_nps)
    probabilities = np.zeros((frames, 3), np.float32)
    probabilities[:, 0] = 1
    if conditioning_nps > 0:
        starts = list(range(0, frames, CORE))
        for offset in range(0, len(starts), batch_size):
            selected = starts[offset:offset + batch_size]
            windows = [timing_window(song, conditions, start)[0] for start in selected]
            inputs = {key: np.concatenate([window[key] for window in windows]) for key in windows[0]}
            output = _distribution(_output(_predict(predictors["timing"], inputs), "counts"),
                                   (len(selected), CORE, 3), "timing")
            for row, start in enumerate(selected):
                valid = min(CORE, frames - start)
                probabilities[start:start + valid] = output[row, :valid]
    counts = probabilities.argmax(axis=-1).astype(np.int32)
    event_frames = np.flatnonzero(counts)
    return {"raw_intensity": raw_shares.copy(), "intensity": shares, "conditioning_nps": conditioning_nps,
            "conditions": conditions, "frame_probabilities": probabilities,
            "counts": counts, "event_frames": event_frames,
            "event_times": song.arrays["frame_times"][event_frames].copy(),
            "_intensity_postprocessing": adjustment,
            "_stage_seconds": {"intensity": intensity_seconds, "timing": perf_counter() - started}}


def _biased(outputs, classes, factor, rng):
    """Mix bounded random preferences into positions/arrows only."""
    result = {}
    for hand in ("red", "blue"):
        values = _distribution(outputs[hand], (1, classes), f"{hand} categories")
        if factor:
            preference = rng.random(values.shape).astype(np.float32)
            preference /= preference.sum(axis=-1, keepdims=True)
            values = (1 - factor) * values + factor * preference
            values /= values.sum(axis=-1, keepdims=True)
        result[hand] = values
    return result


def _inputs(stage, song, state, candidate, history, encoder, settings):
    length = max(settings["history_length"], settings["lighting_block_length"])
    prior = [record for record in history if record["time"] < candidate["time"]][-length:]
    return build_inputs(stage, song, state, candidate, prior, encoder=encoder, settings=settings)


def _history(history, record, length):
    history.append(record)
    # Equal-time queries share strictly preceding history. Keep enough previous
    # records for the two-hand/same-time query groups used by the schedulers.
    if len(history) > length + 4:
        del history[:len(history) - length - 4]


def _rejections(total, diagnostics):
    for key, value in diagnostics.items():
        if key.startswith("rejected_") and isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value


def _atomic_expand(state, anchor, template, duration, rejected):
    """Reject a whole invalid/duplicate expansion and preserve every anchor."""
    if template.get("kind", "none") == "none":
        return True
    trial = {"notes": list(state["notes"]), "chains": list(state["chains"])}
    errors = []
    _expand(trial, anchor, template, duration, errors)
    added = trial["notes"][len(state["notes"]):]
    occupied = {(round(note["time"], 8), note["x"], note["y"]) for note in state["notes"]}
    for note in added:
        key = (round(note["time"], 8), note["x"], note["y"])
        if key in occupied:
            errors.append({"anchor_id": anchor["id"], "reason": "duplicate expansion occupancy"})
            break
        occupied.add(key)
    for chain in trial["chains"][len(state["chains"]):]:
        if any(existing["anchor_id"] == chain["anchor_id"] for existing in state["chains"]):
            errors.append({"anchor_id": anchor["id"], "reason": "duplicate chain anchor"})
    if errors:
        rejected.extend(errors)
        return False
    next_id = max((note["id"] for note in state["notes"]), default=-1) + 1
    for index, note in enumerate(added):
        note["id"] = next_id + index
    state["notes"].extend(added)
    state["chains"].extend(trial["chains"][len(state["chains"]):])
    return True


def _candidates(song, values):
    duration = song.metadata["duration"]
    previous = -1.
    result = []
    for candidate in values:
        now, count = float(candidate["time"]), candidate["count"]
        if not np.isfinite(now) or not previous < now < duration or isinstance(count, bool) or count not in (1, 2):
            raise ValueError("Postprocessed timing must be ordered, inside the song, with count one or two")
        if now < 0:
            raise ValueError("Negative postprocessed timing")
        frame = min(int(np.floor(now / FRAME_SECONDS + .5)), len(song.arrays["frame_times"]) - 1)
        result.append({**candidate, "time": now, "frame": frame, "count": int(count)})
        previous = now
    return result


def _validate_state(state, duration):
    """Check the final object contract after all mutating postprocessors."""
    identities = {}
    all_ids = {note["id"] for note in state["notes"]}
    for note in state["notes"]:
        values = [note[key] for key in ("time", "x", "y", "hand", "d")]
        if (not np.isfinite(values).all() or not 0 <= note["time"] < duration
                or any(not float(note[key]).is_integer() for key in ("x", "y", "hand", "d"))
                or not 0 <= note["x"] < 4 or not 0 <= note["y"] < 3
                or note["hand"] not in (0, 1) or not 0 <= note["d"] < 9
                or note["id"] in identities):
            raise ValueError("Invalid final v2 note or duplicate note ID")
        identities[note["id"]] = note
        if note.get("kind") == "copy" and note.get("anchor_id") not in all_ids:
            raise ValueError("A copied note has no existing anchor")
    for chain in state["chains"]:
        if (chain["anchor_id"] not in identities or not 0 <= chain["time"] < chain["end_time"] < duration
                or not 0 <= chain["tx"] < 4 or not 0 <= chain["ty"] < 3
                or not isinstance(chain["slices"], int) or chain["slices"] < 2
                or not np.isfinite(chain.get("squish", 1.))):
            raise ValueError("Invalid final v2 chain")
    for wall in state["walls"]:
        if (not 0 <= wall["time"] < wall["end_time"] <= duration
                or not np.isfinite([wall[key] for key in ("x", "y", "width", "height")]).all()
                or wall["width"] <= 0 or wall["height"] <= 0):
            raise ValueError("Invalid final v2 wall")
    for bomb in state["bombs"]:
        if (not 0 <= bomb["time"] < duration or not 0 <= bomb["x"] < 4 or not 0 <= bomb["y"] < 3
                or not float(bomb["x"]).is_integer() or not float(bomb["y"]).is_integer()):
            raise ValueError("Invalid final v2 bomb")
    for arc in state["arcs"]:
        ids = arc.get("endpoint_ids", [])
        if len(ids) != 2 or any(identity not in identities for identity in ids):
            raise ValueError("A final v2 arc has missing endpoints")
        head, tail = (identities[identity] for identity in ids)
        if head["hand"] != tail["hand"] or head["time"] >= tail["time"]:
            raise ValueError("Invalid final v2 arc endpoints")
    for event in state["lighting"]:
        if not 0 <= event["time"] < duration or not np.isfinite([event["type"], event["value"]]).all():
            raise ValueError("Invalid final v2 lighting event")


def generate(song, requested_nps, predictors, encoder, settings, vocabularies, *,
             lighting_generator, audio_path=None, logger=None, postprocessors=None, random_factor=None,
             intensity=None, rng=None):
    """Compose seven v2 stages, retained v1 lighting, and application validators."""
    if isinstance(requested_nps, bool) or not np.isfinite(requested_nps) or requested_nps < 0:
        raise ValueError("Requested retained NPS must be finite and nonnegative")
    if postprocessors is None:
        from . import v2_postprocessing as postprocessors
    factor = getattr(config, "random_note_map_factor", 0.) if random_factor is None else random_factor
    if isinstance(factor, bool) or not np.isfinite(factor) or factor < 0:
        raise ValueError("Random factor must be finite and nonnegative")
    factor = min(float(factor), 1.)
    rng = np.random.default_rng() if rng is None else rng
    duration, bpm = float(song.metadata["duration"]), float(song.metadata["bpm"])
    options = deepcopy(settings)
    options["vocabularies"] = vocabularies
    schedule = options["schedule"]
    status = "dummy" if all(getattr(predictor, "status", "trained") == "dummy"
                            for stage, predictor in predictors.items() if stage != "lighting") else "trained"
    result = PipelineResult(status, {key: [] for key in ("notes", "chains", "walls", "bombs", "arcs", "lighting")},
                            diagnostics={"requested_retained_nps": float(requested_nps),
                                         "duration_seconds": duration, "bpm": bpm,
                                         "rejected_expansions": [], "export_ready": False,
                                         "random_factor": factor, "scheduler": deepcopy(schedule)})
    state = result.state
    raw = result.diagnostics["raw_decisions"] = {
        "position_events": 0, "position_notes": 0, "arrow_notes": 0,
        "multiplier_non_none": 0, "wall_proposals": 0, "bomb_proposals": 0,
        "arc_attachments": 0, "lighting_commands": 0}
    _log(logger, "V2 intensity and timing")
    beats = infer_beats(song, requested_nps, predictors,
                       getattr(config, "v2_timing_batch_size", 8), intensity=intensity)
    beat_seconds = beats.pop("_stage_seconds")
    result.diagnostics["intensity_postprocessing"] = beats.pop("_intensity_postprocessing")
    state.update(beats, requested_nps=float(requested_nps),
                 multiplier_classes=len(vocabularies["multiplier"]))
    cleanup_started = perf_counter()
    candidates, diagnostics = postprocessors.postprocess_timing(song, beats, audio_path, logger=logger)
    candidates = _candidates(song, candidates)
    state.update(raw_event_times=beats["event_times"], raw_event_frames=beats["event_frames"], raw_counts=beats["counts"],
                 event_times=np.asarray([candidate["time"] for candidate in candidates], np.float64),
                 event_frames=np.asarray([candidate["frame"] for candidate in candidates], np.int64))
    state["counts"] = np.zeros_like(beats["counts"])
    for candidate in candidates:
        state["counts"][candidate["frame"]] = candidate["count"]
    result.diagnostics["timing_postprocessing"] = diagnostics
    result.stages["intensity"] = {"bins": len(beats["intensity"]), "elapsed_seconds": beat_seconds["intensity"]}
    result.stages["timing"] = {"raw_events": len(beats["event_frames"]), "events": len(candidates),
                                "raw_retained_notes": int(beats["counts"].sum()),
                                "elapsed_seconds": beat_seconds["timing"] + perf_counter() - cleanup_started}
    started = perf_counter()
    encoder(np.asarray([candidate["time"] for candidate in candidates], np.float64))
    history, length = [], options["history_length"]
    for index, candidate in enumerate(candidates):
        output = _predict(predictors["positions"], _inputs("positions", song, state, candidate, history, encoder, options))
        pair = _decode_positions(_biased(output, 13, factor, rng), candidate["count"])
        for hand, category in enumerate(pair):
            if category:
                cell = category - 1
                state["notes"].append({"id": len(state["notes"]), "time": candidate["time"], "hand": hand,
                                       "x": cell % 4, "y": cell // 4, "kind": "retained"})
        _history(history, {"time": candidate["time"], "value": pair}, length)
        _progress(logger, "positions", index, len(candidates))
    result.stages["positions"] = {"events": len(candidates), "retained_notes": len(state["notes"]),
                                   "elapsed_seconds": perf_counter() - started}
    raw.update(position_events=len(candidates), position_notes=len(state["notes"]))
    started = perf_counter()
    index_state(state)
    history = []
    for index, candidate in enumerate(candidates):
        output = _biased(_predict(predictors["arrows"], _inputs("arrows", song, state, candidate, history, encoder, options)), 9, factor, rng)
        pair = [9, 9]
        for hand in (0, 1):
            times, track = state["_tracks"][hand]
            location = int(np.searchsorted(times, candidate["time"]))
            if location < len(track) and track[location]["time"] == candidate["time"]:
                direction = int(output[("red", "blue")[hand]][0].argmax())
                track[location]["d"] = direction
                pair[hand] = direction
        _history(history, {"time": candidate["time"], "value": pair}, length)
        _progress(logger, "arrows", index, len(candidates))
    result.stages["arrows"] = {"directed_notes": len(state["notes"])}
    raw["arrow_notes"] = len(state["notes"])
    state["raw_model_notes"] = np.asarray([[note[key] for key in ("id", "time", "hand", "x", "y", "d")]
                                           for note in state["notes"]], np.float64).reshape(-1, 6)
    result.diagnostics["raw_model_note_columns"] = ["id", "time_seconds", "hand", "x", "y", "direction"]
    state["notes"], diagnostics = postprocessors.postprocess_notes(state["notes"],
                                             [candidate["time"] for candidate in candidates], logger=logger)
    result.diagnostics["note_postprocessing"] = diagnostics
    result.stages["arrows"]["elapsed_seconds"] = perf_counter() - started
    started = perf_counter()
    retained = deepcopy([note for note in state["notes"] if note.get("kind") == "retained"])
    upstream = index_state({**state, "notes": retained, "chains": []})
    by_time = {}
    for note in retained:
        by_time.setdefault(note["time"], []).append(note)
    multiplier_candidates = [{"time": time, "count": len(notes)} for time, notes in sorted(by_time.items())]
    encoder(np.asarray(list(sorted(by_time)), np.float64))
    history, expansion_rejections = [], {}
    _log(logger, "V2 multiplier")
    for index, candidate in enumerate(multiplier_candidates):
        output = _predict(predictors["multiplier"], _inputs("multiplier", song, upstream, candidate, history, encoder, options))
        pair = [0, 0]
        for anchor in by_time[candidate["time"]]:
            hand = ("red", "blue")[anchor["hand"]]
            category = int(_distribution(output[hand], (1, len(vocabularies["multiplier"])), "expansion")[0].argmax())
            raw["multiplier_non_none"] += int(category != 0)
            if getattr(config, "single_notes_only_flag", False):
                category = 0
            old_notes, old_chains = list(state["notes"]), list(state["chains"])
            accepted = _atomic_expand(state, anchor, vocabularies["multiplier"][category], duration,
                                      result.diagnostics["rejected_expansions"])
            if accepted and category:
                proposed = len(state["notes"]), len(state["chains"])
                _rejections(expansion_rejections, postprocessors.validate_expansions(state, song, logger=None))
                if (len(state["notes"]), len(state["chains"])) != proposed:
                    state["notes"], state["chains"] = old_notes, old_chains
                    result.diagnostics["rejected_expansions"].append({"anchor_id": anchor["id"], "reason": "expansion clearance"})
                    accepted = False
            if accepted:
                pair[anchor["hand"]] = category
        _history(history, {"time": candidate["time"], "value": pair}, length)
        _progress(logger, "multiplier", index, len(multiplier_candidates))
    result.stages["multiplier"] = {"retained_notes": len(retained), "added_notes": sum(note.get("kind") == "copy" for note in state["notes"]),
                                    "chains": len(state["chains"]), "chain_slices": sum(chain["slices"] for chain in state["chains"])}
    result.diagnostics["expansion_clearance"] = postprocessors.validate_expansions(state, song, logger=logger)
    _rejections(expansion_rejections, result.diagnostics["expansion_clearance"])
    result.diagnostics["expansion_clearance"].update(expansion_rejections)
    result.stages["multiplier"]["elapsed_seconds"] = perf_counter() - started
    started = perf_counter()
    index_state(state)
    if repaired(options, "hazards"):
        from training.pipeline_v2.hazard_sections import rollout
        report = rollout(song, state, vocabularies, options,
                         lambda inputs: _predict(predictors["hazards"], inputs), encoder,
                         progress=lambda index, total: _progress(logger, "hazards", index, total))
        raw["wall_proposals"] = report["wall_section_proposals"]
        raw["bomb_proposals"] = report["bomb_proposals"]
        report["application_filters"] = postprocessors.validate_hazards(state, song, logger=logger)
        result.diagnostics["hazard_clearance"] = report
        result.stages["hazards"] = {**report, "walls": len(state["walls"]), "bombs": len(state["bombs"]),
                                     "elapsed_seconds": perf_counter() - started}
    else:
        queries = hazard_candidates(expansion_occupancy_notes(state), duration, bpm, schedule)
        encoder(np.asarray([query["time"] for query in queries], np.float64))
        history, hazard_rejections = [], {}
        _log(logger, "V2 hazards")
        for index, candidate in enumerate(queries):
            output = _predict(predictors["hazards"], _inputs("hazards", song, state, candidate, history, encoder, options))
            wall = int(_distribution(output["wall"], (1, len(vocabularies["walls"])), "wall")[0].argmax())
            bombs = _multilabel(output["bombs"], (1, 12), "bomb")[0] >= options["bomb_threshold"]
            previous_walls, previous_bombs = len(state["walls"]), len(state["bombs"])
            if candidate["kind"] == "wall" and wall:
                raw["wall_proposals"] += 1
                state["walls"].append({**deepcopy(vocabularies["walls"][wall]), "time": candidate["time"],
                                       "end_time": candidate["end_time"], "template_id": wall})
            elif candidate["kind"] == "bomb":
                raw["bomb_proposals"] += int(bombs.sum())
                state["bombs"].extend({"time": candidate["time"], "x": int(cell % 4), "y": int(cell // 4)} for cell in np.flatnonzero(bombs))
            # Clearance is applied before accepted outputs enter the next history.
            _rejections(hazard_rejections, postprocessors.validate_hazards(state, song, logger=None))
            accepted_wall = state["walls"][-1]["template_id"] if len(state["walls"]) > previous_walls else 0
            accepted_bombs = np.zeros(12, np.int32)
            for bomb in state["bombs"][previous_bombs:]:
                accepted_bombs[4 * bomb["y"] + bomb["x"]] = 1
            for record in history:
                if record["time"] == candidate["time"] and record.get("kind") == "bomb":
                    record["value"][1:] = [int(any(bomb["time"] == candidate["time"] and 4 * bomb["y"] + bomb["x"] == cell
                                                    for bomb in state["bombs"])) for cell in range(12)]
            _history(history, {"time": candidate["time"], "kind": candidate["kind"],
                               "value": [accepted_wall, *accepted_bombs.tolist()]}, length)
            _progress(logger, "hazards", index, len(queries))
        result.diagnostics["hazard_clearance"] = postprocessors.validate_hazards(state, song, logger=logger)
        _rejections(hazard_rejections, result.diagnostics["hazard_clearance"])
        result.diagnostics["hazard_clearance"].update(hazard_rejections)
        result.stages["hazards"] = {"queries": len(queries), "walls": len(state["walls"]), "bombs": len(state["bombs"]),
                                     "elapsed_seconds": perf_counter() - started}
    started = perf_counter()
    queries = arc_candidates(state["notes"], options)
    state["_arc_options"] = options
    encoder(np.asarray([query["time"] for query in queries], np.float64))
    history, arc_rejections = [], {}
    _log(logger, "V2 arcs")
    lower, upper = options["arc_shape_bounds"]
    for index, candidate in enumerate(queries):
        output = _predict(predictors["arcs"], _inputs("arcs", song, state, candidate, history, encoder, options))
        attach_probabilities = _distribution(output["attach"], (1, 2), "arc attachment")[0]
        attach = int(attach_probabilities[1] >= options.get("arc_threshold", .5)) if repaired(options, "arcs") else int(attach_probabilities.argmax())
        raw["arc_attachments"] += attach
        shape = np.asarray(output["shape"], np.float32)
        if shape.shape != (1, 2) or not np.isfinite(shape).all() or np.any((shape < lower) | (shape > upper)):
            raise ValueError("Invalid arc controls")
        anchor = int(_distribution(output["anchor"], (1, 3), "arc anchor")[0].argmax())
        previous = len(state["arcs"])
        if attach:
            state["arcs"].append({**deepcopy(candidate), "mu": float(shape[0, 0]), "tmu": float(shape[0, 1]), "anchor_mode": anchor})
            _rejections(arc_rejections, postprocessors.validate_arcs(state, song, logger=None))
        attach = int(len(state["arcs"]) > previous)
        _history(history, {"time": candidate["time"], "value": history_value({"attach": attach, "shape": shape[0].tolist(), "anchor": anchor})}, length)
        _progress(logger, "arcs", index, len(queries))
    result.diagnostics["arc_clearance"] = postprocessors.validate_arcs(state, song, logger=logger)
    _rejections(arc_rejections, result.diagnostics["arc_clearance"])
    result.diagnostics["arc_clearance"].update(arc_rejections)
    result.stages["arcs"] = {"queries": len(queries), "arcs": len(state["arcs"]), "elapsed_seconds": perf_counter() - started}
    started = perf_counter()
    _log(logger, "V1 lighting")
    state["lighting"], lighting_report = lighting_generator(state["event_times"], encoder)
    raw["lighting_commands"] = len(state["lighting"])
    result.stages["lighting"] = {**lighting_report, "events": len(state["lighting"]),
                                 "elapsed_seconds": perf_counter() - started}
    if [note for note in state["notes"] if note.get("kind") == "retained"] != retained:
        raise ValueError("A downstream stage changed retained notes")
    _validate_state(state, duration)
    result.diagnostics["final_counts"] = {
        "colored_notes": len(state["notes"]),
        "retained_notes": sum(note.get("kind") == "retained" for note in state["notes"]),
        "heuristic_notes": sum(note.get("kind") == "heuristic" for note in state["notes"]),
        "learned_copies": sum(note.get("kind") == "copy" for note in state["notes"]),
        "unique_onsets": len({note["time"] for note in state["notes"]}),
        "chains": len(state["chains"]), "chain_slices": sum(chain["slices"] for chain in state["chains"]),
        "walls": len(state["walls"]), "bombs": len(state["bombs"]),
        "arcs": len(state["arcs"]), "lighting_events": len(state["lighting"])}
    state.pop("_arc_options", None)
    state.pop("_tracks", None)
    state.pop("_occupancy", None)
    result.diagnostics.update(export_ready=True, production_validated=True, safety_reviewed=False,
                               gameplay_validated=False,
                               encoder=getattr(encoder, "metadata", {}),
                               requested_count_guaranteed=False)
    return result
