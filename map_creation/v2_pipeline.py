"""Production v2 inference with chronological histories and reviewed cleanup.

Training targets and source beatmaps are never inputs. The concept command keeps
its independent review stops; this path uses concrete application postprocessors.
"""

from copy import deepcopy
from numbers import Real
from time import perf_counter
from types import SimpleNamespace

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
    if config.verbose_level <= 1:
        return
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
    bias = getattr(config, "v2_timing_event_bias", 0.)
    if (isinstance(bias, (bool, np.bool_)) or not isinstance(bias, Real)
            or not np.isfinite(bias) or not 0 <= bias < 1):
        raise ValueError("v2_timing_event_bias must be a finite numeric value within [0, 1)")
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
    counts = (probabilities + [0, bias, bias]).argmax(axis=-1).astype(np.int32)
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


def _sample_category(probabilities, temperature, rng):
    """Sample one index from a categorical distribution; nonpositive temperature argmaxes."""
    probs = np.asarray(probabilities, np.float64)
    if not np.isfinite(probs).all() or (probs < 0).any() or not probs.sum() > 0:
        raise ValueError("Invalid note category probabilities")
    if temperature <= 0:
        return int(probs.argmax())
    scaled = np.power(probs / probs.sum(), 1. / temperature)
    scaled /= scaled.sum()
    return int(rng.choice(len(scaled), p=scaled))


def _decode_positions_sampled(outputs, count, temperature, rng):
    """Sample position categories while preserving the required note count.

    A single-note event samples jointly over the 24 hand/cell combinations,
    weighted exactly like the count-one argmax rule; a two-note event samples
    each hand's cell independently.
    """
    red = _distribution(outputs["red"], (1, 13), "red position")[0]
    blue = _distribution(outputs["blue"], (1, 13), "blue position")[0]
    if count == 2:
        return (1 + _sample_category(red[1:], temperature, rng),
                1 + _sample_category(blue[1:], temperature, rng))
    weights = np.concatenate([red[1:] * blue[0], red[0] * blue[1:]])
    choice = _sample_category(weights, temperature, rng)
    return (choice + 1, 0) if choice < 12 else (0, choice - 11)


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


_DIRECTION_VECTORS = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
                      4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1), 8: (0, 0)}


def _heuristic_wall_candidates(notes, event_times):
    """Legacy rule-based obstacles on the final note stream.

    ``gen_obstacles`` fills spans where a lane stayed note-free (the original
    v1 obstacle model); ``validate_hazards`` then applies the legacy margins
    and note clearance. Grid events anchor obstacle spans as before; notes off
    the event grid (onset-snapped notes, multiplier copies) add their own
    one-event groups so the lane scan can never span them, while grid events
    left empty keep their ``times_empty`` split-marker role.
    """
    from .gen_obstacles import calculate_obstacles
    eps = max(float(getattr(config, "min_time_diff", .02)), 1e-6)
    groups = {float(time): [] for time in event_times}
    extra = []  # (time, flat fields) clusters for notes not on the event grid
    for note in sorted(notes, key=lambda note: (note["time"], note.get("id", 0))):
        fields = [int(note[key]) for key in ("x", "y", "hand", "d")]
        group = groups.get(float(note["time"]))
        if group is not None:
            group.extend(fields)
        elif extra and float(note["time"]) - extra[-1][0] <= eps:
            extra[-1][1].extend(fields)
        else:
            extra.append((float(note["time"]), fields))
    clustered = dict(extra)
    timings = np.asarray(sorted(list(groups) + list(clustered)), np.float64)
    if not len(timings):
        return []
    try:
        obstacles = calculate_obstacles([groups.get(time, clustered.get(time, []))
                                         for time in timings], timings)
    except IndexError:
        return []  # no note-free spans: the legacy sorter cannot handle empty
    return [{"x": int(obst[1]), "y": int(obst[2]), "width": int(obst[4]),
             "height": int(obst[5]), "time": float(obst[0]),
             "end_time": float(obst[0] + obst[3]), "kind": "wall"}
            for obst in np.asarray(obstacles, np.float64).reshape(-1, 6)]


def _heuristic_bomb_candidates(state, rng):
    """Bombs at note times, in cells clear of every near-time swing's safe area.

    Each anchor note rolls ``v2_bombs_prob`` and may propose one bomb at its
    own time, in a cell at least ``v2_bomb_note_cell_gap`` rings away from
    every swing cell within ``v2_bomb_note_time_gap`` — the same box
    ``validate_hazards`` enforces, so proposals only drop to wall or
    other-bomb conflicts. Placement draws from the generation RNG and stays
    deterministic per seed.
    """
    from .v2_postprocessing import _bomb_safe_cells, _occupied_notes
    density = float(getattr(config, "v2_bombs_prob", 0.) or 0.)
    time_gap = max(float(getattr(config, "v2_bomb_note_time_gap", .8) or 0.),
                   float(getattr(config, "min_time_diff", .02) or 0.))
    cell_gap = max(float(getattr(config, "v2_bomb_note_cell_gap", 1) or 0.), 0.)
    occupied = sorted(_occupied_notes(state), key=lambda note: (note["time"], note.get("id", 0)))
    times = np.asarray([entry["time"] for entry in occupied])
    proposals, free_cells = [], {}
    for note in sorted(state["notes"], key=lambda note: (note["time"], note.get("id", 0))):
        if density <= 0 or rng.random() > density:
            continue
        lo = int(np.searchsorted(times, note["time"] - time_gap, side="left"))
        hi = int(np.searchsorted(times, note["time"] + time_gap, side="right"))
        if (lo, hi) not in free_cells:
            free_cells[lo, hi] = sorted(_bomb_safe_cells(occupied[lo:hi], cell_gap))
        if not free_cells[lo, hi]:
            continue
        x, y = free_cells[lo, hi][int(rng.integers(len(free_cells[lo, hi])))]
        proposals.append({"time": float(note["time"]), "x": x, "y": y, "kind": "bomb"})
    return proposals


def _heuristic_arc_candidates(notes, radius):
    """All adjacent same-hand pairs; validate_arcs applies its own gap bounds.

    The legacy turbo-start ends a synthetic-head guidance arc at each hand's
    first note; v3 allows it because the head is arc geometry, not a note.
    """
    candidates = []
    for hand in (0, 1):
        ordered = sorted((note for note in notes if note["hand"] == hand),
                         key=lambda note: (note["time"], note["id"]))
        if ordered and config.slider_turbo_start and ordered[0]["time"] > 0:
            first = ordered[0]
            anchor = 2 if first["d"] in ({4, 0, 5} if hand == 0 else {6, 1, 7}) else 1
            candidates.append({"kind": "arc", "time": .3 * first["time"],
                               "end_time": float(first["time"]), "hand": hand,
                               "endpoint_ids": [first["id"]],
                               "synthetic_head": {"x": first["x"], "y": first["y"], "d": first["d"]},
                               "gap": .7 * first["time"],
                               "mu": radius, "tmu": radius, "anchor_mode": anchor})
        for head, tail in zip(ordered, ordered[1:]):
            gap = float(tail["time"] - head["time"])
            if gap <= 0:
                continue
            candidates.append({"kind": "arc", "time": float(head["time"]),
                               "end_time": float(tail["time"]), "hand": hand,
                               "endpoint_ids": [head["id"], tail["id"]], "gap": gap,
                               "mu": radius, "tmu": radius, "anchor_mode": 0})
    return sorted(candidates, key=lambda candidate: (candidate["time"], candidate["hand"]))


def _approach_snap(exit_state, note):
    """Deterministic cut prior: the direction an axis-aligned hand approach implies."""
    if exit_state is None:
        return None
    vx, vy, last_dir = exit_state
    dx, dy = note["x"] - vx, note["y"] - vy
    for _ in range(6):
        if dx or dy:
            break
        vx, vy = vx + _DIRECTION_VECTORS[last_dir][0], vy + _DIRECTION_VECTORS[last_dir][1]
        dx, dy = note["x"] - vx, note["y"] - vy
    if dx == 0 and dy == 0 or (dx != 0 and dy != 0):
        return None  # unresolved or diagonal approach — the model decides
    return (0 if dy > 0 else 1) if dx == 0 else (3 if dx > 0 else 2)


def _atomic_expand(state, anchor, template, duration, rejected):
    """Reject a whole invalid/duplicate expansion and preserve every anchor."""
    if template.get("kind", "none") == "none":
        return True
    trial = {"notes": list(state["notes"]), "chains": list(state["chains"])}
    errors = []
    _expand(trial, anchor, template, duration, errors)
    added = trial["notes"][len(state["notes"]):]
    occupied = {(round(note["time"], 8), note["x"], note["y"]) for note in state["notes"]}
    occupied_hands = {(round(note["time"], 8), note["hand"]) for note in state["notes"]}
    for note in added:
        key = (round(note["time"], 8), note["x"], note["y"])
        hand_key = (round(note["time"], 8), note["hand"])
        if hand_key in occupied_hands:
            errors.append({"anchor_id": anchor["id"], "reason": "duplicate same-hand expansion"})
            break
        if key in occupied:
            errors.append({"anchor_id": anchor["id"], "reason": "duplicate expansion occupancy"})
            break
        occupied.add(key)
        occupied_hands.add(hand_key)
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
        head_spec = arc.get("synthetic_head")
        if head_spec is not None:
            if (len(ids) != 1 or ids[0] not in identities or not isinstance(head_spec, dict)
                    or any(key not in head_spec or not float(head_spec[key]).is_integer()
                           for key in ("x", "y", "d"))
                    or not 0 <= head_spec.get("x", -1) < 4 or not 0 <= head_spec.get("y", -1) < 3
                    or not 0 <= head_spec.get("d", -1) < 9
                    or not 0 <= arc["time"] < arc["end_time"] < duration):
                raise ValueError("Invalid final v2 synthetic arc")
            tail = identities[ids[0]]
            if tail["hand"] != arc["hand"] or tail["time"] != arc["end_time"]:
                raise ValueError("A final v2 synthetic arc has a mismatched tail")
            continue
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
        if not getattr(config, "v2_cleanup_note_heuristics", True):
            # Skip the legacy v1 note mutators; structural dedup, the
            # same-direction recut and the clearance fixpoint still run.
            base = postprocessors
            postprocessors = SimpleNamespace(
                postprocess_timing=base.postprocess_timing,
                postprocess_notes=lambda *args, **kwargs: base.postprocess_notes(
                    *args, skip=base.LEGACY_NOTE_MUTATORS, **kwargs),
                validate_expansions=base.validate_expansions,
                validate_hazards=base.validate_hazards,
                validate_arcs=base.validate_arcs,
                enforce_note_flow_and_clearance=base.enforce_note_flow_and_clearance,
                snap_note_times=base.snap_note_times,
                apply_intro_floor=base.apply_intro_floor,
                apply_fast_direction_rules=base.apply_fast_direction_rules,
                apply_fast_counter_sweep=base.apply_fast_counter_sweep,
                apply_diagonal_cap=base.apply_diagonal_cap,
                apply_speed_cap=base.apply_speed_cap,
                calm_area_pulses=base.calm_area_pulses,
                add_calm_effects=base.add_calm_effects,
                add_calm_lighting=base.add_calm_lighting)
    factor = getattr(config, "random_note_map_factor", 0.) if random_factor is None else random_factor
    if isinstance(factor, bool) or not np.isfinite(factor) or factor < 0:
        raise ValueError("Random factor must be finite and nonnegative")
    factor = min(float(factor), 1.)
    rng = np.random.default_rng() if rng is None else rng
    duration, bpm = float(song.metadata["duration"]), float(song.metadata["bpm"])
    options = deepcopy(settings)
    temperature = float(options.get("note_sample_temperature", 0.) or 0.)
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Note sample temperature must be finite and nonnegative")
    sampling = temperature > 0
    options["vocabularies"] = vocabularies
    schedule = options["schedule"]
    status = "dummy" if all(getattr(predictor, "status", "trained") == "dummy"
                            for stage, predictor in predictors.items() if stage != "lighting") else "trained"
    result = PipelineResult(status, {key: [] for key in ("notes", "chains", "walls", "bombs", "arcs", "lighting")},
                            diagnostics={"requested_retained_nps": float(requested_nps),
                                         "duration_seconds": duration, "bpm": bpm,
                                         "bpm_detected": float(song.metadata.get("bpm_detected", bpm)),
                                         "rejected_expansions": [], "export_ready": False,
                                         "random_factor": factor, "note_sample_temperature": temperature,
                                         "scheduler": deepcopy(schedule)})
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
        if sampling:
            pair = _decode_positions_sampled(_biased(output, 13, 0, rng), candidate["count"], temperature, rng)
        else:
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
    history, kinematics = [], {}
    approach_prior = float(getattr(config, "v2_arrow_approach_prior", 0.) or 0.)
    approach_overrides = 0
    arrows_reference = predictors.get("arrows_reference")
    arrows_variety = float(getattr(config, "v2_arrows_variety", 0.) or 0.) if arrows_reference is not None else 0.
    variety_picks = 0
    for index, candidate in enumerate(candidates):
        inputs = _inputs("arrows", song, state, candidate, history, encoder, options)
        # Per-event model pick interpolates the resulting statistics exactly;
        # mixing softmaxes would flatten disagreement and overshoot the tails.
        model = (arrows_reference if arrows_variety > 0 and rng.random() < arrows_variety
                 else predictors["arrows"])
        variety_picks += int(model is arrows_reference)
        output = _biased(_predict(model, inputs), 9, 0 if sampling else factor, rng)
        pair = [9, 9]
        for hand in (0, 1):
            times, track = state["_tracks"][hand]
            location = int(np.searchsorted(times, candidate["time"]))
            if location < len(track) and track[location]["time"] == candidate["time"]:
                probabilities = output[("red", "blue")[hand]][0]
                direction = (_sample_category(probabilities, temperature, rng)
                             if sampling else int(probabilities.argmax()))
                prior = _approach_snap(kinematics.get(hand), track[location])
                if prior is not None and float(probabilities.max()) < approach_prior:
                    direction = prior
                    approach_overrides += 1
                note = track[location]
                note["d"] = direction
                pair[hand] = direction
                vector = _DIRECTION_VECTORS[direction]
                kinematics[hand] = (note["x"] + vector[0], note["y"] + vector[1],
                                    direction if direction != 8 else (kinematics.get(hand) or (0, 0, 1))[2])
        _history(history, {"time": candidate["time"], "value": pair}, length)
        _progress(logger, "arrows", index, len(candidates))
    raw["arrow_approach_overrides"] = approach_overrides
    raw["arrow_variety"] = round(variety_picks / len(candidates), 4) if candidates else 0.
    result.stages["arrows"] = {"directed_notes": len(state["notes"])}
    raw["arrow_notes"] = len(state["notes"])
    state["raw_model_notes"] = np.asarray([[note[key] for key in ("id", "time", "hand", "x", "y", "d")]
                                           for note in state["notes"]], np.float64).reshape(-1, 6)
    result.diagnostics["raw_model_note_columns"] = ["id", "time_seconds", "hand", "x", "y", "direction"]
    state["notes"], diagnostics = postprocessors.postprocess_notes(state["notes"],
                                             [candidate["time"] for candidate in candidates], logger=logger)
    result.diagnostics["note_postprocessing"] = diagnostics
    result.stages["arrows"]["elapsed_seconds"] = perf_counter() - started
    snap_times = getattr(postprocessors, "snap_note_times", None)
    if snap_times is not None and getattr(config, "v2_notes_onset_snap", False) and state["notes"]:
        snap_started = perf_counter()
        result.diagnostics["onset_snap"] = snap_times(state["notes"], state["chains"],
                                                      audio_path, duration, logger=logger)
        result.diagnostics["onset_snap"]["elapsed_seconds"] = perf_counter() - snap_started
    intro_fn = getattr(postprocessors, "apply_intro_floor", None)
    intro_floor = float(getattr(config, "v2_map_intro_no_notes_seconds", 0.) or 0.)
    # TODO: multiplier expansion copies with a negative dt would land back
    # inside this note-free zone (no second floor runs after expansion);
    # design-accepted today — revisit if a fitted vocabulary gains negative
    # offsets.
    if intro_fn is not None and intro_floor > 0:
        result.diagnostics["intro_zones"] = {
            "no_notes_seconds": intro_floor,
            "notes_dropped": intro_fn(state["notes"], intro_floor)}
    fast_dir = getattr(postprocessors, "apply_fast_direction_rules", None)
    if fast_dir is not None and state["notes"] \
            and float(getattr(config, "v2_fast_gap_seconds", .2) or 0.) > 0:
        result.diagnostics["fast_directions"] = fast_dir(state["notes"], rng=rng,
                                                         logger=logger)
    diag_fn = getattr(postprocessors, "apply_diagonal_cap", None)
    diag_cap = float(getattr(config, "v2_diagonal_max_fraction", 1.) or 0.)
    if diag_fn is not None and state["notes"] and 0 <= diag_cap < 1:
        result.diagnostics["diagonal_cap"] = diag_fn(state["notes"], state["chains"], rng=rng)
    speed_cap = getattr(postprocessors, "apply_speed_cap", None)
    speed_scale = float(getattr(config, "v2_speed_cap_scale", 1.) or 0.)
    if speed_cap is not None and state["notes"] and speed_scale > 0:
        # Primary pass while every note is still mutable; the retained
        # snapshot below records the capped set.
        result.diagnostics["speed_cap"] = speed_cap(state["notes"], state["chains"],
                                                    rng=rng, logger=logger)
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
    enforce_flow = getattr(postprocessors, "enforce_note_flow_and_clearance", None)
    if enforce_flow is not None:
        clearance = enforce_flow(state["notes"], state["chains"])
        result.diagnostics["same_direction_notes_flipped"] = clearance["flow_recuts"]
        result.diagnostics["note_clearance"] = clearance
    if speed_cap is not None and state["notes"] and speed_scale > 0:
        # Post-expansion pass: retained notes must stay dict-equal to the
        # snapshot until the end-of-run check, so only copies and heuristic
        # notes may still drop or recut.
        result.diagnostics["speed_cap_post_expansion"] = speed_cap(
            state["notes"], state["chains"], rng=rng, logger=logger,
            drop_retained=False)
    counter_sweep = getattr(postprocessors, "apply_fast_counter_sweep", None)
    if (counter_sweep is not None and state["notes"]
            and getattr(config, "v2_fast_counter_direction", "recut") == "recut"
            and float(getattr(config, "v2_fast_gap_seconds", .2) or 0.) > 0):
        # Copies inherit their anchor's cut, so expansion and the speed cap's
        # deletions can leave fast pairs the pre-snapshot rules never saw:
        # sweep counter directions over non-retained notes only, keeping the
        # retained-snapshot invariant verified at the end of generate().
        result.diagnostics["fast_counter_sweep"] = counter_sweep(
            state["notes"], rng=rng, logger=logger)
    started = perf_counter()
    index_state(state)
    if getattr(config, "v2_walls_heuristic", False):
        _log(logger, "V2 hazards (heuristic)")
        proposals = (_heuristic_wall_candidates(state["notes"], state["event_times"])
                     if config.add_obstacle_flag else [])
        bomb_proposals = (_heuristic_bomb_candidates(state, rng)
                          if config.add_bombs_flag else [])
        state["walls"], state["bombs"] = proposals, bomb_proposals
        raw["wall_proposals"] = len(proposals)
        raw["bomb_proposals"] = len(bomb_proposals)
        result.diagnostics["hazard_clearance"] = postprocessors.validate_hazards(state, song, logger=logger)
        result.stages["hazards"] = {"mode": "heuristic", "walls": len(state["walls"]),
                                    "bombs": len(state["bombs"]),
                                    "elapsed_seconds": perf_counter() - started}
    elif repaired(options, "hazards"):
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
    if getattr(config, "v2_arcs_heuristic", False):
        _log(logger, "V2 arcs (heuristic)")
        radius = float(config.slider_radius_multiplier)
        proposed = _heuristic_arc_candidates(state["notes"], radius)
        state["arcs"] = proposed
        raw["arc_attachments"] = len(proposed)
        result.diagnostics["arc_clearance"] = postprocessors.validate_arcs(
            state, song, logger=logger, settings={"stage_policies": {}})
        result.stages["arcs"] = {"mode": "heuristic", "queries": len(proposed),
                                 "arcs": len(state["arcs"]),
                                 "elapsed_seconds": perf_counter() - started}
    else:
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
        result.stages["arcs"] = {"mode": "model", "queries": len(queries), "arcs": len(state["arcs"]),
                                 "elapsed_seconds": perf_counter() - started}
    started = perf_counter()
    calm_pulses = []
    pulse_fn, place_fn = (getattr(postprocessors, "calm_area_pulses", None),
                          getattr(postprocessors, "add_calm_effects", None))
    if (pulse_fn is not None and place_fn is not None
            and getattr(config, "v2_calm_effects", False)
            and any(getattr(config, name, True) for name in
                    ("v2_calm_effects_walls", "v2_calm_effects_bombs", "v2_calm_effects_lighting"))):
        # Decorative beat pulses in note-free stretches: edge wall flashes,
        # bottom-corner bombs and later laser flashes. Notes stay untouched.
        _log(logger, "V2 calm effects")
        calm_pulses, diagnostics = pulse_fn(state, song)
        diagnostics.update(place_fn(state, song, calm_pulses))
        if diagnostics["walls"] or diagnostics["bombs"]:
            diagnostics["hazard_clearance"] = postprocessors.validate_hazards(state, song, logger=logger)
        result.diagnostics["calm_effects"] = diagnostics
        result.stages["calm_effects"] = {"pulses": len(calm_pulses),
                                         "elapsed_seconds": perf_counter() - started}
    started = perf_counter()
    _log(logger, "V1 lighting")
    state["lighting"], lighting_report = lighting_generator(state["event_times"], encoder)
    if calm_pulses and getattr(config, "v2_calm_effects_lighting", False) \
            and getattr(postprocessors, "add_calm_lighting", None) is not None:
        added = postprocessors.add_calm_lighting(state, calm_pulses)
        diagnostics["lighting_events"] = added
        result.stages["calm_effects"]["lighting_events"] = added
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
