"""Production v2 inference with chronological histories and reviewed cleanup.

Training targets and source beatmaps are never inputs. The concept command keeps
its independent review stops; this path uses concrete application postprocessors.
"""

from contextlib import contextmanager, nullcontext
from functools import wraps
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Real
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from tools.config import config
from training.beat_models.contracts import CORE, FRAME_SECONDS, frame_conditions, intensity_inputs, timing_window
from training.pipeline_v2.contracts import (build_inputs, expansion_occupancy_notes, index_state,
                                            preparation_defaults)
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


def _recover_single_hand(pair, outputs, last_time, time, support_ratio, *,
                         recent_notes=None, following_event=None, last_recovery_time=-np.inf):
    """Recover a crowded single using supported opposite-hand positions.

    Supplying ``recent_notes`` selects the complete combined policy: improve
    two-second hand balance, retain nearby own-side placement, keep 500ms
    between recoveries, and protect the next scheduled double. Legacy
    hand_75/90 experiments keep their original behavior without that context.
    The original event time and count are never changed.
    """
    if support_ratio <= 0 or sum(bool(cell) for cell in pair) != 1:
        return pair
    hand = 0 if pair[0] else 1
    other = 1-hand
    if time-last_time[hand] >= .25 or time-last_time[other] < .25:
        return pair
    probabilities = [outputs[color][0] for color in ("red", "blue")]
    candidates = range(1, 13)
    if recent_notes is not None:
        if time-last_recovery_time < .5-1e-9:
            return pair
        if (following_event is not None and following_event['count'] == 2
                and following_event['time']-time < .25-1e-9):
            return pair
        recent = [n for n in recent_notes if time-2. <= n['time'] < time]
        counts = [sum(n['hand'] == h for n in recent) for h in (0, 1)]
        if counts[hand] < counts[other]+2:
            return pair
        prior = max((n for n in recent if n['hand'] == other),
                    key=lambda n:n['time'], default=None)
        if prior is None:
            return pair
        candidates = [cell for cell in candidates
                      if ((cell-1) % 4 < 2) == (other == 0)
                      and ((cell-1) % 4-prior['x'])**2
                          + ((cell-1) // 4-prior['y'])**2 <= 2]
        if not candidates:
            return pair
    proposed = min(candidates, key=lambda cell:(-probabilities[other][cell], cell))
    existing_weight = probabilities[hand][pair[hand]]*probabilities[other][0]
    proposed_weight = probabilities[other][proposed]*probabilities[hand][0]
    if proposed_weight <= 0 or proposed_weight < support_ratio*existing_weight:
        return pair
    return (proposed, 0) if other == 0 else (0, proposed)


def _deferred_recovery_alternatives(pair, outputs, support_ratio):
    """Capture supported own-side alternatives without changing decoder history."""
    if support_ratio <= 0 or sum(bool(cell) for cell in pair) != 1:
        return []
    hand = 0 if pair[0] else 1
    other = 1-hand
    probabilities = [outputs[color][0] for color in ('red', 'blue')]
    existing = float(probabilities[hand][pair[hand]]*probabilities[other][0])
    alternatives = []
    for category in range(1, 13):
        x, y = (category-1) % 4, (category-1) // 4
        proposed = float(probabilities[other][category]*probabilities[hand][0])
        if ((x < 2) == (other == 0) and proposed > 0
                and proposed >= support_ratio*existing):
            alternatives.append(dict(hand=other, x=x, y=y,
                                     support=proposed/max(existing, 1e-12)))
    return sorted(alternatives, key=lambda n:(-n['support'], n['x'], n['y']))[:3]


def _inputs(stage, song, state, candidate, history, encoder, settings, options=None):
    resolved = settings if options is None else options
    length = max(resolved["history_length"], resolved["lighting_block_length"])
    prior = [record for record in history if record["time"] < candidate["time"]][-length:]
    return build_inputs(stage, song, state, candidate, prior, encoder=encoder, settings=settings,
                        options=options)


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


@dataclass(eq=False)
class VariantRun:
    """One generation variant: resolved inputs plus mutable decode state.

    The stage functions below read and mutate this context so identical runs
    can later advance in lockstep; ``generate`` composes the same steps for
    a single run.
    """
    song: object
    requested_nps: float
    predictors: dict
    encoder: object
    options: dict
    prepared: dict          # preparation_defaults(options), resolved once per variant
    vocabularies: dict
    lighting_generator: object
    audio_path: object = None
    logger: object = None
    intensity: object = None
    postprocessors: object = None
    playability_mode: str = "search"
    factor: float = 0.
    rng: object = None
    duration: float = 0.
    bpm: float = 0.
    temperature: float = 0.
    sampling: bool = False
    schedule: dict = None
    result: object = None
    state: dict = None
    raw: dict = None
    started: float = 0.
    candidates: list = field(default_factory=list)
    history: list = field(default_factory=list)
    length: int = 0
    last_hand_time: list = field(default_factory=lambda: [-np.inf, -np.inf])
    recovery: float = 0.
    recovery_swaps: int = 0
    recovery_opportunities: int = 0
    recovery_alternatives: dict = field(default_factory=dict)
    recovery_balanced: bool = False
    kinematics: dict = field(default_factory=dict)
    approach_prior: float = 0.
    approach_overrides: int = 0
    arrows_reference: object = None
    arrows_variety: float = 0.
    variety_picks: int = 0
    speed_cap: object = None
    speed_scale: float = 0.
    combined_reach: bool = False
    retained: list = field(default_factory=list)
    upstream: dict = None
    by_time: dict = field(default_factory=dict)
    multiplier_candidates: list = field(default_factory=list)
    expansion_rejections: dict = field(default_factory=dict)
    calm_pulses: list = field(default_factory=list)
    scope: object = None       # batched runs: CM factory replaying this variant's settings
    late_turns: bool = False   # batched runs: replay @_scope_playability_flow in that scope
    late_ready: bool = False   # batched runs: sticky _LATE_TURNS_READY across pieces
    clock_share: float = 0.    # batched runs: attributed stage seconds for honest elapsed


def variant_run(song, requested_nps, predictors, encoder, settings, vocabularies, *,
                lighting_generator, audio_path=None, logger=None, postprocessors=None, random_factor=None,
                intensity=None, rng=None):
    """Resolve one variant's postprocessors and seed its run context."""
    if isinstance(requested_nps, bool) or not np.isfinite(requested_nps) or requested_nps < 0:
        raise ValueError("Requested retained NPS must be finite and nonnegative")
    playability_mode = "search"
    if postprocessors is None:
        from . import v2_postprocessing as postprocessors
        mode = getattr(config, "v2_note_postprocessing_mode", "search")
        if mode not in ("forward", "search"):
            raise ValueError(f"Unknown v2_note_postprocessing_mode: {mode}")
        forward = (mode == "forward"
                   and not getattr(config, "v2_cleanup_note_heuristics", True)
                   and getattr(config, "v2_playability_enabled", False)
                   and (getattr(config, "v2_playability_priority", False)
                        or getattr(config, "v2_playability_reach", False)))
        if forward:
            playability_mode = "forward"
            from . import v2_forward_postprocessing as forward_core
            from . import v2_forward_expansion as forward_late
            base = postprocessors
            postprocessors = SimpleNamespace(
                postprocess_timing=base.postprocess_timing,
                postprocess_notes=forward_core.normalize_notes,
                repair_notes=forward_core.repair_notes,
                validate_expansions=forward_late.validate_expansions,
                drop_guard=forward_late.drop_guard,
                validate_hazards=base.validate_hazards,
                validate_arcs=base.validate_arcs,
                snap_note_times=base.snap_note_times,
                apply_intro_floor=base.apply_intro_floor,
                apply_note_emphasis=forward_late.apply_note_emphasis,
                calm_area_pulses=base.calm_area_pulses,
                add_calm_effects=base.add_calm_effects,
                add_calm_lighting=base.add_calm_lighting)
        elif not getattr(config, "v2_cleanup_note_heuristics", True):
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
                speed_cap_scale=base.speed_cap_scale,
                finalize_playability_reach=base.finalize_playability_reach,
                apply_note_emphasis=base.apply_note_emphasis,
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
    # Resolved once per variant here inside the caller's scoped settings rather
    # than per candidate inside build_inputs; an invalid setting now raises at
    # variant setup instead of mid-decode.
    prepared = preparation_defaults(options)
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
    raw = result.diagnostics["raw_decisions"] = {
        "position_events": 0, "position_notes": 0, "arrow_notes": 0,
        "multiplier_non_none": 0, "wall_proposals": 0, "bomb_proposals": 0,
        "arc_attachments": 0, "lighting_commands": 0}
    return VariantRun(song, requested_nps, predictors, encoder, options, prepared, vocabularies,
                      lighting_generator, audio_path=audio_path, logger=logger, intensity=intensity,
                      postprocessors=postprocessors, playability_mode=playability_mode, factor=factor,
                      rng=rng, duration=duration, bpm=bpm, temperature=temperature, sampling=sampling,
                      schedule=schedule, result=result, state=result.state, raw=raw)


def _beats_stage(run):
    """Intensity, timing and postprocessed candidates into ``run.state``."""
    _log(run.logger, "V2 intensity and timing")
    beats = infer_beats(run.song, run.requested_nps, run.predictors,
                       getattr(config, "v2_timing_batch_size", 8), intensity=run.intensity)
    beat_seconds = beats.pop("_stage_seconds")
    run.result.diagnostics["intensity_postprocessing"] = beats.pop("_intensity_postprocessing")
    run.state.update(beats, requested_nps=float(run.requested_nps),
                     multiplier_classes=len(run.vocabularies["multiplier"]))
    cleanup_started = perf_counter()
    candidates, diagnostics = run.postprocessors.postprocess_timing(
        run.song, beats, run.audio_path, logger=run.logger)
    run.candidates = _candidates(run.song, candidates)
    run.state.update(raw_event_times=beats["event_times"], raw_event_frames=beats["event_frames"],
                     raw_counts=beats["counts"],
                     event_times=np.asarray([candidate["time"] for candidate in run.candidates], np.float64),
                     event_frames=np.asarray([candidate["frame"] for candidate in run.candidates], np.int64))
    run.state["counts"] = np.zeros_like(beats["counts"])
    for candidate in run.candidates:
        run.state["counts"][candidate["frame"]] = candidate["count"]
    run.result.diagnostics["timing_postprocessing"] = diagnostics
    run.result.stages["intensity"] = {"bins": len(beats["intensity"]), "elapsed_seconds": beat_seconds["intensity"]}
    run.result.stages["timing"] = {"raw_events": len(beats["event_frames"]), "events": len(run.candidates),
                                   "raw_retained_notes": int(beats["counts"].sum()),
                                   "elapsed_seconds": beat_seconds["timing"] + perf_counter() - cleanup_started}


def _positions_begin(run):
    """Reset positions decode state; event encodings are prefetched by the stage."""
    run.started = perf_counter()
    run.history, run.length = [], run.options["history_length"]
    run.last_hand_time = [-np.inf, -np.inf]
    run.recovery = config.v2_playability_recovery_ratio if config.v2_playability_enabled else 0.
    if not np.isfinite(run.recovery) or not 0 <= run.recovery <= 1:
        raise ValueError("v2_playability_recovery_ratio must be in [0, 1]")
    run.recovery_swaps = run.recovery_opportunities = 0
    run.recovery_alternatives = {}
    run.recovery_balanced = bool(getattr(config, "v2_playability_recovery_balanced", False))


def _positions_inputs(run, candidate):
    return _inputs("positions", run.song, run.state, candidate, run.history, run.encoder, run.options,
                   options=run.prepared)


def _positions_apply(run, candidate, output):
    """Decode one event's cells, apply single-hand recovery, update notes."""
    if run.sampling:
        pair = _decode_positions_sampled(_biased(output, 13, 0, run.rng), candidate["count"],
                                         run.temperature, run.rng)
    else:
        pair = _decode_positions(_biased(output, 13, run.factor, run.rng), candidate["count"])
    if run.recovery and candidate["count"] == 1:
        hand = 0 if pair[0] else 1
        run.recovery_opportunities += int(candidate["time"] - run.last_hand_time[hand] < .25)
        if run.recovery_balanced:
            alternatives = _deferred_recovery_alternatives(pair, output, run.recovery)
            if alternatives:
                cell = pair[hand]-1
                key = candidate["time"], hand, cell % 4, cell // 4
                run.recovery_alternatives[key] = alternatives
        else:
            selected = _recover_single_hand(pair, output, run.last_hand_time, candidate["time"], run.recovery)
            if tuple(pair) != tuple(selected):
                run.recovery_swaps += 1
            pair = selected
    for hand, category in enumerate(pair):
        if category:
            run.last_hand_time[hand] = candidate["time"]
            cell = category - 1
            run.state["notes"].append({"id": len(run.state["notes"]), "time": candidate["time"], "hand": hand,
                                       "x": cell % 4, "y": cell // 4, "kind": "retained"})
    _history(run.history, {"time": candidate["time"], "value": pair}, run.length)


def _positions_end(run):
    if run.recovery:
        run.raw["recovery_hand_swaps"] = run.recovery_swaps
        run.raw["recovery_policy"] = "deferred-balanced-v2" if run.recovery_balanced else "legacy"
        run.raw["recovery_crowded_single_candidates"] = run.recovery_opportunities
        run.raw["recovery_unresolved_crowded_singles"] = run.recovery_opportunities-run.recovery_swaps
        run.raw["recovery_deferred_notes"] = len(run.recovery_alternatives)
        run.raw["recovery_deferred_alternatives"] = sum(len(v) for v in run.recovery_alternatives.values())
    run.result.stages["positions"] = {"events": len(run.candidates), "retained_notes": len(run.state["notes"]),
                                      "elapsed_seconds": perf_counter() - run.started}
    run.raw.update(position_events=len(run.candidates), position_notes=len(run.state["notes"]))


def _positions_stage(run):
    _positions_begin(run)
    run.encoder(np.asarray([candidate["time"] for candidate in run.candidates], np.float64))
    for index, candidate in enumerate(run.candidates):
        output = _predict(run.predictors["positions"], _positions_inputs(run, candidate))
        _positions_apply(run, candidate, output)
        _progress(run.logger, "positions", index, len(run.candidates))
    _positions_end(run)


def _arrows_begin(run):
    """Index decoded notes and reset the kinematics/direction decode state."""
    run.started = perf_counter()
    index_state(run.state)
    run.history, run.kinematics = [], {}
    run.approach_prior = float(getattr(config, "v2_arrow_approach_prior", 0.) or 0.)
    run.approach_overrides = 0
    run.arrows_reference = run.predictors.get("arrows_reference")
    run.arrows_variety = float(getattr(config, "v2_arrows_variety", 0.) or 0.) if run.arrows_reference is not None else 0.
    run.variety_picks = 0


def _arrows_inputs(run, candidate):
    return _inputs("arrows", run.song, run.state, candidate, run.history, run.encoder, run.options,
                   options=run.prepared)


def _arrows_model(run):
    # Per-event model pick interpolates the resulting statistics exactly;
    # mixing softmaxes would flatten disagreement and overshoot the tails.
    model = (run.arrows_reference if run.arrows_variety > 0 and run.rng.random() < run.arrows_variety
             else run.predictors["arrows"])
    run.variety_picks += int(model is run.arrows_reference)
    return model


def _arrows_apply(run, candidate, output):
    """Assign decoded directions to this event's notes and track kinematics."""
    output = _biased(output, 9, 0 if run.sampling else run.factor, run.rng)
    pair = [9, 9]
    for hand in (0, 1):
        times, track = run.state["_tracks"][hand]
        location = int(np.searchsorted(times, candidate["time"]))
        if location < len(track) and track[location]["time"] == candidate["time"]:
            probabilities = output[("red", "blue")[hand]][0]
            direction = (_sample_category(probabilities, run.temperature, run.rng)
                         if run.sampling else int(probabilities.argmax()))
            prior = _approach_snap(run.kinematics.get(hand), track[location])
            if prior is not None and float(probabilities.max()) < run.approach_prior:
                direction = prior
                run.approach_overrides += 1
            note = track[location]
            note["d"] = direction
            pair[hand] = direction
            vector = _DIRECTION_VECTORS[direction]
            run.kinematics[hand] = (note["x"] + vector[0], note["y"] + vector[1],
                                    direction if direction != 8 else (run.kinematics.get(hand) or (0, 0, 1))[2])
    _history(run.history, {"time": candidate["time"], "value": pair}, run.length)


def _arrows_end(run):
    run.raw["arrow_approach_overrides"] = run.approach_overrides
    run.raw["arrow_variety"] = round(run.variety_picks / len(run.candidates), 4) if run.candidates else 0.
    run.result.stages["arrows"] = {"directed_notes": len(run.state["notes"])}
    run.raw["arrow_notes"] = len(run.state["notes"])
    run.state["raw_model_notes"] = np.asarray([[note[key] for key in ("id", "time", "hand", "x", "y", "d")]
                                               for note in run.state["notes"]], np.float64).reshape(-1, 6)
    run.result.diagnostics["raw_model_note_columns"] = ["id", "time_seconds", "hand", "x", "y", "direction"]
    run.state["notes"], diagnostics = run.postprocessors.postprocess_notes(
        run.state["notes"], [candidate["time"] for candidate in run.candidates], logger=run.logger)
    if run.recovery_alternatives:
        if run.playability_mode == "forward":
            run.result.diagnostics["decoder_recovery"] = {"ran": False}
        else:
            for note in run.state["notes"]:
                alternatives = run.recovery_alternatives.get(tuple(note[k] for k in ("time", "hand", "x", "y")))
                if alternatives:
                    note["_recovery_alternatives"] = deepcopy(alternatives)
    run.result.diagnostics["note_postprocessing"] = diagnostics
    run.result.stages["arrows"]["elapsed_seconds"] = perf_counter() - run.started


def _arrows_stage(run):
    _arrows_begin(run)
    for index, candidate in enumerate(run.candidates):
        inputs = _arrows_inputs(run, candidate)
        output = _predict(_arrows_model(run), inputs)
        _arrows_apply(run, candidate, output)
        _progress(run.logger, "arrows", index, len(run.candidates))
    _arrows_end(run)


def _note_cleanup(run):
    """Conditional application validators between arrow decode and expansion."""
    snap_times = getattr(run.postprocessors, "snap_note_times", None)
    if snap_times is not None and getattr(config, "v2_notes_onset_snap", False) and run.state["notes"]:
        snap_started = perf_counter()
        run.result.diagnostics["onset_snap"] = snap_times(run.state["notes"], run.state["chains"],
                                                          run.audio_path, run.duration, logger=run.logger)
        run.result.diagnostics["onset_snap"]["elapsed_seconds"] = perf_counter() - snap_started
    intro_fn = getattr(run.postprocessors, "apply_intro_floor", None)
    intro_floor = float(getattr(config, "v2_map_intro_no_notes_seconds", 0.) or 0.)
    # TODO: multiplier expansion copies with a negative dt would land back
    # inside this note-free zone (no second floor runs after expansion);
    # design-accepted today — revisit if a fitted vocabulary gains negative
    # offsets.
    if intro_fn is not None and intro_floor > 0:
        run.result.diagnostics["intro_zones"] = {
            "no_notes_seconds": intro_floor,
            "notes_dropped": intro_fn(run.state["notes"], intro_floor)}
    if run.playability_mode == "forward":
        repair = getattr(run.postprocessors, "repair_notes", None)
        if repair is not None and run.state["notes"]:
            run.result.diagnostics["forward_postprocessing"] = repair(
                run.state["notes"], rng=run.rng, logger=run.logger)
    fast_dir = getattr(run.postprocessors, "apply_fast_direction_rules", None)
    if fast_dir is not None and run.state["notes"] \
            and float(getattr(config, "v2_fast_gap_seconds", .2) or 0.) > 0:
        run.result.diagnostics["fast_directions"] = fast_dir(run.state["notes"], rng=run.rng,
                                                             logger=run.logger)
    diag_fn = getattr(run.postprocessors, "apply_diagonal_cap", None)
    diag_cap = float(getattr(config, "v2_diagonal_max_fraction", 1.) or 0.)
    if diag_fn is not None and run.state["notes"] and 0 <= diag_cap < 1:
        run.result.diagnostics["diagonal_cap"] = diag_fn(run.state["notes"], run.state["chains"], rng=run.rng)
    run.speed_cap = getattr(run.postprocessors, "apply_speed_cap", None)
    scale_fn = getattr(run.postprocessors, "speed_cap_scale", None)
    run.speed_scale = scale_fn() if scale_fn is not None \
        else float(getattr(config, "v2_speed_cap_scale", 1.) or 0.)
    run.combined_reach = (getattr(config, "v2_playability_enabled", False)
                          and (getattr(config, "v2_playability_reach", False)
                               or getattr(config, "v2_playability_priority", False)))
    if run.speed_cap is not None and run.state["notes"] and (run.speed_scale > 0 or run.combined_reach):
        # Primary pass while every note is still mutable; the retained
        # snapshot below records the capped set.
        run.result.diagnostics["speed_cap"] = run.speed_cap(run.state["notes"], run.state["chains"],
                                                            rng=run.rng, logger=run.logger)


def _multiplier_begin(run):
    """Snapshot retained notes and build the expansion anchor index."""
    run.started = perf_counter()
    run.retained = deepcopy([note for note in run.state["notes"] if note.get("kind") == "retained"])
    run.upstream = index_state({**run.state, "notes": run.retained, "chains": []})
    run.by_time = {}
    for note in run.retained:
        run.by_time.setdefault(note["time"], []).append(note)
    run.multiplier_candidates = [{"time": time, "count": len(notes)} for time, notes in sorted(run.by_time.items())]
    run.history, run.expansion_rejections = [], {}
    _log(run.logger, "V2 multiplier")


def _multiplier_inputs(run, candidate):
    return _inputs("multiplier", run.song, run.upstream, candidate, run.history, run.encoder, run.options,
                   options=run.prepared)


def _multiplier_apply(run, candidate, output):
    """Expand each anchor at this onset and roll back rejected expansions."""
    pair = [0, 0]
    for anchor in run.by_time[candidate["time"]]:
        hand = ("red", "blue")[anchor["hand"]]
        category = int(_distribution(output[hand], (1, len(run.vocabularies["multiplier"])), "expansion")[0].argmax())
        run.raw["multiplier_non_none"] += int(category != 0)
        if getattr(config, "single_notes_only_flag", False):
            category = 0
        old_notes, old_chains = list(run.state["notes"]), list(run.state["chains"])
        accepted = _atomic_expand(run.state, anchor, run.vocabularies["multiplier"][category], run.duration,
                                  run.result.diagnostics["rejected_expansions"])
        if accepted and category:
            proposed = len(run.state["notes"]), len(run.state["chains"])
            _rejections(run.expansion_rejections, run.postprocessors.validate_expansions(run.state, run.song, logger=None))
            if (len(run.state["notes"]), len(run.state["chains"])) != proposed:
                run.state["notes"], run.state["chains"] = old_notes, old_chains
                run.result.diagnostics["rejected_expansions"].append({"anchor_id": anchor["id"], "reason": "expansion clearance"})
                accepted = False
        if accepted:
            pair[anchor["hand"]] = category
    _history(run.history, {"time": candidate["time"], "value": pair}, run.length)


def _multiplier_end(run):
    run.result.stages["multiplier"] = {"retained_notes": len(run.retained),
                                       "added_notes": sum(note.get("kind") == "copy" for note in run.state["notes"]),
                                       "chains": len(run.state["chains"]),
                                       "chain_slices": sum(chain["slices"] for chain in run.state["chains"])}
    run.result.diagnostics["expansion_clearance"] = run.postprocessors.validate_expansions(run.state, run.song, logger=run.logger)
    _rejections(run.expansion_rejections, run.result.diagnostics["expansion_clearance"])
    run.result.diagnostics["expansion_clearance"].update(run.expansion_rejections)
    run.result.stages["multiplier"]["elapsed_seconds"] = perf_counter() - run.started


def _multiplier_stage(run):
    _multiplier_begin(run)
    run.encoder(np.asarray(list(sorted(run.by_time)), np.float64))
    for index, candidate in enumerate(run.multiplier_candidates):
        output = _predict(run.predictors["multiplier"], _multiplier_inputs(run, candidate))
        _multiplier_apply(run, candidate, output)
        _progress(run.logger, "multiplier", index, len(run.multiplier_candidates))
    _multiplier_end(run)


def _expansion_cleanup(run):
    """Post-expansion validators that keep the retained snapshot frozen."""
    enforce_flow = getattr(run.postprocessors, "enforce_note_flow_and_clearance", None)
    if enforce_flow is not None:
        clearance = enforce_flow(run.state["notes"], run.state["chains"])
        run.result.diagnostics["same_direction_notes_flipped"] = clearance["flow_recuts"]
        run.result.diagnostics["note_clearance"] = clearance
    if run.speed_cap is not None and run.state["notes"] and run.speed_scale > 0:
        # Post-expansion pass: retained notes must stay dict-equal to the
        # snapshot until the end-of-run check, so only copies and heuristic
        # notes may still drop or recut.
        run.result.diagnostics["speed_cap_post_expansion"] = run.speed_cap(
            run.state["notes"], run.state["chains"], rng=run.rng, logger=run.logger,
            drop_retained=False)
    counter_sweep = getattr(run.postprocessors, "apply_fast_counter_sweep", None)
    if (counter_sweep is not None and run.state["notes"]
            and getattr(config, "v2_fast_counter_direction", "recut") == "recut"
            and float(getattr(config, "v2_fast_gap_seconds", .2) or 0.) > 0):
        # Copies inherit their anchor's cut, so expansion and the speed cap's
        # deletions can leave fast pairs the pre-snapshot rules never saw:
        # sweep counter directions over non-retained notes only, keeping the
        # retained-snapshot invariant verified at the end of generate().
        run.result.diagnostics["fast_counter_sweep"] = counter_sweep(
            run.state["notes"], rng=run.rng, logger=run.logger)
    reach_finalize = getattr(run.postprocessors, "finalize_playability_reach", None)
    if run.combined_reach and reach_finalize is not None:
        run.result.diagnostics["reach_post_expansion"] = reach_finalize(
            run.state["notes"], run.state["chains"], protect_retained=True)
    emphasis = getattr(run.postprocessors, "apply_note_emphasis", None)
    if emphasis is not None:
        emphasis_started = perf_counter()
        run.result.diagnostics["note_emphasis"] = emphasis(
            run.state, run.song, rng=run.rng, logger=run.logger)
        run.result.stages["emphasis"] = {
            "added_notes": run.result.diagnostics["note_emphasis"].get("added_notes", 0),
            "elapsed_seconds": perf_counter() - emphasis_started}
    drop_guard = getattr(run.postprocessors, "drop_guard", None)
    if run.playability_mode == "forward" and drop_guard is not None:
        drop_guard(run.state)


def _hazards_stage(run):
    """Heuristic or learned hazard proposals; decode semantics unchanged."""
    started = perf_counter()
    index_state(run.state)
    if getattr(config, "v2_walls_heuristic", False):
        _log(run.logger, "V2 hazards (heuristic)")
        proposals = (_heuristic_wall_candidates(run.state["notes"], run.state["event_times"])
                     if config.add_obstacle_flag else [])
        bomb_proposals = (_heuristic_bomb_candidates(run.state, run.rng)
                          if config.add_bombs_flag else [])
        run.state["walls"], run.state["bombs"] = proposals, bomb_proposals
        run.raw["wall_proposals"] = len(proposals)
        run.raw["bomb_proposals"] = len(bomb_proposals)
        run.result.diagnostics["hazard_clearance"] = run.postprocessors.validate_hazards(run.state, run.song, logger=run.logger)
        run.result.stages["hazards"] = {"mode": "heuristic", "walls": len(run.state["walls"]),
                                        "bombs": len(run.state["bombs"]),
                                        "elapsed_seconds": perf_counter() - started}
    elif repaired(run.options, "hazards"):
        from training.pipeline_v2.hazard_sections import rollout
        report = rollout(run.song, run.state, run.vocabularies, run.options,
                         lambda inputs: _predict(run.predictors["hazards"], inputs), run.encoder,
                         progress=lambda index, total: _progress(run.logger, "hazards", index, total))
        run.raw["wall_proposals"] = report["wall_section_proposals"]
        run.raw["bomb_proposals"] = report["bomb_proposals"]
        report["application_filters"] = run.postprocessors.validate_hazards(run.state, run.song, logger=run.logger)
        run.result.diagnostics["hazard_clearance"] = report
        run.result.stages["hazards"] = {**report, "walls": len(run.state["walls"]), "bombs": len(run.state["bombs"]),
                                        "elapsed_seconds": perf_counter() - started}
    else:
        queries = hazard_candidates(expansion_occupancy_notes(run.state), run.duration, run.bpm, run.schedule)
        run.encoder(np.asarray([query["time"] for query in queries], np.float64))
        run.history, hazard_rejections = [], {}
        _log(run.logger, "V2 hazards")
        for index, candidate in enumerate(queries):
            output = _predict(run.predictors["hazards"],
                              _inputs("hazards", run.song, run.state, candidate, run.history, run.encoder,
                                      run.options, options=run.prepared))
            wall = int(_distribution(output["wall"], (1, len(run.vocabularies["walls"])), "wall")[0].argmax())
            bombs = _multilabel(output["bombs"], (1, 12), "bomb")[0] >= run.options["bomb_threshold"]
            previous_walls, previous_bombs = len(run.state["walls"]), len(run.state["bombs"])
            if candidate["kind"] == "wall" and wall:
                run.raw["wall_proposals"] += 1
                run.state["walls"].append({**deepcopy(run.vocabularies["walls"][wall]), "time": candidate["time"],
                                           "end_time": candidate["end_time"], "template_id": wall})
            elif candidate["kind"] == "bomb":
                run.raw["bomb_proposals"] += int(bombs.sum())
                run.state["bombs"].extend({"time": candidate["time"], "x": int(cell % 4), "y": int(cell // 4)} for cell in np.flatnonzero(bombs))
            # Clearance is applied before accepted outputs enter the next history.
            _rejections(hazard_rejections, run.postprocessors.validate_hazards(run.state, run.song, logger=None))
            accepted_wall = run.state["walls"][-1]["template_id"] if len(run.state["walls"]) > previous_walls else 0
            accepted_bombs = np.zeros(12, np.int32)
            for bomb in run.state["bombs"][previous_bombs:]:
                accepted_bombs[4 * bomb["y"] + bomb["x"]] = 1
            for record in run.history:
                if record["time"] == candidate["time"] and record.get("kind") == "bomb":
                    record["value"][1:] = [int(any(bomb["time"] == candidate["time"] and 4 * bomb["y"] + bomb["x"] == cell
                                                    for bomb in run.state["bombs"])) for cell in range(12)]
            _history(run.history, {"time": candidate["time"], "kind": candidate["kind"],
                                   "value": [accepted_wall, *accepted_bombs.tolist()]}, run.length)
            _progress(run.logger, "hazards", index, len(queries))
        run.result.diagnostics["hazard_clearance"] = run.postprocessors.validate_hazards(run.state, run.song, logger=run.logger)
        _rejections(hazard_rejections, run.result.diagnostics["hazard_clearance"])
        run.result.diagnostics["hazard_clearance"].update(hazard_rejections)
        run.result.stages["hazards"] = {"queries": len(queries), "walls": len(run.state["walls"]), "bombs": len(run.state["bombs"]),
                                        "elapsed_seconds": perf_counter() - started}


def _arcs_stage(run):
    """Heuristic or learned arc attachments; decode semantics unchanged."""
    started = perf_counter()
    if getattr(config, "v2_arcs_heuristic", False):
        _log(run.logger, "V2 arcs (heuristic)")
        radius = float(config.slider_radius_multiplier)
        proposed = _heuristic_arc_candidates(run.state["notes"], radius)
        run.state["arcs"] = proposed
        run.raw["arc_attachments"] = len(proposed)
        run.result.diagnostics["arc_clearance"] = run.postprocessors.validate_arcs(
            run.state, run.song, logger=run.logger, settings={"stage_policies": {}})
        run.result.stages["arcs"] = {"mode": "heuristic", "queries": len(proposed),
                                     "arcs": len(run.state["arcs"]),
                                     "elapsed_seconds": perf_counter() - started}
    else:
        queries = arc_candidates(run.state["notes"], run.options)
        run.state["_arc_options"] = run.options
        run.encoder(np.asarray([query["time"] for query in queries], np.float64))
        run.history, arc_rejections = [], {}
        _log(run.logger, "V2 arcs")
        lower, upper = run.options["arc_shape_bounds"]
        for index, candidate in enumerate(queries):
            output = _predict(run.predictors["arcs"],
                              _inputs("arcs", run.song, run.state, candidate, run.history, run.encoder,
                                      run.options, options=run.prepared))
            attach_probabilities = _distribution(output["attach"], (1, 2), "arc attachment")[0]
            attach = (int(attach_probabilities[1] >= run.options.get("arc_threshold", .5))
                      if repaired(run.options, "arcs") else int(attach_probabilities.argmax()))
            run.raw["arc_attachments"] += attach
            shape = np.asarray(output["shape"], np.float32)
            if shape.shape != (1, 2) or not np.isfinite(shape).all() or np.any((shape < lower) | (shape > upper)):
                raise ValueError("Invalid arc controls")
            anchor = int(_distribution(output["anchor"], (1, 3), "arc anchor")[0].argmax())
            previous = len(run.state["arcs"])
            if attach:
                run.state["arcs"].append({**deepcopy(candidate), "mu": float(shape[0, 0]),
                                          "tmu": float(shape[0, 1]), "anchor_mode": anchor})
                _rejections(arc_rejections, run.postprocessors.validate_arcs(run.state, run.song, logger=None))
            attach = int(len(run.state["arcs"]) > previous)
            _history(run.history, {"time": candidate["time"],
                                   "value": history_value({"attach": attach, "shape": shape[0].tolist(),
                                                           "anchor": anchor})}, run.length)
            _progress(run.logger, "arcs", index, len(queries))
        run.result.diagnostics["arc_clearance"] = run.postprocessors.validate_arcs(run.state, run.song, logger=run.logger)
        _rejections(arc_rejections, run.result.diagnostics["arc_clearance"])
        run.result.diagnostics["arc_clearance"].update(arc_rejections)
        run.result.stages["arcs"] = {"mode": "model", "queries": len(queries), "arcs": len(run.state["arcs"]),
                                     "elapsed_seconds": perf_counter() - started}


def _calm_stage(run):
    """Decorative wall/bomb/lighting pulses inside intentionally note-free gaps."""
    started = perf_counter()
    run.calm_pulses = []
    pulse_fn, place_fn = (getattr(run.postprocessors, "calm_area_pulses", None),
                          getattr(run.postprocessors, "add_calm_effects", None))
    if (pulse_fn is not None and place_fn is not None
            and getattr(config, "v2_calm_effects", False)
            and any(getattr(config, name, True) for name in
                    ("v2_calm_effects_walls", "v2_calm_effects_bombs", "v2_calm_effects_lighting"))):
        # Decorative beat pulses in note-free stretches: edge wall flashes,
        # bottom-corner bombs and later laser flashes. Notes stay untouched.
        _log(run.logger, "V2 calm effects")
        run.calm_pulses, diagnostics = pulse_fn(run.state, run.song)
        diagnostics.update(place_fn(run.state, run.song, run.calm_pulses))
        if diagnostics["walls"] or diagnostics["bombs"]:
            diagnostics["hazard_clearance"] = run.postprocessors.validate_hazards(run.state, run.song, logger=run.logger)
        run.result.diagnostics["calm_effects"] = diagnostics
        run.result.stages["calm_effects"] = {"pulses": len(run.calm_pulses),
                                             "elapsed_seconds": perf_counter() - started}


def _lighting_stage(run):
    """Retained v1 lighting plus optional calm laser flashes."""
    started = perf_counter()
    _log(run.logger, "V1 lighting")
    run.state["lighting"], lighting_report = run.lighting_generator(run.state["event_times"], run.encoder)
    if run.calm_pulses and getattr(config, "v2_calm_effects_lighting", False) \
            and getattr(run.postprocessors, "add_calm_lighting", None) is not None:
        added = run.postprocessors.add_calm_lighting(run.state, run.calm_pulses)
        run.result.diagnostics["calm_effects"]["lighting_events"] = added
        run.result.stages["calm_effects"]["lighting_events"] = added
    run.raw["lighting_commands"] = len(run.state["lighting"])
    run.result.stages["lighting"] = {**lighting_report, "events": len(run.state["lighting"]),
                                     "elapsed_seconds": perf_counter() - started}


def _finalize(run):
    """Audit the frozen retained snapshot and finish the diagnostics record."""
    if [note for note in run.state["notes"] if note.get("kind") == "retained"] != run.retained:
        raise ValueError("A downstream stage changed retained notes")
    playability_findings = _validate_state(
        run.state, run.duration, playability_mode=run.playability_mode)
    run.result.diagnostics["playability_policy"] = run.playability_mode
    run.result.diagnostics["playability_strict_validated"] = (
        run.playability_mode == "search"
        and getattr(config, "v2_playability_enabled", False)
        and (getattr(config, "v2_playability_reach", False)
             or getattr(config, "v2_playability_priority", False)))
    if run.playability_mode == "forward":
        run.result.diagnostics["playability_findings"] = playability_findings or {}
        run.result.diagnostics["hard_direction_validated"] = True
        run.result.diagnostics["hard_dropped_final"] = (
            playability_findings or {}).get("hard_dropped_final", 0)
    run.result.diagnostics["final_counts"] = {
        "colored_notes": len(run.state["notes"]),
        "retained_notes": sum(note.get("kind") == "retained" for note in run.state["notes"]),
        "heuristic_notes": sum(note.get("kind") == "heuristic" for note in run.state["notes"]),
        "learned_copies": sum(note.get("kind") == "copy" and note.get("source") != "emphasis"
                              for note in run.state["notes"]),
        "emphasis_copies": sum(note.get("kind") == "copy" and note.get("source") == "emphasis"
                               for note in run.state["notes"]),
        "unique_onsets": len({note["time"] for note in run.state["notes"]}),
        "chains": len(run.state["chains"]), "chain_slices": sum(chain["slices"] for chain in run.state["chains"]),
        "walls": len(run.state["walls"]), "bombs": len(run.state["bombs"]),
        "arcs": len(run.state["arcs"]), "lighting_events": len(run.state["lighting"])}
    run.state.pop("_arc_options", None)
    run.state.pop("_tracks", None)
    run.state.pop("_occupancy", None)
    run.result.diagnostics.update(export_ready=True, production_validated=True, safety_reviewed=False,
                                  gameplay_validated=False,
                                  encoder=getattr(run.encoder, "metadata", {}),
                                  requested_count_guaranteed=False)
    return run.result


def _validate_state(state, duration, *, playability_mode="search"):
    """Check the final object contract after all mutating postprocessors.

    In forward mode residual flow/fast-counter/overlap stragglers are
    removed as the last net (with the chains and arcs headed by them and
    notes anchored on them) instead of failing; the survivors are
    re-audited and only a still non-empty residual or a structural
    contract failure raises."""
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
    findings = {}
    if (getattr(config, "v2_playability_enabled", False)
            and getattr(config, "v2_playability_reach", False)):
        from .v2_playability_reach import forbidden_cut
        forbidden = [note["id"] for note in state["notes"] if forbidden_cut(note)]
        if forbidden:
            if playability_mode != "forward":
                raise ValueError(f"Final reach policy contains forbidden cuts: {forbidden[:10]}")
            findings["forbidden_cut_ids"] = forbidden
    if (getattr(config, "v2_playability_enabled", False)
            and getattr(config, "v2_playability_priority", False)):
        from .v2_playability_priority import audit_priority
        audit = audit_priority(state["notes"])
        if audit["preparation"]["hard"] or audit["crossing"]["hard"]:
            if playability_mode != "forward":
                raise ValueError("Final priority policy contains hard preparation or crossing violations")
            findings["priority_hard"] = {
                "preparation": audit["preparation"]["hard"],
                "crossing": audit["crossing"]["hard"]}
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
    if playability_mode == "forward":
        from .v2_forward_directions import audit_directions, audit_overlaps
        direction = audit_directions(state["notes"])
        overlap = audit_overlaps(state["notes"])
        residual = {*direction["flow"], *direction["fast_counter"],
                    *overlap["overlap"]}
        if residual:
            # Last net: drop the stragglers the bounded passes could not
            # repair instead of failing the package — plus the chains and
            # arcs headed by them and any notes anchored on them.
            dropped_ids = set(residual)
            expanded = True
            while expanded:
                extra = {note["id"] for note in state["notes"]
                         if note["id"] not in dropped_ids
                         and note.get("anchor_id") in dropped_ids}
                expanded = bool(extra)
                dropped_ids |= extra
            state["notes"] = [note for note in state["notes"]
                              if note["id"] not in dropped_ids]
            state["chains"] = [chain for chain in state["chains"]
                               if chain["anchor_id"] not in dropped_ids]
            state["arcs"] = [arc for arc in state["arcs"]
                             if dropped_ids.isdisjoint(
                                 arc.get("endpoint_ids", ()))]
            findings["hard_dropped_final"] = len(dropped_ids)
            direction = audit_directions(state["notes"])
            overlap = audit_overlaps(state["notes"])
            remaining_ids = {note["id"] for note in state["notes"]}
            dangling = (
                any(note.get("anchor_id") is not None
                    and note["anchor_id"] not in remaining_ids
                    for note in state["notes"])
                or any(chain["anchor_id"] not in remaining_ids
                       for chain in state["chains"])
                or any(not set(arc.get("endpoint_ids", ())) <= remaining_ids
                       for arc in state["arcs"]))
            if (direction["flow"] or direction["fast_counter"]
                    or overlap["overlap"] or dangling):
                raise ValueError(
                    "Final forward direction audit contains enabled flow, "
                    "fast-counter or same-cell overlap violations")
        from .v2_postprocessing import clearance_violations
        findings["clearance"] = clearance_violations(state["notes"])
    return findings


def _scope_playability_flow(generate_map):
    @wraps(generate_map)
    def scoped(*args, **kwargs):
        if config.v2_playability_enabled and config.v2_playability_late_turns:
            from map_creation.v2_postprocessing import playability_generation_phase
            with playability_generation_phase():
                return generate_map(*args, **kwargs)
        return generate_map(*args, **kwargs)
    return scoped


@_scope_playability_flow
def generate(song, requested_nps, predictors, encoder, settings, vocabularies, *,
             lighting_generator, audio_path=None, logger=None, postprocessors=None, random_factor=None,
             intensity=None, rng=None):
    """Compose seven v2 stages, retained v1 lighting, and application validators."""
    run = variant_run(song, requested_nps, predictors, encoder, settings, vocabularies,
                      lighting_generator=lighting_generator, audio_path=audio_path, logger=logger,
                      postprocessors=postprocessors, random_factor=random_factor,
                      intensity=intensity, rng=rng)
    _beats_stage(run)
    _positions_stage(run)
    _arrows_stage(run)
    _note_cleanup(run)
    _multiplier_stage(run)
    _expansion_cleanup(run)
    _hazards_stage(run)
    _arcs_stage(run)
    _calm_stage(run)
    _lighting_stage(run)
    return _finalize(run)


@contextmanager
def _late_turns_phase(enabled):
    """Replay ``@_scope_playability_flow`` for one batched variant's own pieces."""
    if enabled:
        from .v2_postprocessing import playability_generation_phase
        with playability_generation_phase():
            yield
    else:
        yield


@contextmanager
def _variant_scope(run):
    """Enter a batched variant's settings scope for one per-variant piece.

    Sequential ``generate`` holds one ``playability_generation_phase`` for
    the whole call, so ``_LATE_TURNS_READY`` stays True once
    ``apply_speed_cap`` flips it inside ``_note_cleanup``. Re-entering the
    phase per piece would reset it to False and suppress eligible slow
    turns in every later validator; the run therefore carries the flag
    value between pieces instead.
    """
    scope = run.scope() if callable(run.scope) else (run.scope or nullcontext())
    with scope:
        if run.late_turns:
            from .v2_postprocessing import _LATE_TURNS_READY, playability_generation_phase
            with playability_generation_phase():
                _LATE_TURNS_READY.set(run.late_ready)
                try:
                    yield
                finally:
                    run.late_ready = _LATE_TURNS_READY.get()
        else:
            yield


def _batched_inputs(rows):
    """Concatenate same-key batch-1 input dicts into one batch-K dict."""
    if len(rows) == 1:
        return rows[0]
    assert all(row.keys() == rows[0].keys() for row in rows), \
        "Lockstep rows for one model must share the same input keys"
    return {key: np.concatenate([row[key] for row in rows], axis=0) for key in rows[0]}


def _batch_row(output, row):
    """Slice one batch-1 output back out of a batched predict result."""
    if isinstance(output, dict):
        return {key: value[row:row + 1] for key, value in output.items()}
    return output[row:row + 1]


def _lockstep(runs, name, candidates, inputs_fn, pick_fn, apply_fn):
    """Advance live variants through one autoregressive stage, batching predicts.

    Per step every live variant builds its inputs inside its own scope, then
    rows group by chosen model (the arrows variety draw happens at the same
    rng position as the sequential loop) into at most one batched predict per
    model; rows keep variant order so outputs split back positionally. A
    variant past its own candidate list contributes no row that step.
    ``run.clock_share`` accumulates each variant's own piece time plus its
    even share of every shared predict so ``elapsed_seconds`` stays honest
    instead of every variant bracketing the whole shared loop.
    """
    total = max((len(candidates(run)) for run in runs), default=0)
    for index in range(total):
        groups = {}
        for run in runs:
            if index >= len(candidates(run)):
                continue
            started = perf_counter()
            with _variant_scope(run):
                inputs = inputs_fn(run, candidates(run)[index])
                predictor = pick_fn(run)
            run.clock_share += perf_counter() - started
            groups.setdefault(predictor, []).append((run, inputs))
        for predictor, rows in groups.items():
            started = perf_counter()
            output = _predict(predictor, _batched_inputs([inputs for _, inputs in rows]))
            share = (perf_counter() - started) / len(rows)
            for row, (run, _) in enumerate(rows):
                started = perf_counter()
                with _variant_scope(run):
                    apply_fn(run, candidates(run)[index], _batch_row(output, row))
                    _progress(run.logger, name, index, len(candidates(run)))
                run.clock_share += share + perf_counter() - started


def _prefetch_union(encoder, runs, times):
    """Warm the shared latent cache once for all live variants of a stage."""
    queries = [np.asarray(list(times(run)), np.float64) for run in runs]
    queries = [query for query in queries if len(query)]
    if queries:
        encoder(np.unique(np.concatenate(queries)))


def generate_variants(song, requested_nps_list, predictors, encoder, settings_list, vocabularies, *,
                      lighting_generator, audio_path=None, logger=None, postprocessors=None,
                      random_factor=None, intensity=None, rngs=None, scopes=None):
    """Compose ``generate`` for several variants, batching the three decode loops.

    Stage order matches the sequential composition exactly; per-variant
    begin/apply/validator pieces run inside that variant's ``scopes`` entry
    (callables producing a fresh context manager, e.g. a replayed
    difficulty-settings delta) so config reads keep their program points —
    only the concatenated ``_predict`` is shared. ``scopes`` default to the
    ambient config, making K=1 equivalent to ``generate``.
    """
    count = len(requested_nps_list)
    if (len(settings_list) != count or rngs is not None and len(rngs) != count
            or scopes is not None and len(scopes) != count):
        raise ValueError("Variant settings, rngs and scopes must match the requested variants")
    if scopes is not None and any(not callable(scope) for scope in scopes):
        raise ValueError("Variant scopes must be callables producing a context manager, "
                         "not already-entered managers")
    scopes = [nullcontext] * count if scopes is None else scopes
    runs = []
    for index in range(count):
        with scopes[index]():
            # The @_scope_playability_flow flag is read at generate entry, so
            # evaluate it under this variant's scope and replay it per piece.
            late_turns = bool(getattr(config, "v2_playability_enabled", False)
                              and getattr(config, "v2_playability_late_turns", False))
            with _late_turns_phase(late_turns):
                run = variant_run(song, requested_nps_list[index], predictors, encoder,
                                  settings_list[index], vocabularies, lighting_generator=lighting_generator,
                                  audio_path=audio_path, logger=logger, postprocessors=postprocessors,
                                  random_factor=random_factor, intensity=intensity,
                                  rng=None if rngs is None else rngs[index])
            run.scope, run.late_turns = scopes[index], late_turns
            runs.append(run)
    def each(function):
        for run in runs:
            with _variant_scope(run):
                function(run)

    def timed_each(function):
        """Per-variant piece whose own duration feeds the stage's elapsed."""
        for run in runs:
            started = perf_counter()
            with _variant_scope(run):
                function(run)
            run.clock_share += perf_counter() - started

    def prefetch(times):
        # The union prefetch is shared work: split it evenly so per-variant
        # stage elapsed_seconds still sum to the real wall time.
        started = perf_counter()
        _prefetch_union(encoder, runs, times)
        share = (perf_counter() - started) / len(runs)
        for run in runs:
            run.clock_share += share

    def close(function):
        """End a lockstep stage with the variant's attributed share of it."""
        for run in runs:
            run.started = perf_counter() - run.clock_share
            with _variant_scope(run):
                function(run)
            run.clock_share = 0.

    for index, run in enumerate(runs):
        # Raw intensity shares from the first variant seed the rest, matching
        # runtime.generate's prepared.intensity reuse across difficulties.
        if index and run.intensity is None:
            run.intensity = runs[0].state["raw_intensity"].copy()
        with _variant_scope(run):
            _beats_stage(run)
    timed_each(_positions_begin)
    prefetch(lambda run: (candidate["time"] for candidate in run.candidates))
    _lockstep(runs, "positions", lambda run: run.candidates, _positions_inputs,
              lambda run: run.predictors["positions"], _positions_apply)
    close(_positions_end)
    timed_each(_arrows_begin)
    _lockstep(runs, "arrows", lambda run: run.candidates, _arrows_inputs, _arrows_model, _arrows_apply)
    close(_arrows_end)
    each(_note_cleanup)
    timed_each(_multiplier_begin)
    prefetch(lambda run: (candidate["time"] for candidate in run.multiplier_candidates))
    _lockstep(runs, "multiplier", lambda run: run.multiplier_candidates, _multiplier_inputs,
              lambda run: run.predictors["multiplier"], _multiplier_apply)
    close(_multiplier_end)
    each(_expansion_cleanup)
    each(_hazards_stage)
    each(_arcs_stage)
    each(_calm_stage)
    each(_lighting_stage)
    results = []
    each(lambda run: results.append(_finalize(run)))
    return results
