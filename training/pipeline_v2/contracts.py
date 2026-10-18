"""Independent v2 contracts and shared teacher-forced/inference input builders.

Times are elapsed seconds. The existing beat-model Song and its targets remain
unchanged. Source JSON is retained in object records, never rewritten.
"""

from dataclasses import dataclass, field

import numpy as np

from training.beat_models.contracts import FRAME_SECONDS, Song, validate_song

HISTORY = 32
LIGHTING_BLOCK = 16
NEW_STAGES = ("positions", "arrows", "multiplier", "hazards", "arcs", "lighting")
SCHEMA_VERSION = 1


@dataclass
class StageExamples:
    inputs: dict
    targets: dict
    weights: dict
    times: np.ndarray


@dataclass
class V2Song:
    base: Song
    objects: dict = field(default_factory=dict)
    examples: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    latents: dict = field(default_factory=dict)


@dataclass
class V2Dataset:
    songs: list
    vocabularies: dict
    manifest: dict


def preparation_defaults(settings=None):
    from .settings import preparation_settings

    required = ("history_length", "lighting_block_length", "encoder_batch_size", "encoder_grid_seconds",
                "max_multiplier_templates", "max_wall_templates", "max_light_classes", "max_copies",
                "max_chain_slices", "max_chain_span_seconds", "expansion_time_quantum", "chain_span_quantum",
                "lighting_match_tolerance", "arc_shape_bounds", "schedule", "bomb_threshold", "lighting_threshold")
    if settings is not None and all(key in settings for key in required):
        defaults = dict(settings)
    else:
        defaults = preparation_settings()
        if settings is not None and "stage_policies" not in settings:
            defaults.pop("stage_policies", None)
        defaults.update(settings or {})
    for key in ("history_length", "lighting_block_length", "encoder_batch_size",
                "max_multiplier_templates", "max_wall_templates", "max_light_classes",
                "max_copies", "max_chain_slices"):
        value = defaults[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("encoder_grid_seconds", "max_chain_span_seconds", "expansion_time_quantum", "chain_span_quantum"):
        if not np.isfinite(defaults[key]) or defaults[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not np.isfinite(defaults["lighting_match_tolerance"]) or defaults["lighting_match_tolerance"] < 0:
        raise ValueError("lighting_match_tolerance must be nonnegative")
    bounds = defaults["arc_shape_bounds"]
    if len(bounds) != 2 or not np.isfinite(bounds).all() or not 0 <= bounds[0] < bounds[1]:
        raise ValueError("arc_shape_bounds must be ordered nonnegative bounds")
    return defaults


def _base(song):
    return song.base if isinstance(song, V2Song) else song


def _history_value(stage, record, width):
    if stage == "arcs":
        from .arc_policy import history_value
        return np.asarray(history_value(record), np.float32)
    if "value" in record:
        return np.asarray(record["value"]).reshape(width)
    if stage in ("positions", "arrows", "multiplier"):
        return np.asarray([record.get("red", 0), record.get("blue", 0)])
    if stage == "hazards":
        return np.r_[record.get("wall", 0), record.get("bombs", np.zeros(12))]
    if stage == "arcs":
        return np.r_[record.get("attach", 0), record.get("shape", [0., 0.]), record.get("anchor", 0)]
    return np.asarray(record.get("events", np.zeros(width)))


def expansion_occupancy_notes(state):
    """Chain slices are clearance samples, not extra ordinary-note targets."""
    notes = list(state.get("notes", []))
    for chain in state.get("chains", []):
        slices = chain.get("virtual_slices")
        if slices is not None:
            notes.extend(slices)
            continue
        count = max(2, int(chain.get("slices", 2)))
        for index in range(1, count):
            alpha = index / (count - 1)
            notes.append({"time": chain["time"] + alpha * (chain["end_time"] - chain["time"]),
                          "hand": chain["hand"],
                          "x": float(chain["x"]) + alpha * (chain["tx"] - chain["x"]),
                          "y": float(chain["y"]) + alpha * (chain["ty"] - chain["y"])})
    return notes


def index_state(state):
    """Attach optional prepared indexes for repeated input queries on a pass.

    Call again after upstream note/chain structure changes. Note references stay
    live when the arrow pass fills directions. Composition may omit indexes;
    the same input semantics then use a direct scan.
    """
    state["_tracks"] = {}
    for hand in (0, 1):
        notes = sorted((item for item in state.get("notes", []) if item["hand"] == hand), key=lambda item: item["time"])
        state["_tracks"][hand] = (np.asarray([item["time"] for item in notes], np.float64), notes)
    occupied = sorted(expansion_occupancy_notes(state), key=lambda item: item["time"])
    state["_occupancy"] = (np.asarray([item["time"] for item in occupied], np.float64), occupied)
    return state


def build_inputs(stage, song, state, candidate, own_history, encoder=None, settings=None):
    """One input path for both training and composition, without target leakage.

    State contains upstream notes {id,time,hand,x,y,d} and optional chains. Each
    history record contains time and the stage's output keys (or ``value``).
    Only records strictly before the candidate/block start enter history.
    """
    if stage not in NEW_STAGES:
        raise ValueError(f"Unknown v2 stage: {stage}")
    options = preparation_defaults(settings)
    base = _base(song)
    now = float(candidate["time"])
    if not np.isfinite(now) or not 0 <= now < base.metadata["duration"]:
        raise ValueError("Candidate time is outside decoded audio")
    # Select strictly preceding records first so equal-time sibling decisions do
    # not evict valid history. Stop as soon as the bounded input is full.
    histories = []
    limit = max(options["history_length"], options["lighting_block_length"])
    for item in reversed(own_history):
        if float(item["time"]) < now:
            histories.append(item)
            if len(histories) == limit:
                break
    histories.reverse()
    get_audio = encoder or (lambda times: np.zeros((len(times), 16), np.float32))
    if stage == "lighting":
        length = options["lighting_block_length"]
        times = np.asarray(candidate.get("times", [now]), dtype=np.float64)
        if times.ndim != 1 or len(times) > length or not np.isfinite(times).all():
            raise ValueError("Invalid lighting block times")
        valid = np.asarray(candidate.get("mask", np.ones(len(times), bool)), bool)
        if valid.shape == (length,) and len(times) < length and not valid[len(times):].any():
            valid = valid[:len(times)]
        if valid.shape != times.shape or np.any(times[valid] < now) or np.any(times[valid] >= base.metadata["duration"]):
            raise ValueError("Invalid lighting block mask/timeline")
        classes = int(state.get("light_classes", 1))
        audio = np.zeros((1, length, 16, 1), np.float32)
        gaps = np.zeros((1, length, 1), np.float32)
        mask = np.zeros((1, length), bool)
        mask[0, :len(times)] = valid
        if valid.any():
            audio[0, np.flatnonzero(valid), :, 0] = get_audio(times[valid])
            previous = histories[-1]["time"] if histories else 0.
            gaps[0, np.flatnonzero(valid), 0] = np.diff(np.r_[previous, times[valid]])
        history = np.zeros((1, length, classes), np.float32)
        history_mask = np.zeros((1, length), bool)
        previous = histories[-length:]
        if previous:
            history[0, -len(previous):] = [_history_value(stage, item, classes) for item in previous]
            history_mask[0, -len(previous):] = True
        return {"audio": audio, "gaps": gaps, "history": history, "history_mask": history_mask, "mask": mask}
    length = options["history_length"]
    width = 13 if stage == "hazards" else 4 if stage == "arcs" else 2
    categorical = stage in ("positions", "arrows", "multiplier")
    history = np.zeros((1, length, width), np.int32 if categorical else np.float32)
    if categorical:
        start = 13 if stage == "positions" else 10 if stage == "arrows" else int(state.get("multiplier_classes", 1))
        history.fill(start + 1)
    age = np.zeros((1, length, 1), np.float32)
    mask = np.zeros((1, length), bool)
    preceding = histories[-length:]
    if preceding:
        history[0, :len(preceding)] = [_history_value(stage, item, width) for item in preceding]
        age[0, :len(preceding), 0] = [now - item["time"] for item in preceding]
        mask[0, :len(preceding)] = True
    elif categorical:
        history[0, 0] = start
        mask[0, 0] = True
    shares = state.get("shares", state.get("intensity"))
    if shares is None:
        shares = base.arrays.get("intensity", np.zeros_like(base.arrays["bin_durations"]))
    if not np.sum(shares):
        shares = base.arrays["bin_durations"] / base.metadata["duration"]
    density = float(state.get("requested_nps", base.metadata.get("difficulty_nps", 0.)))
    frame = min(int(np.floor(now / FRAME_SECONDS + .5)), len(base.arrays["frame_times"]) - 1)
    bin_index = min(int(base.arrays["frame_times"][frame]), len(shares) - 1)
    relative = float(shares[bin_index]) * base.metadata["duration"] / float(base.arrays["bin_durations"][bin_index])
    conditions = np.asarray([relative, density / 10, relative * density / 10, base.metadata["bpm"] / 120], np.float32)
    gap = now - histories[-1]["time"] if histories else now
    count = candidate.get("count", 0)
    out = {"audio": np.asarray(get_audio([now]), np.float32).reshape(1, 16), "history": history,
           "history_age": age, "history_mask": mask,
           "conditions": np.r_[gap, count, conditions][None].astype(np.float32)}
    notes = state.get("notes", [])
    tracks = state.get("_tracks")
    if stage not in ("arrows", "multiplier"):
        current = []
    elif tracks is None:
        current = [next((item for item in notes if item.get("hand") == hand and abs(item["time"] - now) < 1e-8), None)
                   for hand in (0, 1)]
    else:
        current = []
        for hand in (0, 1):
            timestamps, track = tracks[hand]
            index = int(np.searchsorted(timestamps, now - 1e-8))
            current.append(track[index] if index < len(track) and abs(track[index]["time"] - now) < 1e-8 else None)
    if stage in ("arrows", "multiplier"):
        out["current_positions"] = np.asarray([[0 if item is None else 1 + 4 * item["y"] + item["x"] for item in current]], np.int32)
        if stage == "arrows":
            out["occupied"] = out["current_positions"] != 0
        else:
            out.pop("current_positions")
            out["current_notes"] = np.asarray([[[0, 9] if item is None else [1 + 4 * item["y"] + item["x"], item.get("d", 8)] for item in current]], np.int32)
        neighbor_width = 2 if stage == "arrows" else 3
        neighbors = np.zeros((1, 2, 2, neighbor_width), np.float32)
        for hand in (0, 1):
            for side in (0, 1):
                if tracks is None:
                    eligible = [item for item in notes if item.get("hand") == hand and
                                (item["time"] < now if side == 0 else item["time"] > now)]
                    item = (max if side == 0 else min)(eligible, key=lambda item: item["time"]) if eligible else None
                else:
                    timestamps, track = tracks[hand]
                    index = int(np.searchsorted(timestamps, now, side="left" if side == 0 else "right")) - (side == 0)
                    item = track[index] if 0 <= index < len(track) else None
                if item is not None:
                    values = [1 + 4 * item["y"] + item["x"], abs(item["time"] - now)]
                    if stage == "multiplier":
                        values.insert(1, item.get("d", 8))
                    neighbors[0, side, hand] = values
        out["neighbors"] = neighbors
    elif stage == "hazards":
        end = float(candidate.get("end_time", now))
        occupancy = np.zeros((1, 2, 2, 12), np.float32)
        radius = float(options["schedule"].get("clearance_seconds", .1))
        if "_occupancy" in state:
            timestamps, occupancy_notes = state["_occupancy"]
            lo, hi = np.searchsorted(timestamps, [now - radius, end + radius], side="left")
            hi = int(np.searchsorted(timestamps, end + radius, side="right"))
            occupancy_notes = occupancy_notes[int(lo):hi]
        else:
            occupancy_notes = expansion_occupancy_notes(state)
        for item in occupancy_notes:
            if now - radius <= item["time"] <= end + radius:
                side = 0 if item["time"] < now else 1
                x, y = int(np.clip(round(item["x"]), 0, 3)), int(np.clip(round(item["y"]), 0, 2))
                occupancy[0, side, int(item["hand"]), 4 * y + x] = 1.
        out.update({"occupancy": occupancy, "interval": np.asarray([[now, end]], np.float32)})
    elif stage == "arcs":
        ids = candidate.get("endpoint_ids", candidate.get("note_ids", []))
        if len(ids) != 2:
            raise ValueError("Arc candidates require two existing endpoint IDs")
        indices = [next((i for i, item in enumerate(notes) if item["id"] == identity), -1) for identity in ids]
        if min(indices) < 0:
            raise ValueError("Arc endpoint is missing from upstream notes")
        endpoints = [notes[index] for index in indices]
        out.update({"endpoints": np.asarray([indices], np.int32),
                    "endpoint_features": np.asarray([[[1 + 4 * item["y"] + item["x"], item.get("d", 8), item["hand"]] for item in endpoints]], np.float32),
                    "gap": np.asarray([[endpoints[1]["time"] - endpoints[0]["time"]]], np.float32)})
    return out


def validate_examples(examples, stage):
    if stage not in NEW_STAGES or not isinstance(examples, StageExamples):
        raise ValueError("Invalid stage/examples")
    times = examples.times
    if not isinstance(times, np.ndarray) or times.dtype != np.float64 or times.ndim != (2 if stage == "lighting" else 1):
        raise ValueError("Example times must be float64, with a block axis for lighting")
    count = len(times)
    if not np.isfinite(times).all():
        raise ValueError("Nonfinite example times")
    for family in (examples.inputs, examples.targets, examples.weights):
        if not isinstance(family, dict) or not family:
            raise ValueError("Examples require inputs, targets and weights")
        for key, array in family.items():
            if not isinstance(array, np.ndarray) or array.dtype.hasobject or array.ndim < 1 or len(array) != count or not np.isfinite(array).all():
                raise ValueError(f"Invalid example array: {stage}.{key}")
    for key, weights in examples.weights.items():
        if key not in examples.targets or weights.dtype != np.float32 or np.any(weights < 0):
            raise ValueError(f"Invalid target weights: {stage}.{key}")
    if stage == "lighting":
        length = times.shape[1]
        events = examples.targets.get("events")
        if events is None or events.ndim != 3 or events.shape[:2] != (count, length):
            raise ValueError("Lighting targets require [N,L,C]")
        classes = events.shape[2]
        shapes = {"audio": (count, length, 16, 1), "gaps": (count, length, 1),
                  "history": (count, length, classes), "history_mask": (count, length), "mask": (count, length)}
        target_shapes = {"events": (count, length, classes)}
        weight_shapes = {"events": examples.weights["events"].shape}
        if weight_shapes["events"] not in ((count, length), (count, length, classes)):
            raise ValueError("Lighting masks require [N,L] or [N,L,C]")
    else:
        history = examples.inputs.get("history")
        if history is None or history.ndim != 3 or not history.shape[1]:
            raise ValueError("Stage history requires [N,H,F]")
        length = history.shape[1]
        width = 13 if stage == "hazards" else 4 if stage == "arcs" else 2
        shapes = {"audio": (count, 16), "history": (count, length, width), "history_age": (count, length, 1),
                  "history_mask": (count, length), "conditions": (count, 6)}
        if stage in ("positions", "arrows", "multiplier"):
            target_shapes = {"red": (count,), "blue": (count,)}
        elif stage == "hazards":
            target_shapes = {"wall": (count,), "bombs": (count, 12)}
            shapes.update(occupancy=(count, 2, 2, 12), interval=(count, 2))
        else:
            target_shapes = {"attach": (count,), "shape": (count, 2), "anchor": (count,)}
            shapes.update(endpoints=(count, 2), endpoint_features=(count, 2, 3), gap=(count, 1))
        weight_shapes = {key: (count,) for key in target_shapes}
        if stage == "hazards" and examples.weights["bombs"].ndim == 2:
            weight_shapes["bombs"] = (count, 12)
        if stage == "arrows":
            shapes.update(current_positions=(count, 2), occupied=(count, 2), neighbors=(count, 2, 2, 2))
        elif stage == "multiplier":
            shapes.update(current_notes=(count, 2, 2), neighbors=(count, 2, 2, 3))
    for family, expected in ((examples.inputs, shapes), (examples.targets, target_shapes), (examples.weights, weight_shapes)):
        if set(family) != set(expected):
            raise ValueError(f"Unexpected {stage} array keys")
        for key, shape in expected.items():
            if family[key].shape != shape:
                raise ValueError(f"Wrong {stage}.{key} shape: {family[key].shape}, expected {shape}")
    for key, array in examples.inputs.items():
        if key in ("mask", "history_mask", "occupied") and array.dtype != np.bool_:
            raise ValueError(f"{stage}.{key} must be boolean")
        if key in ("audio", "history_age", "conditions", "neighbors", "occupancy", "interval", "gap", "gaps", "endpoint_features") and array.dtype != np.float32:
            raise ValueError(f"{stage}.{key} must use float32")
    for key, target in examples.targets.items():
        categorical = key in ("red", "blue", "wall", "attach", "anchor")
        if categorical and (target.dtype.kind not in "iu" or np.any(target < 0)):
            raise ValueError(f"Invalid categorical target {stage}.{key}")
        if not categorical and target.dtype != np.float32:
            raise ValueError(f"{stage}.{key} must use float32")
        upper = 13 if stage == "positions" else 9 if stage == "arrows" else 2 if key == "attach" else 3 if key == "anchor" else None
        if upper is not None and np.any(target >= upper):
            raise ValueError(f"Out-of-range categorical target {stage}.{key}")
        if key in ("events", "bombs") and np.any((target < 0) | (target > 1)):
            raise ValueError(f"Invalid multi-label target {stage}.{key}")
    if stage == "lighting":
        mask = examples.inputs["mask"]
        if np.any(times[mask] < 0) or np.any(times[~mask] != -1) or np.any(examples.weights["events"][~mask] != 0):
            raise ValueError("Lighting padding times/masks/weights disagree")
    return True


def validate(song):
    if not isinstance(song, V2Song):
        raise ValueError("Expected V2Song")
    validate_song(song.base)
    for stage, examples in song.examples.items():
        validate_examples(examples, stage)
    return True
