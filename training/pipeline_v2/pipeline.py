"""V2 concept composition with explicit stops at unreviewed gameplay boundaries.

Only upstream predictions and preceding own-stage outputs enter stage histories.
Source target objects are deliberately never read by this module.
"""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from training.beat_models.contracts import (CORE, frame_conditions, intensity_inputs,
                                          timing_inputs)

from .scheduling import arc_candidates, hazard_candidates, lighting_candidates, schedule_settings


STAGES = ("intensity", "timing", "positions", "arrows", "multiplier", "hazards", "arcs", "lighting")
REVIEW_BOUNDARIES = {
    "bpm_alignment": "Final BPM alignment and musical-grid snapping need review; timing remains raw frame argmax.",
    "position_movement": "Position movement constraints and their fallback policy need review.",
    "arrow_swing": "Swing-flow constraints and their fallback policy need review.",
    "hazard_clearance": "Expanded notes, chains, walls and bombs need a reviewed clearance validator.",
    "arc_clearance": "Arc clearance against expanded notes and hazards needs review.",
    "export": "Concept objects require a reviewed seconds-to-beats package export adapter.",
}


@dataclass
class PipelineResult:
    status: str
    state: dict
    stages: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    review_stops: list = field(default_factory=list)

    def to_dict(self):
        return {"status": self.status, "state": self.state, "stages": self.stages,
                "diagnostics": self.diagnostics, "review_stops": self.review_stops}


class ReviewRequired(RuntimeError):
    """The partial structured result is reviewable; no legacy mutator was run."""

    def __init__(self, boundary, partial_result):
        self.boundary = boundary
        self.reason = REVIEW_BOUNDARIES[boundary]
        self.partial_result = partial_result
        super().__init__(f"ReviewRequired[{boundary}]: {self.reason}")


def _predict(predictor, inputs):
    value = predictor.predict(inputs, verbose=0) if hasattr(predictor, "predict") else predictor(inputs)
    if isinstance(value, dict):
        return {key: np.asarray(item) for key, item in value.items()}
    return np.asarray(value)


def _distribution(value, shape, name):
    values = np.asarray(value, dtype=np.float32)
    if (values.shape != shape or not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(axis=-1), 1, atol=1e-5)):
        raise ValueError(f"Invalid {name} probability distribution; expected {shape}")
    return values


def _multilabel(value, shape, name):
    values = np.asarray(value, dtype=np.float32)
    if values.shape != shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"Invalid {name} independent probabilities; expected {shape}")
    return values


def _output(value, key):
    return value[key] if isinstance(value, dict) else value


def infer_beats(base, requested_nps, predictors):
    """Use the retained intensity/timing contracts without a joint-note model."""
    bins = len(base.arrays["bin_durations"])
    shares = _distribution(_output(_predict(predictors["intensity"], intensity_inputs(base)), "shares"),
                           (1, bins), "intensity")[0]
    conditions = frame_conditions(base, shares, requested_nps)
    frames = len(base.arrays["frame_times"])
    probabilities = np.zeros((frames, 3), dtype=np.float32)
    probabilities[:, 0] = 1
    if requested_nps > 0:
        for start in range(0, frames, CORE):
            inputs, _, _ = timing_inputs(base, conditions, start)
            output = _distribution(_output(_predict(predictors["timing"], inputs), "counts"),
                                   (1, CORE, 3), "timing")[0]
            valid = min(CORE, frames - start)
            probabilities[start:start + valid] = output[:valid]
    counts = probabilities.argmax(axis=-1).astype(np.int32)
    event_frames = np.flatnonzero(counts)
    return {"intensity": shares, "conditions": conditions, "frame_probabilities": probabilities,
            "counts": counts, "event_frames": event_frames,
            "event_times": base.arrays["frame_times"][event_frames].copy()}


class DummyPredictor:
    """Deterministic contract output; no mapping-quality claim or trainable state."""

    status = "dummy"

    def __init__(self, stage, vocabularies):
        self.stage, self.vocabularies = stage, vocabularies

    def predict(self, inputs, verbose=0):
        batch = len(inputs["audio"])

        def head(classes, category):
            result = np.zeros((batch, classes), dtype=np.float32)
            result[:, min(category, classes - 1)] = 1
            return result

        if self.stage == "intensity":
            mass = inputs["durations"] * inputs["mask"]
            return {"shares": (mass / mass.sum(axis=-1, keepdims=True)).astype(np.float32)}
        if self.stage == "timing":
            from training.beat_models.contracts import FRAME_SECONDS, HALO
            result = np.zeros((batch, CORE, 3), dtype=np.float32)
            result[:, :, 0] = 1
            for row in range(batch):
                density = float(np.mean(inputs["conditions"][row, HALO:HALO + CORE, 2])) * 10
                if density > 0:
                    stride = max(1, int(round(2 / (density * FRAME_SECONDS))))
                    result[row, ::stride, 0] = 0
                    result[row, ::stride, 2] = 1
            return {"counts": result}
        if self.stage == "positions":
            return {"red": head(13, 9), "blue": head(13, 12)}
        if self.stage == "arrows":
            return {"red": head(9, 1), "blue": head(9, 1)}
        if self.stage == "multiplier":
            classes = len(self.vocabularies["multiplier"])
            red = head(classes, 0)
            if classes > 1:
                for row in range(batch):
                    category = 1 + int(np.count_nonzero(inputs["history_mask"][row])) % (classes - 1)
                    red[row] = 0
                    red[row, category] = 1
            return {"red": red, "blue": head(classes, 0)}
        if self.stage == "hazards":
            return {"wall": head(len(self.vocabularies["walls"]), 1),
                    "bombs": head(12, 3)}
        if self.stage == "arcs":
            return {"attach": head(2, 1), "shape": np.ones((batch, 2), dtype=np.float32),
                    "anchor": head(3, 0)}
        if self.stage == "lighting":
            classes = len(self.vocabularies["lighting"])
            result = np.zeros((batch, inputs["audio"].shape[1], classes), dtype=np.float32)
            result[..., :min(2, classes)] = 1
            return {"events": result}
        raise ValueError(f"Unknown dummy stage {self.stage}")


def dummy_predictors(vocabularies):
    return {stage: DummyPredictor(stage, vocabularies) for stage in STAGES}


def _review(result, boundary, bypass):
    result.review_stops.append({"boundary": boundary, "reason": REVIEW_BOUNDARIES[boundary],
                                "bypassed": bypass})
    result.diagnostics["safety_reviewed"] = False
    result.diagnostics["export_ready"] = False
    if not bypass:
        raise ReviewRequired(boundary, result)


def _inputs(stage, song, state, candidate, history, encoder, settings):
    from .contracts import build_inputs
    prior = [record for record in history if record["time"] < candidate["time"]]
    return build_inputs(stage, song, state, candidate, prior, encoder=encoder, settings=settings)


def _decode_positions(outputs, count):
    red = _distribution(outputs["red"], (1, 13), "red position")[0]
    blue = _distribution(outputs["blue"], (1, 13), "blue position")[0]
    r, b = 1 + int(red[1:].argmax()), 1 + int(blue[1:].argmax())
    if count == 2:
        return r, b
    return (r, 0) if red[r] * blue[0] >= red[0] * blue[b] else (0, b)


def _notes_at(state, time):
    return [note for note in state["notes"] if note["time"] == time and note.get("kind") == "retained"]


def _expand(state, anchor, template, duration, rejected):
    kind = template.get("kind", "none")
    if kind == "none":
        return
    if kind == "copies":
        copies = []
        for relative in template["copies"]:
            x, y = anchor["x"] + relative["dx"], anchor["y"] + relative["dy"]
            time = anchor["time"] + relative["dt"]
            if not (0 <= x < 4 and 0 <= y < 3 and 0 <= time < duration):
                rejected.append({"anchor_id": anchor["id"], "reason": "copy outside song/grid", "template": template})
                return
            copies.append({"id": len(state["notes"]) + len(copies), "time": float(time),
                           "hand": anchor["hand"], "x": int(x), "y": int(y), "d": anchor["d"],
                           "anchor_id": anchor["id"], "kind": "copy"})
        state["notes"].extend(copies)
    elif kind == "chain":
        tx, ty = anchor["x"] + template["dx"], anchor["y"] + template["dy"]
        end = anchor["time"] + template["span"]
        if not (0 <= tx < 4 and 0 <= ty < 3 and anchor["time"] < end < duration and template["slices"] >= 2):
            rejected.append({"anchor_id": anchor["id"], "reason": "chain outside song/grid or invalid span", "template": template})
            return
        state["chains"].append({**deepcopy(anchor), "id": len(state["chains"]), "kind": "chain",
                                "anchor_id": anchor["id"], "end_time": float(end), "tx": int(tx), "ty": int(ty),
                                "slices": int(template["slices"]), "squish": float(template.get("squish", 1.))})
    else:
        rejected.append({"anchor_id": anchor["id"], "reason": "unknown expansion kind", "template": template})


def generate(song, requested_nps, predictors=None, encoder=None, settings=None, *, diagnostic=False,
             contract_mode=False, max_events=None, vocabularies=None):
    """Compose structured concept objects; stop or explicitly record each review.

    ``contract_mode`` is restricted to dummy predictors. ``diagnostic`` explicitly
    permits unreviewed neural output. Neither option permits export or claims
    playability. Existing timing targets and source objects are never modified.
    """
    if not np.isfinite(requested_nps) or requested_nps < 0:
        raise ValueError("Requested retained NPS must be finite and nonnegative")
    if max_events is not None and (isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1):
        raise ValueError("max_events must be a positive integer")
    from .settings import preparation_settings
    selected = preparation_settings()
    selected.update(deepcopy(settings or {}))
    vocabularies = vocabularies or selected.get("vocabularies")
    if not vocabularies or any(not vocabularies.get(key) for key in ("multiplier", "walls", "lighting")):
        raise ValueError("Explicit frozen multiplier, wall and lighting vocabularies are required")
    selected["vocabularies"] = vocabularies
    predictors = dummy_predictors(vocabularies) if predictors is None else predictors
    if {"intensity", "timing"} - set(predictors):
        raise ValueError("Provide the existing intensity and timing predictors")
    statuses = {getattr(predictor, "status", "untrained") for predictor in predictors.values()}
    status = "dummy" if statuses == {"dummy"} else "trained" if statuses == {"trained"} else "untrained"
    if contract_mode and status != "dummy":
        raise ValueError("Full contract mode requires explicitly dummy predictors")
    base = getattr(song, "base", song)
    duration, bpm = float(base.metadata["duration"]), float(base.metadata["bpm"])
    schedule = schedule_settings(selected.get("schedule", selected))
    selected["schedule"] = schedule
    result = PipelineResult(status, {"notes": [], "chains": [], "walls": [], "bombs": [], "arcs": [], "lighting": []},
                            diagnostics={"requested_retained_nps": requested_nps, "duration_seconds": duration,
                                         "bpm": bpm, "scheduler": schedule, "contract_mode": bool(contract_mode),
                                         "diagnostic_bypass": bool(diagnostic), "safety_reviewed": False,
                                         "export_ready": False, "partial_diagnostic": False,
                                         "rejected_expansions": []})
    result.diagnostics["effective_settings"] = {key: value for key, value in selected.items() if key != "vocabularies"}
    result.diagnostics["vocabularies_fingerprint"] = vocabularies.get("fingerprint")
    beats = infer_beats(base, requested_nps, predictors)
    result.state.update(beats)
    result.state.update(requested_nps=float(requested_nps), light_classes=len(vocabularies["lighting"]),
                        multiplier_classes=len(vocabularies["multiplier"]))
    result.stages["intensity"] = {"bins": len(beats["intensity"])}
    result.stages["timing"] = {"events": len(beats["event_frames"]), "retained_notes": int(beats["counts"].sum())}
    frames = beats["event_frames"]
    if max_events is not None and len(frames) > max_events:
        frames = frames[:max_events]
        result.diagnostics["partial_diagnostic"] = True
    candidates = [{"time": float(base.arrays["frame_times"][frame]), "frame": int(frame),
                   "count": int(beats["counts"][frame])} for frame in frames]
    bypass = bool(diagnostic or contract_mode)
    _review(result, "bpm_alignment", bypass)
    if set(STAGES) - set(predictors):
        raise ValueError("Diagnostic composition requires every v2 predictor; joint notes cannot replace positions/arrows")
    dummy_encoder = encoder is None
    if encoder is None:
        if status != "dummy":
            raise ValueError("Neural composition requires an explicit standalone retained encoder")
        encoder = lambda times: np.zeros((len(times), 16), dtype=np.float32)
    history = []
    for candidate in candidates:
        output = _predict(predictors["positions"], _inputs("positions", song, result.state, candidate, history, encoder, selected))
        pair = _decode_positions(output, candidate["count"])
        for hand, category in enumerate(pair):
            if category:
                cell = category - 1
                result.state["notes"].append({"id": len(result.state["notes"]), "time": candidate["time"],
                                               "hand": hand, "x": cell % 4, "y": cell // 4, "kind": "retained"})
        history.append({"time": candidate["time"], "value": pair})
    result.stages["positions"] = {"events": len(candidates), "retained_notes": len(result.state["notes"])}
    _review(result, "position_movement", bypass)
    history = []
    for candidate in candidates:
        output = _predict(predictors["arrows"], _inputs("arrows", song, result.state, candidate, history, encoder, selected))
        pair = [9, 9]
        for note in _notes_at(result.state, candidate["time"]):
            hand = ("red", "blue")[note["hand"]]
            direction = int(_distribution(output[hand], (1, 9), f"{hand} arrow")[0].argmax())
            note["d"] = direction
            pair[note["hand"]] = direction
        history.append({"time": candidate["time"], "value": pair})
    result.stages["arrows"] = {"directed_notes": len(result.state["notes"])}
    _review(result, "arrow_swing", bypass)
    retained_snapshot = deepcopy(result.state["notes"])
    multiplier_state = {**result.state, "notes": retained_snapshot, "chains": []}
    history = []
    for candidate in candidates:
        output = _predict(predictors["multiplier"], _inputs("multiplier", song, multiplier_state, candidate, history, encoder, selected))
        pair = [0, 0]
        for anchor in _notes_at(multiplier_state, candidate["time"]):
            hand = ("red", "blue")[anchor["hand"]]
            category = int(_distribution(output[hand], (1, len(vocabularies["multiplier"])), f"{hand} expansion")[0].argmax())
            pair[anchor["hand"]] = category
            _expand(result.state, anchor, vocabularies["multiplier"][category], duration,
                    result.diagnostics["rejected_expansions"])
        history.append({"time": candidate["time"], "value": pair})
    result.stages["multiplier"] = {"retained_notes": len(retained_snapshot),
                                    "added_notes": len(result.state["notes"]) - len(retained_snapshot),
                                    "chains": len(result.state["chains"]),
                                    "chain_slices": sum(chain["slices"] for chain in result.state["chains"])}
    from .contracts import expansion_occupancy_notes
    hazard_queries = hazard_candidates(expansion_occupancy_notes(result.state), duration, bpm, schedule)
    result.stages["hazards"] = {"queries": len(hazard_queries),
                                "wall_queries": sum(query["kind"] == "wall" for query in hazard_queries),
                                "strobe_queries": sum(bool(query.get("strobe")) for query in hazard_queries),
                                "bomb_queries": sum(query["kind"] == "bomb" for query in hazard_queries)}
    _review(result, "hazard_clearance", bypass)
    history = []
    bomb_threshold = float(selected.get("bomb_threshold", .5))
    if not 0 <= bomb_threshold <= 1:
        raise ValueError("bomb_threshold must be between zero and one")
    for candidate in hazard_queries:
        output = _predict(predictors["hazards"], _inputs("hazards", song, result.state, candidate, history, encoder, selected))
        wall = int(_distribution(output["wall"], (1, len(vocabularies["walls"])), "wall")[0].argmax())
        bombs = _multilabel(output["bombs"], (1, 12), "bomb")[0] >= bomb_threshold
        if candidate["kind"] == "wall" and wall:
            template = vocabularies["walls"][wall]
            result.state["walls"].append({**deepcopy(template), "time": candidate["time"],
                                           "end_time": candidate["end_time"], "template_id": wall})
        elif candidate["kind"] == "bomb":
            for cell in np.flatnonzero(bombs):
                result.state["bombs"].append({"time": candidate["time"], "x": int(cell % 4), "y": int(cell // 4)})
        actual_wall = wall if candidate["kind"] == "wall" else 0
        actual_bombs = bombs.astype(int).tolist() if candidate["kind"] == "bomb" else [0] * 12
        history.append({"time": candidate["time"], "value": [actual_wall, *actual_bombs]})
    result.stages["hazards"].update(walls=len(result.state["walls"]), bombs=len(result.state["bombs"]))
    _review(result, "arc_clearance", bypass)
    history = []
    arc_queries = arc_candidates(result.state["notes"], schedule)
    lower_shape, upper_shape = selected.get("arc_shape_bounds", [0., 2.])
    if not np.isfinite([lower_shape, upper_shape]).all() or not 0 <= lower_shape < upper_shape:
        raise ValueError("Arc shape bounds must be finite, nonnegative and ordered")
    for candidate in arc_queries:
        output = _predict(predictors["arcs"], _inputs("arcs", song, result.state, candidate, history, encoder, selected))
        attach = int(_distribution(output["attach"], (1, 2), "arc attachment")[0].argmax())
        shape = np.asarray(output["shape"], dtype=np.float32)
        if shape.shape != (1, 2) or not np.isfinite(shape).all() or np.any((shape < lower_shape) | (shape > upper_shape)):
            raise ValueError("Arc shape must contain mu/tmu within the configured bounds")
        anchor = int(_distribution(output["anchor"], (1, 3), "arc anchor")[0].argmax())
        if attach:
            result.state["arcs"].append({**deepcopy(candidate), "mu": float(shape[0, 0]),
                                          "tmu": float(shape[0, 1]), "anchor_mode": anchor})
        history.append({"time": candidate["time"], "value": [attach, *shape[0].tolist(), anchor]})
    result.stages["arcs"] = {"queries": len(arc_queries), "arcs": len(result.state["arcs"])}
    history = []
    light_queries = lighting_candidates(song, duration, bpm, schedule)
    light_threshold = float(selected.get("lighting_threshold", .5))
    if not 0 <= light_threshold <= 1:
        raise ValueError("lighting_threshold must be between zero and one")
    block = selected.get("lighting_block_length", 16)
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)) or block < 3:
        raise ValueError("Lighting block length must be an integer of at least three")
    block = int(block)
    for start in range(0, len(light_queries), block):
        queries = light_queries[start:start + block]
        candidate = {"time": queries[0]["time"], "times": [query["time"] for query in queries],
                     "mask": [True] * len(queries) + [False] * (block - len(queries))}
        output = _predict(predictors["lighting"], _inputs("lighting", song, result.state, candidate, history, encoder, selected))
        probabilities = _multilabel(_output(output, "events"), (1, block, len(vocabularies["lighting"])), "lighting")[0]
        for index, query in enumerate(queries):
            labels = probabilities[index] >= light_threshold
            for category in np.flatnonzero(labels):
                result.state["lighting"].append({"time": query["time"], **deepcopy(vocabularies["lighting"][int(category)])})
            history.append({"time": query["time"], "value": labels.astype(np.float32).tolist()})
    result.stages["lighting"] = {"queries": len(light_queries), "events": len(result.state["lighting"]),
                                 "blocks": (len(light_queries) + block - 1) // block,
                                 "tail_queries": len(light_queries) % block}
    if [note for note in result.state["notes"] if note.get("kind") == "retained"] != retained_snapshot:
        raise ValueError("Downstream stage changed retained anchors")
    result.diagnostics.update({"bomb_threshold": bomb_threshold, "lighting_threshold": light_threshold,
                               "encoder": {"status": "dummy_zero_latents"} if dummy_encoder else
                                          {"status": "explicit_retained_adapter", "metadata": getattr(encoder, "metadata", {})},
                               "unreviewed_boundaries": list(REVIEW_BOUNDARIES)})
    _review(result, "export", bypass)
    return result
