"""Independent inference with the exact stage dictionaries used by real models."""

from dataclasses import dataclass, field

import numpy as np

from .contracts import (CLASSES, CORE, HALO, STAGES, frame_conditions, intensity_inputs,
                        local_rates, note_inputs, timing_inputs, validate_song)
from .metrics import generation_metrics


@dataclass
class Generation:
    intensity: np.ndarray
    local_nps: np.ndarray
    frame_probabilities: np.ndarray
    counts: np.ndarray
    event_frames: np.ndarray
    event_times: np.ndarray
    note_classes: np.ndarray
    metrics: dict = field(default_factory=dict)


class DummyIntensity:
    def predict(self, inputs, verbose=0):
        evidence = np.mean(np.abs(inputs["audio"]), axis=-1, dtype=np.float64)
        mass = np.exp(np.clip(evidence - evidence.max(axis=1, keepdims=True), -20, 0))
        mass *= inputs["durations"] * inputs["mask"]
        return mass / mass.sum(axis=1, keepdims=True)


class DummyTiming:
    def predict(self, inputs, verbose=0):
        core = slice(HALO, HALO + CORE)
        onset = inputs["audio"][:, core, -1]
        density = inputs["conditions"][:, core, 2] * 10
        one = np.clip(0.15 + np.maximum(onset, 0) * np.minimum(density, 10) / 12, 0.05, 0.80)
        two = np.clip(one * np.minimum(density, 10) / 30, 0.01, 0.18)
        zero = np.maximum(1 - one - two, 0.01)
        probabilities = np.stack((zero, one, two), axis=-1)
        return probabilities / probabilities.sum(axis=-1, keepdims=True)


class DummyNotes:
    def predict(self, inputs, verbose=0):
        batch = len(inputs["conditions"])
        red = np.full((batch, CLASSES), 1e-4, dtype=np.float32)
        blue = red.copy()
        for index in range(batch):
            cue = int(abs(float(inputs["audio"][index].mean())) * 100)
            previous = inputs["history"][index][inputs["history_mask"][index]]
            turn = int(previous[-1, 0]) % 9
            red[index, 1 + (cue + turn) % 9] = 0.7
            blue[index, 28 + (cue + turn + 4) % 9] = 0.7
            red[index, 0] = 0.2 if turn % 2 else 0.1
            blue[index, 0] = 0.1 if turn % 2 else 0.2
        return {"red": red / red.sum(axis=1, keepdims=True),
                "blue": blue / blue.sum(axis=1, keepdims=True)}


def dummy_predictors():
    return {"intensity": DummyIntensity(), "timing": DummyTiming(), "notes": DummyNotes()}


def _predict(predictor, inputs):
    return predictor.predict(inputs, verbose=0) if hasattr(predictor, "predict") else predictor(inputs)


def _probabilities(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"Invalid {name} probabilities: expected {shape}, got {value.shape}")
    if not np.allclose(value.sum(axis=-1), 1, atol=1e-5):
        raise ValueError(f"Unnormalized {name} probabilities")
    return value


def decode_pair(red, blue, count):
    """Maximize legal head likelihood while conserving the timing count exactly."""
    red = _probabilities(np.asarray(red), (CLASSES,), "red")
    blue = _probabilities(np.asarray(blue), (CLASSES,), "blue")
    if count not in (1, 2):
        raise ValueError("An onset requires one or two colored notes")
    r, b = 1 + int(np.argmax(red[1:])), 1 + int(np.argmax(blue[1:]))
    if count == 2:
        return r, b
    if float(red[r]) * float(blue[0]) >= float(red[0]) * float(blue[b]):
        return r, 0
    return 0, b


def generate(song, requested_nps, predictors=None, max_events=None):
    """Decode frame argmax without snapping, forcing budgets, or dropping tails."""
    validate_song(song)
    if not np.isfinite(requested_nps) or requested_nps < 0:
        raise ValueError("Requested NPS must be finite and nonnegative")
    if max_events is not None and (isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0):
        raise ValueError("Diagnostic event limit must be a positive integer")
    predictors = dummy_predictors() if predictors is None else predictors
    if set(STAGES) - set(predictors):
        raise ValueError("Provide all three stage predictors")
    bins, frames = len(song.arrays["bin_durations"]), len(song.arrays["frame_times"])
    intensity = _probabilities(_predict(predictors["intensity"], intensity_inputs(song)), (1, bins), "intensity")[0]
    conditions = frame_conditions(song, intensity, requested_nps)
    probabilities = np.zeros((frames, 3), dtype=np.float32)
    probabilities[:, 0] = 1
    if requested_nps > 0:
        for start in range(0, frames, CORE):
            inputs, _, _ = timing_inputs(song, conditions, start)
            core = _probabilities(_predict(predictors["timing"], inputs), (1, CORE, 3), "timing")[0]
            valid = min(CORE, frames - start)
            probabilities[start:start + valid] = core[:valid]
    counts = probabilities.argmax(axis=1).astype(np.int32)
    event_frames = np.flatnonzero(counts).astype(np.int64)
    uncapped_events = len(event_frames)
    if max_events is not None and uncapped_events > max_events:
        counts[event_frames[max_events:]] = 0
        event_frames = event_frames[:max_events]
    times = song.arrays["frame_times"][event_frames].copy()
    classes, history_times = [], []
    for frame, time in zip(event_frames, times):
        inputs = note_inputs(song, int(frame), int(counts[frame]), conditions, history_times, classes)
        outputs = _predict(predictors["notes"], inputs)
        if isinstance(outputs, dict):
            red, blue = outputs["red"], outputs["blue"]
        else:
            red, blue = outputs
        red = _probabilities(red, (1, CLASSES), "red")[0]
        blue = _probabilities(blue, (1, CLASSES), "blue")[0]
        classes.append(decode_pair(red, blue, int(counts[frame])))
        history_times.append(float(time))
    output = Generation(intensity, local_rates(intensity, song.arrays["bin_durations"], requested_nps,
                                               song.metadata["duration"]), probabilities, counts,
                        event_frames, times, np.asarray(classes, dtype=np.int32).reshape(-1, 2))
    output.metrics = generation_metrics(song, output, requested_nps)
    output.metrics.update({"partial_diagnostic": len(event_frames) < uncapped_events,
                           "event_limit": max_events, "uncapped_unique_onsets": uncapped_events})
    return output
