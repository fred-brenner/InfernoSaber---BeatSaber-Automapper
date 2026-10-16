"""Shared array contracts; all timestamps and gaps are elapsed seconds.

Song arrays: features[T,82], coarse_features[S,82], frame_times[T],
bin_durations[S], intensity[S], timing[T] (0/1/2), event_times[E],
event_frames[E], note_classes[E,2], original_notes[N,8], retained_indices[R].
original_event_indices[N] links every original note to its synchronized event.
Original-note columns: source index, source beat, seconds, color, x, y,
direction, angle. retained_indices indexes rows of original_notes.
metadata includes duration, bpm, split, group_id, intensity_target_valid,
sources (path/hash records), reduction diagnostics and provenance.

Model inputs (batch dimension first):
intensity: audio[B,S,82], durations[B,S], mask[B,S] -> shares[B,S].
timing: audio[B,1020,82], conditions[B,1020,4], mask[B,1020]
        -> count probabilities[B,512,3].
notes: audio[B,200,80,1], audio_mask[B,200], history[B,32,2], history_age[B,32,1],
       history_mask[B,32], conditions[B,6] -> red/blue[B,109].
Timing conditions: relative profile, global NPS/10, local NPS/10, BPM/120.
Note conditions: previous gap seconds, required count, then timing conditions.
History ages are seconds before the current event. Tokens 109/110 are START/PAD.
"""

from dataclasses import dataclass, field

import numpy as np
from tools.config import config

SCHEMA_VERSION = 1
FEATURE_VERSION = 1
TARGET_VERSION = 2
SAMPLE_RATE = 16000
HOP = 160
FFT = 512
MELS = 80
FEATURES = 82
FRAME_SECONDS = HOP / SAMPLE_RATE
BIN_SECONDS = 1.0
CORE = 512
HALO = 254
TIMING_WINDOW = CORE + 2 * HALO
PATCH = 200
HISTORY = 32
CLASSES = 109
START = 109
PAD = 110
SEED = 3
STAGES = ("intensity", "timing", "notes")
REPRESENTATION = {"schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
                  "target_version": TARGET_VERSION, "sample_rate": SAMPLE_RATE,
                  "hop": HOP, "fft": FFT, "mels": MELS, "features": FEATURES,
                  "bin_seconds": BIN_SECONDS, "core": CORE, "halo": HALO,
                  "patch": PATCH, "note_audio_mask": True, "history": HISTORY, "classes": CLASSES,
                  "time_unit": "seconds", "density_unit": "retained_colored_notes_per_second",
                  "event_grouping": "first_onset_inclusive",
                  "min_time_diff_seconds": float(config.min_time_diff)}


@dataclass
class Song:
    song_id: str
    metadata: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)


def synchronize_events(times, min_time_diff):
    """Group within the first onset's inclusive window; preserve source row IDs."""
    if isinstance(min_time_diff, (bool, np.bool_)) or not np.isfinite(min_time_diff) or min_time_diff < 0:
        raise ValueError("Synchronization min_time_diff must be finite and nonnegative")
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("Synchronization requires finite source timestamps")
    indices = np.empty(len(times), dtype=np.int64)
    anchors = []
    for row in np.argsort(times, kind="stable"):
        when = float(times[row])
        if not anchors or when > anchors[-1] + min_time_diff:
            anchors.append(when)
        indices[row] = len(anchors) - 1
    return indices, np.asarray(anchors, dtype=np.float64)


def encode_note(x, y, direction):
    values = (x, y, direction)
    if any(isinstance(v, (bool, np.bool_)) or not float(v).is_integer() for v in values):
        raise ValueError("Note coordinates and direction must be integers")
    if not (0 <= x < 4 and 0 <= y < 3 and 0 <= direction < 9):
        raise ValueError("Unsupported note position/direction")
    return 1 + (int(y) * 4 + int(x)) * 9 + int(direction)


def decode_note(category):
    if isinstance(category, (bool, np.bool_)) or not float(category).is_integer() or not 0 <= category < CLASSES:
        raise ValueError("Invalid note category")
    if category == 0:
        return None
    position, direction = divmod(int(category) - 1, 9)
    return position % 4, position // 4, direction


def local_rates(shares, durations, requested_nps, duration):
    shares, durations = np.asarray(shares, dtype=float), np.asarray(durations, dtype=float)
    if shares.ndim != 1 or shares.shape != durations.shape or not len(shares):
        raise ValueError("Intensity must contain one share per bin")
    if not np.isfinite(shares).all() or not np.isfinite(durations).all() or np.any(shares < 0) or np.any(durations <= 0):
        raise ValueError("Invalid intensity shares/bin durations")
    if not np.isfinite(requested_nps) or requested_nps < 0 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("Density must be nonnegative and duration positive")
    if not np.isclose(durations.sum(), duration, rtol=1e-6, atol=1e-6):
        raise ValueError("Bin durations do not cover the song")
    if not np.isclose(shares.sum(), 1.0, atol=1e-6):
        raise ValueError("Intensity shares must sum to one")
    return (shares * requested_nps * duration / durations).astype(np.float32)


def frame_conditions(song, shares, requested_nps):
    arrays, meta = song.arrays, song.metadata
    durations = arrays["bin_durations"]
    rates = local_rates(shares, durations, requested_nps, meta["duration"])
    relative = np.asarray(shares) * meta["duration"] / durations
    bins = np.minimum((arrays["frame_times"] / BIN_SECONDS).astype(int), len(durations) - 1)
    return np.column_stack((relative[bins], np.full(len(bins), requested_nps / 10),
                            rates[bins] / 10, np.full(len(bins), meta["bpm"] / 120))).astype(np.float32)


def intensity_inputs(song):
    return {"audio": song.arrays["coarse_features"][None].astype(np.float32),
            "durations": song.arrays["bin_durations"][None].astype(np.float32),
            "mask": np.ones((1, len(song.arrays["bin_durations"])), dtype=bool)}


def timing_inputs(song, conditions, start):
    """Return one context window, core labels and mask, including partial tails."""
    features = song.arrays["features"]
    if start < 0 or start >= len(features):
        raise ValueError("Timing start is outside the song")
    audio = np.zeros((1, TIMING_WINDOW, FEATURES), dtype=np.float32)
    cond = np.zeros((1, TIMING_WINDOW, 4), dtype=np.float32)
    mask = np.zeros((1, TIMING_WINDOW), dtype=bool)
    lo, hi = max(0, start - HALO), min(len(features), start + CORE + HALO)
    dest = lo - (start - HALO)
    audio[0, dest:dest + hi - lo] = features[lo:hi]
    cond[0, dest:dest + hi - lo] = conditions[lo:hi]
    mask[0, dest:dest + hi - lo] = True
    labels = np.zeros((1, CORE), dtype=np.int32)
    valid = min(CORE, len(features) - start)
    labels[0, :valid] = song.arrays["timing"][start:start + valid]
    return {"audio": audio, "conditions": cond, "mask": mask}, labels, mask[:, HALO:HALO + CORE]


def note_inputs(song, frame, count, conditions, history_times, history_classes):
    """Build current-event input from strictly preceding events, with right padding."""
    times = song.arrays["frame_times"]
    if not 0 <= frame < len(times) or count not in (1, 2):
        raise ValueError("Invalid note event frame/count")
    now = float(times[frame])
    ht = np.asarray(history_times, dtype=float)
    hc = np.asarray(history_classes, dtype=np.int32).reshape(-1, 2)
    if len(ht) != len(hc) or not np.isfinite(ht).all() or np.any(ht >= now) or np.any(np.diff(ht) <= 0):
        raise ValueError("History must contain only ordered preceding events")
    if np.any(hc < 0) or np.any(hc >= CLASSES):
        raise ValueError("Invalid history class")
    patch = np.zeros((1, PATCH, MELS, 1), dtype=np.float32)
    audio_mask = np.zeros((1, PATCH), dtype=bool)
    lo, hi = max(0, frame - PATCH // 2), min(len(times), frame + PATCH // 2)
    offset = lo - (frame - PATCH // 2)
    patch[0, offset:offset + hi - lo, :, 0] = song.arrays["features"][lo:hi, :MELS]
    audio_mask[0, offset:offset + hi - lo] = True
    history = np.full((1, HISTORY, 2), PAD, dtype=np.int32)
    ages = np.zeros((1, HISTORY, 1), dtype=np.float32)
    mask = np.zeros((1, HISTORY), dtype=bool)
    if len(ht):
        ht, hc = ht[-HISTORY:], hc[-HISTORY:]
        history[0, :len(ht)] = hc
        ages[0, :len(ht), 0] = now - ht
        mask[0, :len(ht)] = True
        gap = now - ht[-1]
    else:
        history[0, 0] = START
        mask[0, 0] = True
        gap = now
    return {"audio": patch, "audio_mask": audio_mask, "history": history, "history_age": ages, "history_mask": mask,
            "conditions": np.concatenate(([gap, count], conditions[frame]))[None].astype(np.float32)}


def validate_song(song):
    """Validate persisted targets independently of TensorFlow or model code."""
    a, m = song.arrays, song.metadata
    duration, bpm = m.get("duration", 0), m.get("bpm", 0)
    if not song.song_id or not np.isfinite([duration, bpm]).all() or duration <= 0 or bpm <= 0:
        raise ValueError("Song requires an identifier, positive duration and BPM")
    required = ("features", "coarse_features", "frame_times", "bin_durations", "intensity",
                "timing", "event_times", "event_frames", "note_classes", "original_notes", "retained_indices",
                "original_event_indices")
    for key in required:
        if key not in a or not isinstance(a[key], np.ndarray) or a[key].dtype.hasobject or not np.isfinite(a[key]).all():
            raise ValueError(f"Invalid or missing array: {key}")
    t, s, e = len(a["frame_times"]), len(a["bin_durations"]), len(a["event_times"])
    shapes = {"features": (t, FEATURES), "coarse_features": (s, FEATURES), "frame_times": (t,),
              "timing": (t,), "intensity": (s,), "bin_durations": (s,), "event_times": (e,),
              "event_frames": (e,), "note_classes": (e, 2)}
    for key, shape in shapes.items():
        if a[key].shape != shape:
            raise ValueError(f"Wrong {key} shape: {a[key].shape}, expected {shape}")
    for key in ("features", "coarse_features", "intensity", "bin_durations"):
        if a[key].dtype != np.float32:
            raise ValueError(f"{key} must use float32")
    if not t or not s or np.any(a["frame_times"] < 0) or np.any(a["frame_times"] >= duration):
        raise ValueError("Invalid frame timeline")
    if not np.allclose(a["frame_times"], np.arange(t) * FRAME_SECONDS, rtol=0, atol=1e-9):
        raise ValueError("Frames must follow the sample-derived time base")
    if np.any(a["bin_durations"] <= 0) or np.any(a["bin_durations"] > BIN_SECONDS + 1e-6) or not np.isclose(a["bin_durations"].sum(), duration, atol=1e-5):
        raise ValueError("Invalid bin durations")
    for key, upper in (("timing", 3), ("note_classes", CLASSES), ("event_frames", t)):
        if a[key].dtype.kind not in "iu" or np.any(a[key] < 0) or np.any(a[key] >= upper):
            raise ValueError(f"Invalid integer categories: {key}")
    if np.any(np.diff(a["event_times"]) <= 0) or np.any(a["event_times"] < 0) or np.any(a["event_times"] >= duration) or np.any(np.diff(a["event_frames"]) <= 0):
        raise ValueError("Events must be ordered, distinct and inside the song")
    nearest = np.minimum(np.floor(a["event_times"] / FRAME_SECONDS + .5).astype(np.int64), t - 1)
    if not np.array_equal(nearest, a["event_frames"]):
        raise ValueError("Event labels must use nearest frames with half-frame ties later")
    counts = np.count_nonzero(a["note_classes"], axis=1)
    if np.any(counts == 0) or not np.array_equal(np.flatnonzero(a["timing"]), a["event_frames"]) or not np.array_equal(a["timing"][a["event_frames"]], counts):
        raise ValueError("Timing and note counts disagree")
    if np.any(a["intensity"] < 0):
        raise ValueError("Negative intensity")
    valid = bool(m.get("intensity_target_valid", e > 0))
    if valid != (e > 0) or not np.isclose(a["intensity"].sum(), 1.0 if valid else 0.0, atol=1e-6):
        raise ValueError("Invalid intensity target mass/mask")
    original = a["original_notes"]
    retained = a["retained_indices"]
    if original.ndim != 2 or original.shape[1] != 8 or retained.ndim != 1 or retained.dtype.kind not in "iu" or np.any(retained < 0) or np.any(retained >= len(original)):
        raise ValueError("Invalid original/retained note records")
    if len(retained) != counts.sum() or len(np.unique(retained)) != len(retained):
        raise ValueError("Retained note provenance/count mismatch")
    event_indices, anchors = synchronize_events(original[:, 2], m.get("reduction_time_epsilon_seconds", config.min_time_diff))
    if (a["original_event_indices"].shape != (len(original),) or a["original_event_indices"].dtype.kind not in "iu"
            or not np.array_equal(a["original_event_indices"], event_indices)
            or not np.array_equal(anchors, a["event_times"])):
        raise ValueError("Original note synchronization disagrees with event targets")
    if len(original):
        if np.any(original[:, 2] < 0) or np.any(original[:, 2] >= duration):
            raise ValueError("Original notes fall outside decoded audio")
        if not np.isin(original[:, 3], (0, 1)).all() or np.any(original[:, 7] != 0):
            raise ValueError("Only ordinary colored-note targets are supported")
        for row in original:
            encode_note(row[4], row[5], row[6])
        selected = original[retained]
        selected_by_event = {}
        for row, event in zip(selected, event_indices[retained]):
            selected_by_event.setdefault(int(event), []).append(row)
        for index in range(e):
            pair = np.zeros(2, dtype=np.int32)
            for row in selected_by_event.get(index, []):
                color = int(row[3])
                if pair[color]:
                    raise ValueError("Retained notes repeat a hand at one timestamp")
                pair[color] = encode_note(row[4], row[5], row[6])
            if not np.array_equal(pair, a["note_classes"][index]):
                raise ValueError("Retained note labels disagree with source records")
    expected_intensity = np.zeros(s, dtype=np.float64)
    for when, count in zip(a["event_times"], counts):
        expected_intensity[min(int(when / BIN_SECONDS), s - 1)] += count
    if expected_intensity.sum():
        expected_intensity /= expected_intensity.sum()
    if not np.allclose(expected_intensity, a["intensity"], rtol=1e-6, atol=1e-7):
        raise ValueError("Intensity target disagrees with retained colored-note counts")
    return True
