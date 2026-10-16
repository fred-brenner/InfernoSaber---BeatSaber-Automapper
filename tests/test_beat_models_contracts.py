"""Temporal and density invariants independent of any neural implementation."""

import numpy as np
import pytest

from training.beat_models.contracts import (
    CLASSES, CORE, FEATURES, FRAME_SECONDS, HALO, PAD, START, Song,
    decode_note, encode_note, frame_conditions, local_rates, note_inputs,
    timing_inputs, validate_song,
)


def song_fixture(duration=6.13):
    times = np.arange(int(np.ceil(duration / FRAME_SECONDS))) * FRAME_SECONDS
    frames = np.array([1, 119, len(times) - 1], dtype=np.int32)
    event_times = times[frames]
    labels = np.array([[1, 0], [2, 3], [0, 108]], dtype=np.int32)
    originals = []
    for when, pair in zip(event_times, labels):
        for color, category in enumerate(pair):
            if category:
                x, y, direction = decode_note(category)
                originals.append([len(originals), when * 2, when, color, x, y, direction, 0])
    bins = np.minimum(1.0, duration - np.arange(int(np.ceil(duration))))
    intensity = np.zeros(len(bins), dtype=np.float32)
    for when, pair in zip(event_times, labels):
        intensity[int(when)] += np.count_nonzero(pair) / 4
    timing = np.zeros(len(times), dtype=np.int32)
    timing[frames] = np.count_nonzero(labels, axis=1)
    return Song("contract-fixture", {"duration": duration, "bpm": 120,
                                    "intensity_target_valid": True}, {
        "features": np.repeat(times[:, None], FEATURES, axis=1).astype(np.float32),
        "coarse_features": np.zeros((len(bins), FEATURES), dtype=np.float32),
        "frame_times": times, "bin_durations": bins.astype(np.float32),
        "intensity": intensity, "timing": timing, "event_times": event_times,
        "event_frames": frames, "note_classes": labels,
        "original_notes": np.asarray(originals, dtype=np.float64),
        "original_event_indices": np.array([0, 1, 1, 2], dtype=np.int64),
        "retained_indices": np.arange(4, dtype=np.int32),
    })


def test_note_categories_are_bijective_including_absent_hand():
    assert decode_note(0) is None
    categories = {encode_note(x, y, d) for x in range(4) for y in range(3) for d in range(9)}
    assert categories == set(range(1, CLASSES))
    for category in categories:
        assert encode_note(*decode_note(category)) == category
    for value in (-1, 109, 1.5, True):
        with pytest.raises(ValueError):
            decode_note(value)


def test_partial_bin_difficulty_conserves_note_budget_independent_of_bpm():
    song = song_fixture()
    shares = song.arrays["intensity"]
    rates = local_rates(shares, song.arrays["bin_durations"], 3.25, song.metadata["duration"])
    assert np.dot(rates, song.arrays["bin_durations"]) == pytest.approx(3.25 * 6.13)
    before = frame_conditions(song, shares, 3.25)
    song.metadata["bpm"] = 240
    after = frame_conditions(song, shares, 3.25)
    np.testing.assert_array_equal(before[:, :3], after[:, :3])
    np.testing.assert_array_equal(before[:, 3] * 2, after[:, 3])
    assert not local_rates(shares, song.arrays["bin_durations"], 0, 6.13).any()


def test_chunk_context_core_and_partial_tail_cover_each_frame_once():
    song = song_fixture()
    conditions = frame_conditions(song, song.arrays["intensity"], 4)
    cores, labels = [], []
    for start in range(0, len(conditions), CORE):
        inputs, target, valid = timing_inputs(song, conditions, start)
        cores.append(inputs["audio"][:, HALO:HALO + CORE][valid])
        labels.append(target[valid])
        assert np.all(inputs["audio"][~inputs["mask"]] == 0)
    np.testing.assert_array_equal(np.concatenate(cores), song.arrays["features"])
    np.testing.assert_array_equal(np.concatenate(labels), song.arrays["timing"])


def test_note_history_is_causal_and_pad_is_distinct_from_absent_hand():
    song = song_fixture()
    cond = frame_conditions(song, song.arrays["intensity"], 4)
    first = note_inputs(song, 1, 1, cond, [], [])
    assert first["audio_mask"].shape == (1, 200)
    assert first["audio_mask"].sum() == 101
    assert not first["audio_mask"][0, :99].any()
    assert not first["audio"][~first["audio_mask"]].any()
    assert np.all(first["history"][0, 0] == START)
    assert np.all(first["history"][0, 1:] == PAD)
    current = note_inputs(song, 119, 2, cond, [0.01], [[1, 0]])
    assert current["history"][0, 0, 1] == 0
    assert current["history_mask"][0, 0]
    assert not current["history_mask"][0, 1:].any()
    assert current["history_age"][0, 0, 0] == pytest.approx(1.18)
    last = note_inputs(song, len(cond) - 1, 1, cond, [1.19], [[1, 0]])
    assert last["audio_mask"].sum() == 101
    assert not last["audio_mask"][0, 101:].any()
    with pytest.raises(ValueError, match="preceding"):
        note_inputs(song, 119, 2, cond, [1.19], [[1, 0]])


def test_song_contract_rejects_mismatched_counts_and_time_base():
    song = song_fixture()
    assert validate_song(song)
    song.arrays["timing"][119] = 1
    with pytest.raises(ValueError, match="counts"):
        validate_song(song)
    song.arrays["timing"][119] = 2
    song.arrays["frame_times"][0] = 0.001
    with pytest.raises(ValueError, match="time base"):
        validate_song(song)


def test_cache_cannot_shift_label_frame_while_preserving_counts():
    song = song_fixture()
    song.arrays["event_times"][0] = .014
    song.arrays["original_notes"][0, 1:3] = [.028, .014]
    assert validate_song(song)
    song.arrays["event_frames"][0] = 2
    song.arrays["timing"][1] = 0
    song.arrays["timing"][2] = 1
    with pytest.raises(ValueError, match="nearest frames"):
        validate_song(song)
