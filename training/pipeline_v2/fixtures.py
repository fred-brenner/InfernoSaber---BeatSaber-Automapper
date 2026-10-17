"""Synthetic positive targets for every v2 stage; no model inference or training."""

import numpy as np

from training.beat_models.data import assign_splits, build_song
from training.beat_models.features import extract_features


def make_dummy_songs():
    songs = []
    for index, duration in enumerate((6.3, 4.3, 4.3, 4.3)):
        bpm = 120.
        time = np.arange(round(duration * 16000)) / 16000
        audio = (.1 * np.sin(2 * np.pi * (220 + 31 * index) * time)).astype(np.float32)
        arrays, metadata = extract_features(audio)
        notes = []
        for when in (.5, 1., 1.5, 2., 4.):
            if when >= duration:
                continue
            for hand, x in ((0, 0), (1, 3)):
                notes.append({"b": when * 2, "c": hand, "x": x, "y": 2, "d": 1, "a": 0})
        notes.append({"b": 1., "c": 0, "x": 0, "y": 1, "d": 1, "a": 0})
        raw = {"version": "3.2.0", "colorNotes": notes,
               "bombNotes": [{"b": 0., "x": 3, "y": 1}],
               "obstacles": [{"b": 4.6, "x": 0, "y": 0, "d": 2.8, "w": 1, "h": 3}],
               "sliders": [{"b": 2., "c": 0, "x": 0, "y": 2, "d": 1, "mu": .75,
                            "tb": 3., "tx": 0, "ty": 2, "tc": 1, "tmu": 1.25, "m": 1}],
               "burstSliders": [{"b": 2., "c": 0, "x": 0, "y": 2, "d": 1,
                                  "tb": 2.2, "tx": 0, "ty": 1, "sc": 3, "s": 1.}],
               "basicBeatmapEvents": [{"b": 0., "et": 0, "i": 1, "f": 1.},
                                       {"b": 0., "et": 1, "i": 2, "f": 1.},
                                       {"b": 1., "et": 0, "i": 1, "f": 1.}]}
        rows = [[source, note["b"], note["b"] / 2, note["c"], note["x"], note["y"], note["d"], 0]
                for source, note in enumerate(notes)]
        metadata.update({"bpm": bpm, "audio_sha256": metadata["decoded_sha256"], "sources": [],
                         "raw_beatmap": raw, "original_note_records": notes,
                         "original_note_records_provenance": "synthetic_v3_json",
                         "provenance": {"adapter": "v2_synthetic_fixture", "metadata_verified": True}})
        songs.append(build_song(f"v2-dummy-{index}", rows, arrays, metadata))
    return assign_splits(songs)
