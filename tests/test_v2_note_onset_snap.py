"""Bounded per-note onset snap: guardrails and pipeline wiring."""

import numpy as np
import pytest

from map_creation import v2_postprocessing
from map_creation.v2_postprocessing import snap_note_times, _snap_violated
from tools.config import config
from training.beat_models.contracts import CORE, Song
from training.pipeline_v2.pipeline import dummy_predictors
from training.pipeline_v2.settings import preparation_settings


def _note(id_, time, x, y, hand, d=8):
    return {"id": id_, "time": float(time), "x": x, "y": y, "hand": hand, "d": d}


def _audio_song(frames=1100):
    duration = frames / 100
    bins = int(np.ceil(duration))
    return Song("audio-only", {"duration": duration, "bpm": 120.},
                {"features": np.zeros((frames, 82), np.float32),
                 "coarse_features": np.zeros((bins, 82), np.float32),
                 "frame_times": np.arange(frames, dtype=np.float64) / 100,
                 "bin_durations": np.minimum(1., duration - np.arange(bins)).astype(np.float32)})


def _vocabulary():
    return {"multiplier": [{"kind": "none"}, {"kind": "copies", "copies": [{"dx": 0, "dy": -1, "dt": .05}]}],
            "walls": [{"kind": "none"}, {"kind": "wall", "x": 0, "y": 0, "width": 1, "height": 3}],
            "lighting": [{"type": 0, "value": 1}, {"type": 1, "value": 2}]}


class _Timing:
    status = "dummy"

    def __call__(self, inputs):
        values = np.zeros((len(inputs["audio"]), CORE, 3), np.float32)
        values[..., 0] = 1
        values[:, 50, 0] = 0
        values[:, 50, 2] = 1
        return {"counts": values}


def _lighting(times, encoder):
    return ([{"time": float(time), "type": 0, "value": 2000000804} for time in times],
            {"backend": "v1", "queries": len(times)})


def _generate(monkeypatch, snap_enabled):
    from map_creation.v2_pipeline import generate

    monkeypatch.setattr(config, "v2_notes_onset_snap", snap_enabled)
    # Real postprocess_* run (postprocessors=None); only the audio externals are
    # stubbed — the onset detector itself is patched to a dense 20 ms grid.
    monkeypatch.setattr(v2_postprocessing, "sanity_check_timing2",
                        lambda name, times, audio_path=None: np.asarray(times))
    monkeypatch.setattr(v2_postprocessing, "align_beats_on_bpm",
                        lambda times, bpm: np.asarray(times))
    monkeypatch.setattr(v2_postprocessing, "fill_map_times_scale",
                        lambda times, index=5: np.asarray(times))
    monkeypatch.setattr(v2_postprocessing, "_legacy_timing_features",
                        lambda audio_path: (np.zeros((2, 2)), np.zeros((2, 2)), np.empty(0)))
    monkeypatch.setattr(v2_postprocessing, "remove_silent_times",
                        lambda times, silent: np.asarray(times))
    monkeypatch.setattr(v2_postprocessing, "_start_end_times",
                        lambda times, volume, onset: np.asarray(times))
    _patch_onsets(monkeypatch, np.arange(0., 11., .02))
    song, vocab = _audio_song(), _vocabulary()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = _Timing()
    settings = preparation_settings()
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    return generate(song, 4., predictors, lambda times: np.zeros((len(times), 16), np.float32),
                    settings, vocab, lighting_generator=_lighting, audio_path="song.egg",
                    random_factor=0, logger=lambda message: None)


def _effective(notes):
    return {note["id"]: (note["d"] if note["d"] != 8 else None) for note in notes}


def _patch_onsets(monkeypatch, onsets):
    monkeypatch.setattr(v2_postprocessing, "_detect_onsets",
                        lambda audio_path: np.asarray(onsets, np.float64))


def _snap(notes, onsets, monkeypatch, chains=None, duration=60.):
    _patch_onsets(monkeypatch, onsets)
    return snap_note_times(notes, chains or [], "song.egg", duration)


def test_snap_moves_notes_to_nearest_onset_within_per_hand_ioi(monkeypatch):
    notes = [_note(1, 1.00, 0, 0, 0), _note(2, 1.40, 1, 1, 0), _note(3, 1.20, 2, 2, 1)]
    stats = _snap(notes, [1.05, 1.20, 1.45], monkeypatch)
    # hand-0 IOI is .40 → bound .06: both outer onsets are reachable; the lone
    # hand-1 note has no IOI and is skipped entirely.
    assert notes[0]["time"] == pytest.approx(1.05)
    assert notes[1]["time"] == pytest.approx(1.45)
    assert notes[2]["time"] == pytest.approx(1.20)
    assert stats["moved"] == 2


def test_no_onset_inside_bound_leaves_the_note(monkeypatch):
    notes = [_note(1, 1.0, 0, 0, 0), _note(2, 1.4, 1, 1, 0)]
    stats = _snap(notes, [1.09], monkeypatch)  # .09 > .06 bound
    assert [note["time"] for note in notes] == [1.0, 1.4]
    assert stats["moved"] == 0 and stats["no_onset"] == 2


def test_single_note_per_hand_has_no_ioi_and_stays(monkeypatch):
    notes = [_note(1, 1.0, 0, 0, 0)]
    stats = _snap(notes, [1.01], monkeypatch)
    assert notes[0]["time"] == 1.0 and stats["moved"] == 0


def test_move_into_other_hand_cell_is_blocked(monkeypatch):
    # B sits on (0,0) at 1.2 — moving A there lands inside its swing window.
    notes = [_note(1, 1.0, 0, 0, 0), _note(2, 2.0, 1, 1, 0), _note(3, 1.2, 0, 0, 1)]
    stats = _snap(notes, [1.06], monkeypatch)
    assert notes[0]["time"] == 1.0 and stats["blocked"] == 1


def test_move_into_other_hand_swing_volume_is_blocked():
    # A up-cut at (0,1) sweeps (0,0)+(0,2); B stands at (0,0) 0.14s after t'.
    notes = [_note(1, 1.0, 0, 1, 0, d=0), _note(2, 1.3, 0, 0, 1)]
    effective = _effective(notes)
    assert _snap_violated(notes[0], 1.16, [notes[1]], notes, 0, effective, .02, .15, .1)
    # Just outside the forward horizon the same move is clear.
    assert not _snap_violated(notes[0], 1.14, [notes[1]], notes, 0, effective, .02, .15, .1)


def test_simultaneous_same_cell_in_either_direction_is_blocked():
    notes = [_note(1, 1.0, 0, 0, 0), _note(2, 1.01, 0, 0, 1)]
    effective = _effective(notes)
    assert _snap_violated(notes[0], 1.005, [notes[1]], notes, 0, effective, .02, .15, .1)


def test_crossing_the_same_stroke_gap_with_same_direction_is_blocked():
    notes = [_note(1, 1.0, 0, 0, 0, d=0), _note(2, 1.09, 1, 1, 0, d=0), _note(3, 2.0, 2, 2, 0, d=1)]
    effective = _effective(notes)
    # Moving note 2 to 1.102 makes the same-direction gap 0.102 > 0.1.
    assert _snap_violated(notes[1], 1.102, [], notes, 1, effective, .02, .15, .1)
    assert not _snap_violated(notes[1], 1.095, [], notes, 1, effective, .02, .15, .1)


def test_chain_span_moves_with_its_anchor_and_duration_guard(monkeypatch):
    notes = [_note(1, 1.0, 0, 0, 0), _note(2, 1.4, 1, 1, 0)]
    chain = {"anchor_id": 1, "time": 1.0, "end_time": 1.2, "x": 0, "y": 0,
             "tx": 1, "ty": 1, "d": 0, "slices": 3, "squish": 1.}
    stats = _snap(notes, [1.05], monkeypatch, chains=[chain])
    assert stats["moved"] == 1
    assert chain["time"] == pytest.approx(1.05) and chain["end_time"] == pytest.approx(1.25)
    notes2 = [_note(3, 59.8, 0, 0, 0), _note(4, 58.0, 1, 1, 0)]
    tail = {"anchor_id": 3, "time": 59.8, "end_time": 59.95, "x": 0, "y": 0,
            "tx": 1, "ty": 1, "d": 0, "slices": 3, "squish": 1.}
    stats = _snap(notes2, [59.9], monkeypatch, chains=[tail], duration=59.99)
    assert stats["moved"] == 0 and stats["blocked"] == 1  # would push the chain past duration


def test_snapped_stream_keeps_zero_clearance_violations(monkeypatch):
    from map_creation.v2_postprocessing import clearance_violations
    rng = np.random.default_rng(7)
    notes = []
    # Dots plus disjoint hand halves guarantee a violation-free baseline.
    for hand, xs in ((0, (0, 1)), (1, (2, 3))):
        times = np.cumsum(rng.uniform(.25, .8, 30)) + .5
        notes += [_note(len(notes), t, int(rng.choice(xs)), int(rng.integers(3)), hand)
                  for t in times]
    onsets = np.round(np.linspace(0.2, float(max(note["time"] for note in notes)), 500), 3)
    _patch_onsets(monkeypatch, onsets)
    stats = snap_note_times(notes, [], "song.egg", 60.)
    assert stats["moved"] > 0
    assert clearance_violations(notes) == {"same_cell_pairs": 0, "swing_into_pairs": 0,
                                         "flow_repeats": 0}


def test_generate_runs_snap_after_clearance_when_enabled(monkeypatch):
    result = _generate(monkeypatch, True)
    stats = result.diagnostics["onset_snap"]
    counts = result.diagnostics["final_counts"]
    assert stats["notes"] == counts["retained_notes"] + counts["heuristic_notes"]
    assert stats["elapsed_seconds"] >= 0
    assert result.diagnostics["export_ready"]


def test_flag_off_skips_snap_entirely(monkeypatch):
    result = _generate(monkeypatch, False)
    assert "onset_snap" not in result.diagnostics
