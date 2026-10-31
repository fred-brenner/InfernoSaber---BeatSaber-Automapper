"""Decorative calm-area pulses: gap detection, placement, clearance and wiring."""

from types import SimpleNamespace

import numpy as np
import pytest

from map_creation.v2_postprocessing import (_arc_sample_points, add_calm_effects,
                                            add_calm_lighting, calm_area_pulses)
from tools.config import config
from training.beat_models.contracts import CORE, Song
from training.pipeline_v2.pipeline import dummy_predictors
from training.pipeline_v2.settings import preparation_settings


def _song(duration=11.):
    frames, bins = int(round(duration * 100)), int(np.ceil(duration))
    return Song("calm-effects", {"duration": duration, "bpm": 120.},
                {"features": np.zeros((frames, 82), np.float32),
                 "coarse_features": np.zeros((bins, 82), np.float32),
                 "frame_times": np.arange(frames, dtype=np.float64) / 100,
                 "bin_durations": np.minimum(1., duration - np.arange(bins)).astype(np.float32)})


def _vocab():
    return {"multiplier": [{"kind": "none"}, {"kind": "copies", "copies": [{"dx": 0, "dy": -1, "dt": .05}]}],
            "walls": [{"kind": "none"}, {"kind": "wall", "x": 0, "y": 0, "width": 1, "height": 3}],
            "lighting": [{"type": 0, "value": 1}, {"type": 1, "value": 2}]}


def _note(note_id, time, x=1, y=1, hand=0, d=8, kind="retained"):
    return {"id": note_id, "time": float(time), "x": x, "y": y, "hand": hand, "d": d, "kind": kind}


def _state(notes, **families):
    state = {"notes": notes, "chains": [], "walls": [], "bombs": [], "arcs": [], "lighting": []}
    state.update(families)
    return state


def _times(pulses):
    return [pulse["time"] for pulse in pulses]


class _Timing:
    status = "dummy"

    def __call__(self, inputs):
        values = np.zeros((len(inputs["audio"]), CORE, 3), np.float32)
        values[..., 0] = 1
        for frame in (50, 75, 100):
            values[:, frame, 0] = 0
            values[:, frame, 2] = 1
        return {"counts": values}


def _lighting(times, encoder):
    return ([{"time": float(time), "type": 0, "value": 1} for time in times],
            {"backend": "v1", "queries": len(times)})


def test_pulses_sit_on_beat_grid_inside_long_gaps(monkeypatch):
    monkeypatch.setattr(config, "v2_calm_effects_min_gap_seconds", 4.)
    monkeypatch.setattr(config, "v2_calm_effects_edge_seconds", 1.)
    notes = [_note(index, time) for index, time in enumerate((1., 1.5, 2., 8.))]
    pulses, diagnostics = calm_area_pulses(_state(notes), _song())
    # only (2., 8.) clears min_gap; interior [3., 7.) on the 0.5 s beat grid
    assert _times(pulses) == [3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5]
    assert [pulse["index"] for pulse in pulses] == list(range(8))
    assert diagnostics["gaps"] == 1 and diagnostics["pulses"] == 8
    # pulses never come within the edge margin of a gameplay object
    for note in notes:
        assert all(abs(pulse["time"] - note["time"]) >= 1. for pulse in pulses)


def test_pulses_respect_chain_spans_and_song_boundaries(monkeypatch):
    monkeypatch.setattr(config, "v2_calm_effects_min_gap_seconds", 4.)
    monkeypatch.setattr(config, "v2_calm_effects_edge_seconds", 1.)
    chain = {"anchor_id": 0, "time": 5., "end_time": 7., "x": 0, "y": 0,
             "tx": 1, "ty": 1, "d": 0, "slices": 3, "squish": 1., "hand": 0, "id": 0}
    notes = [_note(0, 1.), _note(1, 12.)]
    pulses, diagnostics = calm_area_pulses(_state(notes, chains=[chain]), _song(13.))
    # (1., 5.) interior [2., 4.); (7., 12.) interior [8., 11.); trailing 1 s too short
    assert _times(pulses) == [2., 2.5, 3., 3.5, 8., 8.5, 9., 9.5, 10., 10.5]
    assert diagnostics["gaps"] == 2
    # a note at t=0 still guards the leading edge of its following gap; the
    # trailing gap runs to the song end with no trailing edge margin, and no
    # pulse lands before v2_calm_effects_start_seconds (1.5 s)
    notes = [_note(0, 0.), _note(1, 6.)]
    pulses, _ = calm_area_pulses(_state(notes), _song())
    assert _times(pulses) == [1.5, 2., 2.5, 3., 3.5, 4., 4.5,
                             7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5]


def test_no_notes_or_short_gaps_emit_no_pulses(monkeypatch):
    monkeypatch.setattr(config, "v2_calm_effects_min_gap_seconds", 4.)
    assert calm_area_pulses(_state([]), _song())[0] == []
    notes = [_note(index, .5 * index) for index in range(1, 20)]
    assert calm_area_pulses(_state(notes), _song())[0] == []


def test_no_pulses_before_start_seconds(monkeypatch):
    monkeypatch.setattr(config, "v2_calm_effects_min_gap_seconds", 4.)
    monkeypatch.setattr(config, "v2_calm_effects_edge_seconds", 1.)
    pulses, _ = calm_area_pulses(_state([_note(0, 6.)]), _song())
    # the leading gap starts at 0 but pulses wait for the 1.5 s start margin;
    # the trailing gap (6., 11.) runs to the song end
    assert _times(pulses) == [1.5, 2., 2.5, 3., 3.5, 4., 4.5,
                             7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5]
    monkeypatch.setattr(config, "v2_calm_effects_start_seconds", 0.)
    pulses, _ = calm_area_pulses(_state([_note(0, 6.)]), _song())
    assert _times(pulses)[0] == 0.


def test_pulses_follow_the_division_grid(monkeypatch):
    monkeypatch.setattr(config, "v2_calm_effects_min_gap_seconds", 4.)
    monkeypatch.setattr(config, "v2_calm_effects_edge_seconds", 1.)
    monkeypatch.setattr(config, "v2_calm_effects_division", 2)
    notes = [_note(0, 1.), _note(1, 8.)]
    pulses, _ = calm_area_pulses(_state(notes), _song())
    assert _times(pulses) == [2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75,
                             4., 4.25, 4.5, 4.75, 5., 5.25, 5.5, 5.75,
                             6., 6.25, 6.5, 6.75]


def test_add_calm_effects_alternates_sides_and_flashes_walls(monkeypatch):
    monkeypatch.setattr(config, "v2_calm_effects_walls", True)
    monkeypatch.setattr(config, "v2_calm_effects_bombs", True)
    monkeypatch.setattr(config, "v2_calm_effects_wall_beats", .25)
    pulses = [{"time": time, "index": index, "gap": 0}
              for index, time in enumerate((3., 3.5, 4., 4.5))]
    state = _state([])
    counts = add_calm_effects(state, _song(), pulses)
    assert counts == {"walls": 4, "bombs": 4}
    assert [wall["x"] for wall in state["walls"]] == [0, 3, 0, 3]
    assert [bomb["x"] for bomb in state["bombs"]] == [3, 0, 3, 0]
    assert all(bomb["y"] == 0 for bomb in state["bombs"])
    for wall in state["walls"]:
        # quarter beat at 120 bpm, clamped by the legacy minimum duration
        assert wall["end_time"] - wall["time"] == pytest.approx(.125)
        assert (wall["x"], wall["y"], wall["width"], wall["height"]) in ((0, 0, 1, 3), (3, 0, 1, 3))


def test_add_calm_effects_honours_element_toggles(monkeypatch):
    pulses = [{"time": 3., "index": 0, "gap": 0}]
    state = _state([])
    monkeypatch.setattr(config, "v2_calm_effects_walls", False)
    monkeypatch.setattr(config, "v2_calm_effects_bombs", True)
    assert add_calm_effects(state, _song(), pulses) == {"walls": 0, "bombs": 1}
    assert state["walls"] == [] and len(state["bombs"]) == 1
    monkeypatch.setattr(config, "v2_calm_effects_walls", True)
    monkeypatch.setattr(config, "v2_calm_effects_bombs", False)
    state = _state([])
    assert add_calm_effects(state, _song(), pulses) == {"walls": 1, "bombs": 0}
    assert len(state["walls"]) == 1 and state["bombs"] == []
    # the obstacle switch still gates every wall, including decorative ones
    monkeypatch.setattr(config, "v2_calm_effects_bombs", True)
    monkeypatch.setattr(config, "add_obstacle_flag", False)
    state = _state([])
    assert add_calm_effects(state, _song(), pulses) == {"walls": 0, "bombs": 1}
    assert state["walls"] == [] and len(state["bombs"]) == 1


def test_calm_motif_themes():
    from map_creation.v2_postprocessing import _calm_motif
    # every theme resolves to edge lanes, corner bombs and laser types only
    for theme in ("alternate", "pairs", "phrase", "stagger", "rest4", "sparse", "twin"):
        for index in range(8):
            motif = _calm_motif(theme, index)
            if motif is None:
                continue
            walls, bomb, flashes, slab = motif
            assert all(lane in (0, 3) for lane in walls)
            assert bomb in (None, 0, 3) and all(t in (2, 3) for t in flashes)
    # pairs holds a side for two pulses, phrase for four
    assert _calm_motif("pairs", 0)[0] == _calm_motif("pairs", 1)[0]
    assert _calm_motif("pairs", 1)[0] != _calm_motif("pairs", 2)[0]
    assert _calm_motif("phrase", 0)[0] == _calm_motif("phrase", 3)[0]
    assert _calm_motif("phrase", 3)[0] != _calm_motif("phrase", 4)[0]
    # stagger steps between pillar and top slab, rest4/sparse leave beats empty
    assert _calm_motif("stagger", 0)[3] is False and _calm_motif("stagger", 1)[3] is True
    assert _calm_motif("rest4", 3) is None and _calm_motif("rest4", 0) is not None
    assert _calm_motif("sparse", 1) is None and _calm_motif("sparse", 2) is not None
    # twin accent frames hit both edge walls and both lasers, no bomb
    walls, bomb, flashes, _ = _calm_motif("twin", 4)
    assert walls == (0, 3) and bomb is None and flashes == (2, 3)
    assert _calm_motif("twin", 1)[0] == (3,)


def test_gap_themes_rotate_per_gap(monkeypatch):
    monkeypatch.setattr(config, "v2_calm_effects_variety", True)
    pulses = ([{"time": 3. + .5 * i, "index": i, "gap": 0} for i in range(4)]
              + [{"time": 8. + .5 * i, "index": i, "gap": 1} for i in range(4)])
    state = _state([])
    add_calm_effects(state, _song(), pulses)
    by_gap = {0: [], 1: []}
    for wall in state["walls"]:
        by_gap[0 if wall["time"] < 8 else 1].append(wall["x"])
    # gap 0 alternates every pulse; gap 1 (pairs) holds each side for two
    assert by_gap[0] == [0, 3, 0, 3]
    assert by_gap[1] == [0, 0, 3, 3]
    monkeypatch.setattr(config, "v2_calm_effects_variety", False)
    state = _state([])
    add_calm_effects(state, _song(), pulses)
    by_gap = {0: [], 1: []}
    for wall in state["walls"]:
        by_gap[0 if wall["time"] < 8 else 1].append(wall["x"])
    assert by_gap[0] == [0, 3, 0, 3] and by_gap[1] == [0, 3, 0, 3]


def test_rest_pulses_stay_dark_and_empty(monkeypatch):
    monkeypatch.setattr(config, "v2_calm_effects_lighting", True)
    # gap index 4 maps to the rest4 theme: the fourth pulse rests completely
    pulses = [{"time": 3. + .5 * i, "index": i, "gap": 4} for i in range(4)]
    state = _state([])
    counts = add_calm_effects(state, _song(), pulses)
    assert counts == {"walls": 3, "bombs": 3}
    added = add_calm_lighting(state, pulses)
    assert added == 3 and all(event["time"] < 4.5 for event in state["lighting"])


def test_add_calm_effects_skips_cells_swept_by_arcs():
    pulses = [{"time": 3.5, "index": 0, "gap": 0}, {"time": 4., "index": 1, "gap": 0}]
    notes = [_note(0, 3., x=0, y=1, hand=0), _note(1, 4., x=0, y=1, hand=0),
             _note(2, 3.4, x=3, y=0, hand=1), _note(3, 4., x=2, y=0, hand=1)]
    arcs = [{"kind": "arc", "time": 3., "end_time": 4., "hand": 0,
             "endpoint_ids": [0, 1], "mu": 1., "tmu": 1., "anchor_mode": 0},
            {"kind": "arc", "time": 3.4, "end_time": 4., "hand": 1,
             "endpoint_ids": [2, 3], "mu": 1., "tmu": 1., "anchor_mode": 0}]
    samples = _arc_sample_points(_state(notes, arcs=arcs))
    assert any(point["time"] == pytest.approx(3.5) and point["x"] == pytest.approx(0)
               for point in samples)
    state = _state(notes, arcs=arcs)
    counts = add_calm_effects(state, _song(), pulses)
    # pulse 3.5: wall x=0 sits on the held arc lane, bomb x=3 on the arc's cell
    # pulse 4.0: wall x=3 and bomb x=0 are both clear of every arc sample
    assert counts == {"walls": 1, "bombs": 1}
    assert [wall["x"] for wall in state["walls"]] == [3]
    assert [wall["time"] for wall in state["walls"]] == [4.]
    assert [bomb["x"] for bomb in state["bombs"]] == [0]
    assert [bomb["time"] for bomb in state["bombs"]] == [4.]


def test_add_calm_lighting_alternates_lasers_and_keeps_existing_priority(monkeypatch):
    monkeypatch.setattr(config, "v2_calm_effects_lighting", True)
    events = [{"time": 3.5, "type": 2, "value": 1}]
    pulses = [{"time": 3.5, "index": 0, "gap": 0}, {"time": 4., "index": 1, "gap": 0},
              {"time": 4.5, "index": 2, "gap": 0}]
    state = _state([], lighting=events)
    assert add_calm_lighting(state, pulses) == 2
    assert [(event["time"], event["type"], event["value"]) for event in state["lighting"]] == [
        (3.5, 2, 1), (4., 3, 6), (4.5, 2, 6)]
    monkeypatch.setattr(config, "v2_calm_effects_lighting", False)
    assert add_calm_lighting(state, pulses) == 0


def _generate(song, monkeypatch):
    from map_creation import v2_postprocessing, v2_pipeline

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
    song, vocab = song, _vocab()
    predictors = dummy_predictors(vocab)
    predictors["timing"] = _Timing()
    settings = preparation_settings()
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    return v2_pipeline.generate(song, 4., predictors,
                                lambda times: np.zeros((len(times), 16), np.float32),
                                settings, vocab, lighting_generator=_lighting,
                                random_factor=0, logger=lambda message: None)


def test_generate_fills_long_note_free_gaps_with_decorative_pulses(monkeypatch):
    result = _generate(_song(), monkeypatch)
    diagnostics = result.diagnostics["calm_effects"]
    assert diagnostics["gaps"] >= 1 and diagnostics["pulses"] > 0
    assert diagnostics["walls"] >= 1 and diagnostics["bombs"] >= 1
    note_times = [note["time"] for note in result.state["notes"]]

    def clear_of_notes(time):
        return min(abs(time - note_time) for note_time in note_times) >= 1. - 1e-9

    # every bomb is a calm pulse (heuristic mode emits none) — all ≥ edge margin
    assert all(clear_of_notes(bomb["time"]) for bomb in result.state["bombs"])
    flashes = [event for event in result.state["lighting"] if event["value"] == 6]
    assert flashes and all(event["type"] in (2, 3) and clear_of_notes(event["time"])
                           for event in flashes)
    # decorative walls carry the legacy edge-lane vocabulary
    assert all(wall["x"] in (0, 3) and wall["width"] == 1 for wall in result.state["walls"])
    # every wall is clear of every note in time and lane box
    for wall in result.state["walls"]:
        assert not any(wall["time"] <= note["time"] <= wall["end_time"]
                       and wall["x"] - .5 <= note["x"] <= wall["x"] + wall["width"] - .5
                       for note in result.state["notes"])
    assert result.diagnostics["export_ready"]


def test_generate_skips_calm_effects_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "v2_calm_effects", False)
    result = _generate(_song(), monkeypatch)
    assert "calm_effects" not in result.diagnostics
    assert not any(event["value"] == 6 and event["type"] in (2, 3)
                   for event in result.state["lighting"])
