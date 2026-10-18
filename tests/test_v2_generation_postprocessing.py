"""Production cleanup contracts without model inference or corpus evaluation."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from map_creation import v2_postprocessing as post
from tools.config import config


@pytest.fixture
def song():
    return SimpleNamespace(metadata={"duration": 10., "bpm": 120.},
                           arrays={"frame_times": np.arange(1000) / 100})


@pytest.fixture
def timing_defaults(monkeypatch):
    for key, value in {"use_fixed_bpm": 120, "min_beat_time": .0625, "min_time_diff": .02,
                       "map_filler_iters": 0, "add_start_end_beats": False,
                       "add_silence_flag": False}.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setattr(post, "_legacy_timing_features", lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), []))
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kwargs: times)


def beats(times, counts):
    frames = np.floor(np.asarray(times) * 100 + .5).astype(int)
    labels = np.zeros(1000, int)
    labels[frames] = counts
    return {"event_frames": frames, "event_times": np.asarray(times), "counts": labels,
            "frame_probabilities": np.tile([.8, .15, .05], (1000, 1))}


def note(identity=0, time=1., hand=0, x=0, y=0, direction=1, kind="retained"):
    return {"id": identity, "time": time, "hand": hand, "x": x, "y": y, "d": direction, "kind": kind}


def state(notes=(), chains=(), walls=(), bombs=(), arcs=()):
    return {"notes": list(notes), "chains": list(chains), "walls": list(walls), "bombs": list(bombs), "arcs": list(arcs)}


def chain():
    return {**note(), "id": 0, "anchor_id": 0, "kind": "chain", "end_time": 3., "tx": 0, "ty": 2,
            "slices": 2, "squish": 1.}


def test_timing_moves_preserve_counts_and_near_merges_keep_largest(song, timing_defaults, monkeypatch):
    inputs = beats([.5, .51, .8], [1, 2, 1])
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kwargs: times + .2)
    candidates, report = post.postprocess_timing(song, inputs, "unused.egg")
    assert [item["count"] for item in candidates] == [2, 1]
    assert [item["time"] for item in candidates] == pytest.approx([.7, 1.])
    assert inputs["event_times"].tolist() == [.5, .51, .8]
    operations = {item["stage"]: item for item in report["operations"]}
    assert (operations["minimum_spacing"]["before_events"], operations["minimum_spacing"]["after_events"]) == (3, 2)
    assert operations["onset_snap"]["moved_event_times"] == 2
    assert operations["onset_snap"]["event_count_delta"] == 0


def test_moved_near_events_merge_without_summing_counts(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kwargs: np.asarray([.5, .515]))
    candidates, report = post.postprocess_timing(song, beats([.4, .8], [1, 1]), "unused.egg")
    assert candidates == [{"time": .5, "frame": 40, "count": 1}]
    assert report["merged_events"] == 1


def test_rejected_onset_zero_does_not_absorb_a_valid_early_onset(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kwargs: np.asarray([0., .015]))
    candidates, _ = post.postprocess_timing(song, beats([.4, .8], [2, 1]), "unused.egg")
    assert [(item["time"], item["count"]) for item in candidates] == [(.015, 1)]


def test_filler_uses_strongest_nonzero_class_at_nearest_frame(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 20.)
    monkeypatch.setattr(config, "add_beat_max_bounds", [.1, .4, 3., 4.])
    inputs = beats([1., 3.], [1, 1])
    inputs["frame_probabilities"][200] = [.98, .001, .019]
    candidates, report = post.postprocess_timing(song, inputs, "unused.egg")
    assert [(item["time"], item["count"]) for item in candidates] == [(1., 1), (2., 2), (3., 1)]
    fill = next(item for item in report["operations"] if item["stage"] == "density_fill_0")
    assert fill == {"stage": "density_fill_0", "before_events": 2, "after_events": 3, "event_count_delta": 1,
                    "before_retained_notes": 2, "after_retained_notes": 4, "moved_event_times": 0}


def test_timing_diagnostics_record_alignment_and_silence_steps(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "use_fixed_bpm", 0)
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(post, "_legacy_timing_features", lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), [2.]))
    candidates, report = post.postprocess_timing(song, beats([1., 2.], [1, 2]), "unused.egg")
    assert [item["time"] for item in candidates] == [1.]
    operations = {item["stage"]: item for item in report["operations"]}
    assert operations["initial_bpm_alignment"]["before_events"] == operations["initial_bpm_alignment"]["after_events"] == 2
    assert operations["silence_removal"]["event_count_delta"] == -1
    assert operations["silence_removal"]["after_retained_notes"] == 1
    assert operations["final_bpm_alignment"]["before_events"] == operations["final_bpm_alignment"]["after_events"] == 1


def test_zero_density_never_loads_audio_or_fills(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "map_filler_iters", 10)
    monkeypatch.setattr(post, "_legacy_timing_features", lambda _: pytest.fail("empty timing must not decode audio"))
    candidates, report = post.postprocess_timing(song, beats([], []), "unused.egg")
    assert candidates == []
    assert report["retained_notes"] == 0


def test_training_first_cut_selects_anchor_after_heuristic_cleanup(monkeypatch):
    monkeypatch.setattr(config, "gimme_more_notes_flag", False)
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    monkeypatch.setattr(post, "sanity_check_notes", lambda notes, _: notes)
    original = [note(4, y=0), note(9, y=2)]
    before = deepcopy(original)
    actual, report = post.postprocess_notes(original, [1.])
    assert [item["id"] for item in actual] == [0, 1]
    assert [item["kind"] for item in actual] == ["heuristic", "retained"]
    assert report["heuristic_notes"] == 1
    assert original == before


@pytest.mark.parametrize("length", [0, 1, 4, 20, 80])
@pytest.mark.parametrize("hand", [0, 1])
def test_short_and_empty_hand_maps_use_real_legacy_flow(length, hand, monkeypatch):
    monkeypatch.setattr(config, "gimme_more_notes_flag", False)
    monkeypatch.setattr(config, "add_waveform_pattern_flag", 2)
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    original = [note(index, time=1. + index, hand=hand, y=index % 3) for index in range(length)]
    actual, _ = post.postprocess_notes(original, [item["time"] for item in original])
    assert all(item["hand"] == hand for item in actual)
    assert all(0 <= item["x"] < 4 and 0 <= item["y"] < 3 and 0 <= item["d"] <= 8 for item in actual)


def test_mirroring_is_applied_and_zero_probability_returns_notes(monkeypatch):
    from map_creation.artificial_mod import gimme_more_notes
    monkeypatch.setattr(config, "gimme_more_notes_flag", True)
    monkeypatch.setattr(config, "gimme_more_notes_prob", 1.)
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    monkeypatch.setattr(post, "sanity_check_notes", lambda notes, _: notes)
    actual, report = post.postprocess_notes([note()], [1.])
    assert {item["hand"] for item in actual} == {0, 1}
    assert report["note_count_differences"] == {"added_notes": 1, "removed_notes": 0,
                                               "basis": "per_elapsed_time_and_hand_multiplicity"}
    mirror = next(item for item in report["operations"] if item["stage"] == "mirroring")
    assert mirror == {"stage": "mirroring", "before_notes": 1, "after_notes": 2, "delta_notes": 1,
                      "added_notes": 1, "removed_notes": 0}
    monkeypatch.setattr(config, "gimme_more_notes_prob", 0.)
    monkeypatch.setattr(config, "max_speed", 4.)
    assert gimme_more_notes([[0, 0, 0, 1]]) == [[0, 0, 0, 1]]


def test_single_mode_survives_emphasis(monkeypatch):
    monkeypatch.setattr(config, "gimme_more_notes_flag", True)
    monkeypatch.setattr(config, "gimme_more_notes_prob", 1.)
    monkeypatch.setattr(config, "single_notes_only_flag", True)
    monkeypatch.setattr(config, "single_notes_only_strict_flag", True)
    monkeypatch.setattr(post, "sanity_check_notes", lambda notes, _: [values * 2 for values in notes])
    actual, report = post.postprocess_notes([note()], [1.])
    assert len(actual) == 1
    operations = {item["stage"]: item for item in report["operations"]}
    assert operations["initial_single_note_filter"]["removed_notes"] == 1
    assert operations["legacy_note_flow"]["added_notes"] == 1
    assert operations["final_single_note_filter"]["removed_notes"] == 1


def test_geometry_differences_do_not_claim_identity_additions_for_direction_changes(monkeypatch):
    monkeypatch.setattr(config, "gimme_more_notes_flag", False)
    monkeypatch.setattr(config, "single_notes_only_flag", False)

    def edit_geometry(groups, times):
        groups[0][0], groups[0][3] = 1, 0
        return groups

    monkeypatch.setattr(post, "sanity_check_notes", edit_geometry)
    actual, report = post.postprocess_notes([note()], [1.])
    assert actual[0]["x"] == 1 and actual[0]["d"] == 0
    assert report["note_count_differences"]["added_notes"] == report["note_count_differences"]["removed_notes"] == 0
    assert report["geometry_record_differences"] == {
        "added_records": 1, "removed_records": 1,
        "semantics": "multiset differences include position/direction changes; not identity moves"}


def test_empty_and_single_timing_alignment_and_silent_removal():
    from map_creation.sanity_check import remove_silent_times
    assert post.align_beats_on_bpm(np.asarray([]), 120).size == 0
    assert post.align_beats_on_bpm(np.asarray([.6]), 120).tolist() == [.5]
    assert remove_silent_times(np.asarray([1.]), [1., 2.]).size == 0


def test_no_audio_onsets_returns_empty_timing_mask(monkeypatch):
    from map_creation import sanity_check
    monkeypatch.setattr(sanity_check.librosa, "load", lambda *args, **kwargs: (np.zeros(500), 44100))
    monkeypatch.setattr(sanity_check.librosa.onset, "onset_strength", lambda **kwargs: np.zeros(2))
    monkeypatch.setattr(sanity_check.librosa.onset, "onset_detect", lambda **kwargs: np.asarray([], int))
    assert sanity_check.sanity_check_timing2(None, np.asarray([1., 2.]), audio_path="unused").tolist() == [0., 0.]


def test_production_onset_errors_do_not_exit_the_application(monkeypatch):
    from map_creation import sanity_check
    monkeypatch.setattr(sanity_check.librosa, "load", lambda *args, **kwargs: (np.zeros(500), 44100))
    monkeypatch.setattr(sanity_check.librosa.onset, "onset_strength", lambda **kwargs: np.zeros(2))

    def fail(**kwargs):
        raise ValueError("invalid onset parameters")

    monkeypatch.setattr(sanity_check.librosa.onset, "onset_detect", fail)
    with pytest.raises(ValueError, match="Onset detection failed"):
        sanity_check.sanity_check_timing2(None, np.asarray([1.]), audio_path="unused")


def test_expansion_conflicts_drop_only_added_objects(song):
    anchor = note()
    other = note(1, time=2., x=0, y=1)
    duplicate = {**note(2, kind="copy"), "anchor_id": 0}
    data = state([anchor, other, duplicate], [chain()])
    original = deepcopy([anchor, other])
    report = post.validate_expansions(data, song)
    assert data["notes"] == original
    assert data["chains"] == []
    assert report == {"rejected_copies": 1, "rejected_chains": 1}


def test_later_copy_cannot_displace_an_already_accepted_chain(song):
    data = state([note()], [chain()])
    assert post.validate_expansions(data, song)["rejected_chains"] == 0
    data["notes"].append({**note(1, time=2., x=0, y=1, kind="copy"), "anchor_id": 0})
    report = post.validate_expansions(data, song)
    assert report == {"rejected_copies": 1, "rejected_chains": 0}
    assert len(data["chains"]) == 1
    assert len(data["notes"]) == 1


def test_hazards_see_chain_interiors_and_preserve_colored_notes(song, monkeypatch):
    monkeypatch.setattr(config, "add_obstacle_flag", True)
    monkeypatch.setattr(config, "sporty_obstacles", False)
    head = note()
    bombs = [{"time": 2., "x": 0, "y": 1}, {"time": 5., "x": 3, "y": 0}]
    walls = [{"time": 1.9, "end_time": 2.1, "x": 0, "y": 0, "width": 1, "height": 3},
             {"time": 4., "end_time": 4.5, "x": 3, "y": 0, "width": 1, "height": 3}]
    data = state([head], [chain()], walls, bombs)
    before = deepcopy(data["notes"])
    report = post.validate_hazards(data, song)
    assert data["notes"] == before
    assert data["walls"] == [walls[1]]
    assert data["bombs"] == [bombs[1]]
    assert report == {"rejected_walls": 1, "rejected_bombs": 1}


def test_obstacle_flag_only_disables_walls(song, monkeypatch):
    monkeypatch.setattr(config, "add_obstacle_flag", False)
    data = state(walls=[{"time": 1., "end_time": 2., "x": 0, "y": 0, "width": 1, "height": 3}],
                 bombs=[{"time": 4., "x": 3, "y": 0}])
    post.validate_hazards(data, song)
    assert data["walls"] == []
    assert len(data["bombs"]) == 1


@pytest.mark.parametrize("geometry", [{"x": 3, "width": 20}, {"y": float("nan")}, {"x": .5}, {"width": True}])
def test_unsupported_wall_templates_are_rejected_even_in_sporty_mode(song, monkeypatch, geometry):
    monkeypatch.setattr(config, "add_obstacle_flag", True)
    monkeypatch.setattr(config, "sporty_obstacles", True)
    data = state(walls=[{"time": 2., "end_time": 3., "x": 0, "y": 0, "width": 1, "height": 5, **geometry}])
    assert post.validate_hazards(data, song)["rejected_walls"] == 1
    assert data["walls"] == []


def test_arc_flags_gap_and_movement_filter_learned_connections(song, monkeypatch):
    endpoints = [note(0), note(1, time=2., x=2)]
    for key, value in {"add_slider_flag": True, "slider_time_gap": [.1, 2.],
                       "slider_movement_minimum": 0, "slider_probability": 1.}.items():
        monkeypatch.setattr(config, key, value)
    for setting, value in (("add_slider_flag", False), ("slider_time_gap", [2., 5.]),
                           ("slider_movement_minimum", 1000)):
        with monkeypatch.context() as scoped:
            scoped.setattr(config, setting, value)
            data = state(endpoints, arcs=[{"endpoint_ids": [0, 1], "time": 1., "end_time": 2., "hand": 0}])
            assert post.validate_arcs(data, song)["rejected_arcs"] == 1
            assert data["arcs"] == []


def test_legacy_timing_audio_handles_a_short_flat_recording(tmp_path):
    import wave

    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(14800)
        handle.writeframes(np.zeros(100, dtype="<i2").tobytes())
    volume, onset, silent = post._legacy_timing_features(path)
    assert volume.shape == onset.shape == (1, config.tcn_len)
    assert np.isfinite(volume).all() and np.isfinite(onset).all()
    assert np.asarray(silent).ndim == 1


def test_arcs_keep_existing_endpoints_and_do_not_reroll_probability(song, monkeypatch):
    for key, value in {"add_slider_flag": True, "slider_time_gap": [.1, 2.],
                       "slider_movement_minimum": 0, "slider_probability": .5, "slider_turbo_start": True}.items():
        monkeypatch.setattr(config, key, value)
    draws = []
    monkeypatch.setattr(post.random, "random", lambda: draws.append(1) or .1)
    endpoints = [note(3), note(7, time=2., x=2)]
    arc = {"endpoint_ids": [3, 7], "time": 1., "end_time": 2., "hand": 0, "mu": 1., "tmu": 1., "anchor_mode": 0}
    data = state(endpoints, arcs=[arc])
    before = deepcopy(endpoints)
    post.validate_arcs(data, song)
    post.validate_arcs(data, song)
    assert data["notes"] == before
    assert len(data["arcs"]) == 1
    assert draws == [1]
    assert data["arcs"][0]["endpoint_ids"] == [3, 7]


@pytest.mark.parametrize("family", ["bombs", "walls", "chains", "missing_endpoint"])
def test_arc_conflicts_are_rejected_without_moving_notes(song, monkeypatch, family):
    monkeypatch.setattr(config, "add_slider_flag", True)
    monkeypatch.setattr(config, "slider_time_gap", [.1, 2.])
    monkeypatch.setattr(config, "slider_movement_minimum", 0)
    monkeypatch.setattr(config, "slider_probability", 1.)
    endpoints = [note(0), note(1, time=2., x=2)]
    arc = {"endpoint_ids": [0, 1], "time": 1., "end_time": 2., "hand": 0}
    data = state(endpoints, arcs=[arc])
    if family == "bombs":
        data[family] = [{"time": 1.5, "x": 1, "y": 0}]
    elif family == "walls":
        data[family] = [{"time": 1.4, "end_time": 1.6, "x": 1, "y": 0, "width": 1, "height": 3}]
    elif family == "chains":
        data[family] = [chain()]
    else:
        arc["endpoint_ids"] = [0, 99]
    before = deepcopy(endpoints)
    post.validate_arcs(data, song)
    assert data["notes"] == before
    assert data["arcs"] == []
