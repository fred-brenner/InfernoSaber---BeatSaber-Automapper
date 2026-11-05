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
    for key, value in {"min_beat_time": .0625, "min_time_diff": .02,
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
    candidates, report = post.postprocess_timing(song, inputs, "unused.egg", skip=("bpm_alignment",))
    assert [item["count"] for item in candidates] == [2, 1]
    assert [item["time"] for item in candidates] == pytest.approx([.7, 1.])
    assert inputs["event_times"].tolist() == [.5, .51, .8]
    operations = {item["stage"]: item for item in report["operations"]}
    assert (operations["minimum_spacing"]["before_events"], operations["minimum_spacing"]["after_events"]) == (3, 2)
    assert operations["onset_snap"]["moved_event_times"] == 2
    assert operations["onset_snap"]["event_count_delta"] == 0


def test_moved_near_events_merge_without_summing_counts(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kwargs: np.asarray([.5, .515]))
    candidates, report = post.postprocess_timing(song, beats([.4, .8], [1, 1]), "unused.egg", skip=("bpm_alignment",))
    assert candidates == [{"time": .5, "frame": 40, "count": 1}]
    assert report["merged_events"] == 1


def test_rejected_onset_zero_does_not_absorb_a_valid_early_onset(song, timing_defaults, monkeypatch):
    # pins the legacy drop-all behavior this test was written to document;
    # the "non_silent" default would restore the unmatched event instead
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "always")
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kwargs: np.asarray([0., .015]))
    candidates, _ = post.postprocess_timing(song, beats([.4, .8], [2, 1]), "unused.egg", skip=("bpm_alignment",))
    assert [(item["time"], item["count"]) for item in candidates] == [(.015, 1)]


def test_filler_uses_strongest_nonzero_class_at_nearest_frame(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 20.)
    monkeypatch.setattr(config, "add_beat_max_bounds", [.1, .4, 3., 4.])
    inputs = beats([1., 3.], [1, 1])
    inputs["frame_probabilities"][200] = [.98, .001, .019]
    candidates, report = post.postprocess_timing(song, inputs, "unused.egg", skip=("bpm_alignment",))
    assert [(item["time"], item["count"]) for item in candidates] == [(1., 1), (2., 2), (3., 1)]
    fill = next(item for item in report["operations"] if item["stage"] == "density_fill_0")
    assert fill == {"stage": "density_fill_0", "before_events": 2, "after_events": 3, "event_count_delta": 1,
                    "before_retained_notes": 2, "after_retained_notes": 4, "moved_event_times": 0}


def test_timing_diagnostics_record_alignment_and_silence_steps(song, timing_defaults, monkeypatch):
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
    candidates, report = post.postprocess_timing(song, beats([], []), "unused.egg", skip=("bpm_alignment",))
    assert candidates == []
    assert report["retained_notes"] == 0


def test_entry_ramp_drops_fill_inside_delay_window(song, timing_defaults, monkeypatch):
    # 120 bpm -> 4 beats = 2.0 s pause threshold; entries: 1.0, 5.0 and 9.0
    # (song duration is 10 s, so the brief's 10.0 tail is shifted to 9.0)
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 1.0)
    monkeypatch.setattr(config, "v2_fill_entry_pause_beats", 4.0)
    insert = np.asarray([1., 5., 5.5, 6.2, 9.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, report = post.postprocess_timing(song, beats([1., 5., 9.], [1, 1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    # 5.5 is inside (5.0, 6.0) -> dropped; 6.2 outside -> kept; model times kept
    assert [item["time"] for item in candidates] == [1., 5., 6.2, 9.]


def test_entry_ramp_first_event_is_an_entry(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 1.0)
    insert = np.asarray([3., 3.4, 8.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, _ = post.postprocess_timing(song, beats([3., 8.], [1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    # 3.4 inside (3.0, 4.0) -> dropped
    assert [item["time"] for item in candidates] == [3., 8.]


def test_entry_ramp_disabled_at_zero_delay(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 0.0)
    insert = np.asarray([1., 5., 5.5, 9.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, _ = post.postprocess_timing(song, beats([1., 5., 9.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 5., 5.5, 9.]


def test_density_fill_skips_silent_spans(song, timing_defaults, monkeypatch):
    # the brief's 10.0 tail is shifted to 9.0: duration is 10 s, so 10.0 maps
    # out of the 1000 frame_times and _merge_candidates drops time >= duration
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_respect_silence", True)
    silent_frames = np.asarray([4.0 + i * (config.hop_size / config.samplerate_music)
                                for i in range(30)])  # silent span around t=4.0..5.0
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), silent_frames))
    insert = np.asarray([1., 4.15, 6., 9.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, report = post.postprocess_timing(song, beats([1., 6., 9.], [1, 1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 6., 9.]  # 4.15 silent -> dropped


def test_silent_span_merge_covers_small_frame_gaps(song, timing_defaults, monkeypatch):
    # A fill inside a <4*hop hole between silent frame runs is still inside the
    # merged span; nearest-frame removal (0.03 s) cannot reach it there, so this
    # pins the span merge instead of remove_silent_times' point matching.
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_respect_silence", True)
    hop = config.hop_size / config.samplerate_music
    silent_frames = np.asarray([4.0 + i * hop for i in range(15)]
                               + [4.0 + (17 + i) * hop for i in range(15)])  # 3*hop hole ~4.5
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), silent_frames))
    insert = np.asarray([1., 4.54, 6., 9.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, _ = post.postprocess_timing(song, beats([1., 6., 9.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 6., 9.]  # 4.54 inside merged span -> dropped


def test_silence_removal_runs_before_density_fill(song, timing_defaults, monkeypatch):
    # density_fill_0 needs iterations and a sub-threshold density to run —
    # mirrors test_filler_uses_strongest_nonzero_class_at_nearest_frame's setup
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 20.)
    monkeypatch.setattr(config, "add_beat_max_bounds", [.1, .4, 3., 4.])
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), [2.]))
    _, report = post.postprocess_timing(song, beats([1., 2., 8.], [1, 1, 1]),
                                        "unused.egg", skip=("bpm_alignment",))
    stages = [op["stage"] for op in report["operations"]]
    assert stages.index("silence_removal") < stages.index("density_fill_0")


def test_onset_snap_unmatched_non_silent_keeps_event(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "non_silent")
    # silent span at 8.5 keeps >0.03 s distance to every event, so the later
    # silence_removal step cannot eat the snapped 8.01 (the brief's 8.0 would)
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), [8.5]))
    # second event unmatched (0); t=4.0 not inside silent span -> restored to original
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    candidates, _ = post.postprocess_timing(song, beats([1., 4., 8.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1.02, 4., 8.01]


def test_onset_snap_unmatched_non_silent_drops_in_silence(song, timing_defaults, monkeypatch):
    # same setup but silent span covers 4.0 -> event still dropped
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "non_silent")
    frames = np.asarray([4.0 + i * (config.hop_size / config.samplerate_music) for i in range(30)])
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), frames))
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    candidates, _ = post.postprocess_timing(song, beats([1., 4., 8.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1.02, 8.01]


def test_onset_snap_unmatched_always_is_legacy(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "always")
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    candidates, _ = post.postprocess_timing(song, beats([1., 4., 8.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1.02, 8.01]


def test_onset_snap_unmatched_unknown_policy_raises(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "bogus")
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    with pytest.raises(ValueError, match="v2_onset_snap_unmatched_policy"):
        post.postprocess_timing(song, beats([1., 4., 8.], [1, 1, 1]),
                                "unused.egg", skip=("bpm_alignment",))


def test_onset_snap_non_silent_ignores_fill_silence_gate(song, timing_defaults, monkeypatch):
    # v2_fill_respect_silence gates fills only; the non_silent onset policy
    # detects silence via add_silence_flag alone. The unmatched event sits in a
    # merged-span hole >0.03 s from any silent frame, so silence_removal cannot
    # cull it downstream — only the onset-snap policy itself can.
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_fill_respect_silence", False)
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "non_silent")
    hop = config.hop_size / config.samplerate_music
    frames = np.asarray([4.0 + i * hop for i in range(15)]
                        + [4.0 + (17 + i) * hop for i in range(15)])  # 3*hop hole ~4.5
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), frames))
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    candidates, _ = post.postprocess_timing(song, beats([1., 4.54, 8.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1.02, 8.01]


def test_training_first_cut_selects_anchor_after_heuristic_cleanup(monkeypatch):
    monkeypatch.setattr(config, "gimme_more_notes_flag", False)
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    monkeypatch.setattr(post, "sanity_check_notes", lambda notes, _: notes)
    original = [note(4, y=0), note(9, y=2)]
    before = deepcopy(original)
    actual, report = post.postprocess_notes(original, [1.])
    # same-hand notes at one timestep collapse before anchoring (never playable);
    # the survivor is marked retained by the source-order reduce contract
    assert [item["id"] for item in actual] == [0]
    assert [item["kind"] for item in actual] == ["retained"]
    assert report["heuristic_notes"] == 0 and report["retained_notes"] == 1
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
    assert report == {"rejected_walls": 1, "rejected_bombs": 1, "rejected_intro": 0}


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


def test_synthetic_turbo_arc_guides_first_note_and_keeps_clearance(song, monkeypatch):
    for key, value in {"add_slider_flag": True, "slider_time_gap": [.5, 12.],
                       "slider_movement_minimum": 3, "slider_probability": 0.}.items():
        monkeypatch.setattr(config, key, value)
    first = note(0, time=.5)
    turbo = {"endpoint_ids": [0], "time": .15, "end_time": .5, "hand": 0,
             "synthetic_head": {"x": 0, "y": 0, "d": 1}, "mu": 1., "tmu": 1., "anchor_mode": 1}
    # Gap .35 is outside the window, movement is zero, probability is off —
    # the guidance arc survives all three like the legacy turbo start.
    data = state([first], arcs=[deepcopy(turbo)])
    diagnostics = post.validate_arcs(data, song)
    assert diagnostics["synthetic_start_arc"] == 1
    assert data["arcs"][0]["endpoint_ids"] == [0]
    data = state([first], bombs=[{"time": .3, "x": 0, "y": 0}], arcs=[deepcopy(turbo)])
    diagnostics = post.validate_arcs(data, song)
    assert diagnostics["synthetic_start_arc"] == 0 and data["arcs"] == []
    bad = deepcopy(turbo)
    bad["synthetic_head"] = {"x": 1, "y": 0, "d": 1}
    data = state([first], arcs=[bad])
    post.validate_arcs(data, song)
    assert data["arcs"] == []


def test_dots_recut_to_implied_direction_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "allow_dot_notes", False)
    monkeypatch.setattr(config, "gimme_more_notes_flag", False)
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    monkeypatch.setattr(post, "sanity_check_notes", lambda notes, _: notes)
    notes = [note(0, time=1., x=1, y=1, direction=1), note(1, time=2., x=1, y=0, direction=8)]
    actual, _ = post.postprocess_notes(notes, [1., 2.])
    assert all(item["d"] != 8 for item in actual)
    expected = post._dot_virtual_direction(1, 1, 1, 1, 0)
    assert next(item["d"] for item in actual if item["time"] == 2.) == expected


def test_first_dot_uses_legacy_vertical_fallback(monkeypatch):
    monkeypatch.setattr(config, "allow_dot_notes", False)
    monkeypatch.setattr(config, "gimme_more_notes_flag", False)
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    monkeypatch.setattr(config, "first_note_layer_threshold", 1)
    monkeypatch.setattr(post, "sanity_check_notes", lambda notes, _: notes)
    upper, _ = post.postprocess_notes([note(0, time=1., y=1, direction=8)], [1.])
    lower, _ = post.postprocess_notes([note(0, time=1., y=0, direction=8)], [1.])
    assert upper[0]["d"] == 0 and lower[0]["d"] == 1


def test_dots_pass_through_when_allowed(monkeypatch):
    monkeypatch.setattr(config, "allow_dot_notes", True)
    monkeypatch.setattr(config, "gimme_more_notes_flag", False)
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    monkeypatch.setattr(post, "sanity_check_notes", lambda notes, _: notes)
    actual, _ = post.postprocess_notes([note(0, time=1., y=1, direction=8)], [1.])
    assert actual[0]["d"] == 8


def test_bomb_flag_disables_all_bombs(song, monkeypatch):
    monkeypatch.setattr(config, "add_bombs_flag", False)
    data = state(walls=[{"time": 1., "end_time": 2., "x": 0, "y": 0, "width": 1, "height": 3}],
                 bombs=[{"time": 4., "x": 3, "y": 0}])
    post.validate_hazards(data, song)
    assert data["walls"] != [] and data["bombs"] == []


def test_validate_hazards_keeps_a_safe_area_around_notes(song, monkeypatch):
    monkeypatch.setattr(config, "v2_bomb_note_time_gap", .4)
    monkeypatch.setattr(config, "v2_bomb_note_cell_gap", 1)
    head = note(0, time=5., x=0, y=1, direction=0)
    adjacent = {"time": 5., "x": 1, "y": 1}   # beside the note, off the cut axis
    corner = {"time": 5., "x": 3, "y": 2}     # more than one cell ring away
    late = {"time": 5.5, "x": 1, "y": 1}      # adjacent cell, outside the time gap
    data = state([head], bombs=[adjacent, corner, late])
    report = post.validate_hazards(data, song)
    assert data["bombs"] == [corner, late]
    assert report == {"rejected_walls": 0, "rejected_bombs": 1, "rejected_intro": 0}


def test_bomb_safe_area_settings_restore_legacy_margins(song, monkeypatch):
    monkeypatch.setattr(config, "v2_bomb_note_time_gap", .1)
    monkeypatch.setattr(config, "v2_bomb_note_cell_gap", 0)
    head = note(0, time=5., x=0, y=1, direction=0)
    adjacent = {"time": 5., "x": 1, "y": 1}   # adjacent is legal under legacy margins
    same = {"time": 5., "x": 0, "y": 1}       # same-cell overlap is still rejected
    data = state([head], bombs=[adjacent, same])
    post.validate_hazards(data, song)
    assert data["bombs"] == [adjacent]


def test_intro_floor_drops_notes_before_cutoff():
    notes = [note(0, .5), note(1, 1.9), note(2, 2.0), note(3, 3.)]
    assert post.apply_intro_floor(notes, 2.0) == 2
    assert [n["time"] for n in notes] == [2.0, 3.]


def test_intro_floor_zero_keeps_all():
    notes = [note(0, 0.), note(1, .5)]
    assert post.apply_intro_floor(notes, 0.) == 0
    assert [n["time"] for n in notes] == [0., .5]


def test_hazards_intro_floor_drops_early_walls_and_bombs(song, monkeypatch):
    monkeypatch.setattr(config, "add_obstacle_flag", True)
    monkeypatch.setattr(config, "sporty_obstacles", False)
    monkeypatch.setattr(config, "v2_map_intro_no_hazards_seconds", 1.)
    walls = [{"time": .4, "end_time": .9, "x": 0, "y": 0, "width": 1, "height": 3},
             {"time": 1.2, "end_time": 1.6, "x": 0, "y": 0, "width": 1, "height": 3}]
    bombs = [{"time": .9, "x": 3, "y": 0}, {"time": 1.5, "x": 3, "y": 0}]
    data = state(walls=walls, bombs=bombs)
    report = post.validate_hazards(data, song)
    assert [wall["time"] for wall in data["walls"]] == [1.2]
    assert [bomb["time"] for bomb in data["bombs"]] == [1.5]
    assert report == {"rejected_walls": 1, "rejected_bombs": 1, "rejected_intro": 2}
    monkeypatch.setattr(config, "v2_map_intro_no_hazards_seconds", 0.)
    data = state(walls=walls, bombs=bombs)
    report = post.validate_hazards(data, song)
    assert len(data["walls"]) == 2 and len(data["bombs"]) == 2
    assert report["rejected_intro"] == 0


def test_heuristic_bomb_candidates_stay_outside_the_note_safe_area(monkeypatch):
    from map_creation.v2_pipeline import _heuristic_bomb_candidates
    monkeypatch.setattr(config, "v2_bombs_prob", 1.)
    monkeypatch.setattr(config, "v2_bomb_note_time_gap", .4)
    monkeypatch.setattr(config, "v2_bomb_note_cell_gap", 1)
    state = {"chains": [], "notes": [
        note(0, time=1., x=0, y=1, direction=0),   # column swing -> only x>=2 free
        note(1, time=2., x=1, y=0, direction=3),   # row swing -> only y=2 free
        note(2, time=3., x=2, y=2, direction=5),   # diagonal swing -> only (3,0) free
        note(3, time=4., x=1, y=1, direction=8)]}  # dot -> only x=3 free
    proposals = _heuristic_bomb_candidates(state, np.random.default_rng(0))
    assert [item["time"] for item in proposals] == [1., 2., 3., 4.]
    assert proposals[0]["x"] >= 2
    assert proposals[1]["y"] == 2
    assert (proposals[2]["x"], proposals[2]["y"]) == (3, 0)
    assert proposals[3]["x"] == 3
    monkeypatch.setattr(config, "v2_bombs_prob", 0.)
    assert _heuristic_bomb_candidates(state, np.random.default_rng(0)) == []


def test_heuristic_bomb_candidates_skip_fully_covered_times(monkeypatch):
    from map_creation.v2_pipeline import _heuristic_bomb_candidates
    monkeypatch.setattr(config, "v2_bombs_prob", 1.)
    state = {"chains": [], "notes": [     # simultaneous swings cover the grid
        note(0, x=0, y=1, direction=0), note(1, x=3, y=1, direction=1),
        note(2, x=0, y=0, direction=2), note(3, x=2, y=2, direction=3),
        note(4, x=0, y=0, direction=4), note(5, x=0, y=0, direction=8)]}
    assert _heuristic_bomb_candidates(state, np.random.default_rng(0)) == []


def test_flow_legal_requires_135_degrees_by_default(monkeypatch):
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 3)
    # middle down-right -> left up-right: 90° over a 2-cell travel (was legal)
    assert not post._flow_legal(5, (0., 2, 1, 7), 0, 1, .5, .1)
    # down-right -> up is 135°, still legal
    assert post._flow_legal(0, (0., 2, 1, 7), 0, 1, .5, .1)
    # same direction stays illegal
    assert not post._flow_legal(7, (0., 0, 1, 7), 1, 1, .5, .1)
    # same-stroke exemption unchanged
    assert post._flow_legal(5, (0., 2, 1, 7), 0, 1, .05, .1)


def test_flow_min_turn_two_restores_travel_exemption(monkeypatch):
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 2)
    assert post._flow_legal(5, (0., 2, 1, 7), 0, 1, .5, .1)       # 90° + 2 cells: legacy legal
    assert not post._flow_legal(5, (0., 2, 1, 7), 1, 1, .5, .1)   # 90° + 1 cell: still illegal


def test_dedup_survivor_keeps_a_135_degree_turn(monkeypatch):
    for key, value in {"allow_dot_notes": True, "gimme_more_notes_flag": False,
                       "single_notes_only_flag": False, "v2_cleanup_same_dir_recut": 0.,
                       "v2_flow_min_turn_steps": 3}.items():
        monkeypatch.setattr(config, key, value)
    notes = [note(0, time=1., x=2, y=1, direction=7),
             note(1, time=2., x=0, y=1, direction=5),   # 90° over a 2-cell travel: legacy-legal
             note(2, time=2., x=3, y=0, direction=8)]   # dot on the exit cell: implied reversal
    actual, _ = post.postprocess_notes(notes, [1., 2.],
                                       skip=post.LEGACY_NOTE_MUTATORS + ("clearance",))
    survivor = next(item for item in actual if item["time"] == 2.)
    # the 90°+travel cut used to outrank the dot; now the dot's implied
    # reversal (>=135° from down-right) is the only legal survivor
    assert (survivor["x"], survivor["y"], survivor["d"]) == (3, 0, 8)
    effective = post._dot_virtual_direction(7, 2, 1, survivor["x"], survivor["y"])
    assert post._step_distance(effective, 7) >= 3


def test_diagonal_cap_limits_fraction():
    # 20 notes 0.5 s apart; each hand alternates antipodal diagonals (d5/d6,
    # 180 deg apart) so each note's adjacent cardinals sit 135 deg from both
    # same-hand neighbours -> legal under the strict Task-2 rule
    notes = [note(i, 1. + .5 * i, hand=i % 2, x=i % 4, y=1,
                 direction=5 if (i // 2) % 2 == 0 else 6) for i in range(20)]
    stats = post.apply_diagonal_cap(notes, rng=np.random.default_rng(0))
    remaining = sum(n["d"] in (4, 5, 6, 7) for n in notes)
    assert remaining <= int(0.10 * 20)
    assert stats["diagonal_before"] == 20 and stats["diagonal_after"] == remaining


def test_diagonal_cap_prefers_vertical_on_tie():
    # lone diagonal: no neighbours -> both cardinals legal, equal cost -> vertical wins
    notes = [note(0, 1., direction=7)]  # down-right -> candidates down(1) / right(3)
    post.apply_diagonal_cap(notes, rng=np.random.default_rng(0))
    assert notes[0]["d"] == 1


def test_diagonal_cap_keeps_when_no_legal_cardinal():
    # candidate up(0) is 135°-legal vs the d7 predecessor but only 45° from
    # the d4 successor; candidate right(3) is 135°-legal vs that successor but
    # 45° from the predecessor — no cardinal is legal against both neighbours
    notes = [note(0, 1., direction=7), note(1, 2., x=0, y=1, direction=5),
             note(2, 3., x=3, y=1, direction=4)]
    post.apply_diagonal_cap(notes, rng=np.random.default_rng(0))
    assert notes[1]["d"] == 5


def test_diagonal_cap_deterministic_per_seed():
    build = lambda: [note(i, 1. + .5 * i, hand=i % 2, x=i % 4, y=1,
                          direction=5 if (i // 2) % 2 == 0 else 6) for i in range(20)]
    a, b = build(), build()
    post.apply_diagonal_cap(a, rng=np.random.default_rng(42))
    post.apply_diagonal_cap(b, rng=np.random.default_rng(42))
    assert [n["d"] for n in a] == [n["d"] for n in b]


def test_diagonal_cap_disabled_at_one(monkeypatch):
    monkeypatch.setattr(config, "v2_diagonal_max_fraction", 1.0)
    notes = [note(i, 1. + .5 * i, hand=i % 2, x=i % 4, y=1,
                 direction=5 if (i // 2) % 2 == 0 else 6) for i in range(20)]
    post.apply_diagonal_cap(notes, rng=np.random.default_rng(42))
    assert [n["d"] for n in notes] == [5 if (i // 2) % 2 == 0 else 6 for i in range(20)]


def test_generate_wires_intro_floor_and_diagonal_cap(monkeypatch):
    """generate() runs the intro floor and diagonal cap with floors enabled —
    stub predictors and stubbed legacy audio helpers, no model assets."""
    from map_creation import v2_pipeline
    from training.beat_models.contracts import CORE, Song
    from training.pipeline_v2.pipeline import dummy_predictors
    from training.pipeline_v2.settings import preparation_settings

    monkeypatch.setattr(post, "sanity_check_timing2",
                        lambda name, times, audio_path=None: np.asarray(times))
    monkeypatch.setattr(post, "align_beats_on_bpm", lambda times, bpm: np.asarray(times))
    monkeypatch.setattr(post, "fill_map_times_scale",
                        lambda times, index=5: np.asarray(times))
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda audio_path: (np.zeros((2, 2)), np.zeros((2, 2)), np.empty(0)))
    monkeypatch.setattr(post, "remove_silent_times",
                        lambda times, silent: np.asarray(times))
    monkeypatch.setattr(post, "_start_end_times",
                        lambda times, volume, onset: np.asarray(times))
    for key, value in {"v2_map_intro_no_notes_seconds": 2.,
                       "v2_map_intro_no_hazards_seconds": 1.,
                       "v2_diagonal_max_fraction": .1,
                       "v2_cleanup_note_heuristics": False}.items():
        monkeypatch.setattr(config, key, value)
    song = Song("wiring", {"duration": 11., "bpm": 120.},
                {"features": np.zeros((1100, 82), np.float32),
                 "coarse_features": np.zeros((11, 82), np.float32),
                 "frame_times": np.arange(1100, dtype=np.float64) / 100,
                 "bin_durations": np.ones(11, np.float32)})
    vocab = {"multiplier": [{"kind": "none"}],
             "walls": [{"kind": "none"}],
             "lighting": [{"type": 0, "value": 1}]}

    class _Timing:
        status = "dummy"

        def __call__(self, inputs):
            values = np.zeros((len(inputs["audio"]), CORE, 3), np.float32)
            values[..., 0] = 1
            # window-local frames: 0.5/1.0 s land below the intro floor,
            # 3.0 s stays above it (later windows repeat the pattern)
            for frame in (50, 100, 300):
                values[:, frame, 0] = 0
                values[:, frame, 1] = 1
            return {"counts": values}

    def lighting(times, encoder):
        return ([{"time": float(time), "type": 0, "value": 1} for time in times],
                {"backend": "v1", "queries": len(times)})

    predictors = dummy_predictors(vocab)
    predictors["timing"] = _Timing()
    settings = preparation_settings()
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    result = v2_pipeline.generate(song, 4., predictors,
                                  lambda times: np.zeros((len(times), 16), np.float32),
                                  settings, vocab, lighting_generator=lighting,
                                  random_factor=0, logger=lambda message: None)
    assert result.diagnostics["intro_zones"]["notes_dropped"] >= 1
    assert "diagonal_cap" in result.diagnostics
    assert min(note["time"] for note in result.state["notes"]) >= 2.0
