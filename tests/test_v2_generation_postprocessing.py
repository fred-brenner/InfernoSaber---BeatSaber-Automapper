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
                       "add_silence_flag": False,
                       "v2_fill_entry_min_notes": 0, "v2_suppress_start_fill": False,
                       "v2_fill_min_interval_seconds": 0.,
                       "v2_onset_fill_flag": False}.items():
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
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the ramp gate
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


def test_entry_count_gate_blocks_until_kth_kept_candidate(song, timing_defaults, monkeypatch):
    # count gate only (delay off): 4-beat pause = 2.0 s at 120 bpm -> entries
    # 1.0 and 7.0; each window runs to the 4th kept candidate at/after it
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 0.)
    monkeypatch.setattr(config, "v2_fill_entry_min_notes", 4)
    monkeypatch.setattr(config, "v2_fill_entry_pause_beats", 4.0)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the count gate
    insert = np.asarray([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 7., 7.5, 8., 9., 9.5])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, report = post.postprocess_timing(song, beats([1., 2., 3., 4., 7., 8., 9.], [1] * 7),
                                                 "unused.egg", skip=("bpm_alignment",))
    # [1,4) blocked (1.5/2.5/3.5 dropped; 4.5/5. released by the 4th note);
    # the >=pause gap restarts the count at 7.0 — only three kept candidates
    # follow it, so [7,inf) is gated and 7.5/9.5 stay out
    assert [item["time"] for item in candidates] == [1., 2., 3., 4., 4.5, 5., 7., 8., 9.]
    assert report["entry_blocked_windows"] == 2


def test_entry_count_gate_unions_with_delay_window(song, timing_defaults, monkeypatch):
    # union of both windows: the 6 s delay outreaches the count gate at entry
    # 1.0, so 4.5 (already past the 4th candidate) is still blocked; 7.5 passes
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 6.)
    monkeypatch.setattr(config, "v2_fill_entry_min_notes", 4)
    monkeypatch.setattr(config, "v2_fill_entry_pause_beats", 4.0)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the union
    insert = np.asarray([1., 2., 2.5, 3., 4., 4.5, 7.5, 8.9, 9., 9.4])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, _ = post.postprocess_timing(song, beats([1., 2., 3., 4., 9.], [1] * 5),
                                            "unused.egg", skip=("bpm_alignment",))
    # entry 1: [1,4) union [1,7) -> 2.5/4.5 blocked; entry 9: [9,15) union
    # [9,inf) -> 9.4 blocked; 7.5/8.9 kept
    assert [item["time"] for item in candidates] == [1., 2., 3., 4., 7.5, 8.9, 9.]


def test_entry_count_gate_zero_preserves_delay_only(song, timing_defaults, monkeypatch):
    # 0 disables the count gate: only the seconds-delay windows apply
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 1.0)
    monkeypatch.setattr(config, "v2_fill_entry_min_notes", 0)
    monkeypatch.setattr(config, "v2_fill_entry_pause_beats", 4.0)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)
    insert = np.asarray([1., 5., 5.5, 6.2, 9.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, report = post.postprocess_timing(song, beats([1., 5., 9.], [1, 1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 5., 6.2, 9.]
    assert report["entry_blocked_windows"] == 3


def test_start_fill_suppressed_drops_only_start_segment(song, timing_defaults, monkeypatch):
    # flag on + intro floor active: inserts before the first event are dropped,
    # the end segment survives
    monkeypatch.setattr(config, "v2_suppress_start_fill", True)
    monkeypatch.setattr(config, "v2_map_intro_no_notes_seconds", 2.)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)
    monkeypatch.setattr(post, "_start_end_times",
                        lambda times, volume, onset: np.sort(np.r_[times, [.2, .4, 9.5]]))
    candidates, report = post.postprocess_timing(song, beats([2., 8.], [1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [2., 8., 9.5]
    assert report["start_fill_suppressed"] == 2


def test_start_fill_suppression_needs_flag_and_intro_floor(song, timing_defaults, monkeypatch):
    # flag off or a zero floor both restore the legacy opening inserts
    monkeypatch.setattr(config, "v2_map_intro_no_notes_seconds", 2.)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)
    monkeypatch.setattr(post, "_start_end_times",
                        lambda times, volume, onset: np.sort(np.r_[times, [.2, 9.5]]))
    candidates, report = post.postprocess_timing(song, beats([2., 8.], [1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [.2, 2., 8., 9.5]
    assert report["start_fill_suppressed"] == 0

    monkeypatch.setattr(config, "v2_suppress_start_fill", True)
    monkeypatch.setattr(config, "v2_map_intro_no_notes_seconds", 0.)
    candidates, report = post.postprocess_timing(song, beats([2., 8.], [1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [.2, 2., 8., 9.5]
    assert report["start_fill_suppressed"] == 0


def test_density_fill_skips_silent_spans(song, timing_defaults, monkeypatch):
    # the brief's 10.0 tail is shifted to 9.0: duration is 10 s, so 10.0 maps
    # out of the 1000 frame_times and _merge_candidates drops time >= duration
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_fill_respect_silence", True)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the density fill
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
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the density fill
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
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the step order
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), [2.]))
    _, report = post.postprocess_timing(song, beats([1., 2., 8.], [1, 1, 1]),
                                        "unused.egg", skip=("bpm_alignment",))
    stages = [op["stage"] for op in report["operations"]]
    assert stages.index("silence_removal") < stages.index("density_fill_0")


def test_onset_snap_unmatched_non_silent_keeps_event(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "non_silent")
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the snap policy
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
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the snap policy
    frames = np.asarray([4.0 + i * (config.hop_size / config.samplerate_music) for i in range(30)])
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), frames))
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    candidates, _ = post.postprocess_timing(song, beats([1., 4., 8.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1.02, 8.01]


def test_onset_snap_unmatched_always_is_legacy(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "always")
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the snap policy
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
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the snap policy
    hop = config.hop_size / config.samplerate_music
    frames = np.asarray([4.0 + i * hop for i in range(15)]
                        + [4.0 + (17 + i) * hop for i in range(15)])  # 3*hop hole ~4.5
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), frames))
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    candidates, _ = post.postprocess_timing(song, beats([1., 4.54, 8.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1.02, 8.01]


def test_onset_snap_unmatched_never_restores_inside_silence(song, timing_defaults, monkeypatch):
    # "never" restores the unmatched event even inside a detected silent
    # span; 4.54 sits in a merged-span hole so silence_removal cannot cull
    # it downstream — same fixture as the non_silent drop test above
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "never")
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)  # isolate the snap policy
    hop = config.hop_size / config.samplerate_music
    frames = np.asarray([4.0 + i * hop for i in range(15)]
                        + [4.0 + (17 + i) * hop for i in range(15)])  # 3*hop hole ~4.5
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), frames))
    monkeypatch.setattr(post, "sanity_check_timing2", lambda _, times, **kw: np.asarray([1.02, 0., 8.01]))
    candidates, _ = post.postprocess_timing(song, beats([1., 4.54, 8.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1.02, 4.54, 8.01]


def test_fill_silence_gate_off_never_decodes_silence(song, timing_defaults, monkeypatch):
    # fills under respect=False must not burn the legacy decode: the cached
    # detection stays empty until a gated-or-forced call asks for it
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_fill_respect_silence", False)
    monkeypatch.setattr(config, "v2_onset_snap_unmatched_policy", "always")
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 40.)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 0.)
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: pytest.fail("fills-only respect=False must not decode"))
    insert = np.asarray([1., 4.15, 6., 9.])
    monkeypatch.setattr(post, "fill_map_times_scale", lambda times, idx: insert)
    candidates, _ = post.postprocess_timing(song, beats([1., 6., 9.], [1, 1, 1]),
                                            "unused.egg",
                                            skip=("bpm_alignment", "silence_removal", "start_end_fill"))
    assert [item["time"] for item in candidates] == [1., 4.15, 6., 9.]


def test_max_gap_fill_inserts_onset_times(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_edge_seconds", 0.5)
    monkeypatch.setattr(post, "_detect_onsets", lambda _: np.asarray([2.6, 3.3, 4.9, 7.2]))
    candidates, report = post.postprocess_timing(song, beats([1., 8.], [1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    # gap 1->8 exceeds 4 s: onsets in (1.5, 7.5) inserted -> 2.6, 3.3, 4.9, 7.2
    assert [item["time"] for item in candidates] == [1., 2.6, 3.3, 4.9, 7.2, 8.]


def test_max_gap_fill_respects_silence_and_edge(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_edge_seconds", 1.0)
    frames = np.asarray([3. + i * (config.hop_size / config.samplerate_music) for i in range(30)])
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), frames))
    monkeypatch.setattr(post, "_detect_onsets", lambda _: np.asarray([1.6, 3.1, 7.8]))
    candidates, _ = post.postprocess_timing(song, beats([1., 9.], [1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    # interior (2.0, 8.0): 1.6 outside edge, 3.1 silent, 7.8 kept
    assert [item["time"] for item in candidates] == [1., 7.8, 9.]


def test_max_gap_fill_bpm_grid_source(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_source", "bpm_grid")
    candidates, _ = post.postprocess_timing(song, beats([1., 8.], [1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    # 120 bpm -> 0.5 s grid inside (1.5, 7.5)
    assert [item["time"] for item in candidates] == [1., 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 8.]


def test_gap_fill_v1_beat_source_inserts_predicted_times(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_source", "v1_beat")
    monkeypatch.setattr(post, "_v1_beat_times", lambda path: np.asarray([3.0, 5.5]))
    candidates, _ = post.postprocess_timing(song, beats([1., 8.], [1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 3., 5.5, 8.]


def test_gap_fill_v1_beat_falls_back_to_onsets(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_source", "v1_beat")
    def boom(path):
        raise FileNotFoundError("missing bundle")
    monkeypatch.setattr(post, "_v1_beat_times", boom)
    monkeypatch.setattr(post, "_detect_onsets", lambda _: np.asarray([4.0]))
    candidates, report = post.postprocess_timing(song, beats([1., 8.], [1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 4., 8.]
    assert report.get("gap_fill_source_used") == "onsets"  # fallback recorded


def test_max_gap_fill_never_refills_detected_silence(song, timing_defaults, monkeypatch):
    # the fill-respect flag scopes to subdivision fills: the net must stay
    # out of detected silence even with v2_fill_respect_silence=False, or it
    # would refill spans silence_removal just emptied (it runs earlier)
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(config, "v2_fill_respect_silence", False)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_edge_seconds", 0.5)
    frames = np.asarray([3. + i * (config.hop_size / config.samplerate_music) for i in range(30)])
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), frames))
    monkeypatch.setattr(post, "_detect_onsets", lambda _: np.asarray([2., 3.5, 7.8]))
    candidates, _ = post.postprocess_timing(song, beats([1., 9.], [1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    # interior (1.5, 8.5): 3.5 sits inside the detected span -> skipped
    assert [item["time"] for item in candidates] == [1., 2., 7.8, 9.]


def test_max_gap_fill_unknown_source_raises_before_any_gap(song, timing_defaults, monkeypatch):
    # eager validation: a bogus source raises even when no dead zone exists
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_source", "bogus")
    with pytest.raises(ValueError, match="v2_gap_fill_source"):
        post.postprocess_timing(song, beats([1., 2.], [1, 1]), "unused.egg",
                                skip=("bpm_alignment",))


def test_max_gap_fill_requires_audio_unless_bpm_grid(song, timing_defaults, monkeypatch):
    # production always resolves an audio path; without one only the bpm
    # lattice can still fill (it decodes nothing)
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    with pytest.raises(ValueError, match="audio_path"):
        post.postprocess_timing(song, beats([1., 8.], [1, 1]), None,
                                skip=("bpm_alignment",))
    monkeypatch.setattr(config, "v2_gap_fill_source", "bpm_grid")
    candidates, _ = post.postprocess_timing(song, beats([1., 8.], [1, 1]), None,
                                            skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == [1., 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 8.]


def test_max_gap_fill_edge_floors_at_min_beat_time(song, timing_defaults, monkeypatch):
    # the edge alone keeps inserts min_beat_time off the bounding events:
    # values below it (incl. negative) clamp up to the spacing floor
    monkeypatch.setattr(config, "v2_max_note_gap_seconds", 4.0)
    monkeypatch.setattr(config, "v2_gap_fill_edge_seconds", -1.)
    monkeypatch.setattr(post, "_detect_onsets", lambda _: np.asarray([1.04, 2., 7.9]))
    candidates, _ = post.postprocess_timing(song, beats([1., 8.], [1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    # interior (1.0625, 7.9375): 1.04 is too close to the bound, 7.9 kept
    assert [item["time"] for item in candidates] == [1., 2., 7.9, 8.]


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


def test_flow_legal_requires_135_degrees_when_legacy_flow_configured(monkeypatch):
    monkeypatch.setattr(config, "v2_playability_enabled", False)
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
    monkeypatch.setattr(config, "v2_playability_enabled", False)
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 2)
    assert post._flow_legal(5, (0., 2, 1, 7), 0, 1, .5, .1)       # 90° + 2 cells: legacy legal
    assert not post._flow_legal(5, (0., 2, 1, 7), 1, 1, .5, .1)   # 90° + 1 cell: still illegal


def test_dedup_survivor_keeps_a_135_degree_turn(monkeypatch):
    for key, value in {"v2_playability_enabled": False,
                       "allow_dot_notes": True, "gimme_more_notes_flag": False,
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


@pytest.fixture
def speed_defaults(monkeypatch):
    # pin the legacy speed cost and the cap knobs so expectations are exact
    for key, value in {"max_speed": 32., "reaction_time": 1.15,
                       "reaction_time_fact": .013, "cdf": 1.2, "cdf_lr": 1.15,
                       "min_time_diff": .02, "v2_speed_cap_scale": 1.,
                       "add_beat_intensity_orig": 100.,
                       "v2_speed_cap_start_notes": 0, "v2_speed_cap_start_factor": .35,
                       "v2_cleanup_same_dir_gap": .1, "v2_cleanup_swing_window": .15,
                       "v2_flow_min_turn_steps": 3}.items():
        monkeypatch.setattr(config, key, value)


def test_speed_cap_drops_later_note_of_overspeed_pair(speed_defaults):
    notes = [note(0, 1., x=0, y=1, direction=1),
             note(1, 1.05, x=3, y=1, direction=1),               # ~148 cells/s at cap ~24.5
             note(2, 1.05, hand=1, x=1, y=1, direction=8)]      # other hand untouched
    stats = post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0, 2]
    assert stats["speed_cap_dropped"] == 1
    assert stats["speed_cap_value"] == pytest.approx(
        32. * post.get_factor_from_max_speed(32., .5, 1))


def test_speed_cap_keeps_a_playable_pair(speed_defaults):
    notes = [note(0, 1., x=0, y=1, direction=1),
             note(1, 1.5, x=0, y=1, direction=0)]                # same cell, .5 s: ~3 cells/s
    stats = post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0, 1]
    assert stats["speed_cap_dropped"] == 0


def test_speed_cap_start_ramp_tightens_early_pairs(speed_defaults, monkeypatch):
    # a 3-cell jump in .4 s (~18.5 cells/s) is legal past the ramp but over the
    # ~35 % cap applied to the second combined note position
    monkeypatch.setattr(config, "v2_speed_cap_start_notes", 30)
    early = [note(0, 1., x=0, y=1, direction=1), note(1, 1.4, x=3, y=1, direction=1)]
    post.apply_speed_cap(early)
    assert [item["id"] for item in early] == [0]
    fillers = [note(10 + i, .5 + .6 * i, hand=1, x=1, y=1, direction=8) for i in range(30)]
    late = fillers + [note(0, 19., x=0, y=1, direction=1), note(1, 19.4, x=3, y=1, direction=1)]
    post.apply_speed_cap(late)
    assert [item["id"] for item in late[-2:]] == [0, 1]


def test_speed_cap_scale_scales_the_cap(speed_defaults, monkeypatch):
    build = lambda: [note(0, 1., x=0, y=1, direction=1), note(1, 1.4, x=3, y=1, direction=1)]
    monkeypatch.setattr(config, "v2_speed_cap_scale", .5)
    notes = build()
    post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0]
    monkeypatch.setattr(config, "v2_speed_cap_scale", 2.)
    notes = build()
    post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0, 1]


def test_speed_cap_scales_with_beat_intensity(speed_defaults, monkeypatch):
    # the app's Beat Intensity field multiplies v2_speed_cap_scale linearly:
    # 100% keeps the configured reference, 50% halves the allowed speed
    from tools.utils.numpy_shorts import get_factor_from_max_speed
    base = 32. * get_factor_from_max_speed(32., .5, 1)
    build = lambda: [note(0, 1., x=0, y=1, direction=1),
                     note(1, 1.1, x=3, y=1, direction=1)]
    monkeypatch.setattr(config, "v2_speed_cap_scale", 2.)
    monkeypatch.setattr(config, "add_beat_intensity_orig", 200.)
    notes = build()
    stats = post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0, 1]
    assert stats["speed_cap_value"] == pytest.approx(base * 4.)
    monkeypatch.setattr(config, "add_beat_intensity_orig", 50.)
    notes = build()
    stats = post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0]
    assert stats["speed_cap_value"] == pytest.approx(base)


def test_speed_cap_disabled_at_zero_scale(speed_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_speed_cap_scale", 0.)
    notes = [note(0, 1., x=0, y=1, direction=1), note(1, 1.05, x=3, y=1, direction=1)]
    stats = post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0, 1]
    assert stats["speed_cap_dropped"] == 0 and stats["speed_cap_value"] == 0.


def test_speed_cap_drops_chain_and_copies_with_their_anchor(speed_defaults):
    notes = [note(0, 1., x=0, y=1, direction=1),
             note(1, 1.05, x=3, y=1, direction=1),
             {**note(2, 3., x=2, y=2, direction=4), "kind": "copy", "anchor_id": 1}]
    chains = [{**chain(), "id": 0, "anchor_id": 1}, {**chain(), "id": 1, "anchor_id": 0}]
    stats = post.apply_speed_cap(notes, chains)
    assert [item["id"] for item in notes] == [0]
    assert [item["anchor_id"] for item in chains] == [0]
    assert stats["speed_cap_dropped"] == 2 and stats["speed_cap_chains_dropped"] == 1


def test_speed_cap_recuts_flow_violation_after_deletion(speed_defaults):
    # B dies to the cap; the revealed A->C pair repeats direction 0 -> recut
    notes = [note(0, 1., x=0, y=1, direction=0),
             note(1, 1.05, x=3, y=2, direction=1),               # overspeed vs A
             note(2, 2., x=1, y=1, direction=0)]
    stats = post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0, 2]
    assert notes[1]["d"] != 0 and post._step_distance(notes[1]["d"], 0) >= 3
    assert stats["speed_cap_recuts"] == 1


def test_speed_cap_drops_when_no_flow_legal_recut(speed_defaults, monkeypatch):
    # no legal recut for the revealed pair -> the successor is dropped too
    monkeypatch.setattr(post, "_flow_direction", lambda *args, **kwargs: 8)
    notes = [note(0, 1., x=0, y=1, direction=0),
             note(1, 1.05, x=3, y=2, direction=1),
             note(2, 2., x=1, y=1, direction=0)]
    stats = post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0]
    assert stats["speed_cap_dropped"] == 2


def test_speed_cap_iterates_a_chain_of_violations(speed_defaults):
    # every consecutive same-hand pair is over the cap: each deletion reveals
    # the next pair -> the fixpoint strips back to the first note
    notes = [note(0, 1., x=0, y=1, direction=1),
             note(1, 1.06, x=3, y=1, direction=1),
             note(2, 1.12, x=0, y=1, direction=1),
             note(3, 1.18, x=3, y=1, direction=1)]
    stats = post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0]
    assert stats["speed_cap_dropped"] == 3


def test_speed_cap_drop_retained_false_spares_retained(speed_defaults):
    # post-expansion mode: a retained later-note is neither dropped nor recut
    # — counted once and kept as the next predecessor — while a copy in the
    # same position still drops
    notes = [note(0, 1., x=0, y=1, direction=1),
             note(1, 1.05, x=3, y=1, direction=1),                    # retained, over cap vs 0
             note(2, 1.10, x=0, y=1, direction=1, kind="copy")]       # copy, over cap vs 1
    stats = post.apply_speed_cap(notes, drop_retained=False)
    assert [item["id"] for item in notes] == [0, 1]
    assert stats["speed_cap_retained_skipped"] == 1
    assert stats["speed_cap_dropped"] == 1


def test_speed_cap_drop_retained_false_skips_retained_recuts(speed_defaults):
    # B (a copy) dies to the cap; the revealed A->C repeat would recut, but
    # both notes are retained -> the pair counts as a flow skip, unmutated
    notes = [note(0, 1., x=0, y=1, direction=0),
             {**note(1, 1.05, x=3, y=2, direction=1), "kind": "copy", "anchor_id": 0},
             note(2, 2., x=1, y=1, direction=0)]
    stats = post.apply_speed_cap(notes, drop_retained=False)
    assert [item["id"] for item in notes] == [0, 2]
    assert notes[1]["d"] == 0
    assert stats["speed_cap_dropped"] == 1
    assert stats["speed_cap_recuts"] == 0
    assert stats["speed_cap_retained_skipped"] == 0
    assert stats["speed_cap_flow_skipped"] == 1


def test_speed_cap_start_ramp_composes_like_v1(speed_defaults, monkeypatch):
    # v1 evaluates get_factor_from_max_speed on the ramped speed: position 1
    # caps at 32 * r * factor(32 * r) ~= 7.14 with r = ramp[1] ~= .3724,
    # where scaling the finished base cap by r would allow ~9.14 — the 8.0
    # cells/s pair below sits between the two compositions.
    monkeypatch.setattr(config, "v2_speed_cap_start_notes", 30)
    ramp = .35 + .65 / 29  # np.linspace(.35, 1., 30)[1]
    expected = 32 * ramp * post.get_factor_from_max_speed(32 * ramp, .5, 1)
    assert expected == pytest.approx(7.14, abs=.01)
    notes = [note(0, 1., x=0, y=1, direction=1),
             note(1, 1.627, x=3, y=1, direction=0)]  # ~8.0 cells/s
    stats = post.apply_speed_cap(notes)
    assert [item["id"] for item in notes] == [0]
    assert stats["speed_cap_dropped"] == 1


def test_speed_cap_frozen_pair_drops_mutable_earlier(speed_defaults):
    # the earlier note of an overspeed pair is equally responsible: with the
    # later note frozen a mutable earlier one drops instead of skipping
    notes = [{**note(0, 1., x=0, y=1, direction=1, kind="copy"), "anchor_id": 1},
             note(1, 1.05, x=3, y=1, direction=1)]
    stats = post.apply_speed_cap(notes, drop_retained=False)
    assert [item["id"] for item in notes] == [1]
    assert stats["speed_cap_dropped"] == 1
    assert stats["speed_cap_retained_skipped"] == 0


def test_speed_cap_retained_pair_counts_speed_skip(speed_defaults):
    # retained->retained overspeed cannot resolve: counted once, both kept
    notes = [note(0, 1., x=0, y=1, direction=1),
             note(1, 1.05, x=3, y=1, direction=1)]
    stats = post.apply_speed_cap(notes, drop_retained=False)
    assert [item["id"] for item in notes] == [0, 1]
    assert stats["speed_cap_dropped"] == 0
    assert stats["speed_cap_retained_skipped"] == 1
    assert stats["speed_cap_flow_skipped"] == 0


def test_speed_cap_frozen_flow_recuts_mutable_earlier(speed_defaults):
    # copy B dies to the cap; the revealed A->C repeat is a flow violation —
    # with C retained the mutable earlier note is recut instead of skipped
    notes = [note(0, 1., x=0, y=1, direction=0, kind="heuristic"),
             {**note(1, 1.05, x=3, y=2, direction=1, kind="copy"), "anchor_id": 2},
             note(2, 2., x=1, y=1, direction=0)]
    stats = post.apply_speed_cap(notes, drop_retained=False)
    assert [item["id"] for item in notes] == [0, 2]
    assert notes[0]["d"] != 0 and post._step_distance(notes[0]["d"], 0) >= 3
    assert stats["speed_cap_dropped"] == 1 and stats["speed_cap_recuts"] == 1
    assert stats["speed_cap_flow_skipped"] == 0


def test_speed_cap_frozen_flow_drops_earlier_without_legal_recut(speed_defaults, monkeypatch):
    # same setup but no legal recut exists -> the mutable earlier note drops
    monkeypatch.setattr(post, "_flow_direction", lambda *args, **kwargs: 8)
    notes = [note(0, 1., x=0, y=1, direction=0, kind="heuristic"),
             {**note(1, 1.05, x=3, y=2, direction=1, kind="copy"), "anchor_id": 2},
             note(2, 2., x=1, y=1, direction=0)]
    stats = post.apply_speed_cap(notes, drop_retained=False)
    assert [item["id"] for item in notes] == [2]
    assert stats["speed_cap_dropped"] == 2
    assert stats["speed_cap_recuts"] == 0 and stats["speed_cap_flow_skipped"] == 0


def test_speed_cap_recut_inside_fast_pair_excludes_counter(speed_defaults):
    # copy B dies to the cap; C's revealed recut must avoid the counter
    # family: without the exclusion the down-left d4 wins on original
    # distance, inside a fast pair the up-left d6 takes it
    notes = [note(0, 1., x=0, y=1, direction=3, kind="heuristic"),
             {**note(1, 1.05, x=3, y=2, direction=1, kind="copy"), "anchor_id": 2},
             note(2, 1.24, x=1, y=0, direction=5, kind="heuristic")]
    stats = post.apply_speed_cap(notes, drop_retained=False)
    assert [item["id"] for item in notes] == [0, 2]
    assert notes[1]["d"] == 6
    assert stats["speed_cap_recuts"] == 1


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
                       "v2_speed_cap_scale": 1.,
                       "v2_max_note_gap_seconds": 0.,   # audio_path=None: keep the net off
                       "v2_cleanup_note_heuristics": False,
                       "v2_note_postprocessing_mode": "search",
                       "emphasize_beats_flag": False}.items():
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
    assert "fast_directions" in result.diagnostics
    assert "diagonal_cap" in result.diagnostics
    assert "speed_cap" in result.diagnostics
    assert "speed_cap_post_expansion" in result.diagnostics
    assert "fast_counter_sweep" in result.diagnostics
    assert min(note["time"] for note in result.state["notes"]) >= 2.0


def test_generate_speed_cap_runs_around_expansion(monkeypatch):
    """Regression for "A downstream stage changed retained notes": the primary
    speed-cap pass runs before the retained snapshot and the post-expansion
    pass only drops mutable notes — overspeed copies die without touching
    retained notes."""
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
    for key, value in {"v2_map_intro_no_notes_seconds": 0.,
                       "v2_map_intro_no_hazards_seconds": 0.,
                       "v2_diagonal_max_fraction": 1.,
                       "v2_speed_cap_scale": 1.,
                       "v2_max_note_gap_seconds": 0.,   # audio_path=None: keep the net off
                       "v2_cleanup_note_heuristics": False,
                       "v2_note_postprocessing_mode": "search",
                       "emphasize_beats_flag": False}.items():
        monkeypatch.setattr(config, key, value)
    song = Song("wiring", {"duration": 11., "bpm": 120.},
                {"features": np.zeros((1100, 82), np.float32),
                 "coarse_features": np.zeros((11, 82), np.float32),
                 "frame_times": np.arange(1100, dtype=np.float64) / 100,
                 "bin_durations": np.ones(11, np.float32)})
    # every retained note gets a copy 3 cells over and .05 s later: overspeed
    vocab = {"multiplier": [{"kind": "none"},
                            {"kind": "copies", "copies": [{"dx": 3, "dy": 0, "dt": .05}]}],
             "walls": [{"kind": "none"}],
             "lighting": [{"type": 0, "value": 1}]}

    class _Timing:
        status = "dummy"

        def __call__(self, inputs):
            values = np.zeros((len(inputs["audio"]), CORE, 3), np.float32)
            values[..., 0] = 1
            for frame in (50, 100, 300):
                values[:, frame, 0] = 0
                values[:, frame, 1] = 1
            return {"counts": values}

    def lighting(times, encoder):
        return ([{"time": float(time), "type": 0, "value": 1} for time in times],
                {"backend": "v1", "queries": len(times)})

    calls = []
    real = post.apply_speed_cap

    def spy(notes, chains=None, rng=None, logger=None, drop_retained=True):
        calls.append({"drop_retained": drop_retained,
                      "copies": sum(item.get("kind") == "copy" for item in notes)})
        return real(notes, chains, rng=rng, logger=logger, drop_retained=drop_retained)

    monkeypatch.setattr(post, "apply_speed_cap", spy)
    predictors = dummy_predictors(vocab)
    predictors["timing"] = _Timing()
    settings = preparation_settings()
    settings["schedule"]["arc_max_gap_seconds"] = 6.
    result = v2_pipeline.generate(song, 4., predictors,
                                  lambda times: np.zeros((len(times), 16), np.float32),
                                  settings, vocab, lighting_generator=lighting,
                                  random_factor=0, logger=lambda message: None)
    # call order proves the positions: the primary pass ran before expansion
    # (no copies exist yet); the post-expansion pass ran with copies present
    assert [call["drop_retained"] for call in calls] == [True, False]
    assert calls[0]["copies"] == 0 and calls[1]["copies"] > 0
    post_expansion = result.diagnostics["speed_cap_post_expansion"]
    assert post_expansion["speed_cap_dropped"] == calls[1]["copies"]
    assert all(item.get("kind") != "copy" for item in result.state["notes"])


@pytest.fixture
def fast_defaults(monkeypatch):
    # pin the legacy speed cost and the fast-direction knobs for exact ratios
    for key, value in {"max_speed": 32., "reaction_time": 1.15,
                       "reaction_time_fact": .013, "cdf": 1.2, "cdf_lr": 1.15,
                       "min_time_diff": .02, "v2_fast_gap_seconds": .2,
                       "v2_fast_counter_direction": "recut",
                       "v2_cleanup_same_dir_gap": .1, "v2_cleanup_swing_window": .15,
                       "v2_flow_min_turn_steps": 3}.items():
        monkeypatch.setattr(config, key, value)


def test_fast_run_collapses_to_ping_pong(fast_defaults):
    # .15 s same-hand chain with the ±45° draw denied: every direction is
    # >=135°-legal input but outside {pred, REVERSE(pred)}; each recut lands
    # on the predecessor's 180° reverse
    notes = [note(0, 1.0, x=0, y=1, direction=1),
             note(1, 1.15, x=1, y=1, direction=5),
             note(2, 1.30, x=2, y=1, direction=6),
             note(3, 1.45, x=3, y=1, direction=5)]
    stats = post.apply_fast_direction_rules(notes, rng=SimpleNamespace(uniform=lambda: 1.))
    assert [item["d"] for item in notes] == [1, 0, 1, 0]
    assert stats["fast_pairs"] == 3 and stats["fast_simplified"] == 3


def test_fast_run_admitted_draw_keeps_off_reversals(fast_defaults):
    # same chain with the ±45° draw admitted: 5 after 1 and 6 after 5 are
    # legal off-reversals (135°/225°) inside the widened set -> kept, while a
    # cut outside the set (4 is +90° from 5) still recuts into the reversal
    # family {REVERSE, REVERSE±45°}
    notes = [note(0, 1.0, x=0, y=1, direction=1),
             note(1, 1.15, x=1, y=1, direction=5),
             note(2, 1.30, x=2, y=1, direction=6),
             note(3, 1.45, x=3, y=1, direction=5)]
    stats = post.apply_fast_direction_rules(notes, rng=SimpleNamespace(uniform=lambda: 0.))
    assert [item["d"] for item in notes] == [1, 5, 6, 5]
    assert stats["fast_pairs"] == 3 and stats["fast_simplified"] == 0

    notes = [note(0, 1.0, x=0, y=1, direction=1),
             note(1, 1.15, x=1, y=1, direction=5),
             note(2, 1.30, x=2, y=1, direction=4)]
    stats = post.apply_fast_direction_rules(notes, rng=SimpleNamespace(uniform=lambda: 0.))
    reverse = post._REVERSE_DIRECTION[5]
    angle = post._DIRECTION_ANGLE[reverse]
    family = {reverse, post._ANGLE_DIRECTION[(angle - 1) % 8],
              post._ANGLE_DIRECTION[(angle + 1) % 8]}
    assert notes[1]["d"] == 5
    assert notes[2]["d"] in family
    assert stats["fast_simplified"] == 1


def test_fast_run_slow_pairs_untouched(fast_defaults):
    notes = [note(0, 1.0, x=0, y=1, direction=1),
             note(1, 1.65, x=1, y=1, direction=5),
             note(2, 2.30, x=2, y=1, direction=6)]
    stats = post.apply_fast_direction_rules(notes, rng=np.random.default_rng(0))
    assert [item["d"] for item in notes] == [1, 5, 6]
    assert stats["fast_pairs"] == 0 and stats["fast_simplified"] == 0
    assert stats["expected_speed"] > 0


def test_fast_run_admits_45_when_local_is_average(fast_defaults):
    # a lone pair has expected == local -> the ±45° option is always admitted
    # (y=1: outside both counter-direction rows)
    notes = [note(0, 1.0, x=0, y=0, direction=2),
             note(1, 1.05, x=1, y=1, direction=4)]   # up-left = left-45°, same-stroke legal
    stats = post.apply_fast_direction_rules(notes, rng=np.random.default_rng(0))
    assert notes[1]["d"] == 4
    assert stats["expected_speed"] == pytest.approx(
        post.calc_note_speed([0, 0, 0, 2], [1, 1, 0, 4], .05))


def test_fast_run_denies_45_at_extreme_local_speed(fast_defaults):
    # expected/local ~= .34: a low draw admits the ±45° keep, a high draw
    # strips it and the recut falls back to the 180° reverse
    build = lambda: [note(0, 1.0, hand=0, x=0, y=0, direction=2),
                     note(1, 1.04, hand=0, x=3, y=2, direction=4),
                     note(10, 2.0, hand=1, x=1, y=1, direction=1),
                     note(11, 4.0, hand=1, x=1, y=1, direction=0),
                     note(12, 5.0, hand=1, x=1, y=1, direction=1),
                     note(13, 7.0, hand=1, x=1, y=1, direction=0)]
    notes = build()
    post.apply_fast_direction_rules(notes, rng=SimpleNamespace(uniform=lambda: .99))
    assert notes[1]["d"] == 3
    notes = build()
    post.apply_fast_direction_rules(notes, rng=SimpleNamespace(uniform=lambda: .01))
    assert notes[1]["d"] == 4


def test_fast_run_off_reversal_reachable_for_separated_pair(fast_defaults):
    # .15 s sits between same_dir_gap (.1) and fast_gap (.2): the pair no
    # longer shares a stroke, so _flow_legal demands >=135° and only the
    # reversal cone can ever be legal — the admitted ±45° draw is what makes
    # an off-reversal cut reachable here at all
    build = lambda: [note(0, 1.0, x=0, y=1, direction=1),
                     note(1, 1.15, x=1, y=1, direction=5)]
    notes = build()
    post.apply_fast_direction_rules(notes, rng=SimpleNamespace(uniform=lambda: 0.))
    assert notes[1]["d"] == 5   # 135° off-reversal of the down cut: kept
    notes = build()
    post.apply_fast_direction_rules(notes, rng=SimpleNamespace(uniform=lambda: 1.))
    assert notes[1]["d"] == 0   # denied draw collapses to the 180° reverse


def test_fast_run_counter_direction_recut(fast_defaults):
    # top-row down-right following an up-left cut: legal input, counter family
    # (the up-left predecessor sits on the top row and is not counter itself)
    notes = [note(0, 1.0, x=0, y=2, direction=4),
             note(1, 1.15, x=0, y=2, direction=7)]
    stats = post.apply_fast_direction_rules(notes, rng=np.random.default_rng(0))
    assert notes[1]["d"] == 3
    assert stats["fast_counter_recuts"] == 1 and stats["fast_simplified"] == 0


def test_fast_run_counter_direction_kept_when_slow(fast_defaults):
    notes = [note(0, 1.0, x=0, y=2, direction=4),
             note(1, 1.6, x=0, y=2, direction=7)]
    stats = post.apply_fast_direction_rules(notes, rng=np.random.default_rng(0))
    assert notes[1]["d"] == 7 and stats["fast_counter_recuts"] == 0


@pytest.mark.parametrize("mode", ["off", "keep"])
def test_fast_run_counter_off_keeps_counter_directions(fast_defaults, monkeypatch, mode):
    monkeypatch.setattr(config, "v2_fast_counter_direction", mode)
    notes = [note(0, 1.0, x=0, y=2, direction=4),
             note(1, 1.15, x=0, y=2, direction=7)]
    stats = post.apply_fast_direction_rules(notes, rng=np.random.default_rng(0))
    # no allowed direction is legal against the up-left cut, so the note stays
    assert notes[1]["d"] == 7
    assert stats["fast_counter_recuts"] == 0 and stats["fast_simplified"] == 0


def test_fast_run_disabled_at_zero_gap(fast_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_fast_gap_seconds", 0.)
    notes = [note(0, 1.0, x=0, y=1, direction=1),
             note(1, 1.15, x=1, y=1, direction=5)]
    stats = post.apply_fast_direction_rules(notes, rng=np.random.default_rng(0))
    assert [item["d"] for item in notes] == [1, 5]
    assert stats == {"fast_pairs": 0, "fast_simplified": 0,
                     "fast_counter_recuts": 0, "expected_speed": 0.}


def test_fast_run_keeps_direction_when_no_legal_candidate(fast_defaults):
    # denied ±45° draw: note1's only allowed recut (up) would sit 90° from
    # note2's right cut and note2's (down-right) 90° from note3's down-left
    # cut, so both keep their chaotic-but-legal directions; note3 recuts to
    # the reverse of right
    notes = [note(0, 1.0, x=0, y=1, direction=1),
             note(1, 1.15, x=1, y=1, direction=4),
             note(2, 1.30, x=2, y=1, direction=3),
             note(3, 1.45, x=3, y=1, direction=6)]
    stats = post.apply_fast_direction_rules(notes, rng=SimpleNamespace(uniform=lambda: 1.))
    assert [item["d"] for item in notes] == [1, 4, 3, 2]
    assert stats["fast_pairs"] == 3 and stats["fast_simplified"] == 1


def test_fast_run_dot_predecessor_feeds_implied_direction(fast_defaults):
    # a dot after a right cut on the opposite side inherits the left reversal;
    # with the ±45° draw denied the chaotic successor is recut against left's
    # allowed set
    notes = [note(0, 1.0, x=3, y=1, direction=3),
             note(1, 1.1, x=0, y=1, direction=8),
             note(2, 1.25, x=2, y=1, direction=5)]
    stats = post.apply_fast_direction_rules(notes, rng=SimpleNamespace(uniform=lambda: 1.))
    assert [item["d"] for item in notes] == [3, 8, 3]
    assert stats["fast_simplified"] == 1


def test_fast_run_deterministic_per_seed(fast_defaults):
    build = lambda: [note(0, 1.0, x=0, y=0, direction=2),
                     note(1, 1.04, x=3, y=2, direction=4),
                     note(2, 1.08, x=0, y=2, direction=7),
                     note(10, 2.0, hand=1, x=1, y=1, direction=1),
                     note(11, 4.0, hand=1, x=1, y=1, direction=0)]
    a, b = build(), build()
    post.apply_fast_direction_rules(a, rng=np.random.default_rng(42))
    post.apply_fast_direction_rules(b, rng=np.random.default_rng(42))
    assert [(n["x"], n["y"], n["d"]) for n in a] == [(n["x"], n["y"], n["d"]) for n in b]


def test_fast_counter_sweep_recuts_copies(fast_defaults):
    # top-row down-right copy after an up-left cut: counter family — a fast
    # pair expansion created after the pre-snapshot rules
    notes = [note(0, 1.0, x=0, y=2, direction=4),
             {**note(1, 1.15, x=0, y=2, direction=7, kind="copy"), "anchor_id": 0}]
    stats = post.apply_fast_counter_sweep(notes)
    assert notes[1]["d"] == 3
    assert stats == {"fast_pairs_checked": 1, "fast_counter_recuts": 1}


def test_fast_counter_sweep_never_touches_retained(fast_defaults):
    # the retained member of the run keeps its contract-fixed cut while a
    # heuristic counter note in the same run still recuts
    notes = [note(0, 1.0, x=0, y=2, direction=4),
             note(1, 1.15, x=0, y=2, direction=7),                  # retained: skipped
             note(2, 1.30, x=3, y=0, direction=4, kind="heuristic")]  # bottom-row up-left
    stats = post.apply_fast_counter_sweep(notes)
    assert notes[1]["d"] == 7
    assert notes[2]["d"] != 4
    assert stats == {"fast_pairs_checked": 2, "fast_counter_recuts": 1}


def test_fast_counter_sweep_ignores_slow_pairs(fast_defaults):
    notes = [note(0, 1.0, x=0, y=2, direction=4),
             {**note(1, 1.6, x=0, y=2, direction=7, kind="copy"), "anchor_id": 0}]
    stats = post.apply_fast_counter_sweep(notes)
    assert notes[1]["d"] == 7
    assert stats == {"fast_pairs_checked": 0, "fast_counter_recuts": 0}


def test_fast_counter_sweep_keeps_counter_when_no_legal_candidate(fast_defaults):
    # the only non-counter direction legal against the down-left predecessor
    # (right, d3) sits 0° from the successor's own cut -> nothing legal, keep
    notes = [note(0, 1.0, x=0, y=1, direction=6),
             {**note(1, 1.15, x=1, y=0, direction=5, kind="copy"), "anchor_id": 0},
             note(2, 1.30, x=2, y=1, direction=3)]
    stats = post.apply_fast_counter_sweep(notes)
    assert notes[1]["d"] == 5
    assert stats == {"fast_pairs_checked": 2, "fast_counter_recuts": 0}


def test_fast_counter_sweep_disabled_knobs_no_op(fast_defaults, monkeypatch):
    build = lambda: [note(0, 1.0, x=0, y=2, direction=4),
                     {**note(1, 1.15, x=0, y=2, direction=7, kind="copy"), "anchor_id": 0}]
    monkeypatch.setattr(config, "v2_fast_counter_direction", "off")
    notes = build()
    stats = post.apply_fast_counter_sweep(notes)
    assert notes[1]["d"] == 7
    assert stats == {"fast_pairs_checked": 0, "fast_counter_recuts": 0}
    monkeypatch.setattr(config, "v2_fast_counter_direction", "recut")
    monkeypatch.setattr(config, "v2_fast_gap_seconds", 0.)
    notes = build()
    stats = post.apply_fast_counter_sweep(notes)
    assert notes[1]["d"] == 7 and stats["fast_counter_recuts"] == 0



def test_fill_min_interval_scales_inversely_with_nps(monkeypatch):
    monkeypatch.setattr(config, "v2_fill_min_interval_seconds", .09)
    monkeypatch.setattr(config, "v2_fill_min_interval_nps", 5.)
    monkeypatch.setattr(config, "v2_fill_min_interval_min_seconds", .035)
    monkeypatch.setattr(config, "max_speed", 20.)
    assert post._fill_min_interval() == pytest.approx(.09)
    monkeypatch.setattr(config, "max_speed", 40.)   # nps 10 -> half the floor
    assert post._fill_min_interval() == pytest.approx(.045)
    monkeypatch.setattr(config, "max_speed", 80.)   # nps 20 -> clamped at the hard floor
    assert post._fill_min_interval() == pytest.approx(.035)
    monkeypatch.setattr(config, "max_speed", 10.)   # nps 2.5 -> never exceeds the base
    assert post._fill_min_interval() == pytest.approx(.09)
    monkeypatch.setattr(config, "v2_fill_min_interval_seconds", 0.)
    assert post._fill_min_interval() == 0.


def test_fill_min_interval_culls_dense_fill_proposals(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "map_filler_iters", 1)
    monkeypatch.setattr(config, "max_speed", 20.)
    monkeypatch.setattr(config, "add_beat_intensity", 50.)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 0.)
    monkeypatch.setattr(config, "v2_fill_entry_pause_beats", 0.)
    monkeypatch.setattr(post, "fill_map_times_scale",
                        lambda times, index=5: np.asarray([1., 1.04, 2., 2.03, 3.]))
    monkeypatch.setattr(config, "v2_fill_min_interval_seconds", .09)
    monkeypatch.setattr(config, "v2_fill_min_interval_nps", 5.)
    candidates, _ = post.postprocess_timing(song, beats([1., 3.], [1, 1]), "unused.egg",
                                            skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == pytest.approx([1., 2., 3.])
    monkeypatch.setattr(config, "v2_fill_min_interval_seconds", 0.)
    candidates, _ = post.postprocess_timing(song, beats([1., 3.], [1, 1]), "unused.egg",
                                            skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == pytest.approx([1., 1.04, 2., 2.03, 3.])


def test_start_end_fill_additions_respect_min_interval(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_fill_min_interval_seconds", .09)
    monkeypatch.setattr(config, "v2_fill_min_interval_nps", 5.)
    monkeypatch.setattr(config, "max_speed", 20.)
    monkeypatch.setattr(post, "_start_end_times",
                        lambda times, volume, onset: np.sort(np.r_[times, .5, .55, 9.5, 9.53]))
    candidates, report = post.postprocess_timing(song, beats([1., 5., 9.], [1, 1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == pytest.approx([.5, 1., 5., 9., 9.5])
    assert next(op for op in report["operations"]
                if op["stage"] == "start_end_fill")["event_count_delta"] == 2


def test_onset_coverage_inserts_missed_strong_onsets_after_alignment(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_onset_fill_flag", True)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 0.)
    monkeypatch.setattr(post, "_detect_onset_peaks",
                        lambda path: (np.asarray([2.5, 4., 6.]), np.asarray([.9, .3, .8]),
                                      np.ones(3)))
    # if coverage ran before the final alignment the insert would be moved too
    monkeypatch.setattr(post, "align_beats_on_bpm", lambda times, bpm: np.asarray(times) + .01)
    candidates, report = post.postprocess_timing(song, beats([1., 2.45, 5., 9.], [1, 1, 1, 1]),
                                                 "unused.egg")
    # events moved +.01 by the alignment, but the 2.5 onset is still covered;
    # 4.0 filtered by the strength quantile; 6.0 inserted un-moved (post-alignment)
    assert [item["time"] for item in candidates] == pytest.approx([1.02, 2.47, 5.02, 6., 9.02])
    assert report["onset_coverage_added"] == 1
    assert any(op["stage"] == "onset_coverage" for op in report["operations"])


def test_onset_coverage_respects_entry_gate_and_silence(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_onset_fill_flag", True)
    monkeypatch.setattr(post, "_detect_onset_peaks",
                        lambda path: (np.asarray([6.]), np.asarray([.9]), np.ones(1)))
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 10.)
    candidates, report = post.postprocess_timing(song, beats([1., 5., 9.], [1, 1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == pytest.approx([1., 5., 9.])
    assert report["onset_coverage_added"] == 0

    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 0.)
    monkeypatch.setattr(config, "add_silence_flag", True)
    monkeypatch.setattr(post, "_legacy_timing_features",
                        lambda _: (np.zeros((100, 24)), np.zeros((100, 24)), np.asarray([6.])))
    candidates, report = post.postprocess_timing(song, beats([1., 5., 9.], [1, 1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == pytest.approx([1., 5., 9.])
    assert report["onset_coverage_added"] == 0


def test_onset_coverage_flag_off_and_skippable(song, timing_defaults, monkeypatch):
    called = []
    monkeypatch.setattr(post, "_detect_onset_peaks",
                        lambda path: called.append(1) or (np.asarray([6.]), np.asarray([.9]),
                                                          np.ones(1)))
    candidates, _ = post.postprocess_timing(song, beats([1., 5., 9.], [1, 1, 1]),
                                            "unused.egg", skip=("bpm_alignment",))
    assert not called  # fixture pins v2_onset_fill_flag off
    monkeypatch.setattr(config, "v2_onset_fill_flag", True)
    candidates, _ = post.postprocess_timing(song, beats([1., 5., 9.], [1, 1, 1]),
                                            "unused.egg",
                                            skip=("bpm_alignment", "onset_coverage"))
    assert not called


def test_onset_coverage_loudness_tier(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_onset_fill_flag", True)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 0.)
    monkeypatch.setattr(config, "v2_onset_fill_loudness_quantile", .6)
    monkeypatch.setattr(config, "v2_onset_fill_loudness_strength_quantile", .25)
    # strengths sorted [.45,.45,.9,.9] -> p50 .675, p25 .45; loudness p60 -> .9
    monkeypatch.setattr(post, "_detect_onset_peaks",
                        lambda path: (np.asarray([2.5, 4., 6., 8.]),
                                      np.asarray([.9, .45, .45, .9]),
                                      np.asarray([.9, .9, .1, .9])))
    candidates, report = post.postprocess_timing(song, beats([1., 2.45, 5., 9.], [1, 1, 1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    # 2.5 covered by the 2.45 event; 4.0 admitted only through the loudness
    # tier; 6.0 has the same strength but sits in a quiet section; 8.0 tier-1
    assert [item["time"] for item in candidates] == pytest.approx([1., 2.45, 4., 5., 8., 9.])
    assert report["onset_coverage_added"] == 2
    assert report["onset_coverage_loud_admitted"] == 1


def test_onset_coverage_bypasses_note_runway_but_keeps_delay(song, timing_defaults, monkeypatch):
    # subdivision fills wait for v2_fill_entry_min_notes after a pause, but
    # onset-anchored coverage is a real audio hit — it only keeps the delay.
    monkeypatch.setattr(config, "v2_onset_fill_flag", True)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 1.)
    monkeypatch.setattr(config, "v2_fill_entry_pause_beats", 4.)
    monkeypatch.setattr(config, "v2_fill_entry_min_notes", 4)
    monkeypatch.setattr(post, "_detect_onset_peaks",
                        lambda path: (np.asarray([5.5, 6.5, 7.5]),
                                      np.asarray([.9, .9, .9]),
                                      np.ones(3)))
    # gap 1.0 -> 5.0 at 120 bpm is 8 beats: entry at 5.0, delay ends at 6.0,
    # while the fill runway would need a 4th candidate that never comes.
    candidates, report = post.postprocess_timing(song, beats([1., 5.], [1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    assert [item["time"] for item in candidates] == pytest.approx([1., 5., 6.5, 7.5])
    assert report["onset_coverage_added"] == 2


def test_onset_coverage_loudness_tier_disabled(song, timing_defaults, monkeypatch):
    monkeypatch.setattr(config, "v2_onset_fill_flag", True)
    monkeypatch.setattr(config, "v2_fill_entry_delay_seconds", 0.)
    monkeypatch.setattr(config, "v2_onset_fill_loudness_quantile", 0.)
    monkeypatch.setattr(post, "_detect_onset_peaks",
                        lambda path: (np.asarray([4., 8.]),
                                      np.asarray([.45, .9]),
                                      np.asarray([.9, .9])))
    candidates, report = post.postprocess_timing(song, beats([1., 5., 9.], [1, 1, 1]),
                                                 "unused.egg", skip=("bpm_alignment",))
    # with the tier off, the sub-median .45 onset at 4.0 is not admitted
    assert [item["time"] for item in candidates] == pytest.approx([1., 5., 8., 9.])
    assert report["onset_coverage_added"] == 1
    assert report["onset_coverage_loud_admitted"] == 0
