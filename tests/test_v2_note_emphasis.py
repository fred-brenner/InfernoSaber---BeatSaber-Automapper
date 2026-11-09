"""Standalone v1-style note emphasis: same-time axis stacks on slow swings."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from map_creation.v2_note_emphasis import apply_note_emphasis
from tools.config import config


class Draws:
    """Scripted scalar RNG; unanticipated draws fail the test."""

    def __init__(self, values=()):
        self.values = list(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        assert self.calls <= len(self.values), "unscripted RNG draw"
        return self.values[self.calls - 1]


def note(identity, time, hand, x, y, direction, kind="retained", **extra):
    return {"id": identity, "time": float(time), "hand": hand, "x": x, "y": y,
            "d": direction, "kind": kind, **extra}


def state(notes=(), chains=()):
    return {"notes": list(notes), "chains": list(chains),
            "walls": [], "bombs": [], "arcs": []}


def song():
    return SimpleNamespace(metadata={"duration": 600., "bpm": 120.},
                           arrays={"frame_times": np.arange(60000) / 100})


def stream(*groups):
    """Retained onsets; each group is (time, [(hand, x, y, d), ...])."""
    notes, identity = [], 0
    for time, members in groups:
        for hand, x, y, direction in members:
            notes.append(note(identity, time, hand, x, y, direction))
            identity += 1
    return notes


REVERSAL = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5, 8: 1}


def filler(direction=0):
    """Four comfortable onsets ending in a reversal of ``direction``."""
    return ((0., [(0, 0, 1, direction)]), (.5, [(1, 3, 1, direction)]),
            (1., [(0, 0, 1, REVERSAL[direction])]),
            (1.5, [(1, 3, 1, REVERSAL[direction])]))


def slow_state(x=1, y=0, direction=0, hand=0, time=5., extra_groups=()):
    """Four warm-up onsets plus a single slow same-hand onset."""
    return state(stream(*filler(direction), *extra_groups,
                        (time, [(hand, x, y, direction)])))


@pytest.fixture(autouse=True)
def emphasis_defaults(monkeypatch):
    monkeypatch.setattr(config, "emphasize_beats_flag", True)
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    monkeypatch.setattr(config, "v2_cleanup_note_heuristics", False)
    monkeypatch.setattr(config, "min_time_diff", .02)
    monkeypatch.setattr(config, "emphasize_beats_3", .1)
    monkeypatch.setattr(config, "emphasize_beats_3_fact", 0.)
    monkeypatch.setattr(config, "emphasize_beats_2", .4)
    monkeypatch.setattr(config, "emphasize_beats_2_fact", 0.)
    monkeypatch.setattr(config, "emphasize_beats_quantile", .65)
    monkeypatch.setattr(config, "v2_playability_enabled", False)
    monkeypatch.setattr(config, "add_beat_intensity_orig", 100.)
    monkeypatch.setattr(config, "max_speed", 32.)
    monkeypatch.setattr(config, "reaction_time", 1.15)
    monkeypatch.setattr(config, "reaction_time_fact", .013)
    monkeypatch.setattr(config, "cdf", 1.2)
    monkeypatch.setattr(config, "cdf_lr", 1.15)
    monkeypatch.setattr(config, "v2_speed_cap_scale", 2.)
    monkeypatch.setattr(config, "v2_speed_cap_start_notes", 30)
    monkeypatch.setattr(config, "v2_speed_cap_start_factor", .35)
    monkeypatch.setattr(config, "v2_fast_gap_seconds", .2)
    monkeypatch.setattr(config, "v2_fast_counter_direction", "recut")
    monkeypatch.setattr(config, "v2_cleanup_same_dir_gap", .1)
    monkeypatch.setattr(config, "v2_cleanup_swing_window", .15)
    monkeypatch.setattr(config, "v2_diagonal_max_fraction", .1)
    monkeypatch.setattr(config, "v2_flow_min_turn_steps", 3)


def added(state):
    return [n for n in state["notes"] if n.get("source") == "emphasis"]


def test_bottom_edge_triple_climbs_the_column():
    s = slow_state(x=1, y=0, direction=0)
    out = apply_note_emphasis(s, song(), rng=Draws([.95]))
    new = added(s)
    assert [n["id"] for n in new] == [5, 6]
    assert [(n["x"], n["y"]) for n in new] == [(1, 1), (1, 2)]
    assert all(n["time"] == 5. and n["hand"] == 0 and n["d"] == 0 for n in new)
    assert all(n["kind"] == "copy" and n["source"] == "emphasis"
               and n["anchor_id"] == 4 for n in new)
    assert out["applied"] and out["added_notes"] == 2 and out["triples"] == 1


def test_horizontal_triple_keeps_center_middle():
    s = slow_state(x=0, y=1, direction=3)
    apply_note_emphasis(s, song(), rng=Draws([.95]))
    assert [(n["x"], n["y"]) for n in added(s)] == [(1, 1), (2, 1)]


def test_double_skips_added_center_middle_but_keeps_original():
    s = slow_state(x=2, y=1, direction=3)
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    new = added(s)
    assert [(n["x"], n["y"]) for n in new] == [(3, 1)]
    assert (s["notes"][4]["x"], s["notes"][4]["y"]) == (2, 1)


def test_edge_triple_extends_along_the_available_side():
    s = slow_state(x=3, y=1, direction=2)
    apply_note_emphasis(s, song(), rng=Draws([.95]))
    assert [(n["x"], n["y"]) for n in added(s)] == [(2, 1), (1, 1)]


def test_double_falls_back_when_a_triple_member_is_counter():
    s = slow_state(x=1, y=1, direction=0)
    out = apply_note_emphasis(s, song(), rng=Draws([.95]))
    assert [(n["x"], n["y"]) for n in added(s)] == [(1, 2)]
    assert out["triples"] == 0 and out["doubles"] == 1


def test_diagonal_double_keeps_the_same_arrow():
    s = slow_state(x=2, y=1, direction=4)
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    new = added(s)
    assert [(n["x"], n["y"]) for n in new] == [(1, 2)]
    assert new[0]["d"] == 4


def test_diagonal_fraction_bound_is_enforced(monkeypatch):
    monkeypatch.setattr(config, "v2_diagonal_max_fraction", .1)
    s = state(stream((0., [(0, 0, 1, 1)]), (.5, [(1, 3, 1, 1)]),
                     (1., [(0, 0, 1, 0)]), (1.5, [(1, 3, 1, 0)]),
                     (5., [(0, 2, 1, 4)])))
    apply_note_emphasis(s, song(), rng=Draws([.95]))
    assert not added(s)


@pytest.mark.parametrize("x,y,direction", [
    (0, 0, 0), (3, 0, 0), (0, 2, 1), (3, 2, 1), (0, 0, 5), (3, 2, 4),
    (0, 1, 2), (3, 1, 3), (0, 2, 7), (3, 0, 6), (0, 0, 2), (3, 2, 3)])
def test_additions_stay_on_grid_and_under_three_per_swing(x, y, direction):
    s = slow_state(x=x, y=y, direction=direction)
    apply_note_emphasis(s, song(), rng=Draws([.95]))
    for n in added(s):
        assert 0 <= n["x"] < 4 and 0 <= n["y"] < 3
    assert len(added(s)) <= 2


def test_first_four_onsets_are_never_expanded():
    s = state(stream((0., [(0, 1, 2, 0)]), (3., [(0, 1, 2, 0)]),
                     (3.3, [(0, 1, 2, 0)]), (3.6, [(0, 1, 2, 0)])))
    rng = Draws()
    apply_note_emphasis(s, song(), rng=rng)
    assert rng.calls == 0 and len(s["notes"]) == 4


def test_the_slow_gap_gate_uses_the_quantile_rule():
    s = state(stream(*filler(0), (1.6, [(0, 1, 2, 0)])))
    rng = Draws()
    apply_note_emphasis(s, song(), rng=rng)
    assert rng.calls == 0


def test_draw_bands_select_triple_double_or_nothing():
    for draw, expected in ((.95, 2), (.55, 1), (.3, 0)):
        s = slow_state(x=2, y=1, direction=3)
        apply_note_emphasis(s, song(), rng=Draws([draw]))
        assert len(added(s)) == expected


def test_speed_factored_draw_expands_only_at_high_speed(monkeypatch):
    monkeypatch.setattr(config, "emphasize_beats_2", .5)
    monkeypatch.setattr(config, "emphasize_beats_2_fact", .01)
    monkeypatch.setattr(config, "max_speed", 10.)
    s = slow_state(x=2, y=1, direction=3)
    apply_note_emphasis(s, song(), rng=Draws([.25]))
    assert not added(s)
    monkeypatch.setattr(config, "max_speed", 32.)
    s = slow_state(x=2, y=1, direction=3)
    apply_note_emphasis(s, song(), rng=Draws([.25]))
    assert len(added(s)) == 1


def test_emphasis_flag_off_consumes_no_draws(monkeypatch):
    monkeypatch.setattr(config, "emphasize_beats_flag", False)
    s = slow_state()
    rng = Draws()
    out = apply_note_emphasis(s, song(), rng=rng)
    assert rng.calls == 0 and not added(s) and not out["applied"]


def test_single_mode_consumes_no_draws(monkeypatch):
    monkeypatch.setattr(config, "single_notes_only_flag", True)
    s = slow_state()
    rng = Draws()
    out = apply_note_emphasis(s, song(), rng=rng)
    assert rng.calls == 0 and not added(s) and not out["applied"]


def test_legacy_cleanup_opt_in_is_a_noop(monkeypatch):
    monkeypatch.setattr(config, "v2_cleanup_note_heuristics", True)
    s = slow_state()
    rng = Draws()
    apply_note_emphasis(s, song(), rng=rng)
    assert rng.calls == 0 and not added(s)


def test_empty_and_short_maps_are_noops():
    for s in (state(), state(stream((0., [(0, 1, 2, 0)])))):
        rng = Draws()
        out = apply_note_emphasis(s, song(), rng=rng)
        assert rng.calls == 0 and not out["applied"]


def test_dot_anchors_are_skipped():
    s = slow_state(x=0, y=1, direction=8)
    rng = Draws()
    apply_note_emphasis(s, song(), rng=rng)
    assert rng.calls == 0 and not added(s)


def test_anchors_with_copies_or_chains_are_skipped():
    s = slow_state(x=0, y=1, direction=3)
    s["notes"].append(note(10, 5.05, 0, 1, 1, 3, kind="copy", anchor_id=4))
    rng = Draws()
    apply_note_emphasis(s, song(), rng=rng)
    assert rng.calls == 0 and len(added(s)) == 0
    s = slow_state(x=0, y=1, direction=3)
    s["chains"].append({**note(4, 5., 0, 0, 1, 3), "id": 0, "anchor_id": 4,
                        "kind": "chain", "end_time": 6., "tx": 2, "ty": 1,
                        "slices": 2, "squish": 1.})
    rng = Draws()
    apply_note_emphasis(s, song(), rng=rng)
    assert rng.calls == 0 and not added(s)


def test_heuristic_note_at_the_swing_counts_as_expanded():
    s = slow_state(x=0, y=1, direction=3)
    s["notes"].append(note(10, 5.01, 0, 2, 1, 3, kind="heuristic"))
    rng = Draws()
    apply_note_emphasis(s, song(), rng=rng)
    assert rng.calls == 0 and not added(s)


def test_second_same_hand_note_in_the_onset_disqualifies():
    s = state(stream(*filler(3), (5., [(0, 0, 1, 3), (0, 3, 1, 3)])))
    rng = Draws()
    apply_note_emphasis(s, song(), rng=rng)
    assert rng.calls == 0 and not added(s)


def test_no_new_onset_times_or_duplicate_cells():
    s = state(stream(*filler(0), (5., [(0, 0, 1, 0), (1, 3, 1, 0)])))
    apply_note_emphasis(s, song(), rng=Draws([.55, .55]))
    assert {n["time"] for n in s["notes"]} == {0., .5, 1., 1.5, 5.}
    cells = [(n["time"], n["x"], n["y"]) for n in s["notes"]]
    assert len(cells) == len(set(cells))


def test_inputs_and_chains_stay_dictionary_equal():
    s = slow_state(x=2, y=1, direction=3)
    s["chains"].append({**note(2, 1., 0, 0, 1, 3), "id": 9, "anchor_id": 2,
                        "kind": "chain", "end_time": 2., "tx": 2, "ty": 1,
                        "slices": 2, "squish": 1.})
    before = deepcopy(s)
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert s["notes"][:5] == before["notes"]
    assert s["chains"] == before["chains"]


def test_fresh_ids_skip_existing_note_and_chain_ids():
    s = slow_state(x=0, y=1, direction=3)
    s["chains"].append({**note(2, 1., 0, 0, 1, 3), "id": 40, "anchor_id": 2,
                        "kind": "chain", "end_time": 2., "tx": 2, "ty": 1,
                        "slices": 2, "squish": 1.})
    apply_note_emphasis(s, song(), rng=Draws([.95]))
    assert [n["id"] for n in added(s)] == [41, 42]


def test_same_cell_conflict_is_rejected():
    s = state(stream((0., [(0, 0, 1, 3)]), (.5, [(1, 3, 1, 1)]),
                     (1., [(0, 0, 1, 2)]), (1.5, [(1, 3, 1, 0)]),
                     (5., [(0, 2, 1, 3), (1, 3, 1, 1)])))
    apply_note_emphasis(s, song(), rng=Draws([.55, .3]))
    assert not added(s)


def test_swept_path_conflict_is_rejected():
    s = state(stream(*filler(0), (5., [(0, 1, 0, 0), (1, 1, 2, 8)])))
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert not added(s)


def test_counter_direction_copies_are_rejected(monkeypatch):
    groups = (*filler(0), (5., [(0, 1, 1, 0), (1, 1, 2, 0)]))
    s = state(stream(*groups))
    apply_note_emphasis(s, song(), rng=Draws([.55, .3]))
    assert not added(s)
    monkeypatch.setattr(config, "v2_fast_counter_direction", "off")
    s = state(stream(*groups))
    apply_note_emphasis(s, song(), rng=Draws([.55, .3]))
    assert [(n["x"], n["y"]) for n in added(s)] == [(1, 0)]


def test_speed_cap_rejects_an_unplayable_arrival(monkeypatch):
    notes = stream(*filler(0), (6., [(0, 0, 1, 0)]))
    notes.append(note(9, 4.6, 0, 3, 2, 1, kind="heuristic"))
    monkeypatch.setattr(config, "v2_speed_cap_scale", .001)
    s = state(deepcopy(notes))
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert not added(s)
    monkeypatch.setattr(config, "v2_speed_cap_scale", 100.)
    s = state(deepcopy(notes))
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert len(added(s)) == 1


def test_zero_speed_scale_disables_the_travel_gate(monkeypatch):
    monkeypatch.setattr(config, "v2_speed_cap_scale", 0.)
    s = slow_state(x=2, y=1, direction=3)
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert len(added(s)) == 1


@pytest.fixture
def priority(monkeypatch):
    monkeypatch.setattr(config, "v2_playability_enabled", True)
    monkeypatch.setattr(config, "v2_playability_priority", True)
    monkeypatch.setattr(config, "v2_playability_reach", True)


def test_hard_crossing_without_same_cell_conflict_is_rejected(priority):
    s = state(stream((0., [(0, 0, 1, 0)]), (.5, [(1, 3, 1, 2)]),
                     (1., [(0, 0, 1, 1)]), (1.5, [(1, 3, 1, 3)]),
                     (5., [(0, 0, 0, 0), (1, 1, 2, 2)])))
    apply_note_emphasis(s, song(), rng=Draws([.55, .3]))
    assert not added(s)


def test_forbidden_inward_cuts_fall_back_to_a_safe_cell(priority):
    s = slow_state(x=1, y=0, direction=3, hand=1)
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert [(n["x"], n["y"]) for n in added(s)] == [(2, 0)]


def test_visibility_regression_is_rejected(priority):
    notes = stream(*filler(0), (3., [(0, 1, 1, 0)]), (3.3, [(0, 2, 2, 1)]),
                   (3.6, [(0, 2, 2, 0)]), (3.9, [(0, 2, 2, 1)]))
    s = state(notes)
    out = apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert not added(s)
    assert out["rejected"].get("visibility", 0) > 0


def test_preexisting_hard_violation_declines_safe_additions(priority):
    groups = (*filler(0), (2., [(0, 0, 1, 4), (1, 1, 1, 5)]),
              (5., [(0, 2, 1, 3)]))
    s = state(stream(*groups))
    out = apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert not added(s)
    assert out["rejected"].get("priority_hard", 0) > 0


def test_two_sequential_same_hand_expansions_preserve_inputs():
    notes = stream(*filler(0), (5., [(0, 0, 1, 0)]), (9., [(0, 0, 1, 1)]))
    s = state(notes)
    before = deepcopy(notes)
    out = apply_note_emphasis(s, song(), rng=Draws([.55, .55]))
    new = added(s)
    assert len(new) == 2 and out["added_notes"] == 2
    assert s["notes"][:6] == before
    assert {n["time"] for n in s["notes"]} == {0., .5, 1., 1.5, 5., 9.}
    assert len({n["id"] for n in new}) == 2


def test_snapped_near_onsets_share_one_group():
    notes = stream(*filler(0), (5., [(0, 0, 1, 0)]), (5.01, [(1, 3, 1, 0)]))
    s = state(notes)
    out = apply_note_emphasis(s, song(), rng=Draws([.55, .55]))
    new = added(s)
    assert out["eligible_anchors"] == 2 and len(new) == 2
    assert {n["time"] for n in s["notes"]} == {0., .5, 1., 1.5, 5., 5.01}


def test_over_threshold_baseline_events_survive_safe_additions(priority):
    groups = [(i * .2, [(0, 0, 1, 0)]) for i in range(19)]
    groups.append((4.1, [(0, 0, 0, 0)]))
    groups.append((5.0, [(0, 1, 1, 1)]))
    s = state(stream(*groups))
    apply_note_emphasis(s, song(), rng=Draws([.3, .55]))
    assert [(n["x"], n["y"]) for n in added(s)] == [(1, 0)]


def test_new_copies_cannot_inherit_an_over_bound_allowance(priority):
    groups = [(i * .2, [(0, 0, 1, 0)]) for i in range(19)]
    groups.append((4.1, [(0, 0, 0, 1)]))
    s = state(stream(*groups))
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert not added(s)


def test_reach_only_mode_preserves_forbidden_and_tails(monkeypatch):
    monkeypatch.setattr(config, "v2_playability_enabled", True)
    monkeypatch.setattr(config, "v2_playability_priority", False)
    monkeypatch.setattr(config, "v2_playability_reach", True)
    s = slow_state(x=1, y=0, direction=3, hand=1)
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert [(n["x"], n["y"]) for n in added(s)] == [(2, 0)]
    s = slow_state(x=0, y=1, direction=0)
    apply_note_emphasis(s, song(), rng=Draws([.55]))
    assert len(added(s)) == 1
