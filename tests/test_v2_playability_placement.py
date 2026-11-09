"""Human-calibrated placement tails preserve ordinary phrase variation."""

from copy import deepcopy
import importlib
import importlib.util
import json
import math

import pytest


def placement():
    name = "map_creation.v2_playability_placement"
    assert importlib.util.find_spec(name) is not None, "placement exposure costs are missing"
    return importlib.import_module(name)


def note(identity, time, hand=0, x=1, y=1, direction=0):
    return dict(id=identity, time=time, hand=hand, x=x, y=y, d=direction)


def test_isolated_top_and_outer_notes_are_allowed_in_every_nps_stratum():
    policy = placement()
    for nps in (0., 2., 3., 5., 7., 12.):
        notes = [note(0, 1., x=0, y=2), note(1, 1., hand=1, x=3, y=2)]
        result = policy.placement_costs(notes, nps)
        assert result["total_excess"] == 0.
        assert all(m["eligible_events"] == 0 for m in result["metrics"].values())
        assert policy.local_placement_cost(notes, nps) == 0.


def test_repeated_high_and_far_opposite_placement_has_positive_tail_cost():
    policy = placement()
    high = [note(i, 1.+i*.25, x=3, y=2) for i in range(16)]
    normal = [note(i, 1.+i*.25, x=1+i%2, y=i%2) for i in range(16)]
    result = policy.placement_costs(high, 4.)
    for metric in ("top_row_exposure", "outer_lane_exposure", "opposite_side_exposure"):
        assert result["metrics"][metric]["excess"] > 0.
    assert result["total_excess"] > policy.local_placement_cost(normal, 4.)
    assert policy.local_placement_cost(high, 4.) == result["total_excess"]


def test_normal_phrase_with_top_and_outer_variation_receives_no_cost():
    policy = placement()
    notes = [note(i, i*.4, x=i%2, y=2 if i == 2 else i%2) for i in range(12)]
    result = policy.placement_costs(notes, 4.)
    assert result["metrics"]["top_row_exposure"]["eligible_events"] == 9
    assert result["total_excess"] == 0.


def test_phrase_denominators_include_zero_events_and_not_stack_or_dot_multiplicity():
    policy = placement()
    notes = [note(i, 1.+i*.25, x=1, y=0, direction=8) for i in range(6)]
    notes[0]["y"] = notes[1]["y"] = 2
    notes += [note(20, 1., x=3, y=2)]
    metrics = policy.placement_costs(notes, 4.)["metrics"]
    assert metrics["top_row_exposure"]["eligible_events"] == 3
    assert metrics["top_row_exposure"]["severities"] == pytest.approx([.5, .4, 1/3])
    assert metrics["outer_lane_exposure"]["severities"] == pytest.approx([.125, .1, 1/12])
    assert metrics["opposite_side_exposure"]["severities"] == pytest.approx([.125, .1, 1/12])
    assert metrics["top_row_exposure"]["event_note_ids"][0] == [0, 20, 1, 2, 3]


def test_recent_event_window_is_bounded_and_long_pause_resets_eligibility():
    policy = placement()
    notes = [note(i, i*.25, y=2 if i < 2 else 0) for i in range(10)]
    notes += [note(10, 9., y=2)]
    result = policy.placement_costs(notes, 4.)
    metric = result["metrics"]["top_row_exposure"]
    assert metric["eligible_events"] == 7
    assert metric["severities"][-1] == 0.
    assert len(metric["event_note_ids"][-1]) == 8
    assert 10 not in {identity for event in metric["event_note_ids"] for identity in event}


def test_twenty_millisecond_groups_are_anchor_based_and_do_not_chain():
    policy = placement()
    notes = [note(i, time, y=2) for i, time in enumerate((0., .015, .03, .28, .53))]
    result = policy.placement_costs(notes, 4.)
    assert result["metrics"]["top_row_exposure"]["eligible_events"] == 1
    assert result["metrics"]["top_row_exposure"]["severities"] == [1.]


def test_mirrored_hands_and_direction_changes_have_identical_placement_costs():
    policy = placement()
    notes = [note(i, 1.+i*.3, x=i%4, y=i%3, direction=i%9) for i in range(15)]
    mirrored = [{**n, "hand": 1, "x": 3-n["x"], "d": 8} for n in notes]
    left = policy.placement_costs(notes, 4.)
    right = policy.placement_costs(mirrored, 4.)
    assert left == right


def test_input_order_and_annotations_are_preserved_and_result_is_json_safe():
    policy = placement()
    notes = [note(i, 2.+i*.25, x=i%4, y=i%3) for i in range(12)]
    notes[0]["custom"] = {"keep": True}
    original = deepcopy(notes)
    expected = policy.placement_costs(notes)
    assert policy.placement_costs(list(reversed(notes))) == expected
    assert notes == original
    assert math.isfinite(expected["total_excess"])
    assert math.isfinite(policy.local_placement_cost([], 0.))
    json.dumps(expected, allow_nan=False)


def test_explicit_nps_strata_keep_local_ranking_in_the_callers_frozen_band():
    policy = placement()
    notes = [note(i, i*.05, y=2) for i in range(8)]
    for nps, band in ((2.9, "<3"), (3., "3-5"), (5., "5-7"), (7., "7+")):
        assert policy.placement_costs(notes, nps)["nps_band"] == band
    assert policy.placement_costs(notes)["nps_band"] == "7+"

