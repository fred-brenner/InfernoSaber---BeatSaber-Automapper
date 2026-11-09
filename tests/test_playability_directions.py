"""M5 direction-variety scores, turn intervals and movement-axis diagnostics."""
from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.playability_directions import m5_direction_variety, movement_axis_diagnostics
from evaluation.playability_metrics import prepare_notes, score_map


def test_movement_axis_folds_reversals_and_tracks_dwell_across_dot_notes():
    # Horizontal out and back, diagonal, vertical. All four moves use dot
    # notes, so the trajectory is measurable even without any cut arrows.
    points = [(0, 0), (2, 0), (0, 0), (1, 1), (1, 2)]
    notes = [(i * 0.3, 0, x, y, 8) for i, (x, y) in enumerate(points)]
    axis = score_map(notes)["m5"]["movement_axis"]
    assert axis["angle_counts"] == {"0": 1, "45": 2, "90": 0}
    assert axis["completed_runs"] == 2
    assert axis["median_run_moves"] == 1.5
    assert axis["median_run_seconds"] == pytest.approx(0.45)


def test_movement_axis_breaks_at_stationary_cells_and_keeps_hands_separate():
    points = [(0, 0), (3, 1), (3, 2), (3, 2), (2, 1)]
    left = [(i * 0.4, 0, x, y, 0) for i, (x, y) in enumerate(points)]
    right = [(0.1, 1, 0, 0, 3), (0.9, 1, 0, 1, 3)]
    axis = movement_axis_diagnostics(prepare_notes(left + right))
    # (3,1) is nearest the horizontal axis. The repeated cell breaks the
    # trajectory before the last diagonal, and the right hand makes one move.
    assert axis["angle_counts"] == {"0": 0, "45": 0, "90": 1}
    assert axis["completed_runs"] == 1
    assert axis["median_run_moves"] == 1
    assert axis["median_run_seconds"] == pytest.approx(0.4)


def test_no_complete_movement_axis_run_has_no_duration():
    axis = movement_axis_diagnostics(prepare_notes([(0, 0, 1, 1, 0)]))
    assert axis["angle_counts"] == {"0": 0, "45": 0, "90": 0}
    assert axis["completed_runs"] == 0
    assert axis["median_run_moves"] is None
    assert axis["median_run_seconds"] is None


@pytest.mark.parametrize("before,after,category,is_reversal", [
    (0, 0, "0", False),
    (0, 4, "45", False),
    (0, 2, "90", False),
    (0, 6, ">=135", True),
    (0, 1, ">=135", True),
    (5, 6, ">=135", True),
    (7, 0, ">=135", True),
    (5, 4, "90", False),
])
def test_m5_raw_arrow_angle_categories(before, after, category, is_reversal):
    # 135 and 180 degrees share the reversal category; a 90-degree turn
    # remains a non-reversal even though its undirected cut axis changes.
    result = score_map([(0.0, 0, 1, 1, before),
                        (0.125, 0, 1, 1, after)])["m5"]
    assert result["transitions"] == 1
    assert result["non_reversal_share"] == (0.0 if is_reversal else 1.0)
    assert result["fast_reversals"] == int(is_reversal)
    assert result["fast_reversals_per_minute"] == (480.0 if is_reversal else 0.0)
    angles = result["diagnostics"]["overall"]["angles"]
    assert set(angles) == {"0", "45", "90", ">=135"}
    for label, observations in angles.items():
        expected_count = int(label == category)
        assert observations["count"] == expected_count
        assert observations["share"] == float(expected_count)
        assert observations["gap_seconds"] == {
            "count": expected_count, "median": 0.125 if expected_count else None}
        assert observations["interval_seconds"] == {"count": 0, "median": None}
        assert observations["interval_transitions"] == {"count": 0, "median": None}


@pytest.mark.parametrize("gap,fast", [(0.249, True), (0.25, False), (0.251, False)])
def test_m5_fast_reversal_threshold_is_strict(gap, fast):
    result = score_map([(0.0, 0, 1, 1, 0), (gap, 0, 1, 1, 1)])["m5"]
    assert result["transitions"] == 1
    assert result["fast_reversals"] == int(fast)
    assert result["fast_reversals_per_minute"] == pytest.approx(
        60.0 / gap if fast else 0.0)


@pytest.mark.parametrize("arriving_direction,category", [(0, "0"), (1, ">=135")])
def test_m5_dots_exclude_pairs_and_break_recurrence(arriving_direction, category):
    notes = [(0.0, 0, 1, 1, 0), (0.125, 0, 1, 1, arriving_direction),
             (0.25, 0, 1, 1, 8), (0.375, 0, 1, 1, 0),
             (0.5, 0, 1, 1, arriving_direction)]
    # Dot geometry resolves in prepare_notes, but M5 must use raw arrows.
    assert prepare_notes(notes)[2]["eff"] is not None
    result = score_map(notes)["m5"]
    assert result["transitions"] == 2
    overall = result["diagnostics"]["overall"]
    assert overall["angles"][category]["count"] == 2
    assert overall["angles"][category]["interval_seconds"] == {"count": 0, "median": None}
    assert overall["non_reversal_intervals"]["seconds"] == {"count": 0, "median": None}
    assert overall["non_reversal_intervals"]["transitions"] == {"count": 0, "median": None}
    assert overall["completed_cut_axis_runs"]["notes"] == {"count": 0, "median": None}


def test_m5_interleaved_hands_keep_slow_gaps_and_local_intervals():
    notes = [(0.0, 0, 1, 1, 0), (0.1, 0, 1, 1, 0),
             (0.3, 0, 1, 1, 1), (0.9, 0, 1, 1, 0), (1.0, 0, 1, 1, 0),
             (0.05, 1, 2, 1, 2), (0.15, 1, 2, 1, 3), (0.25, 1, 2, 1, 3)]
    result = score_map(notes)["m5"]
    assert result["transitions"] == 6
    assert result["non_reversal_share"] == 0.5
    assert result["fast_reversals"] == 2
    assert result["fast_reversals_per_minute"] == 120.0
    diagnostics = result["diagnostics"]
    overall = diagnostics["overall"]
    assert overall["gap_seconds"]["count"] == 6
    assert overall["gap_seconds"]["median"] == pytest.approx(0.1)
    same_direction = overall["angles"]["0"]
    assert same_direction["count"] == 3
    assert same_direction["interval_seconds"]["count"] == 1
    assert same_direction["interval_seconds"]["median"] == pytest.approx(0.9)
    assert same_direction["interval_transitions"] == {"count": 1, "median": 3.0}
    # The 0.6s reversal gap survives; there is no recurrence between hands.
    assert overall["angles"][">=135"]["interval_seconds"]["count"] == 1
    assert overall["angles"][">=135"]["interval_seconds"]["median"] == pytest.approx(0.6)
    assert overall["non_reversal_intervals"]["seconds"]["median"] == pytest.approx(0.9)
    assert overall["non_reversal_intervals"]["transitions"] == {"count": 1, "median": 3.0}
    assert diagnostics["per_hand"]["0"]["angles"]["0"]["share"] == 0.5
    assert diagnostics["per_hand"]["1"]["angles"]["0"]["count"] == 1
    assert diagnostics["per_hand"]["1"]["angles"]["0"]["interval_seconds"] == {
        "count": 0, "median": None}


@pytest.mark.parametrize("notes", [
    [], [(0.0, 0, 1, 1, 0)],
    [(0.0, 0, 1, 1, 0), (0.5, 1, 2, 1, 1)],
    [(0.0, 0, 1, 1, 8), (0.5, 0, 1, 1, 0)],
])
def test_m5_no_eligible_transitions_are_undefined(notes):
    import json

    result = score_map(notes)["m5"]
    assert result["transitions"] == 0
    assert result["non_reversal_share"] is None
    assert result["fast_reversals"] == 0
    assert result["fast_reversals_per_minute"] is None
    overall = result["diagnostics"]["overall"]
    assert overall["gap_seconds"] == {"count": 0, "median": None}
    assert all(angle["share"] is None for angle in overall["angles"].values())
    json.dumps(result, allow_nan=False)


def test_m5_zero_exposure_has_no_reversal_rate():
    prepared = prepare_notes([(0.0, 0, 1, 1, 0), (0.125, 0, 1, 1, 1)])
    result = m5_direction_variety(prepared, minutes=0.0)
    assert result["transitions"] == 1
    assert result["non_reversal_share"] == 0.0
    assert result["fast_reversals"] == 1
    assert result["fast_reversals_per_minute"] is None


def test_m5_uses_the_collapsed_stack_representative():
    notes = [(0.0, 0, 1, 1, 0), (0.01, 0, 2, 1, 4), (0.125, 0, 1, 1, 1)]
    scored = score_map(notes)
    assert scored["notes"] == 2
    assert scored["m5"]["transitions"] == 1
    assert scored["m5"]["non_reversal_share"] == 0.0
    assert scored["m5"]["diagnostics"]["overall"]["angles"][">=135"]["count"] == 1


def test_m5_cut_axis_runs_end_only_at_an_observed_axis_change():
    notes = [(0.0, 0, 1, 1, 0), (0.1, 0, 1, 1, 1),
             (0.2, 0, 1, 1, 0), (0.3, 0, 1, 1, 1), (0.4, 0, 1, 1, 0),
             (0.8, 0, 1, 1, 2), (1.0, 0, 1, 1, 3),
             (1.5, 0, 1, 1, 0), (1.6, 0, 1, 1, 1)]
    runs = score_map(notes)["m5"]["diagnostics"]["overall"]["completed_cut_axis_runs"]
    # Five vertical cuts wait 0.8s for an axis change, then two horizontal
    # cuts wait 0.7s. The final two vertical cuts have no observed change.
    assert runs["notes"] == {"count": 2, "median": 3.5}
    assert runs["seconds"]["count"] == 2
    assert runs["seconds"]["median"] == pytest.approx(0.75)


def test_m5_dot_does_not_complete_or_bridge_cut_axis_runs():
    notes = [(0.0, 0, 1, 1, 0), (0.1, 0, 1, 1, 1), (0.2, 0, 1, 1, 8),
             (0.3, 0, 1, 1, 2), (0.4, 0, 1, 1, 3), (0.5, 0, 1, 1, 0)]
    runs = score_map(notes)["m5"]["diagnostics"]["overall"]["completed_cut_axis_runs"]
    assert runs["notes"] == {"count": 1, "median": 2.0}
    assert runs["seconds"]["count"] == 1
    assert runs["seconds"]["median"] == pytest.approx(0.2)


def test_m5_mixed_angle_changes_have_dwell_without_same_category_recurrence():
    # Arrival categories: 45 at .1, 0 at .2, 90 at .5, 180 at .8,
    # then 0 at .9. The final reversal has no observed next change.
    notes = [(0.0, 0, 1, 1, 0), (0.1, 0, 1, 1, 4), (0.2, 0, 1, 1, 4),
             (0.5, 0, 1, 1, 6), (0.8, 0, 1, 1, 5), (0.9, 0, 1, 1, 5)]
    overall = score_map(notes)["m5"]["diagnostics"]["overall"]
    dwell = overall["direction_change_intervals"]
    assert dwell["seconds"]["count"] == 2
    assert dwell["seconds"]["median"] == pytest.approx(0.35)
    assert dwell["transitions"] == {"count": 2, "median": 1.5}
    angle_45 = overall["angles"]["45"]
    assert angle_45["interval_seconds"] == {"count": 0, "median": None}
    assert angle_45["next_change_seconds"]["count"] == 1
    assert angle_45["next_change_seconds"]["median"] == pytest.approx(0.4)
    assert angle_45["next_change_transitions"] == {"count": 1, "median": 2.0}
    assert overall["angles"]["90"]["next_change_seconds"]["median"] == pytest.approx(0.3)
    assert overall["angles"][">=135"]["next_change_seconds"] == {"count": 0, "median": None}
    assert overall["angles"]["0"]["next_change_seconds"] == {"count": 0, "median": None}


def test_m5_next_change_dwell_never_crosses_a_dot_or_hands():
    notes = [(0.0, 0, 1, 1, 0), (0.1, 0, 1, 1, 4), (0.2, 0, 1, 1, 8),
             (0.3, 0, 1, 1, 6), (0.4, 0, 1, 1, 5),
             (0.05, 1, 2, 1, 0), (0.15, 1, 2, 1, 2)]
    overall = score_map(notes)["m5"]["diagnostics"]["overall"]
    assert overall["direction_change_intervals"]["seconds"] == {"count": 0, "median": None}
    assert overall["direction_change_intervals"]["transitions"] == {"count": 0, "median": None}


def test_m5_quarter_second_boundary_is_independent_of_the_timestamp_origin():
    # Subtracting these timestamps rounds an exact quarter slightly downward.
    notes = [(0.45, 0, 0, 0, 0), (0.70, 0, 0, 0, 1)]
    assert score_map(notes)["m5"]["fast_reversals"] == 0


def test_m5_turn_frequencies_use_map_minutes_for_both_hands():
    notes = [(0.0, 0, 0, 0, 0), (0.5, 0, 0, 1, 4),
             (0.0, 1, 3, 0, 3), (0.5, 1, 3, 1, 2)]
    diagnostics = score_map(notes)["m5"]["diagnostics"]
    assert diagnostics["overall"]["angles"]["45"]["per_minute"] == 120.0
    assert diagnostics["per_hand"]["0"]["angles"]["45"]["per_minute"] == 120.0
    assert diagnostics["per_hand"]["1"]["angles"]["45"]["per_minute"] == 0.0
    empty = score_map([])["m5"]["diagnostics"]["overall"]
    assert empty["angles"]["45"]["per_minute"] is None
