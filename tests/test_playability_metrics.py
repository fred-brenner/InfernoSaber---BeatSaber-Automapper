from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.playability_metrics import (MIRROR_DIR, STEP, corridor,
                                            m1_lateral, m2_occlusion,
                                            m3_divergence, m4_interference,
                                            prepare_notes, score_map,
                                            swept_cells, swing_volume)


def test_step_table_matches_get_cut_dir_xy():
    # sanity_check.get_cut_dir_xy direction→step table (approach offsets:
    # hand enters from pos + step, exits pos - step). The source module pulls
    # in aubio/librosa, so the expected values are pinned here instead.
    assert STEP[0] == (0, -1)  # up
    assert STEP[3] == (-1, 0)  # right
    assert STEP[2] == (1, 0)   # left
    assert STEP[5] == (-1, -1)  # up-right
    assert STEP == {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0),
                    4: (1, -1), 5: (-1, -1), 6: (1, 1), 7: (-1, 1)}
    assert MIRROR_DIR == {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}


def test_prepare_collapses_same_hand_stacks():
    stacked = prepare_notes([(0.0, 0, 1, 1, 0), (0.01, 0, 2, 1, 1)])
    assert len(stacked) == 1
    assert stacked[0]["x"] == 1  # first wins
    split = prepare_notes([(0.0, 0, 1, 1, 0), (0.01, 1, 2, 1, 0)])
    assert len(split) == 2


def test_dot_implied_direction():
    # Down-cut at (1,1) exits at (1,0); the dot sits on the exit cell and
    # inherits the pendulum reversal → effective direction up (0).
    prepared = prepare_notes([(0.0, 0, 1, 1, 1), (0.4, 0, 1, 0, 8)])
    assert prepared[0]["pred"] is None
    assert prepared[0]["eff"] == 1
    assert prepared[0]["exit_cell"] == (1, 0)
    assert prepared[1]["pred"] is prepared[0]
    assert prepared[1]["eff"] == 0


def test_swept_cells():
    # approach = pos + STEP[d], exit = pos - STEP[d]; own cell excluded,
    # in-grid only. d=2/d=3 are mirror steps, so both adjacent cells along
    # the cut axis are swept whenever they lie inside the 4x3 grid.
    assert swept_cells(1, 1, 3) == {(0, 1), (2, 1)}  # approach + exit in grid
    assert swept_cells(3, 1, 3) == {(2, 1)}          # exit (4,1) off-grid
    assert swept_cells(1, 1, 2) == {(0, 1), (2, 1)}


def test_m1_dense_horizontal_stream():
    # d=3 stream at 0.2s gaps alternating lanes: every transition is
    # max-stress horizontal work, s(0.2) = 0.6 → value saturates at 1.
    notes = [(0.2 * i, 0, i % 2, 1, 3) for i in range(10)]
    result = m1_lateral(prepare_notes(notes))
    assert result["transitions"] == 9
    assert result["value"] == pytest.approx(1.0)
    assert result["units"] == pytest.approx([0.6] * 9)


def test_m1_vertical_control():
    # Same density, all vertical cuts → zero lateral load, no units.
    notes = [(0.2 * i, 0, 1, 1, i % 2) for i in range(10)]
    result = m1_lateral(prepare_notes(notes))
    assert result["transitions"] == 9
    assert result["value"] == 0.0
    assert len(result["units"]) == 0


def test_m1_slow_lateral_discounted():
    # Horizontal cuts at 1.0s gaps: s(g) = 0 → value 0 and no units, but
    # the >0.35 stress band still records that the cuts were horizontal.
    notes = [(float(i), 0, i % 2, 1, 3) for i in range(5)]
    result = m1_lateral(prepare_notes(notes))
    assert result["transitions"] == 4
    assert result["value"] == 0.0
    assert len(result["units"]) == 0
    assert result["bands"][">0.35"] == 1.0
    assert result["bands"]["<=0.2"] == 0.0   # empty band → 0.0
    assert result["bands"]["0.2-0.35"] == 0.0


def test_m1_dot_is_relief():
    # The middle note is a dot: its transition carries w=0 even though its
    # implied direction resolves — dots are free-direction relief.
    notes = [(0.0, 0, 1, 1, 3), (0.2, 0, 1, 0, 8), (0.4, 0, 2, 1, 3)]
    result = m1_lateral(prepare_notes(notes))
    assert result["transitions"] == 2
    assert result["value"] == pytest.approx(0.5)
    assert result["units"] == pytest.approx([0.6])


def test_corridor():
    assert corridor((0, 0), (3, 0)) == {(0, 0), (1, 0), (2, 0), (3, 0)}
    # off-grid endpoint (4,1) clamps to (3,1)
    assert corridor((0, 1), (4, 1)) == {(0, 1), (1, 1), (2, 1), (3, 1)}
    # exact diagonal corner crossings touch both flanking cells (supercover)
    assert corridor((0, 0), (2, 2)) == {(0, 0), (1, 0), (0, 1), (1, 1),
                                       (2, 1), (1, 2), (2, 2)}


def test_m2_rising_same_column():
    # Bottom note at t=1.0, top note same column at t=1.2 (other hand):
    # occl = max(0, 2-0)·(1-0/3)·(1-0.2/0.4) = 2·1·0.5 = 1.0 → one incident.
    prepared = prepare_notes([(1.0, 0, 1, 0, 0), (1.2, 1, 1, 2, 0)])
    result = m2_occlusion(prepared)
    assert result["units"] == pytest.approx([1.0])
    assert result["value"] == pytest.approx(0.5)  # Σ occl / N_notes = 1.0/2
    assert len(result["incidents"]) == 1
    incident = result["incidents"][0]
    assert incident["t"] == pytest.approx(1.2)
    assert incident["note"] == (1, 2)
    assert incident["occluder"] == (1, 0)
    assert incident["dt"] == pytest.approx(0.2)
    assert incident["occl"] == pytest.approx(1.0)


def test_m2_simultaneous_not_occluder():
    # Same-time bottom/top pair in one column: simultaneous partners share
    # the note's depth and are excluded by the strict Δt > EPS gate.
    prepared = prepare_notes([(1.0, 0, 1, 0, 0), (1.0, 1, 1, 2, 0)])
    result = m2_occlusion(prepared)
    assert result["value"] == 0.0
    assert len(result["units"]) == 0
    assert result["incidents"] == []


def test_m2_nearest_occluder_only():
    # j at (x=1, y=1), t=1.25 has two in-window candidates: a high note
    # (x=2, y=2) at Δt=0.05 (occl 0 — above j) and a low note (x=0, y=0) at
    # Δt=0.3 (would occl >0). The nearest occluder wins → occl == 0.
    # The candidates sit |Δx| = 2 apart, so they don't occlude each other.
    prepared = prepare_notes([(0.95, 0, 0, 0, 0), (1.2, 1, 2, 2, 0),
                            (1.25, 0, 1, 1, 0)])
    result = m2_occlusion(prepared)
    assert result["value"] == 0.0
    assert len(result["units"]) == 0
    assert result["incidents"] == []


def test_m2_column_gate():
    # Rising pair within T_VIS but |Δx| = 2 exceeds the ±1-lane sightline
    # gate → no occluder.
    prepared = prepare_notes([(1.0, 0, 0, 0, 0), (1.2, 1, 2, 2, 0)])
    result = m2_occlusion(prepared)
    assert result["value"] == 0.0
    assert len(result["units"]) == 0
    assert result["incidents"] == []


def test_m2_incidents_severity_sorted():
    # Two occlusions: the later, stronger one (occl 1.5 at t=2.1, dt=0.1)
    # must lead the incident list — same top-k contract as M4.
    prepared = prepare_notes([(1.0, 0, 1, 0, 0), (1.2, 1, 1, 2, 0),
                            (2.0, 0, 2, 0, 0), (2.1, 1, 2, 2, 0)])
    result = m2_occlusion(prepared)
    assert [event["occl"] for event in result["incidents"]] == \
        pytest.approx([1.5, 1.0])


def test_m3_parallel_vertical():
    # Both hands ride a 2s vertical wave (y 0<->2) on their own side: every
    # displacement vector is purely (0,+-2), so each dominant axis points
    # along y and |sin theta| between them is 0.
    times = [0.25 * i for i in range(9)]  # t = 0..2.0, one segment
    notes = [(t, 0, 0, (i % 2) * 2, 8) for i, t in enumerate(times)]
    notes += [(t, 1, 3, (i % 2) * 2, 8) for i, t in enumerate(times)]
    result = m3_divergence(prepare_notes(notes))
    assert result["segments"] == 1
    assert result["units"] == pytest.approx([0.0])
    assert result["value"] == pytest.approx(0.0)


def test_m3_orthogonal():
    # Same window, left hand on a vertical wave while the right hand sweeps
    # horizontally (x 2<->3): perpendicular axes -> |sin theta| = 1.
    times = [0.25 * i for i in range(9)]
    left = [(t, 0, 0, (i % 2) * 2, 8) for i, t in enumerate(times)]
    right = [(t, 1, 2 + i % 2, 1, 8) for i, t in enumerate(times)]
    result = m3_divergence(prepare_notes(left + right))
    assert result["segments"] == 1
    assert result["units"] == pytest.approx([1.0])
    assert result["value"] == pytest.approx(1.0)


def test_m3_isotropic_skipped():
    # Segment 1 (t = 0..1.0): both hands vertical -> valid. Segment 2 (after
    # a 1.0s gap): left stays vertical but the right hand walks a square,
    # scatter = 2*I, lambda share 0.5 < ISO_MIN -> no dominant axis, so the
    # segment is excluded from units, weights and the segment count.
    times1 = [0.25 * i for i in range(5)]
    notes = [(t, 0, 0, (i % 2) * 2, 8) for i, t in enumerate(times1)]
    notes += [(t, 1, 3, (i % 2) * 2, 8) for i, t in enumerate(times1)]
    times2 = [2.0 + 0.25 * i for i in range(5)]
    notes += [(t, 0, 0, (i % 2) * 2, 8) for i, t in enumerate(times2)]
    square = [(2, 0), (3, 0), (3, 1), (2, 1), (2, 0)]
    notes += [(t, 1, x, y, 8) for t, (x, y) in zip(times2, square)]
    result = m3_divergence(prepare_notes(notes))
    assert result["segments"] == 1
    assert result["units"] == pytest.approx([0.0])
    assert result["value"] == pytest.approx(0.0)


def test_m3_idle_hand_ignored():
    # A left-hand-only segment has no right-hand axis, so it contributes
    # nothing; the second segment (V vs H) is the only one counted.
    times1 = [0.25 * i for i in range(5)]
    notes = [(t, 0, 0, (i % 2) * 2, 8) for i, t in enumerate(times1)]
    times2 = [2.0 + 0.25 * i for i in range(5)]
    notes += [(t, 0, 0, (i % 2) * 2, 8) for i, t in enumerate(times2)]
    notes += [(t, 1, 2 + i % 2, 1, 8) for i, t in enumerate(times2)]
    result = m3_divergence(prepare_notes(notes))
    assert result["segments"] == 1
    assert result["units"] == pytest.approx([1.0])
    assert result["value"] == pytest.approx(1.0)
    solo = m3_divergence(prepare_notes(notes[:5]))  # the idle segment alone
    assert solo["segments"] == 0
    assert len(solo["units"]) == 0
    assert solo["value"] == 0.0


def test_prepare_collapse_float_tolerance():
    # 0.30 - 0.28 == 0.020000000000000004 in IEEE float — the EPS+epsilon
    # window keeps one physical stack from splitting on float noise
    # (v2_postprocessing uses the same epsilon-adjusted comparisons).
    assert len(prepare_notes([(0.28, 0, 1, 1, 0), (0.30, 0, 2, 1, 0)])) == 1


def test_m1_diagonal_stream():
    # d=5 stream at 0.2s gaps: every transition carries diagonal stress
    # w=0.5 at identical speed -> value = 0.5 (pins the diagonal class).
    notes = [(0.2 * i, 0, i % 2, 1, 5) for i in range(10)]
    assert m1_lateral(prepare_notes(notes))["value"] == pytest.approx(0.5)


def test_m4_mirrored_double_zero():
    # Canonical outward double: left (1,1) cut d=2 + right (2,1) cut d=3.
    # x_a+x_b==3, same row, mirrored directions, simultaneous -> severity 0
    # even though the swing volumes overlap on (1,1) and (2,1).
    prepared = prepare_notes([(1.0, 0, 1, 1, 2), (1.0, 1, 2, 1, 3)])
    result = m4_interference(prepared, minutes=1.0)
    assert result["value"] == 0.0
    assert len(result["units"]) == 0
    assert result["incidents"] == []


def test_m4_inward_clash():
    # Deliberately non-mirrored simultaneous pair: the left hand's right
    # cut sweeps (2,1) -- the right note's own cell. An exact inward pair
    # (right cutting d=2, the mirror of d=3) would be discounted to 0, so
    # the right note cuts d=0 here.
    prepared = prepare_notes([(1.0, 0, 1, 1, 3), (1.0, 1, 2, 1, 0)])
    result = m4_interference(prepared, minutes=1.0)
    assert result["units"] == pytest.approx([1.0])
    assert result["value"] == pytest.approx(1.0)
    incident = result["incidents"][0]
    assert incident["cells"] == [(2, 1)]
    assert incident["severity"] == pytest.approx(1.0)


def test_m4_unresolved_dot_own_cell_only():
    # A first-note dot has no predecessor -> eff=None -> the swing volume
    # is just its own cell, and M4 scores the pair without crashing.
    prepared = prepare_notes([(0.0, 0, 1, 1, 8), (0.0, 1, 3, 0, 8)])
    assert prepared[0]["eff"] is None
    assert swing_volume(prepared[0]) == {(1, 1)}
    result = m4_interference(prepared, minutes=1.0)
    assert result["value"] == 0.0  # disjoint single cells


def test_m4_left_cuts_right_into_active_right():
    # Flagship example: left (1,1) d=3 sweeps (2,1) while the right hand
    # plays (2,1)->(2,2) around it. The second right note's travel
    # corridor (pred exit (1,1) -> approach (3,2)) contributes its whole
    # overlap: without it the volumes would be disjoint.
    prepared = prepare_notes([(0.9, 1, 2, 1, 2), (1.0, 0, 1, 1, 3),
                            (1.1, 1, 2, 2, 2)])
    result = m4_interference(prepared, minutes=1.0)
    # each pair shares {(1,1),(2,1)} -> 2 * (1 - 0.1/0.25) = 1.2
    assert sorted(result["units"]) == pytest.approx([1.2, 1.2])
    assert result["value"] == pytest.approx(2.4)
    assert len(result["incidents"]) == 2
    assert result["incidents"][0]["cells"] == [(1, 1), (2, 1)]
    assert result["incidents"][0]["severity"] == pytest.approx(1.2)


def test_m4_crossover():
    # Left note lands right of the right-hand note (x_a=2 > x_b=1) at
    # dt=0.1: zero shared cells, pure wrist-crossover term
    # (2-1) * (1 - 0.1/0.25) = 0.6.
    prepared = prepare_notes([(1.0, 0, 2, 1, 0), (1.1, 1, 1, 1, 0)])
    result = m4_interference(prepared, minutes=1.0)
    assert result["units"] == pytest.approx([0.6])
    assert result["value"] == pytest.approx(0.6)
    incident = result["incidents"][0]
    assert incident["cells"] == []
    assert incident["crossover"] == pytest.approx(0.6)
    assert incident["severity"] == pytest.approx(0.6)


def test_m4_corridor_reset():
    # Flagship setup with the right-hand gap stretched to 1.0s > T_RESET:
    # the hand reset -> b2 has no travel corridor (volume = own + swept),
    # and b1 sits outside TAU_X so no pair scores at all.
    prepared = prepare_notes([(0.1, 1, 2, 1, 2), (1.0, 0, 1, 1, 3),
                            (1.1, 1, 2, 2, 2)])
    assert swing_volume(prepared[2]) == {(1, 2), (2, 2), (3, 2)}
    result = m4_interference(prepared, minutes=1.0)
    assert result["value"] == 0.0
    assert len(result["units"]) == 0


def test_score_map_shape():
    notes = [(0.0, 0, 1, 1, 0), (0.5, 1, 2, 1, 1), (1.0, 0, 2, 1, 0)]
    result = score_map(notes)
    assert set(result) == {"notes", "span_seconds", "nps",
                           "m1", "m2", "m3", "m4"}
    assert result["notes"] == 3
    assert result["span_seconds"] == pytest.approx(1.0)
    assert result["nps"] == pytest.approx(result["notes"]
                                        / result["span_seconds"])
    assert result["m4"]["value"] == 0.0  # pairs 0.5s apart, outside TAU_X
