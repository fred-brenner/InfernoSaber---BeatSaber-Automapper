from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.playability_metrics import (MIRROR_DIR, STEP, corridor,
                                            m1_lateral, prepare_notes,
                                            swept_cells)


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
