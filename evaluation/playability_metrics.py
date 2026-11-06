"""Playability metrics — shared note preparation and swing geometry.

Foundation for the M1-M4 playability metrics. Pure note-list helpers:
stdlib + numpy only, no ``tools.config``/``map_creation``/audio/model
imports so the module stays cheap to import in the evaluation harness.

Notes are ``(t_seconds, hand, x, y, d)`` tuples: ``hand`` 0 = left,
1 = right; ``x`` in 0..3 columns, ``y`` in 0..2 rows, ``d`` in 0..8 cut
direction (0 up, 1 down, 2 left, 3 right, 4 up-left, 5 up-right,
6 down-left, 7 down-right, 8 dot).

The swing-volume geometry reuses the established occupancy model from
``map_creation/v2_postprocessing.py``: a cut with step ``s`` approaches
from ``pos + s`` and exits through ``pos - s``, so both adjacent cells
along the axis belong to the swing. The helpers are copied verbatim
(with a source comment each) because importing ``v2_postprocessing``
pulls in aubio/librosa transitively through ``sanity_check``.
"""

import numpy as np

EPS = 0.02          # same-hand stack collapse window, seconds
T_FAST = 0.5        # fast-transition gap scale for M1 speed weight, seconds
T_VIS = 0.4         # occlusion lookback window for M2, seconds
T_SEG = 0.5         # bimanual segment split gap for M3, seconds
TAU_X = 0.25        # cross-hand interference window for M4, seconds
T_RESET = 0.4       # same-hand idle gap beyond which no travel corridor, seconds
ISO_MIN = 0.6       # dominant-axis eigenvalue share floor for M3
NPS_BANDS = ((0, 3), (3, 5), (5, 7), (7, float("inf")))

GRID_W, GRID_H = 4, 3  # lanes x rows

# Direction index -> one-cell (dx, dy) offset. Same table as
# ``sanity_check.get_cut_dir_xy`` / ``v2_postprocessing._DIRECTION_STEP``:
# the approach vector — the hand enters from ``pos + step`` and exits
# through ``pos - step``.
STEP = {
    0: (0, -1),   # up
    1: (0, 1),    # down
    2: (1, 0),    # left
    3: (-1, 0),   # right
    4: (1, -1),   # up-left
    5: (-1, -1),  # up-right
    6: (1, 1),    # down-left
    7: (-1, 1),   # down-right
}
_STEP_DIRECTION = {step: direction for direction, step in STEP.items()}

# Exact swing reversal — same mapping as ``_REVERSE_DIRECTION``.
REVERSE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 5: 6, 6: 5, 7: 4}

# Left/right mirrored cut direction — same convention as ``_v2_mirroring``:
# the horizontal step flips, the vertical step is preserved, dots stay dots.
MIRROR_DIR = {direction: _STEP_DIRECTION[(-step[0], step[1])]
              for direction, step in STEP.items()}
MIRROR_DIR[8] = 8


def swept_cells(x, y, direction):
    """In-grid cells the saber passes through on this cut: approach and exit.

    Copied from ``v2_postprocessing._swept_cells``; returns a set here so
    swing volumes can be unioned directly."""
    step = STEP.get(direction)
    if step is None:
        return set()
    return {(x + sign * step[0], y + sign * step[1]) for sign in (1, -1)
            if 0 <= x + sign * step[0] < GRID_W and 0 <= y + sign * step[1] < GRID_H}


def _dot_virtual_direction(previous_d, px, py, x, y):
    """Implied direction of a dot note from geometry.

    Copied verbatim from ``v2_postprocessing._dot_virtual_direction``:

    After the previous cut the hand rests near its exit cell ``pos - step``;
    the dot is then hit by whatever motion reaches it from there. A dot on the
    exit cell continues the same stroke; a dot in the return zone inherits the
    pendulum reversal. ``None`` when there is no previous direction."""
    if previous_d is None or previous_d == 8:
        return None
    dx, dy = STEP[previous_d]
    mx, my = x - px + dx, y - py + dy
    if mx == 0 and my == 0:
        return REVERSE[previous_d]
    return _STEP_DIRECTION[(-int(np.sign(mx)), -int(np.sign(my)))]


def corridor(a, b):
    """Supercover cells on the segment between cells ``a`` and ``b``.

    Off-grid endpoints are clamped to the grid first. Where the line
    crosses a cell corner exactly, both flanking cells count (supercover,
    not just Bresenham)."""
    (x0, y0), (x1, y1) = _clamp_cell(a), _clamp_cell(b)
    dx, dy = x1 - x0, y1 - y0
    nx, ny = abs(dx), abs(dy)
    sign_x, sign_y = int(np.sign(dx)), int(np.sign(dy))
    cells = {(x0, y0)}
    x, y, ix, iy = x0, y0, 0, 0
    while ix < nx or iy < ny:
        # next vertical vs horizontal grid-line crossing: (1+2i)*n compares
        # the crossing fractions (0.5+i)/n without dividing.
        decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if decision == 0:  # exact corner crossing: both flanks are touched
            x, y = x + sign_x, y + sign_y
            cells.update(((x - sign_x, y), (x, y - sign_y), (x, y)))
            ix, iy = ix + 1, iy + 1
        elif decision < 0:
            x += sign_x
            cells.add((x, y))
            ix += 1
        else:
            y += sign_y
            cells.add((x, y))
            iy += 1
    return cells


def _clamp_cell(cell):
    """Clamp a possibly off-grid cell to the 4x3 note grid."""
    x, y = cell
    return min(max(int(x), 0), GRID_W - 1), min(max(int(y), 0), GRID_H - 1)


def prepare_notes(notes):
    """Sort, collapse same-hand stacks, and attach swing context.

    Returns one dict per kept note: ``{"t", "hand", "x", "y", "d", "eff",
    "pred", "exit_cell"}``. ``eff`` is the effective direction — dots are
    resolved through the same-hand predecessor's effective direction via
    ``_dot_virtual_direction`` (``None`` when unresolvable). ``pred`` is the
    previous kept same-hand note dict or ``None``. ``exit_cell`` is
    ``(x - STEP[eff][0], y - STEP[eff][1])`` or ``None``; it may lie off-grid
    (callers clamp as needed). Same-hand notes within ``EPS`` collapse to
    their first representative — a stack is one physical stroke."""
    kept, last_t = [], {}
    for t, hand, x, y, d in sorted(notes, key=lambda note: note[0]):
        anchor = last_t.get(hand)
        if anchor is not None and t - anchor <= EPS:
            continue
        last_t[hand] = t
        kept.append((t, hand, x, y, d))
    prepared, predecessor = [], {}
    for t, hand, x, y, d in kept:
        pred = predecessor.get(hand)
        if d != 8:
            eff = d
        elif pred is not None:
            eff = _dot_virtual_direction(pred["eff"], pred["x"], pred["y"], x, y)
        else:
            eff = None
        step = STEP.get(eff)
        note = {"t": t, "hand": hand, "x": x, "y": y, "d": d,
                "eff": eff, "pred": pred,
                "exit_cell": (x - step[0], y - step[1]) if step else None}
        prepared.append(note)
        predecessor[hand] = note
    return prepared
