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
INCIDENT_TOP_K = 10  # cap on M4 severity-sorted incident reports

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
        if anchor is not None and t - anchor <= EPS + 1e-9:
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


# M1 cut stress w(d): vertical cuts (0,1) and dots (8) carry no lateral
# stress, diagonals (4-7) half, horizontal cuts (2,3) full — dots are
# free-direction relief, so the raw ``d`` counts, not ``eff``.
M1_STRESS = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0,
             4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.0}
M1_GAP_BANDS = ("<=0.2", "0.2-0.35", ">0.35")


def _gap_band(gap):
    """M1 reporting band for a same-hand gap, in seconds."""
    if gap <= 0.2:
        return "<=0.2"
    if gap <= 0.35:
        return "0.2-0.35"
    return ">0.35"


def m1_lateral(prepared):
    """M1 lateral load — the share of fast same-hand work on non-vertical
    axes (design §M1).

    Each same-hand transition (``pred`` chain) contributes the arriving
    note's cut stress ``w(d)`` weighted by speed ``s(g) = max(0, 1 -
    g/T_FAST)``; ``value = Σ w·s / Σ s`` in [0,1] — 0 when no transition is
    fast enough. ``units`` holds the positive ``w·s`` contributions (the
    pooled-percentile input); ``bands`` reports mean stress per gap band to
    expose "only bad when fast" (0.0 for empty bands). ``transitions`` is
    the number of same-hand transitions seen."""
    numerator = denominator = 0.0
    units, transitions = [], 0
    band_stress = {band: [] for band in M1_GAP_BANDS}
    for note in prepared:
        pred = note["pred"]
        if pred is None:
            continue
        transitions += 1
        gap = note["t"] - pred["t"]
        stress, speed = M1_STRESS[note["d"]], max(0.0, 1.0 - gap / T_FAST)
        contribution = stress * speed
        numerator += contribution
        denominator += speed
        if contribution > 0:
            units.append(contribution)
        band_stress[_gap_band(gap)].append(stress)
    return {"value": numerator / denominator if denominator else 0.0,
            "units": np.asarray(units, dtype=float),
            "bands": {band: float(np.mean(stresses)) if stresses else 0.0
                      for band, stresses in band_stress.items()},
            "transitions": transitions}


def _nearest_occluder(prepared, index):
    """M2 occluder for ``prepared[index]`` (design §M2).

    The strictly earlier (``Δt > EPS``) note nearest by ``Δt`` within
    ``T_VIS`` and ``|Δx| <= 1`` — any hand; color is irrelevant to
    sightlines. Simultaneous partners share j's depth and are excluded by
    the strict gate. Scanning the time-sorted list backwards, the first
    candidate is the nearest; once it is cut the sightline is clear, so
    older notes behind it never stack penalties."""
    note = prepared[index]
    for earlier in range(index - 1, -1, -1):
        other = prepared[earlier]
        dt = note["t"] - other["t"]
        if dt > T_VIS:
            break
        if dt > EPS and abs(note["x"] - other["x"]) <= 1:
            return other, dt
    return None, 0.0


def m2_occlusion(prepared):
    """M2 occlusion — mean sightline obstruction per note (design §M2).

    Each note's occluder (``_nearest_occluder``) contributes
    ``occl = max(0, y_j - y_i)·(1 - |Δx|/3)·(1 - Δt/T_VIS)`` — taller
    occluders closer to j's lane and to j's cut time hide more.
    ``value = Σ occl / N_notes``; ``units`` holds the positive occl values
    (the pooled-percentile input); ``incidents`` lists ``occl >= 0.5``
    events as ``{"t", "note", "occluder", "dt", "occl"}`` for top-k
    reporting."""
    occlusions, units, incidents = [], [], []
    for index, note in enumerate(prepared):
        occluder, dt = _nearest_occluder(prepared, index)
        occl = 0.0
        if occluder is not None:
            occl = (max(0.0, note["y"] - occluder["y"])
                    * (1.0 - abs(note["x"] - occluder["x"]) / 3.0)
                    * (1.0 - dt / T_VIS))
        occlusions.append(occl)
        if occl > 0:
            units.append(occl)
        if occl >= 0.5 - 1e-9:  # ε absorbs float noise at the threshold
            incidents.append({"t": note["t"],
                              "note": (note["x"], note["y"]),
                              "occluder": (occluder["x"], occluder["y"]),
                              "dt": dt, "occl": occl})
    return {"value": float(np.sum(occlusions) / len(prepared))
            if prepared else 0.0,
            "units": np.asarray(units, dtype=float),
            "incidents": incidents}


def _event_segments(prepared):
    """Split the time-sorted note stream into activity segments.

    Events are the unique note times; same-time notes have zero gap, so
    splitting the note list wherever consecutive notes sit more than
    ``T_SEG`` apart yields the same segmentation as splitting the unique
    event times at gaps > ``T_SEG``."""
    segments, current, previous_t = [], [], None
    for note in prepared:
        if previous_t is not None and note["t"] - previous_t > T_SEG:
            segments.append(current)
            current = []
        current.append(note)
        previous_t = note["t"]
    if current:
        segments.append(current)
    return segments


def _hand_vectors(segment, hand):
    """Displacement vectors between consecutive same-hand notes inside one
    segment — the note pairs never cross a segment boundary."""
    coords = [(note["x"], note["y"]) for note in segment
              if note["hand"] == hand]
    return [(b[0] - a[0], b[1] - a[1])
            for a, b in zip(coords, coords[1:])]


def _dominant_axis(vectors):
    """A hand's dominant motion axis: top eigenvector of the 2x2 scatter
    ``Σvvᵀ``. ``None`` when the hand has fewer than 2 vectors, when the
    scatter is all-zero (stationary hand, eigenvalues sum to 0), or when
    the motion is near-isotropic (``λ1/(λ1+λ2) < ISO_MIN``)."""
    if len(vectors) < 2:
        return None
    v = np.asarray(vectors, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(v.T @ v)
    total = float(eigenvalues.sum())
    if total <= 0.0 or eigenvalues[1] / total < ISO_MIN:
        return None
    return eigenvectors[:, 1]


def m3_divergence(prepared):
    """M3 bimanual divergence — how far the two hands' dominant motion axes
    rotate away from each other within activity segments (design §M3).

    The event stream segments at gaps > ``T_SEG`` (``_event_segments``).
    Per segment each hand's consecutive same-hand displacements
    (``_hand_vectors``) define a dominant axis (``_dominant_axis``); the
    segment counts only when both hands resolve one. Its divergence is the
    2D cross magnitude ``|u_L.x·u_R.y − u_L.y·u_R.x|`` = ``|sin θ|`` —
    unoriented, so eigenvector sign is irrelevant: same-axis or mirrored
    hands → 0, orthogonal → 1. ``value`` is the weighted mean by
    ``min(nvec_L, nvec_R)``; ``units`` holds the per-segment divergences
    (the pooled-percentile input); ``segments`` counts valid segments."""
    units, weights = [], []
    for segment in _event_segments(prepared):
        vectors_l, vectors_r = (_hand_vectors(segment, hand)
                                for hand in (0, 1))
        axis_l, axis_r = (_dominant_axis(vectors)
                          for vectors in (vectors_l, vectors_r))
        if axis_l is None or axis_r is None:
            continue
        units.append(float(abs(axis_l[0] * axis_r[1]
                               - axis_l[1] * axis_r[0])))
        weights.append(min(len(vectors_l), len(vectors_r)))
    total_weight = sum(weights)
    return {"value": float(np.dot(units, weights) / total_weight)
            if total_weight else 0.0,
            "units": np.asarray(units, dtype=float),
            "segments": len(units)}


def swing_volume(note):
    """M4 swing volume — grid cells the saber occupies for this note.

    Own cell ∪ ``swept_cells`` on ``eff`` (approach + exit) ∪ the travel
    corridor: supercover cells between the same-hand predecessor's exit
    cell (its own cell when unresolved) and this note's approach cell (its
    own cell when ``eff`` is ``None``). The corridor applies only while the
    same-hand gap stays within ``T_RESET`` — after an idle beat the hand
    resets and reaches in fresh. ``corridor`` clamps off-grid endpoints."""
    x, y = note["x"], note["y"]
    volume = {(x, y)} | swept_cells(x, y, note["eff"])
    pred = note["pred"]
    if pred is not None and note["t"] - pred["t"] <= T_RESET + 1e-9:
        step = STEP.get(note["eff"])
        approach = (x + step[0], y + step[1]) if step else (x, y)
        start = pred["exit_cell"] or (pred["x"], pred["y"])
        volume |= corridor(start, approach)
    return volume


def _cross_hand_partners(prepared, index):
    """Yield indices of opposite-hand notes within ``TAU_X`` of
    ``prepared[index]``. ``prepared`` is time-sorted, so partners form a
    contiguous window — scan outward both ways and stop at its edge."""
    t = prepared[index]["t"]
    j = index - 1
    while j >= 0 and t - prepared[j]["t"] <= TAU_X:
        if prepared[j]["hand"] != prepared[index]["hand"]:
            yield j
        j -= 1
    j = index + 1
    while j < len(prepared) and prepared[j]["t"] - t <= TAU_X:
        if prepared[j]["hand"] != prepared[index]["hand"]:
            yield j
        j += 1


def _is_mirrored_double(a, b):
    """Exact mirrored simultaneous pair (design §M4): symmetric cells
    ``x_a + x_b == 3``, same row, mirrored cut directions within ``EPS`` —
    the canonical human double is intentional, not an interference defect."""
    return (abs(a["t"] - b["t"]) <= EPS + 1e-9
            and a["x"] + b["x"] == GRID_W - 1
            and a["y"] == b["y"]
            and b["d"] == MIRROR_DIR.get(a["d"]))


def m4_interference(prepared, minutes):
    """M4 path interference — cross-hand swing-volume clashes per minute
    (design §M4).

    Every hand-0/hand-1 pair within ``TAU_X`` scores
    ``severity = |V_a ∩ V_b|·(1 − |Δt|/TAU_X)
    + max(0, x_a − x_b)·(1 − |Δt|/TAU_X)`` — shared swing cells plus a
    wrist-crossover term when the left note lands right of the right-hand
    note. Exact mirrored doubles (``_is_mirrored_double``) score 0.
    ``value`` = Σ severity / ``minutes`` (0.0 when the span is zero — a
    zero-length stream has no meaningful rate); ``units`` holds the
    positive severities (the pooled-percentile input); ``incidents`` is
    the ``INCIDENT_TOP_K``-capped list of
    ``{"t","a","b","cells","severity","crossover"}`` dicts sorted by
    severity desc, where ``t`` is the later note's time (the moment the
    interference completes) and ``cells`` the sorted shared cells."""
    volumes = [swing_volume(note) for note in prepared]
    total, units, incidents = 0.0, [], []
    for index, a in enumerate(prepared):
        if a["hand"] != 0:
            continue
        for j in _cross_hand_partners(prepared, index):
            b = prepared[j]
            if _is_mirrored_double(a, b):
                continue  # severity 0: no unit, no incident, no total
            weight = max(0.0, 1.0 - abs(a["t"] - b["t"]) / TAU_X)
            shared = volumes[index] & volumes[j]
            crossover = max(0.0, (a["x"] - b["x"]) * weight)
            severity = len(shared) * weight + crossover
            total += severity
            if severity <= 0.0:
                continue
            units.append(severity)
            incidents.append({"t": max(a["t"], b["t"]),
                              "a": (a["x"], a["y"], a["d"]),
                              "b": (b["x"], b["y"], b["d"]),
                              "cells": sorted(shared),
                              "severity": severity,
                              "crossover": crossover})
    incidents.sort(key=lambda event: event["severity"], reverse=True)
    return {"value": total / minutes if minutes > 0 else 0.0,
            "units": np.asarray(units, dtype=float),
            "incidents": incidents[:INCIDENT_TOP_K]}


def score_map(notes):
    """Score one map on all four playability metrics (design calibration).

    Runs ``prepare_notes`` once and shares the result across M1–M4.
    ``notes`` is the kept note count (same-hand stacks within ``EPS`` are
    one physical stroke); ``span_seconds`` = last − first kept note time;
    ``nps`` = notes/span, 0.0 when the span is 0 — a zero-length stream
    has no meaningful rate. ``m4`` receives ``minutes = span/60``."""
    prepared = prepare_notes(notes)
    span = prepared[-1]["t"] - prepared[0]["t"] if prepared else 0.0
    return {"notes": len(prepared),
            "span_seconds": span,
            "nps": len(prepared) / span if span > 0 else 0.0,
            "m1": m1_lateral(prepared),
            "m2": m2_occlusion(prepared),
            "m3": m3_divergence(prepared),
            "m4": m4_interference(prepared, span / 60.0)}
