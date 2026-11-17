from bisect import bisect_left, bisect_right
from statistics import median

from tools.config import config
from map_creation import v2_forward_flow as flow
from map_creation.v2_postprocessing import _swing_cells, _wall_contains


def pattern_wall_candidates(notes):
    diagnostics = {"windows": 0, "proposed": 0}
    if not (flow.enabled()
            and getattr(config, "v2_pattern_obstacles", False)
            and getattr(config, "v2_walls_heuristic", False)
            and getattr(config, "add_obstacle_flag", False)):
        return [], diagnostics

    ordered = sorted(notes, key=lambda note: float(note["time"]))
    eps = float(config.min_time_diff)
    onsets = []
    for note in ordered:
        t = float(note["time"])
        if not onsets or t - onsets[-1] > eps:
            onsets.append(t)
    if len(onsets) < 8:
        return [], diagnostics

    times = [float(note["time"]) for note in ordered]
    cells = [_swing_cells(note) for note in ordered]
    before, after = map(float, config.obstacle_time_gap)
    duration = max(.125, float(config.obstacle_min_duration) + 1e-6)
    walls = []
    last_start = float("-inf")
    for start in range(0, len(onsets) - 7, 8):
        window = onsets[start:start + 8]
        gaps = [b - a for a, b in zip(window, window[1:])]
        mid = median(gaps)
        tolerance = max(.015, .15 * mid)
        if (window[0] < last_start + 4.
                or any(gap < .1 - 1e-9 or gap > .4 + 1e-9
                       or abs(gap - mid) > tolerance + 1e-9 for gap in gaps)):
            continue
        diagnostics["windows"] += 1
        last_start = window[0]
        for t, preferred in ((window[0], 0), (window[4], 3)):
            for lane in (preferred, 3 - preferred):
                wall = dict(kind="wall", pattern="flow_edge_accent", x=lane, y=2,
                            width=1, height=1, time=t, end_time=t + duration)
                lo = bisect_left(times, t - before)
                hi = bisect_right(times, wall["end_time"] + after)
                if not any(_wall_contains(wall, cell)
                           for note_cells in cells[lo:hi] for cell in note_cells):
                    walls.append(wall)
                    break
    diagnostics["proposed"] = len(walls)
    return walls, diagnostics
