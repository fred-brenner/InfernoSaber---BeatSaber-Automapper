"""Version 2 hazard sections, shared by supervision and both composers.

All times are elapsed seconds. Queries depend only on duration, BPM and the
recorded policy. Source objects are never edited. Short *sections* remain valid
decisions; the duration limit applies only after consecutive walls are merged.
"""

from bisect import bisect_left, bisect_right
from collections import Counter
from copy import deepcopy
import math

import numpy as np


def policy_defaults():
    """A serializable snapshot of the approved legacy clearance settings."""
    return {"version": 2, "grid_beats": .25, "max_candidates": 50000,
            "clearance": {"add_obstacle_flag": True, "obstacle_time_gap": [.3, .8],
                          "obstacle_min_duration": .1, "obstacle_max_count": 2,
                          "sporty_obstacles": False, "norm_obstacle_positions": [[0], [3]],
                          "norm_obstacle_allowed_types": [0, 1, 2],
                          "sport_obstacle_positions": [[0, 1, 1], [2, 2, 3]],
                          "sport_obstacle_allowed_types": [0, 1],
                          "obstacle_crouch_width": 4, "max_obstacle_height": 5,
                          "min_time_diff": .02}}


def policy(options=None):
    """Resolve an explicit v2 policy, without consulting live project config.

    Callers select the legacy route when the recorded version is absent. This
    module's explicit API defaults to v2 when no policy was supplied at all.
    """
    selected = (options or {}).get("stage_policies", {}).get("hazards")
    result = policy_defaults() if selected is None else deepcopy(selected)
    if result.get("version") != 2:
        raise ValueError("Hazard sections require policy version 2")
    for key in ("grid_beats", "max_candidates"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid hazard policy {key}")
    if int(result["max_candidates"]) != result["max_candidates"]:
        raise ValueError("Hazard max_candidates must be an integer")
    if set(result.get("clearance", {})) != set(policy_defaults()["clearance"]):
        raise ValueError("Hazard policy must record the full frozen clearance settings")
    clearance = result["clearance"]
    numeric = ("obstacle_min_duration", "min_time_diff", "obstacle_max_count",
               "obstacle_crouch_width", "max_obstacle_height")
    for key in numeric:
        value = clearance[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid hazard clearance {key}")
    gaps = clearance["obstacle_time_gap"]
    if len(gaps) != 2 or not np.isfinite(gaps).all() or min(gaps) < 0:
        raise ValueError("Invalid hazard obstacle_time_gap")
    return result


def candidates(duration, bpm, options=None):
    """A wall section and a bomb query at each in-audio quarter-beat boundary.

    The final wall section is clipped to decoded duration. Candidate overflow
    fails before allocation; no timeline truncation or source augmentation occurs.
    """
    if not np.isfinite([duration, bpm]).all() or duration <= 0 or bpm <= 0:
        raise ValueError("Hazard sections require positive finite duration and BPM")
    selected = policy(options)
    step = 60. * selected["grid_beats"] / bpm
    count = int(math.ceil(duration / step))
    # Floating-point division can put an exact endpoint just beyond its integer.
    if count > 1 and (count - 1) * step >= duration:
        count -= 1
    if count * 2 > selected["max_candidates"]:
        raise ValueError(f"Hazard section schedule has {count * 2} queries; exceeds "
                         f"max_candidates={selected['max_candidates']}; no queries were discarded")
    queries = []
    for index in range(count):
        start, end = index * step, min((index + 1) * step, float(duration))
        queries.extend(({"kind": "wall", "time": start, "end_time": end, "section_index": index,
                         "origin": "scheduler"},
                        {"kind": "bomb", "time": start, "end_time": start, "section_index": index,
                         "origin": "scheduler"}))
    return queries


def _nearest(boundaries, time):
    right = bisect_left(boundaries, time)
    if right == 0:
        return 0
    if right == len(boundaries):
        return right - 1
    # Small roundoff at an exact half-grid does not change the earlier tie rule.
    return right - 1 if time - boundaries[right - 1] <= boundaries[right] - time + 1e-12 else right


def _geometry(record):
    return tuple(record.get(key) for key in ("x", "y", "width", "height"))


def source_family(record):
    """Report rejected legacy bombs by their raw type, without rewriting records."""
    raw = record.get("raw", {})
    if record.get("family") == "notes" and isinstance(raw, dict) and raw.get("_type") == 3:
        return "bombs"
    return record.get("family")


def _integer_geometry(record, keys):
    return all(isinstance(record.get(key), (int, float, np.integer, np.floating))
               and not isinstance(record[key], (bool, np.bool_)) and np.isfinite(record[key])
               and int(record[key]) == record[key] for key in keys)


def _contains(wall, x, y):
    return ((wall["x"] - .5 <= x) & (x <= wall["x"] + wall["width"] - .5)
            & (wall["y"] - .5 <= y) & (y <= wall["y"] + wall["height"] - .5))


def _lanes(wall):
    return set(range(int(wall["x"]), int(wall["x"] + wall["width"])))


def geometry_reason(wall, clearance):
    """The frozen legacy wall geometry/mode contract (without time or duration)."""
    if not _integer_geometry(wall, ("x", "y", "width", "height")):
        return "invalid_geometry"
    if (wall["width"] <= 0 or wall["height"] <= 0 or wall["height"] > clearance["max_obstacle_height"]
            or wall["x"] < 0 or wall["y"] < 0 or wall["x"] + wall["width"] > 4
            or wall["y"] + wall["height"] > 5):
        return "invalid_geometry"
    mode = "sport" if clearance["sporty_obstacles"] else "norm"
    allowed_lanes = {lane for side in clearance[f"{mode}_obstacle_positions"] for lane in side}
    crouch = clearance["sporty_obstacles"] and wall["y"] == 2 and wall["width"] == clearance["obstacle_crouch_width"]
    if not _lanes(wall).issubset(allowed_lanes) or (wall["y"] not in clearance[f"{mode}_obstacle_allowed_types"] and not crouch):
        return "unsupported_geometry_mode"
    return None


class ClearanceIndex:
    """Sorted note/chain samples; queries inspect only their temporal neighborhood."""

    def __init__(self, state, duration, clearance):
        self.duration, self.clearance = duration, clearance
        occupied = list(state.get("notes", []))
        for chain in state.get("chains", []):
            span = chain["end_time"] - chain["time"]
            steps = max(int(chain["slices"]) - 1, int(math.ceil(span / .01)), 1)
            for index in range(1, steps + 1):
                alpha = index / steps
                occupied.append({"time": chain["time"] + alpha * span,
                                 "x": chain["x"] + alpha * (chain["tx"] - chain["x"]),
                                 "y": chain["y"] + alpha * (chain["ty"] - chain["y"]),
                                 "d": chain.get("d", 8)})
        occupied.sort(key=lambda item: item["time"])
        self.times = np.asarray([item["time"] for item in occupied], np.float64)
        self.x = np.asarray([item["x"] for item in occupied], np.float64)
        self.y = np.asarray([item["y"] for item in occupied], np.float64)
        # Exactly the legacy calc_note_pos cells, including both swing sides.
        directions = ((0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1), (0, 0))
        swing_times, swing_x, swing_y = [], [], []
        for item in occupied:
            dx, dy = directions[int(item.get("d", 8))]
            points = {(item["x"], item["y"])}
            points.update((item["x"] + sign * dx, item["y"] + sign * dy) for sign in (-1, 1)
                          if 0 <= item["x"] + sign * dx < 4 and 0 <= item["y"] + sign * dy < 3)
            for x, y in points:
                swing_times.append(item["time"]); swing_x.append(x); swing_y.append(y)
        self.swing_times = np.asarray(swing_times, np.float64)
        self.swing_x, self.swing_y = np.asarray(swing_x), np.asarray(swing_y)

    def wall_reason(self, wall, previous=(), *, minimum=False):
        clearance = self.clearance
        start, end = wall["time"], wall["end_time"]
        if not clearance["add_obstacle_flag"]:
            return "walls_disabled"
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= self.duration:
            return "invalid_interval"
        if minimum and end - start + 1e-12 < clearance["obstacle_min_duration"]:
            return "short_merged_wall"
        reason = geometry_reason(wall, clearance)
        if reason:
            return reason
        before, after = clearance["obstacle_time_gap"]
        lanes = _lanes(wall)
        if clearance["sporty_obstacles"] and {1, 2}.issubset(lanes) and wall["y"] >= 2 and start < 10:
            return "opening_crouch_wall"
        if clearance["sporty_obstacles"] and lanes.intersection({1, 2}):
            recovery = max(before, after)
            if any(other["time"] - recovery < end and other["end_time"] + recovery > start
                   and {1, 2}.issubset(lanes | _lanes(other)) for other in previous):
                return "inner_wall_recovery"
        if sum(other["end_time"] > start for other in previous) >= clearance["obstacle_max_count"]:
            return "wall_count_limit"
        lo, hi = np.searchsorted(self.times, [start - before, end + after], side="left")
        hi = np.searchsorted(self.times, end + after, side="right")
        if np.any(_contains(wall, self.x[lo:hi], self.y[lo:hi])):
            return "note_or_chain_collision"
        return None

    def bomb_cells(self, time):
        cells = np.ones(12, bool)
        if not 0 <= time < self.duration:
            cells[:] = False
            return cells
        radius = max(self.clearance["min_time_diff"], .1)
        lo = np.searchsorted(self.swing_times, time - radius, side="left")
        hi = np.searchsorted(self.swing_times, time + radius, side="right")
        for cell in range(12):
            if np.any((np.abs(self.swing_x[lo:hi] - cell % 4) < .5)
                      & (np.abs(self.swing_y[lo:hi] - cell // 4) < .5)):
                cells[cell] = False
        return cells


def coalesce(sections):
    """Merge touching positive sections of identical geometry; never fill gaps."""
    merged = []
    for section in sections:
        item = deepcopy(section)
        if merged and _geometry(merged[-1]) == _geometry(item) and abs(merged[-1]["end_time"] - item["time"]) <= 1e-10:
            merged[-1]["end_time"] = item["end_time"]
        else:
            merged.append(item)
    return merged


class SectionDecoder:
    """Causal cleared-decision history and delayed wall duration validation.

    ``step`` accepts a categorical wall ID and a 12-cell bomb decision. It must
    receive the query order returned by :func:`candidates`. ``finish`` returns
    final merged/cleared objects without rewriting earlier decision history,
    including decisions in pending runs ultimately dropped as too short.
    """

    def __init__(self, state, duration, vocab, options=None):
        self.options = options or {}
        self.policy = policy(options)
        self.clearance = self.policy["clearance"]
        self.vocab = vocab["walls"]
        self.index = ClearanceIndex(state, duration, self.clearance)
        self.history, self.walls, self.bombs = [], [], []
        self._wall_ends, self._bomb_times = [], [[] for _ in range(12)]
        self.rejections = Counter()
        self.raw_wall_sections = self.raw_bombs = self.accepted_wall_sections = 0
        self._last_query = (-math.inf, -1)

    def _nearby_walls(self, time, recovery=0.):
        start = bisect_left(self._wall_ends, time - recovery)
        return self.walls[start:]

    def eligible_bomb_cells(self, time):
        cells = self.index.bomb_cells(time)
        for wall in self._nearby_walls(time):
            if wall["time"] <= time <= wall["end_time"]:
                for cell in range(12):
                    if _contains(wall, cell % 4, cell // 4):
                        cells[cell] = False
        for cell, times in enumerate(self._bomb_times):
            if times and time - times[-1] <= self.clearance["min_time_diff"]:
                cells[cell] = False
        return cells

    def step(self, candidate, wall=0, bombs=None):
        kind, time = candidate["kind"], float(candidate["time"])
        order = (time, 0 if kind == "wall" else 1)
        if kind not in ("wall", "bomb") or order <= self._last_query:
            raise ValueError("Hazard decoder requires chronological wall-then-bomb queries")
        self._last_query = order
        accepted_wall, accepted_bombs = 0, np.zeros(12, np.float32)
        if kind == "wall" and wall:
            if isinstance(wall, bool) or int(wall) != wall or not 0 < wall < len(self.vocab):
                raise ValueError("Wall decision is outside the frozen vocabulary")
            self.raw_wall_sections += 1
            item = {**deepcopy(self.vocab[int(wall)]), "time": time,
                    "end_time": float(candidate["end_time"]), "template_id": int(wall)}
            continuation = bool(self.walls and _geometry(self.walls[-1]) == _geometry(item)
                                and abs(self.walls[-1]["end_time"] - time) <= 1e-10)
            previous = self._nearby_walls(time, max(self.clearance["obstacle_time_gap"]))
            if continuation:
                previous = previous[:-1]
            reason = self.index.wall_reason(item, previous)
            if reason:
                self.rejections[reason] += 1
            else:
                accepted_wall = int(wall)
                self.accepted_wall_sections += 1
                if continuation:
                    self.walls[-1]["end_time"] = item["end_time"]
                    self._wall_ends[-1] = item["end_time"]
                else:
                    self.walls.append(item); self._wall_ends.append(item["end_time"])
        elif kind == "bomb":
            decisions = np.asarray(np.zeros(12) if bombs is None else bombs).reshape(12)
            if not np.isfinite(decisions).all() or np.any((decisions != 0) & (decisions != 1)):
                raise ValueError("Bomb decisions must be twelve binary cells")
            self.raw_bombs += int(decisions.sum())
            accepted_bombs = (decisions.astype(bool) & self.eligible_bomb_cells(time)).astype(np.float32)
            self.rejections["bomb_clearance"] += int(decisions.sum() - accepted_bombs.sum())
            for cell in np.flatnonzero(accepted_bombs):
                self.bombs.append({"time": time, "x": int(cell % 4), "y": int(cell // 4)})
                self._bomb_times[int(cell)].append(time)
        self.history.append({"time": time, "kind": kind,
                             "value": [accepted_wall, *accepted_bombs.tolist()]})
        length = int(self.options.get("history_length", 32))
        # Keep one spare decision: the current wall is a same-time sibling of
        # the bomb query and must not displace any strictly earlier history.
        del self.history[:-(length + 1)]
        return {"wall": accepted_wall, "bombs": accepted_bombs}

    def finish(self):
        walls, ends, reasons = [], [], Counter()
        for wall in self.walls:
            # Full merged walls receive the same frozen validation once more.
            lo = bisect_left(ends, wall["time"] - max(self.clearance["obstacle_time_gap"]))
            reason = self.index.wall_reason(wall, walls[lo:], minimum=True)
            if reason:
                reasons[reason] += 1
            else:
                walls.append(deepcopy(wall))
                ends.append(wall["end_time"])
        bombs = []
        for bomb in self.bombs:
            start = bisect_left(ends, bomb["time"])
            if any(wall["time"] <= bomb["time"] <= wall["end_time"]
                   and _contains(wall, bomb["x"], bomb["y"]) for wall in walls[start:start + 2]):
                reasons["final_bomb_wall_collision"] += 1
            else:
                bombs.append(deepcopy(bomb))
        return {"walls": walls, "bombs": bombs,
                "diagnostics": {"policy_version": 2, "wall_section_proposals": self.raw_wall_sections,
                                "accepted_wall_sections": self.accepted_wall_sections,
                                "coalesced_wall_runs": len(self.walls), "bomb_proposals": self.raw_bombs,
                                "final_walls": len(walls), "final_bombs": len(bombs),
                                "section_rejections": dict(self.rejections), "final_rejections": dict(reasons)}}


def rollout(song, state, vocab, options, predict, encoder=None, progress=None):
    """Run a predictor with the shared decoder, updating only walls and bombs.

    ``predict(inputs)`` returns wall probabilities [1,C] and bomb probabilities
    [1,12]. ``progress(index,total)`` is optional. Returned diagnostics separate
    section proposals, accepted decisions, merged runs and final object counts.
    """
    from .contracts import build_inputs

    base = getattr(song, "base", song)
    queries = candidates(base.metadata["duration"], base.metadata["bpm"], options)
    decoder = SectionDecoder(state, base.metadata["duration"], vocab, options)
    threshold = float(options.get("bomb_threshold", .5))
    if not 0 <= threshold <= 1:
        raise ValueError("Bomb threshold must be between zero and one")
    for index, query in enumerate(queries):
        output = predict(build_inputs("hazards", song, state, query, decoder.history, encoder, options))
        wall, bombs = np.asarray(output["wall"]), np.asarray(output["bombs"])
        if (wall.shape != (1, len(vocab["walls"])) or bombs.shape != (1, 12)
                or not np.isfinite(wall).all() or not np.isfinite(bombs).all()
                or np.any((wall < 0) | (wall > 1)) or np.any((bombs < 0) | (bombs > 1))
                or not np.allclose(wall.sum(axis=-1), 1., atol=1e-4)):
            raise ValueError("Invalid hazard predictor probabilities")
        decoder.step(query, int(wall[0].argmax()), bombs[0] >= threshold)
        if progress:
            progress(index, len(queries))
    final = decoder.finish()
    state.update(walls=final["walls"], bombs=final["bombs"])
    return {**final["diagnostics"], "queries": len(queries), "wall_queries": len(queries) // 2,
            "bomb_queries": len(queries) // 2}


def _overlapping_sections(boundaries, record, bpm=None):
    """Conservative mask for an unsupported original, retaining its true span."""
    numeric = (int, float, np.integer, np.floating)
    start = record.get("time")
    if isinstance(start, (bool, np.bool_)) or not isinstance(start, numeric) or not np.isfinite(start):
        return range(len(boundaries) - 1)
    start, end = float(start), record.get("end_time")
    if isinstance(end, (bool, np.bool_)) or not isinstance(end, numeric) or not np.isfinite(end) or end <= start:
        duration = record.get("duration")
        if isinstance(duration, (bool, np.bool_)) or not isinstance(duration, numeric) or not np.isfinite(duration):
            # Geometry can fail before ingestion stores normalized duration.
            # Read the preserved raw duration only for a conservative mask.
            raw = record.get("raw", {})
            duration = raw.get("_duration", raw.get("d")) if isinstance(raw, dict) else None
            finite = not isinstance(duration, (bool, np.bool_)) and isinstance(duration, numeric) and np.isfinite(duration)
            if finite and duration <= 0:
                duration = 0.
            else:
                valid = (finite and isinstance(bpm, numeric) and not isinstance(bpm, (bool, np.bool_))
                         and np.isfinite(bpm) and bpm > 0)
                duration = float(duration) * 60. / bpm if valid else None
        end = start + max(0., duration) if duration is not None else boundaries[-1]
    lo = max(0, bisect_right(boundaries, start) - 1)
    hi = min(len(boundaries) - 1, bisect_left(boundaries, end))
    return range(lo, max(lo + 1, hi))


def build_examples(song, state, vocab, options, encoder=None):
    """Build masked section targets and causal histories with no source queries.

    Wall weights are [N]; bomb weights are [N,12]. Excluded objects mask their
    affected section/cell, so unsupported positives never become NONE negatives.
    Raw-source coverage and final supported-reference output are separate counts.
    """
    from .contracts import build_inputs
    from .targets import _stack

    duration, bpm = song.base.metadata["duration"], song.base.metadata["bpm"]
    queries = candidates(duration, bpm, options)
    boundaries = [query["time"] for query in queries if query["kind"] == "wall"] + [float(duration)]
    count = len(boundaries) - 1
    complete = float(song.diagnostics.get("source_objects_complete", False))
    decoder = SectionDecoder(state, duration, vocab, options)
    indices = {_geometry(item): index for index, item in enumerate(vocab["walls"]) if index}
    assigned = [dict() for _ in range(count)]
    wall_masks = np.full(count, complete, np.float32)
    bomb_masks = np.full((count, 12), complete, np.float32)
    bomb_targets = np.zeros((count, 12), np.float32)
    bomb_ids = {}
    exclusions, errors = [], []
    wall_source_sections = {}

    def exclude(record, family, reason):
        exclusions.append({"id": record.get("id"), "family": family, "reason": reason})

    for record in song.objects.get("walls", []):
        start, end = float(record["time"]), float(record["end_time"])
        first, last = _nearest(boundaries, start), _nearest(boundaries, end)
        errors.append({"id": record["id"], "original_time": start, "original_end_time": end,
                       "time": boundaries[first], "end_time": boundaries[last],
                       "start_error_seconds": boundaries[first] - start,
                       "end_error_seconds": boundaries[last] - end})
        reason = geometry_reason(record, decoder.clearance)
        if not 0 <= start < end <= duration + 1e-7:
            reason = "invalid_interval"
        elif end - start + 1e-12 < decoder.clearance["obstacle_min_duration"]:
            reason = "short_original_wall"
        elif reason is None and _geometry(record) not in indices:
            reason = "template_outside_training_vocabulary"
        elif reason is None and first >= last:
            reason = "quantization_collapsed_wall"
        if reason:
            exclude(record, "walls", reason)
            for index in set(_overlapping_sections(boundaries, record, bpm)) | set(range(first, last)):
                if index < count:
                    wall_masks[index] = 0
            continue
        wall_source_sections[record["id"]] = set(range(first, last))
        category = indices[_geometry(record)]
        for index in range(first, last):
            assigned[index].setdefault(category, []).append(record["id"])
    for record in song.objects.get("bombs", []):
        index = _nearest(boundaries[:-1], float(record["time"]))
        errors.append({"id": record["id"], "original_time": record["time"], "time": boundaries[index],
                       "start_error_seconds": boundaries[index] - record["time"]})
        if not _integer_geometry(record, ("x", "y")) or not (0 <= record["x"] < 4 and 0 <= record["y"] < 3):
            exclude(record, "bombs", "invalid_geometry"); bomb_masks[index] = 0
            continue
        cell = int(4 * record["y"] + record["x"])
        bomb_targets[index, cell] = 1
        bomb_ids.setdefault((index, cell), []).append(record["id"])
    for record in song.objects.get("ignored", []):
        family = record.get("family")
        if family == "walls":
            for index in _overlapping_sections(boundaries, record, bpm):
                if index < count:
                    wall_masks[index] = 0
            exclude(record, "walls", "ignored_source_object")
        elif family in ("bombs", "notes"):
            rows = range(count) if "time" not in record else [_nearest(boundaries[:-1], record["time"])]
            cell = int(4 * record["y"] + record["x"]) if _integer_geometry(record, ("x", "y")) and 0 <= record["x"] < 4 and 0 <= record["y"] < 3 else None
            for index in rows:
                if cell is None or (family == "notes" and source_family(record) != "bombs"):
                    bomb_masks[index] = 0
                else:
                    bomb_masks[index, cell] = 0
            exclude(record, source_family(record), "ignored_source_object")
    inputs, walls, bombs, ww, bw = [], [], [], [], []
    supported_wall_sections, supported_bombs = {}, set()
    for query in queries:
        index = query["section_index"]
        inputs.append(build_inputs("hazards", song, state, query, decoder.history, encoder, options))
        wall, cells, wall_weight, bomb_weight = 0, np.zeros(12, np.float32), 0., np.zeros(12, np.float32)
        if query["kind"] == "wall":
            wall_weight = wall_masks[index]
            if len(assigned[index]) > 1:
                wall_weight = 0.
                for ids in assigned[index].values():
                    for identity in ids:
                        exclude({"id": identity}, "walls", "ambiguous_geometry_overlap")
            elif assigned[index] and wall_weight:
                wall = next(iter(assigned[index]))
            accepted = decoder.step(query, wall)
            if wall and accepted["wall"] != wall:
                wall_weight = 0.
                for identity in assigned[index][wall]:
                    exclude({"id": identity}, "walls", "section_clearance")
            if wall_weight and accepted["wall"]:
                for identity in assigned[index][wall]:
                    supported_wall_sections.setdefault(identity, set()).add(index)
        else:
            available = decoder.eligible_bomb_cells(query["time"])
            bomb_weight = bomb_masks[index] * available
            cells = bomb_targets[index] * (bomb_weight > 0)
            accepted = decoder.step(query, bombs=cells)
            for cell in np.flatnonzero(bomb_targets[index]):
                if bomb_weight[cell] and accepted["bombs"][cell]:
                    supported_bombs.update(bomb_ids[(index, int(cell))])
                else:
                    for identity in bomb_ids[(index, int(cell))]:
                        exclude({"id": identity}, "bombs", "bomb_clearance_or_source_mask")
        walls.append(wall); bombs.append(cells); ww.append(wall_weight); bw.append(bomb_weight)
    template = inputs[0]
    examples = _stack("hazards", [query["time"] for query in queries], inputs,
                      {"wall": np.asarray(walls, np.int32), "bombs": np.asarray(bombs, np.float32).reshape(-1, 12)},
                      {"wall": np.asarray(ww, np.float32), "bombs": np.asarray(bw, np.float32).reshape(-1, 12)}, template)
    final = decoder.finish()
    # Deduplicate object reasons arising from several adjacent masked sections.
    exclusions = list({(item["family"], item["id"], item["reason"]): item for item in exclusions}.values())
    parsed_counts = {family: len(song.objects.get(family, [])) for family in ("walls", "bombs")}
    ignored_counts = Counter(source_family(record) for record in song.objects.get("ignored", []))
    diagnostics = {"hazard_query_policy": "quarter_beat_sections_and_boundary_bombs_v2",
                   "hazard_query_provenance": deepcopy(queries), "hazard_policy": deepcopy(decoder.policy),
                   "hazard_quantization_errors": errors, "hazard_source_exclusions": exclusions,
                   "hazard_exclusion_counts": dict(Counter(item["reason"] for item in exclusions)),
                   "hazard_raw_source_counts": {family: count + ignored_counts[family] for family, count in parsed_counts.items()},
                   "hazard_parsed_source_counts": parsed_counts,
                   "hazard_supported_source_ids": {"walls": sorted(supported_wall_sections), "bombs": sorted(supported_bombs)},
                   "hazard_supported_reference": {"wall_source_objects": len(supported_wall_sections),
                                                  "fully_supported_wall_source_objects": sum(supported_wall_sections.get(identity, set()) == sections for identity, sections in wall_source_sections.items()),
                                                  "wall_sections": sum(1 for wall, weight in zip(walls, ww) if wall and weight),
                                                  "bomb_source_objects": len(supported_bombs),
                                                  "bomb_cells": int(np.asarray(bombs).sum()),
                                                  "final_walls": len(final["walls"]), "final_bombs": len(final["bombs"])},
                   "hazard_reference_rollout": final["diagnostics"],
                   "hazard_scheduler_coverage": {family: {"source_objects": len(song.objects.get(family, [])),
                                                          "matched_source_objects": len(supported_wall_sections) if family == "walls" else len(supported_bombs),
                                                          "unmatched_source_ids": [record["id"] for record in song.objects.get(family, []) if record["id"] not in (supported_wall_sections if family == "walls" else supported_bombs)]}
                                                  for family in ("walls", "bombs")}}
    diagnostics["unscheduled_walls"] = diagnostics["hazard_scheduler_coverage"]["walls"]["unmatched_source_ids"]
    diagnostics["unscheduled_bombs"] = diagnostics["hazard_scheduler_coverage"]["bombs"]["unmatched_source_ids"]
    return examples, diagnostics
