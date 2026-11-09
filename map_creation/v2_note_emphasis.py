"""Standalone v1-style note emphasis for the production v2 stream.

Retained directed anchors on isolated slow onset groups may grow into
double/triple same-time stacks along the anchor's arrow axis with the
original difficulty-scaled probabilities. Strictly propose/check/append:
no removals or edits, no new onset times, and every stack must clear the
frozen-baseline safety gates (occupancy, clearance, diagonal fraction,
speed cap, counter-direction, enabled playability policies); unsafe
candidates are rejected, never repaired.
"""

from copy import deepcopy
import math

import numpy as np

from tools.config import config
from tools.utils.numpy_shorts import get_factor_from_max_speed
from map_creation.sanity_check import calc_note_pos, calc_note_speed

_WARMUP_GROUPS = 4
_CENTER_CELLS = ((1, 1), (2, 1))
_TOL = 1e-9


def _ordered(notes):
    return sorted(notes, key=lambda n: (n["time"], n["id"]))


def _public(record):
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _onset_groups(retained, eps):
    """Chronological groups anchored to each group's first note."""
    groups = []
    for note in retained:
        if groups and note["time"] - groups[-1][0]["time"] <= eps + _TOL:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def _stack_cells(anchor, target):
    """Axis-contiguous candidate cell lists in v1 preference order."""
    positions = calc_note_pos([anchor["x"], anchor["y"], anchor["hand"], anchor["d"]])
    adjacents = positions[1:]
    candidates = []
    if target < 2:
        return candidates
    if target == 3:
        if len(adjacents) == 2:
            candidates.append(list(adjacents))
        elif len(adjacents) == 1:
            step = (adjacents[0][0] - anchor["x"], adjacents[0][1] - anchor["y"])
            second = [adjacents[0][0] + step[0], adjacents[0][1] + step[1]]
            if 0 <= second[0] < 4 and 0 <= second[1] < 3:
                candidates.append([adjacents[0], second])
    for cell in adjacents:
        if tuple(cell) not in _CENTER_CELLS:
            candidates.append([cell])
    return candidates


class _EmphasisGate:
    """Frozen pre-emphasis baselines; trials compare against unchanged rules."""

    def __init__(self, state, song, eps):
        self.state, self.song, self.eps = state, song, eps
        self.notes = state["notes"]
        self.chains = state.get("chains", [])
        self.baseline = [dict(note) for note in self.notes]
        self.base_notes = {note["id"]: note for note in self.baseline}
        self.expanded = {note["anchor_id"] for note in self.notes
                         if note.get("anchor_id") is not None}
        self.expanded |= {chain.get("anchor_id") for chain in self.chains
                          if chain.get("anchor_id") is not None}
        self.p3 = float(np.clip(config.emphasize_beats_3
                                + config.emphasize_beats_3_fact * config.max_speed, 0., 1.))
        self.p2 = float(np.clip(config.emphasize_beats_2
                                + config.emphasize_beats_2_fact * config.max_speed, 0., 1. - self.p3))
        from map_creation.v2_playability_priority import _clearance_findings
        self.clearance = _clearance_findings(self.baseline)
        diagonals = sum(note["d"] in (4, 5, 6, 7) for note in self.baseline)
        self.diagonal_bound = max(diagonals / len(self.baseline),
                                  float(getattr(config, "v2_diagonal_max_fraction", 1.) or 0.))
        from map_creation.v2_postprocessing import speed_cap_scale
        self.scale = speed_cap_scale()
        self.requested, self.effective, self.base_violations = {}, {}, set()
        if self.scale > 0:
            self._speed_baselines()
        self.playability = bool(getattr(config, "v2_playability_enabled", False))
        self.priority = self.playability and bool(getattr(config, "v2_playability_priority", False))
        self.reach = self.playability and bool(getattr(config, "v2_playability_reach", False))
        if self.priority or self.reach:
            from map_creation.v2_playability_reach import note_span_nps
            self.nps = note_span_nps(self.baseline)
        if self.priority:
            self._priority_baselines()
        elif self.reach:
            from map_creation.v2_playability_reach import reach_costs
            self.base_reach = reach_costs(self.baseline, self.nps)

    def _speed_baselines(self):
        from map_creation.v2_playability_policy import SpeedGroups
        ordered = _ordered(self.baseline)
        count = max(int(getattr(config, "v2_speed_cap_start_notes", 30) or 0), 0)
        first = float(getattr(config, "v2_speed_cap_start_factor", .35) or 0.)
        ramp = np.linspace(first, 1., count) if count else ()
        for index, note in enumerate(ordered):
            speed = config.max_speed * (ramp[index] if index < count else 1.)
            self.requested[note["id"]] = (speed * get_factor_from_max_speed(speed, .5, 1)
                                          * self.scale)
        provisional = SpeedGroups(ordered, lambda note: math.inf, self.eps)
        for index, note in enumerate(ordered):
            arrival = min((calc_note_speed([ordered[p][key] for key in ("x", "y", "hand", "d")],
                                          [note[key] for key in ("x", "y", "hand", "d")],
                                          note["time"] - ordered[p]["time"])
                           for p in provisional.previous[index]), default=0.)
            self.effective[note["id"]] = max(self.requested[note["id"]], arrival)
        groups = SpeedGroups(ordered, lambda note: self.effective[note["id"]], self.eps)
        self.base_violations = groups.violations(range(len(ordered)))

    def _priority_baselines(self):
        from map_creation import v2_playability_priority_metrics as metrics
        from map_creation.v2_playability_priority import _soft_bounds
        from map_creation.v2_playability_visibility import visibility_costs
        self.metrics = metrics
        self.bounds = _soft_bounds(self.nps)
        report = metrics.priority_metrics(self.baseline)
        self.base_prep = {event["id"]: event for event in report["preparation"]["events"]}
        self.base_axis = {event["id"]: event for event in report["axis"]["events"]}
        self.base_cross = {tuple(event["ids"]): event for event in report["crossing"]["events"]}
        visibility = visibility_costs(self.baseline, self.nps)
        self.visibility_cutoff = visibility["cutoff"]
        self.visibility_member = {identity: event["severity"] for event in visibility["events"]
                                  for identity in event["group_ids"]}

    def eligible(self, anchor):
        """A retained directed note, sole same-hand at its unexpanded onset."""
        if anchor.get("kind") != "retained" or anchor.get("d", 8) not in range(8):
            return False
        if anchor["id"] in self.expanded:
            return False
        eps = self.eps
        return not any(other is not anchor and other["hand"] == anchor["hand"]
                       and abs(other["time"] - anchor["time"]) <= eps + _TOL
                       for other in self.notes)

    def copy(self, anchor, cell, identity):
        return {"id": identity, "time": anchor["time"], "hand": anchor["hand"],
                "x": int(cell[0]), "y": int(cell[1]), "d": anchor["d"],
                "kind": "copy", "source": "emphasis", "anchor_id": anchor["id"]}

    def check(self, proposed):
        """Return ``None`` when the whole stack may append, else a reason."""
        trial = self.notes + proposed
        if self._counter_violation(trial, proposed):
            return "counter_direction"
        diagonals = sum(note["d"] in (4, 5, 6, 7) for note in trial)
        if diagonals / len(trial) > self.diagonal_bound + _TOL:
            return "diagonal_fraction"
        reason = self._occupancy(trial, proposed)
        if reason is not None:
            return reason
        from map_creation.v2_playability_priority import _clearance_findings
        findings = _clearance_findings(trial)
        if any(not findings[key] <= self.clearance[key] for key in findings):
            return "clearance"
        if self.scale > 0 and self._speed_violation(trial, proposed):
            return "speed_cap"
        if self.priority:
            return self._priority_violation(trial, proposed)
        if self.reach:
            return self._reach_violation(trial, proposed)
        return None

    def _counter_violation(self, trial, proposed):
        threshold = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
        if (threshold <= 0 or len(trial) < 2
                or getattr(config, "v2_fast_counter_direction", "recut") != "recut"):
            return False
        from map_creation.v2_postprocessing import _counter_direction
        ordered = _ordered(trial)
        fast = set()
        for hand in (0, 1):
            seq = [note for note in ordered if note["hand"] == hand]
            for first, second in zip(seq, seq[1:]):
                if second["time"] - first["time"] < threshold:
                    fast.update((first["id"], second["id"]))
        return any(note["id"] in fast and _counter_direction(note) for note in proposed)

    def _occupancy(self, trial, proposed):
        """Copies/chains survive ``validate_expansions`` on a private scratch."""
        from map_creation.v2_postprocessing import validate_expansions
        scratch = {"notes": deepcopy(trial), "chains": deepcopy(self.chains)}
        validate_expansions(scratch, self.song, logger=None)
        kept = {note["id"]: note for note in scratch["notes"]}
        if len(kept) != len(trial) or any(note["id"] not in kept for note in proposed):
            return "cell_occupied"
        if any(kept[note["id"]] != note for note in trial):
            return "baseline_mutation"
        if [_public(chain) for chain in scratch["chains"]] != [_public(chain) for chain in self.chains]:
            return "cell_occupied"
        return None

    def _speed_violation(self, trial, proposed):
        from map_creation.v2_playability_policy import SpeedGroups
        ordered = sorted(trial, key=lambda note: (note["time"], note["hand"], note["id"]))
        caps = dict(self.effective)
        for note in proposed:
            caps[note["id"]] = self.requested[note["anchor_id"]]
        groups = SpeedGroups(ordered, lambda note: caps[note["id"]], self.eps)
        index = {note["id"]: i for i, note in enumerate(ordered)}
        changed = [index[note["id"]] for note in proposed]
        return bool(groups.violations(changed) - self.base_violations)

    def _priority_violation(self, trial, proposed):
        from map_creation.v2_playability_priority import _specific_components
        from map_creation.v2_playability_reach import forbidden_cut
        from map_creation.v2_playability_visibility import visibility_costs
        if any(forbidden_cut(note) for note in proposed):
            return "forbidden_cut"
        report = self.metrics.priority_metrics(trial)
        prep = {event["id"]: event for event in report["preparation"]["events"]}
        axis = {event["id"]: event for event in report["axis"]["events"]}
        cross = {tuple(event["ids"]): event for event in report["crossing"]["events"]}
        if any(event["hard"] for event in prep.values()):
            return "priority_hard"
        if any(event["hard"] for event in cross.values()):
            return "priority_hard"
        notes_by_id = {note["id"]: note for note in trial}
        for key, events, base in (("preparation", prep, self.base_prep),
                                  ("axis", axis, self.base_axis),
                                  ("crossing", cross, self.base_cross)):
            for identity, event in events.items():
                original = base.get(identity)
                allowance = max(original["total"], self.bounds[key]) if original else self.bounds[key]
                if event["total"] > allowance + 1e-9:
                    return "priority_" + key
        for identity, event in prep.items():
            new = _specific_components(notes_by_id[identity], event)
            if identity in self.base_prep:
                old = _specific_components(self.base_notes[identity], self.base_prep[identity])
                allowed = {key: max(old[key], self.bounds[key]) for key in new}
            else:
                allowed = self.bounds
            if any(new[key] > allowed[key] + 1e-9 for key in new):
                return "specific_tail"
        visibility = visibility_costs(trial, self.nps)
        for event in visibility["events"]:
            original = max((self.visibility_member.get(identity, 0.)
                            for identity in event["group_ids"]), default=0.)
            if event["severity"] > max(original, self.visibility_cutoff) + 1e-9:
                return "visibility"
        return None

    def _reach_violation(self, trial, proposed):
        from map_creation.v2_playability_reach import forbidden_cut, reach_costs
        if any(forbidden_cut(note) for note in proposed):
            return "forbidden_cut"
        report = reach_costs(trial, self.nps)
        if set(report["hard_ban_ids"]) - set(self.base_reach["hard_ban_ids"]):
            return "forbidden_cut"
        for key, metric in report["metrics"].items():
            if metric["excess"] > self.base_reach["metrics"][key]["excess"] + 1e-8:
                return "reach_" + key
        return None


def apply_note_emphasis(state, song, *, rng=None, logger=None, gate_factory=None):
    """Append v1-style same-time stacks to isolated slow swings in place."""
    diagnostics = {"applied": False, "eligible_anchors": 0, "added_notes": 0,
                   "doubles": 0, "triples": 0, "rejected": {}}
    if (getattr(config, "v2_cleanup_note_heuristics", True)
            or not getattr(config, "emphasize_beats_flag", False)
            or getattr(config, "single_notes_only_flag", False)):
        return diagnostics
    notes = state.get("notes")
    if not notes:
        return diagnostics
    eps = float(getattr(config, "min_time_diff", .02))
    retained = _ordered(note for note in notes if note.get("kind") == "retained")
    groups = _onset_groups(retained, eps)
    if len(groups) <= _WARMUP_GROUPS:
        return diagnostics
    group_times = np.asarray([group[0]["time"] for group in groups])
    gaps = np.r_[1., np.diff(group_times)]
    wait = float(np.quantile(gaps, float(getattr(config, "emphasize_beats_quantile", .65)))) + .05
    gate = (gate_factory(state, song, eps) if gate_factory is not None
            else _EmphasisGate(state, song, eps))
    diagnostics["applied"] = True
    next_id = max([note["id"] for note in notes]
                  + [chain["id"] for chain in state.get("chains", [])], default=-1) + 1
    draw = np.random if rng is None else rng
    for index, group in enumerate(groups):
        if index < _WARMUP_GROUPS or gaps[index] < wait:
            continue
        for anchor in group:
            if not gate.eligible(anchor):
                continue
            diagnostics["eligible_anchors"] += 1
            rd = float(np.asarray(draw.random()))
            target = 3 if rd > 1 - gate.p3 else 2 if rd > 1 - gate.p3 - gate.p2 else 0
            if not target:
                continue
            for cells in _stack_cells(anchor, target):
                proposed = [gate.copy(anchor, cell, next_id + offset)
                            for offset, cell in enumerate(cells)]
                reason = gate.check(proposed)
                if reason is None:
                    notes.extend(proposed)
                    next_id += len(proposed)
                    key = "triples" if len(proposed) == 2 else "doubles"
                    diagnostics[key] += 1
                    diagnostics["added_notes"] += len(proposed)
                    break
                diagnostics["rejected"][reason] = diagnostics["rejected"].get(reason, 0) + 1
    return diagnostics
