"""Append-only forward guards for learned expansion and emphasis.

New copies and chains are checked as one atomic batch against a shared
local index seeded from non-copy base notes. A rejected batch is removed
without touching the index, so later legal expansions stay unblocked.
Accepted additions are never moved or recut afterward.
"""
import math

import numpy as np

from tools.config import config
from map_creation.v2_forward_clearance import LocalNoteIndex
from map_creation.v2_forward_directions import DirectionGuard

_TOL = 1e-9
_PRIVATE = "_forward_guard"


def _forbidden(note):
    from map_creation.v2_playability_reach import forbidden_cut
    return forbidden_cut(note)


def _chain_samples(chain):
    span = chain["end_time"] - chain["time"]
    steps = max(int(chain["slices"]) - 1, int(math.ceil(span / .01)), 1)
    return [{"id": f"chain-{chain['id']}-{index}",
             "time": chain["time"] + alpha * span, "hand": chain["hand"],
             "x": chain["x"] + alpha * (chain["tx"] - chain["x"]),
             "y": chain["y"] + alpha * (chain["ty"] - chain["y"]),
             "d": chain["d"], "kind": "chain_sample",
             "chain_id": chain["id"], "anchor_id": chain["anchor_id"]}
            for index, alpha in enumerate((k / steps for k in range(1, steps + 1)), 1)]


def _guard(state):
    guard = state.get(_PRIVATE)
    if guard is None:
        base = [note for note in state.get("notes", [])
                if note.get("kind") != "copy"]
        index = LocalNoteIndex(base)
        guard = {"index": index,
                 "accepted_ids": {note["id"] for note in base},
                 "accepted_chains": set(),
                 "direction": DirectionGuard(base)}
        state[_PRIVATE] = guard
    return guard


def validate_expansions(state, song, logger=None):
    """Check new copies/chains atomically; rejected batches are removed."""
    guard = _guard(state)
    notes = list(state.get("notes", []))
    chains = list(state.get("chains", []))
    batch = [note for note in notes
             if note.get("kind") == "copy" and note["id"] not in guard["accepted_ids"]]
    new_chains = [chain for chain in chains
                  if chain["id"] not in guard["accepted_chains"]]
    samples = [sample for chain in new_chains for sample in _chain_samples(chain)]
    note_ids = {note["id"] for note in notes}
    anchored = (all(chain["anchor_id"] in note_ids for chain in new_chains)
                and all(note.get("anchor_id") in note_ids for note in batch))
    strict = (getattr(config, "v2_playability_enabled", False)
              and (getattr(config, "v2_playability_priority", False)
                   or getattr(config, "v2_playability_reach", False)))
    banned = strict and any(_forbidden(note) for note in batch)
    clear = (anchored and not banned
             and guard["index"].accepts_many([*batch, *samples])
             and guard["direction"].check(batch) is None)
    if not clear:
        drop = {note["id"] for note in batch}
        drop_chains = {chain["id"] for chain in new_chains}
        state["notes"] = [note for note in notes if note["id"] not in drop]
        state["chains"] = [chain for chain in chains if chain["id"] not in drop_chains]
        return {"rejected_copies": len(batch),
                "rejected_chains": len(new_chains)}
    if batch:
        guard["index"].add_many(batch)
        guard["direction"].add_many(batch)
        guard["accepted_ids"].update(note["id"] for note in batch)
    for chain in new_chains:
        guard["accepted_chains"].add(chain["id"])
    if samples:
        guard["index"].add_many(samples)
    return {"rejected_copies": 0, "rejected_chains": 0}


def drop_guard(state):
    state.pop(_PRIVATE, None)


class _ForwardGate:
    """Local-index emphasis gate: same protocol as the strict gate."""

    def __init__(self, state, song, eps):
        self.state, self.song, self.eps = state, song, eps
        self.notes = state["notes"]
        self.chains = state.get("chains", [])
        guard = _guard(state)
        self.index = guard["index"]
        self.direction = guard["direction"]
        for note in self.notes:
            if note["id"] not in guard["accepted_ids"]:
                self.index.add_many([note])
                self.direction.add_many([note])
                guard["accepted_ids"].add(note["id"])
        for chain in self.chains:
            if chain["id"] not in guard["accepted_chains"]:
                self.index.add_many(_chain_samples(chain))
                guard["accepted_chains"].add(chain["id"])
        self.known = set(guard["accepted_ids"])
        self.expanded = {note["anchor_id"] for note in self.notes
                         if note.get("anchor_id") is not None}
        self.expanded |= {chain.get("anchor_id") for chain in self.chains
                          if chain.get("anchor_id") is not None}
        self.p3 = float(np.clip(config.emphasize_beats_3
                                + config.emphasize_beats_3_fact * config.max_speed, 0., 1.))
        self.p2 = float(np.clip(config.emphasize_beats_2
                                + config.emphasize_beats_2_fact * config.max_speed,
                                0., 1. - self.p3))
        bound = float(getattr(config, "v2_diagonal_max_fraction", 1.) or 0.)
        diagonals = sum(note["d"] in (4, 5, 6, 7) for note in self.notes)
        self.diagonal_bound = max(diagonals / max(len(self.notes), 1), bound)

    def _sync(self):
        new = [note for note in self.notes if note["id"] not in self.known]
        if new:
            self.index.add_many(new)
            self.direction.add_many(new)
            self.known.update(note["id"] for note in new)
            guard = self.state.get(_PRIVATE)
            if guard is not None:
                guard["accepted_ids"].update(note["id"] for note in new)

    def eligible(self, anchor):
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
        self._sync()
        if (getattr(config, "v2_playability_enabled", False)
                and (getattr(config, "v2_playability_priority", False)
                     or getattr(config, "v2_playability_reach", False))
                and any(_forbidden(note) for note in proposed)):
            return "forbidden_cut"
        threshold = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
        if (threshold > 0
                and getattr(config, "v2_fast_counter_direction", "recut") == "recut"):
            from map_creation.v2_postprocessing import _counter_direction
            if any(self._fast_pair(note) and _counter_direction(note)
                   for note in proposed):
                return "counter_direction"
        if not self.index.accepts_many(proposed):
            return "cell_occupied"
        reason = self.direction.check(proposed)
        if reason is not None:
            return reason
        diagonals = sum(note["d"] in (4, 5, 6, 7) for note in self.notes) \
            + sum(note["d"] in (4, 5, 6, 7) for note in proposed)
        if diagonals / (len(self.notes) + len(proposed)) > self.diagonal_bound + _TOL:
            return "diagonal_fraction"
        return None

    def _fast_pair(self, note):
        threshold = float(getattr(config, "v2_fast_gap_seconds", .2) or 0.)
        return any(other["hand"] == note["hand"] and other is not note
                   and abs(other["time"] - note["time"]) < threshold
                   and other["id"] != note["anchor_id"]
                   for other in self.notes)


def apply_note_emphasis(state, song, *, rng=None, logger=None):
    from map_creation import v2_note_emphasis
    try:
        return v2_note_emphasis.apply_note_emphasis(
            state, song, rng=rng, logger=logger, gate_factory=_ForwardGate)
    finally:
        drop_guard(state)
