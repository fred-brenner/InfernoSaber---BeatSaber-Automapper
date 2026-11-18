"""Flow signature extraction, specific style counters, reference distance.

Pure reporting surface: nothing here mutates notes or feeds back into
generation. The counters mirror the derivation script
(``devin-artifacts/flow-signature-20261118/derive_metrics.py``) so a
generated map can be compared against the pooled human-reference
constants committed in ``reference_signature.json`` next to this file.

Notes are dicts with keys ``time`` (elapsed seconds), ``hand`` (0/1),
``x`` (0..3), ``y`` (0..2), ``d`` (0..8, 8 = dot) and ``id``. Same-hand
onset groups anchor on their first member: a note joins the open group
while ``time - group[0].time <= EPS + TOL``.

Band upper bounds are mixed by design: ``<.1`` and ``.1-.2`` are
exclusive on top while ``.2-.4`` and ``.4-.6`` include their upper edge.

Deviation from the derivation script, deliberately: its per-hand loop
overwrote each band Counter (``dict.update``), so its ``bands`` blocks
described hand 1 only. That was a bug — ``signature()`` here pools the
transition histograms over both hands, exactly like run lengths,
occupancy, ABAB windows and the specific counters already did. The
committed pooled constants are generated with the corrected pooling.
"""
from collections import Counter
import json
import math
from pathlib import Path

EPS = .02          # onset-group tolerance in seconds (fixed, not config)
TOL = 1e-9         # float comparison slack
FAST_GAP = .4      # "fast follower" / run / ABAB window upper gap
RUN_CAP = 8        # run-length histogram cap

BANDS = (.1, .2, .4, .6, math.inf)              # per-band upper bounds
BAND_NAMES = ("<.1", ".1-.2", ".2-.4", ".4-.6", ">.6")

_LANE_KEYS = ("lane:same", "lane:adjacent", "lane:wide")
_DY_KEYS = tuple(f"dy:{k}" for k in range(-2, 3))
_STEP_KEYS = tuple(f"step:{k}" for k in range(5))
_RUN_KEYS = tuple(str(k) for k in range(1, RUN_CAP + 1))
_CELL_KEYS = tuple(f"{x},{y}" for x in range(4) for y in range(3))
_ABAB_KEYS = ("alternating", "other")

# Distance weights from the flow-signature plan, Task 1.
_W_BAND_MASS = 1.
_W_LANE, _W_STEP, _W_DY = 3., 2., 1.
_W_RUNS, _W_OCCUPANCY, _W_ABAB = 2., 1., 1.

# Mirrors v2_postprocessing._DIRECTION_ANGLE/_step_distance. Replicated
# instead of imported: that module pulls aubio/librosa transitively,
# which a reporting path should not need.
_DIRECTION_ANGLE = {3: 0, 5: 1, 0: 2, 4: 3, 2: 4, 6: 5, 1: 6, 7: 7}

_SPECIFIC_KEYS = ("notes", "inner_y1_fast", "inner_y1_terminal",
                  "inner_y2_fast", "inner_y2_terminal",
                  "top_horizontal", "top_horizontal_fast",
                  "top_horizontal_terminal", "mid_horizontal_fast",
                  "inner_top_horizontal_fast", "bottom_down_fast")


def band_index(gap):
    """Gap band 0..4 for ``BANDS``: <.1, [.1,.2), [.2,.4], (.4,.6], >.6."""
    if gap < BANDS[0] - TOL:
        return 0
    if gap < BANDS[1] - TOL:
        return 1
    if gap <= BANDS[2] + TOL:
        return 2
    if gap <= BANDS[3] + TOL:
        return 3
    return 4


def _step_distance(a, b):
    """Smallest 45-degree angle between cut directions (0..4); dots are free."""
    if a == 8 or b == 8:
        return 4
    pa, pb = _DIRECTION_ANGLE[a], _DIRECTION_ANGLE[b]
    return min((pa - pb) % 8, (pb - pa) % 8)


def _onset_groups(notes, hand):
    """Anchored same-hand onset groups, sorted by (time, id)."""
    groups = []
    for n in sorted((n for n in notes if n.get("hand") == hand),
                    key=lambda n: (float(n["time"]), n.get("id", 0))):
        if not groups or n["time"] - groups[-1][0]["time"] > EPS + TOL:
            groups.append([])
        groups[-1].append(n)
    return groups


def _singleton(group):
    """Onset group usable for transitions: exactly one non-dot note."""
    return len(group) == 1 and group[0]["d"] != 8


def _band_dict(counter):
    out = {"pairs": counter["pairs"]}
    for key in _LANE_KEYS + _DY_KEYS + _STEP_KEYS + ("outer_pair:0", "outer_pair:1"):
        out[key] = counter[key]
    return out


def signature(notes):
    """Per-map flow signature: transition histograms, runs, occupancy, ABAB.

    Returns a JSON-ready dict:

    - ``bands``: per ``BAND_NAMES`` Counters of consecutive
      singleton-to-singleton same-hand transitions (gap > EPS), each with
      ``pairs`` plus lane type (``lane:same|adjacent|wide`` by |dx|),
      ``dy:-2..2``, ``step:0..4`` (direction step clamped to 4) and
      ``outer_pair:0|1`` (outer two-lane pair for that hand); pooled
      over both hands (see module docstring).
    - ``runs``: same-x run-length histogram over singleton notes with
      in-run gaps in (EPS, FAST_GAP], lengths capped at ``RUN_CAP``.
    - ``occupancy``: per-hand 12-cell ``"x,y"`` note counts (all notes,
      dots and stack members included).
    - ``abab4``: ``eligible`` 4-group windows (all singleton, all gaps in
      (EPS, FAST_GAP]) and how many alternate between adjacent lanes.
    - ``notes``: hand-0/1 note count.
    """
    bands = {name: Counter() for name in BAND_NAMES}
    runs = Counter()
    occupancy = {0: Counter(), 1: Counter()}
    abab4 = Counter()
    for hand in (0, 1):
        groups = _onset_groups(notes, hand)
        for group in groups:
            for n in group:
                occupancy[hand][(n["x"], n["y"])] += 1
        for ga, gb in zip(groups, groups[1:]):
            if not (_singleton(ga) and _singleton(gb)):
                continue
            a, b = ga[0], gb[0]
            gap = b["time"] - a["time"]
            if gap <= EPS + TOL:
                continue
            lane = ("same" if a["x"] == b["x"]
                    else "adjacent" if abs(b["x"] - a["x"]) == 1 else "wide")
            step = min(_step_distance(a["d"], b["d"]), 4)
            outer = int({a["x"], b["x"]} == ({0, 1} if hand == 0 else {2, 3}))
            bands[BAND_NAMES[band_index(gap)]].update(
                pairs=1,
                **{f"lane:{lane}": 1, f"dy:{b['y'] - a['y']}": 1,
                   f"step:{step}": 1, f"outer_pair:{outer}": 1})
        run = []
        for group in groups:
            n = group[0] if _singleton(group) else None
            if n is None or (run and (n["x"] != run[-1]["x"]
                             or not EPS < n["time"] - run[-1]["time"]
                             <= FAST_GAP + TOL)):
                if run:
                    runs[min(len(run), RUN_CAP)] += 1
                    run = []
            if n is not None:
                run.append(n)
        if run:
            runs[min(len(run), RUN_CAP)] += 1
        for i in range(len(groups) - 3):
            chunk = groups[i:i + 4]
            if not all(_singleton(g) for g in chunk):
                continue
            ns = [g[0] for g in chunk]
            if not all(EPS < b["time"] - a["time"] <= FAST_GAP + TOL
                       for a, b in zip(ns, ns[1:])):
                continue
            xs = [n["x"] for n in ns]
            abab4["eligible"] += 1
            abab4["alternating"] += int(
                xs[0] == xs[2] and xs[1] == xs[3]
                and abs(xs[0] - xs[1]) == 1)
    n_notes = sum(1 for n in notes if n.get("hand") in (0, 1))
    return {"bands": {name: _band_dict(bands[name]) for name in BAND_NAMES},
            "runs": {key: runs[int(key)] for key in _RUN_KEYS},
            "occupancy": {str(hand): {f"{x},{y}": occupancy[hand][(x, y)]
                                      for x in range(4) for y in range(3)}
                          for hand in (0, 1)},
            "abab4": {"eligible": abab4["eligible"],
                      "alternating": abab4["alternating"]},
            "notes": n_notes}


def specific_counts(notes):
    """Specific style counters (both hands; every group member counted).

    ``<key>_fast`` counts a note whose same-hand *onset group* is followed
    by another onset group within FAST_GAP — the follower's shape does
    not matter (dots and stacks still count as fast followers); the last
    group or a slower follower yields ``_terminal``.
    """
    counts = Counter()
    for hand in (0, 1):
        groups = _onset_groups(notes, hand)
        for i, group in enumerate(groups):
            nxt = groups[i + 1][0] if i + 1 < len(groups) else None
            next_gap = (nxt["time"] - group[0]["time"]
                        if nxt is not None else math.inf)
            continuing_fast = next_gap <= FAST_GAP + TOL
            for n in group:
                counts["notes"] += 1
                inner = n["x"] in (1, 2)
                if inner and n["y"] in (1, 2):
                    counts[f"inner_y{n['y']}_"
                           f"{'fast' if continuing_fast else 'terminal'}"] += 1
                if n["y"] == 2 and n["d"] in (2, 3):
                    counts["top_horizontal"] += 1
                    counts[f"top_horizontal_"
                           f"{'fast' if continuing_fast else 'terminal'}"] += 1
                if n["y"] == 1 and n["d"] in (2, 3) and continuing_fast:
                    counts["mid_horizontal_fast"] += 1
                if n["y"] == 0 and n["d"] == 1 and continuing_fast:
                    counts["bottom_down_fast"] += 1
                if (inner and n["y"] == 2 and n["d"] in (2, 3)
                        and continuing_fast):
                    counts["inner_top_horizontal_fast"] += 1
    return {key: counts[key] for key in _SPECIFIC_KEYS}


def _shares(counts, keys):
    """Histogram counts normalized over ``keys``; empty -> zero vector."""
    total = sum(counts.get(key, 0) for key in keys)
    if total <= 0:
        return dict.fromkeys(keys, 0.)
    return {key: counts.get(key, 0) / total for key in keys}


def _tv(a_counts, b_counts, keys):
    """Total variation .5 * sum|p - q| over the shared support."""
    pa, pb = _shares(a_counts, keys), _shares(b_counts, keys)
    return .5 * sum(abs(pa[key] - pb[key]) for key in keys)


def _occupancy_band(sig, hand):
    occ = sig.get("occupancy", {})
    return occ.get(str(hand)) or occ.get(hand) or {}


def _band_masses(sig):
    """Transition-pair counts per gap band (the band-mass histogram)."""
    bands = sig.get("bands", {})
    return {name: bands.get(name, {}).get("pairs", 0)
            for name in BAND_NAMES}


def signature_distance(sig, reference=None):
    """Weighted total-variation distance of ``sig`` vs the pooled reference.

    Every compared histogram is normalized to shares over its own
    support; TV(p, q) = .5 * sum|p_k - q_k| over the support keys. An
    empty histogram acts as a zero vector, so an absent distribution
    distances .5 from a populated reference one and 0 from an empty one.

        distance = 1 * TV(share of transition pairs across the 5 bands)
                 + sum over the 5 gap bands of
                       [ 3*TV(lane) + 2*TV(step) + 1*TV(dy) ]
                 + 2 * TV(same-x run lengths 1..8)
                 + 1 * sum over both hands of TV(12-cell occupancy)
                 + 1 * TV(abab4 {alternating, other})

    Range [0, 36]; 0 iff every distribution matches. The band-mass term
    captures *where in tempo* transitions sit (fast vs slow share);
    the per-band terms then compare the within-band lane/step/dy
    conditionals. ``outer_pair`` is collected for reporting but carries
    no distance weight. ``reference`` defaults to the committed pooled
    constants.
    """
    if reference is None:
        reference = load_reference()
    distance = _W_BAND_MASS * _tv(_band_masses(sig),
                                  _band_masses(reference), BAND_NAMES)
    for name in BAND_NAMES:
        band = sig.get("bands", {}).get(name, {})
        ref = reference.get("bands", {}).get(name, {})
        distance += (_W_LANE * _tv(band, ref, _LANE_KEYS)
                     + _W_STEP * _tv(band, ref, _STEP_KEYS)
                     + _W_DY * _tv(band, ref, _DY_KEYS))
    distance += _W_RUNS * _tv(sig.get("runs", {}),
                              reference.get("runs", {}), _RUN_KEYS)
    for hand in (0, 1):
        distance += _W_OCCUPANCY * _tv(_occupancy_band(sig, hand),
                                       _occupancy_band(reference, hand),
                                       _CELL_KEYS)
    distance += _W_ABAB * _tv(_abab_counts(sig.get("abab4", {})),
                              _abab_counts(reference.get("abab4", {})),
                              _ABAB_KEYS)
    return distance


def _abab_counts(abab4):
    eligible = abab4.get("eligible", 0)
    return {"alternating": abab4.get("alternating", 0),
            "other": eligible - abab4.get("alternating", 0)}


_REFERENCE = None
_REFERENCE_FILE = Path(__file__).resolve().parent / "reference_signature.json"


def load_reference(path=None):
    """Pooled reference signature; defaults to the committed JSON.

    The JSON is exactly the ``signature()`` output shape summed over the
    seven reference maps (see the module docstring for the band quirk).
    """
    global _REFERENCE
    if path is None:
        if _REFERENCE is None:
            _REFERENCE = load_reference(_REFERENCE_FILE)
        return _REFERENCE
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def signature_report(notes):
    """Bundle ``{"signature":…, "distance":…, "specific":…}`` for a note list."""
    sig = signature(notes)
    return {"signature": sig, "distance": signature_distance(sig),
            "specific": specific_counts(notes)}
