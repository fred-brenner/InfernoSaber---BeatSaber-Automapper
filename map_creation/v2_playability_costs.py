"""Versioned, pure geometry costs shared by combined generation and evaluation.

The legacy scorer keeps its original definitions. The current definition sees
temporally crossed upper notes at the same height and scores mirrored pairs
with the same overlap calculation as every other pair.
No configuration, audio, model, or corpus is imported here.
"""

VERSION = "combined-geometry-v3-20261108"
EPS, VISIBILITY_SECONDS, PAIR_SECONDS = .02, .4, .25
LATERAL_DIRECTIONS = frozenset((2, 3, 4, 5, 6, 7))

# Human-only fit: geometry-human-thresholds-v2.json in the external
# combined-reach-playtest-20261108 results. 51 curated highest-Standard maps;
# 7/27/7/10 maps by retained note-span NPS. Event denominators include zeros.
# The v2 fit met pooled/equal-map 10% M2 / 3% M4 budgets. M2 is unchanged;
# inherited M4 cutoffs are provisional after removing the mirror exemption.
# Refit M4 once the ongoing metric-definition review is complete.
HUMAN_POPULATION_SHA256 = "881fbe99af8046194ab89908ae5fc3ec0cb5084c74e4a98ef50c015e8f22f62d"
_M2_LIMIT = (0., .23188405797101555, .5413333333333413, .846153846153868)
_M4_LIMIT = (1., 2., 2., 2.)
_M3_NORMAL = ((.0264008922350484, .12177220153074939),
              (.09833399816010788, .22022142426113267),
              (.08257547186283129, .13744642908129692),
              (.06154479108944497, .11167191065108628))


def calibrated_limits(nps):
    """Frozen event cutoffs and the human map-level M3 normal range."""
    band = sum(nps >= bound for bound in (3, 5, 7))
    return {"m2": _M2_LIMIT[band], "m4": _M4_LIMIT[band], "m3_normal": _M3_NORMAL[band]}


def soft_event_cost(value, cutoff):
    """Prioritize the calibrated excess tail, retaining a weak raw tie-break.

    A subthreshold event is allowed, not intrinsically defective. The 2% raw
    term only prefers a clearer otherwise-equivalent placement; its scale is
    deliberately much smaller than an event exceeding the calibrated cutoff.
    """
    return (max(0., value - cutoff) + .02 * value) / (1 + cutoff)


def occlusion_severity(note, before, dt, direction=None, before_direction=None):
    """Nearest earlier note's time-weighted rise or crossed-upper obstruction.

    Normal hand order and same-hand streams receive the unchanged rise cost.
    For crossed opposite hands at equal upper height the coarse obstruction is
    half the row, with 50% extra overlap for a prescribed lateral swing axis.
    Simultaneous partners share depth and remain outside this sightline cost.
    """
    if before is None or not EPS < dt <= VISIBILITY_SECONDS:
        return 0.
    dx = abs(note["x"] - before["x"])
    if dx > 1:
        return 0.
    obstruction = max(0, note["y"] - before["y"])
    if note["hand"] != before["hand"] and note["y"] == before["y"] and note["y"] > 0:
        left, right = (note, before) if note["hand"] == 0 else (before, note)
        if left["x"] > right["x"]:
            direction = note["d"] if direction is None else direction
            before_direction = before["d"] if before_direction is None else before_direction
            lateral = direction in LATERAL_DIRECTIONS or before_direction in LATERAL_DIRECTIONS
            obstruction = max(obstruction, note["y"] / 2 * (1 + .5 * lateral))
    return obstruction * (1 - dx / 3) * (1 - dt / VISIBILITY_SECONDS)


def interference_severity(left, right, dt, left_volume, right_volume):
    """Cross-hand swept/travel intersection plus crossed-hand position cost."""
    return (len(left_volume & right_volume) + max(0, left["x"] - right["x"])) * max(0., 1 - dt / PAIR_SECONDS)


def m3_segment_cutoff(nps):
    """Human-only strict upper segment tail, <=3% pooled and equal-map.

    51 highest-Standard curated charts, 7/27/7/10 supported maps; fit and
    individual skips in complete-combined-playability-20261108/placement-calibration.
    Historical map-level quartiles remain descriptive, not event thresholds.
    """
    return (.6710053207609463, .8944271909999159,
            .6407474392457675, .7626860094880923)[sum(nps >= b for b in (3, 5, 7))]
