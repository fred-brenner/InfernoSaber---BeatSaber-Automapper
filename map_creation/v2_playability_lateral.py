"""Human-calibrated repeated M1 horizontal stress, without runtime corpus I/O.

The original single-arrival M1 stress remains unchanged. This component sees
consecutive fast horizontal reversals even when each arrival stays below its
individual stress threshold. Times are elapsed seconds; dots reset exposure.
"""

import json
from pathlib import Path


_REFERENCE = json.loads(Path(__file__).with_name("v2_playability_lateral_reference.json").read_text())
_BANDS = ("<3", "3-5", "5-7", "7+")


def horizontal_run_cutoff(nps):
    """Frozen run-tail cutoff in the caller's original full-map NPS band."""
    band = _BANDS[sum(nps >= bound for bound in (3,5,7))]
    return _REFERENCE["cutoffs_by_nps_band"][band]


def consecutive_horizontal_load(notes, nps=None):
    """Count consecutive fast d2/d3 reversals at every eligible arrow arrival.

    Eligible arrivals are every retained non-dot note with a prior same-hand
    retained note, including zero load and arrival after a dot. Anchored 20ms
    stacks keep their first representative. A reversal needs the exact other
    horizontal direction and gap <=.25 seconds. Twelve alternating horizontal
    notes therefore peak at 11, while any slow, non-horizontal, same-direction
    or dot interruption resets the load. Notes and annotations are untouched.
    """
    ordered = sorted(notes, key=lambda note:(note['time'],note.get('id',0)))
    previous, previous_ids, loads = {}, {}, {0:0,1:0}
    severities, identities = [], []
    retained = 0
    for i, note in enumerate(ordered):
        hand = note['hand']
        before = previous.get(hand)
        gap = note['time']-before['time'] if before is not None else float('inf')
        if gap <= .02+1e-9:
            continue
        retained += 1
        reversing = (before is not None and before['d'] in (2,3)
                     and note['d'] == 5-before['d'] and gap <= .25+1e-9)
        loads[hand] = loads[hand]+1 if reversing else 0
        if before is not None and note['d'] != 8:
            severities.append(loads[hand])
            identities.append([previous_ids[hand],note.get('id',i)])
        previous[hand] = note
        previous_ids[hand] = note.get('id',i)
    if nps is None:
        span = ordered[-1]['time']-ordered[0]['time'] if ordered else 0.
        nps = retained/span if span > 0 else 0.
    nps = float(nps)
    band = _BANDS[sum(nps >= bound for bound in (3,5,7))]
    cutoff = horizontal_run_cutoff(nps)
    return dict(reference_version=_REFERENCE['version'],nps=nps,nps_band=band,
        budget=_REFERENCE['budget'],cutoff=cutoff,eligible_events=len(severities),
        flagged_events=sum(value > cutoff for value in severities),severities=severities,
        event_note_ids=identities,max_reversals=max(severities,default=0),
        total_excess=sum(max(0.,value-cutoff)/max(1.,cutoff) for value in severities))
