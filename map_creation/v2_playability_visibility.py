"""Pure sustained upper-center exposure shared by priority repair and reports.

Times are elapsed seconds. Each eligible same-hand onset anchors a trailing
1.5-second phrase of four to eight 20ms-anchored onset groups. Stack members
share their onset's weight; all eligible zero-exposure phrases are retained.
This placement proxy is separate from the historical pairwise M2 definition.
"""
from collections import deque
from functools import lru_cache
import json
import math
from pathlib import Path

VERSION = 'visibility-upper-center-v1-20261108'
GROUP_SECONDS = .02
WINDOW_SECONDS = 1.5
WINDOW_ONSETS = 8
MIN_ONSETS = 4
BANDS = ('<3', '3-5', '5-7', '7+')
_TOL = 1e-9


def central_top(note):
    """Whether the real note cell lies in the upper-center sightline proxy."""
    return note['y'] == 2 and note['x'] in (1, 2)


def _groups(notes):
    groups = [[], []]
    for note in sorted(notes, key=lambda n: (n['time'], n['id'])):
        hand_groups = groups[note['hand']]
        if not hand_groups or note['time']-hand_groups[-1][0]['time'] > GROUP_SECONDS+_TOL:
            hand_groups.append([])
        hand_groups[-1].append(note)
    return groups


def _events(groups):
    events = []
    for hand, hand_groups in enumerate(groups):
        recent = deque()
        for group in hand_groups:
            anchor = group[0]
            time = anchor['time']
            recent.append(group)
            while len(recent) > WINDOW_ONSETS or recent[0][0]['time'] < time-WINDOW_SECONDS-_TOL:
                recent.popleft()
            if len(recent) < MIN_ONSETS:
                continue
            count = len(recent)
            members = [n for onset in recent for n in onset]
            weights = [1./(count*len(onset)) for onset in recent for _ in onset]
            severity = math.fsum(weight for n, weight in zip(members, weights) if central_top(n))
            events.append(dict(id=anchor['id'], time=time, start_time=recent[0][0]['time'],
                               hand=hand, onsets=count, group_ids=[n['id'] for n in group],
                               note_ids=[n['id'] for n in members], weights=weights, severity=severity))
    return sorted(events, key=lambda e: (e['time'], e['hand'], e['id']))


def visibility_events(notes):
    """Eligible phrase events with edit-stable endpoint IDs and note weights.

    ``note_ids`` and aligned ``weights`` describe the whole phrase; weights sum
    to one. ``group_ids`` contains only the endpoint onset's members, allowing
    a later inserted note to merge into an existing 20ms group. Eligibility and
    weights depend on time and hand, never on position or prescribed arrow.
    """
    return _events(_groups(notes))


@lru_cache(maxsize=1)
def _default_reference():
    return json.loads(Path(__file__).with_name('v2_playability_visibility_reference.json').read_text(encoding='utf8'))


def visibility_costs(notes, nps=None, reference=None):
    """JSON-safe calibrated excess, retaining every eligible zero event.

    A caller can freeze the full-map retained note-span ``nps`` during repair.
    Strict ``severity > cutoff`` flags the human tail; allowed exposure has
    exactly zero excess. Unsupported fits report unavailable flags and excess.
    No corpus, configuration, model or evaluation module is imported here.
    """
    notes = list(notes)
    groups = _groups(notes)
    if nps is None:
        span = max(n['time'] for n in notes)-min(n['time'] for n in notes) if notes else 0.
        nps = sum(map(len, groups))/span if span > 0 else 0.
    nps = float(nps)
    reference = _default_reference() if reference is None else reference
    if reference['event_version'] != VERSION:
        raise ValueError('Visibility reference and event definition versions differ')
    band = BANDS[sum(nps >= bound for bound in (3, 5, 7))]
    profile = reference['bands'][band]
    supported, cutoff = profile['supported'], profile['cutoff']
    events = _events(groups)
    for event in events:
        event['flagged'] = event['severity'] > cutoff if supported else None
        event['excess'] = max(0., event['severity']-cutoff) if supported else None
    return dict(version=VERSION, reference_version=reference['reference_version'],
                nps=nps, nps_band=band, cutoff=cutoff, budget=reference['budget'],
                supported=supported, fallback=profile.get('fallback', False),
                eligible_events=len(events), flagged_events=sum(e['flagged'] for e in events) if supported else None,
                excess=math.fsum(e['excess'] for e in events) if supported else None,
                max_severity=max((e['severity'] for e in events), default=0.), events=events)
