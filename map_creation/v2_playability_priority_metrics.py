"""Pure player-priority events shared by offline reports and opt-in repair.

Elapsed times are seconds. Human-reference thresholds are fitted separately;
the explicit hard preparation and crossing rules do not use those thresholds.
"""
from bisect import bisect_right
from functools import lru_cache
import math

VERSION = 'priority-events-v2-segments-20261108'
FAST_SECONDS = .25
PATH_SECONDS = 1.5
STACK_EPS = 1e-7
TIME_TOLERANCE = 1e-9
GEOMETRY_TOLERANCE = 1e-9
NEAR_MISS_MARGIN = .5
PREPARATION_WEIGHT = 4.0
SPECIFIC_WEIGHT = 1.0
EXPOSURE_TAU = .75
AXIS_BASE = .25
AXIS_FREQUENCY_WEIGHT = 3.0
AXIS_IRREGULARITY_WEIGHT = 2.0

STEP = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0),
        4: (1, -1), 5: (-1, -1), 6: (1, 1), 7: (-1, 1)}
AXIS = {3: 0, 5: 1, 0: 2, 4: 3, 2: 0, 6: 1, 1: 2, 7: 3}


def _ordered(notes):
    return sorted(notes, key=lambda n: (n['time'], n['hand'], n['id']))


def _outside(v, low, high):
    return max(low - v, v - high, 0)



def _onset_groups(ordered, hand):
    """Same-hand groups anchored to the first note, not transitive gaps."""
    seq = [note for note in ordered if note['hand'] == hand]
    start = 0
    while start < len(seq):
        end = start + 1
        while end < len(seq) and seq[end]['time'] - seq[start]['time'] <= STACK_EPS:
            end += 1
        yield seq[start:end]
        start = end


def preparation_events(notes):
    """One event per note, with shared context for each same-hand onset."""
    ordered = _ordered(notes)
    result = []
    for hand in (0, 1):
        previous_time, exposure = None, 0.
        for group in _onset_groups(ordered, hand):
            t = group[0]['time']
            gap = None if previous_time is None else t - previous_time
            load = 0. if gap is None else exposure * math.exp(-gap / EXPOSURE_TAU)
            speed = 0. if gap is None else EXPOSURE_TAU / (EXPOSURE_TAU + gap)
            awkward = False
            for note in group:
                d = note['d']
                eligible = d in STEP
                dx = dy = 0
                placement = SPECIFIC_WEIGHT * int(
                    (hand == 0 and note['x'] == 0) or (hand == 1 and note['x'] == 3))
                preparation_specific = 0.
                if eligible:
                    sx, sy = STEP[d]
                    low, high = (0, 2) if hand == 0 else (1, 3)
                    dx = _outside(note['x'] + sx, low, high)
                    dy = _outside(note['y'] + sy, 0, 2)
                    preparation_specific = SPECIFIC_WEIGHT * (
                        int((hand == 0 and note['x'] == 1 and sx == -1)
                            or (hand == 1 and note['x'] == 2 and sx == 1))
                        + int((note['y'] == 2 and sy == 1) or (note['y'] == 0 and sy == -1)))
                excursion = max(dx, dy)
                specific = placement + preparation_specific
                excursion_cost = PREPARATION_WEIGHT * int(excursion > 0)
                raw = excursion_cost + specific
                components = {'excursion': excursion_cost, 'placement': placement,
                              'preparation_specific': preparation_specific, 'specific': specific,
                              'speed': raw * speed, 'repetition': raw * load}
                result.append(dict(id=note['id'], time=note['time'], hand=hand, eligible=eligible,
                                   preparation_eligible=eligible, placement_eligible=True,
                                   hard=excursion >= 2, excursion=excursion, offsets=[dx, dy],
                                   components=components,
                                   total=raw + components['speed'] + components['repetition']))
                awkward |= eligible and raw > 0
            previous_time, exposure = t, load + int(awkward)
    return sorted(result, key=lambda e: (e['time'], e['hand'], e['id']))


def note_paths(notes):
    """ID-keyed path records: previous exit to approach, then approach to exit.

    Dots inherit a virtual direction if possible; unresolved dots are points.
    All endpoints remain in unbounded grid coordinates. Same-onset notes share
    their predecessor, and the last sorted member becomes the next anchor.
    """
    ordered = _ordered(notes)
    paths = {}
    for hand in (0, 1):
        previous = None
        for group in _onset_groups(ordered, hand):
            directions = []
            for n in group:
                x, y = n['x'], n['y']
                direction = n['d']
                if direction == 8 and previous is not None and previous[1] in STEP:
                    prior, prior_direction = previous
                    psx, psy = STEP[prior_direction]
                    mx, my = x-prior['x']+psx, y-prior['y']+psy
                    if mx == 0 and my == 0:
                        sx, sy = -psx, -psy
                    else:
                        sx, sy = -(int(mx > 0)-int(mx < 0)), -(int(my > 0)-int(my < 0))
                    direction = next(d for d, step in STEP.items() if step == (sx, sy))
                sx, sy = STEP.get(direction, (0, 0))
                approach, exit_point = [x+sx, y+sy], [x-sx, y-sy]
                segments = []
                predecessor_id = None
                if previous is not None:
                    prior, prior_direction = previous
                    predecessor_id = prior['id']
                    if n['time'] - prior['time'] <= PATH_SECONDS + TIME_TOLERANCE:
                        px, py = STEP.get(prior_direction, (0, 0))
                        segments.append([[prior['x']-px, prior['y']-py], approach])
                segments.append([approach, exit_point])
                paths[n['id']] = {'predecessor_id': predecessor_id, 'segments': segments}
                directions.append(direction)
            previous = (group[-1], directions[-1])
    return paths


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _on_segment(a, b, p):
    return (abs(_cross(a, b, p)) <= GEOMETRY_TOLERANCE
            and min(a[0], b[0])-GEOMETRY_TOLERANCE <= p[0] <= max(a[0], b[0])+GEOMETRY_TOLERANCE
            and min(a[1], b[1])-GEOMETRY_TOLERANCE <= p[1] <= max(a[1], b[1])+GEOMETRY_TOLERANCE)


def _intersects(a, b, c, d):
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    tol = GEOMETRY_TOLERANCE
    if (((ab_c > tol and ab_d < -tol) or (ab_c < -tol and ab_d > tol))
            and ((cd_a > tol and cd_b < -tol) or (cd_a < -tol and cd_b > tol))):
        return True
    return any((_on_segment(a, b, c), _on_segment(a, b, d),
                _on_segment(c, d, a), _on_segment(c, d, b)))


def _point_segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length2 = dx*dx + dy*dy
    if length2 == 0:
        return math.hypot(p[0]-a[0], p[1]-a[1])
    t = max(0., min(1., ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / length2))
    return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


def _segment_distance(first, second):
    return _cached_segment_distance(tuple(tuple(p) for p in first),
                                    tuple(tuple(p) for p in second),
                                    GEOMETRY_TOLERANCE)


@lru_cache(maxsize=32768)
def _cached_segment_distance(first, second, tolerance):
    a, b = first
    c, d = second
    if _intersects(a, b, c, d):
        return 0.
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def crossing_pair_event(left, right, left_path, right_path):
    """Score two supplied paths in constant bounded work, without rescanning notes.

    Path arguments are note_paths records or nonempty lists of segments. The
    caller must pass left then right; eligibility reflects the pair time window.
    """
    left_segments = left_path['segments'] if isinstance(left_path, dict) else left_path
    right_segments = right_path['segments'] if isinstance(right_path, dict) else right_path
    pred_ids = [left_path.get('predecessor_id') if isinstance(left_path, dict) else None,
                right_path.get('predecessor_id') if isinstance(right_path, dict) else None]
    dt = abs(left['time'] - right['time'])
    eligible = dt <= PATH_SECONDS + TIME_TOLERANCE
    distance = min(_segment_distance(a, b) for a in left_segments for b in right_segments)
    detected = distance == 0.
    raw = 1. if detected else max(0., 1. - distance / NEAR_MISS_MARGIN)
    weighted = raw * max(0., (PATH_SECONDS-dt) / (PATH_SECONDS-FAST_SECONDS)) if eligible else 0.
    return dict(ids=[left['id'], right['id']], times=[left['time'], right['time']],
                predecessor_ids=pred_ids, eligible=eligible, dt=dt, distance=distance,
                crossover=max(0, left['x']-right['x']), detected=detected,
                hard=eligible and detected and dt <= FAST_SECONDS + TIME_TOLERANCE,
                components={'intersection': float(detected),
                            'near_miss': raw if not detected else 0., 'raw': raw,
                            'weighted': weighted}, total=weighted)


def crossing_events(notes):
    """Each eligible opposite-hand pair once, including zero-cost pairs."""
    ordered = _ordered(notes)
    times = [n['time'] for n in ordered]
    paths = note_paths(ordered)
    events = []
    for i, a in enumerate(ordered):
        end = bisect_right(times, a['time'] + PATH_SECONDS + TIME_TOLERANCE)
        for b in ordered[i+1:end]:
            if a['hand'] == b['hand']:
                continue
            left, right = (a, b) if a['hand'] == 0 else (b, a)
            events.append(crossing_pair_event(left, right, paths[left['id']], paths[right['id']]))
    return events


def axis_change_events(notes):
    """Per-note events; dots and simultaneous same-hand stacks reset cadence."""
    ordered = _ordered(notes)
    result, state = [], {}
    stack_ids = {n['id'] for hand in (0, 1)
                 for group in _onset_groups(ordered, hand) if len(group) > 1
                 for n in group}
    for n in ordered:
        h = n['hand']
        old = state.get(h)
        axis = AXIS.get(n['d']) if n['id'] not in stack_ids else None
        eligible = axis is not None and old is not None
        magnitude = min((axis-old['axis']) % 4, (old['axis']-axis) % 4) if eligible else 0
        base = AXIS_BASE * magnitude
        frequency = irregularity = 0.0
        prior_change = old['change_time'] if old else None
        interval = old['since'] + 1 if magnitude and prior_change is not None else None
        duration = n['time'] - prior_change if interval is not None else None
        previous_id = old['id'] if eligible else None
        if axis is None:
            state.pop(h, None)
        else:
            current = dict(axis=axis, id=n['id'], since=0, change_time=None, interval=None)
            if old is not None:
                current.update(since=old['since'], change_time=prior_change, interval=old['interval'])
                if magnitude:
                    if interval is not None:
                        frequency = AXIS_FREQUENCY_WEIGHT / interval
                        if old['interval'] is not None:
                            cadence = abs(interval-old['interval']) / max(interval, old['interval'])
                            irregularity = (AXIS_IRREGULARITY_WEIGHT * cadence / min(interval, old['interval'])
                                            / max(1., duration))
                    current.update(since=0, change_time=n['time'], interval=interval)
                else:
                    current['since'] += 1
            state[h] = current
        components = {'base': base, 'frequency': frequency, 'irregularity': irregularity}
        result.append(dict(id=n['id'], time=n['time'], hand=h, eligible=eligible,
                           previous_id=previous_id, previous_change_time=prior_change,
                           interval_notes=interval, interval_seconds=duration,
                           hard=False, magnitude_degrees=45*magnitude,
                           components=components, total=sum(components.values())))
    return result


def priority_metrics(notes):
    """JSON-compatible event lists and totals with no calibration claims."""
    notes = list(notes)
    result = {'version': VERSION}
    for name, function in (('preparation', preparation_events), ('crossing', crossing_events),
                           ('axis', axis_change_events)):
        events = function(notes)
        result[name] = {'events': events, 'eligible': sum(e['eligible'] for e in events),
                        'hard': sum(e['hard'] for e in events),
                        'total': sum(e['total'] for e in events)}
        if name == 'preparation':
            result[name]['preparation_eligible'] = sum(e['preparation_eligible'] for e in events)
            result[name]['placement_eligible'] = sum(e['placement_eligible'] for e in events)
        if name != 'crossing':
            result[name]['per_hand'] = {str(h): {
                'eligible': sum(e['eligible'] for e in events if e['hand'] == h),
                'hard': sum(e['hard'] for e in events if e['hand'] == h),
                'total': sum(e['total'] for e in events if e['hand'] == h)} for h in (0, 1)}
    return result
