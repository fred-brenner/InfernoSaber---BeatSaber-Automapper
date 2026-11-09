"""Bounded cell bridges for already-proposed M1 arrow repairs.

The caller owns the direction proposal and final commit. This search changes
only nearby cells temporarily, returns a mapping, and restores every cell even
when validation raises. It never relaxes the physical or shared pattern guard.
"""

from bisect import bisect_left, bisect_right


def _event_excess(report):
    values = {}
    for key,metric in report['metrics'].items():
        cutoff = metric['cutoff']
        for value,identities in zip(metric['severities'],metric['event_note_ids']):
            values[key,tuple(identities)] = max(0.,value-cutoff)/max(1.,cutoff)
    return values


def find_lateral_bridge(notes,cap,mutable_ids,guard):
    """Return <=8 one-Manhattan cell edits passing physical and shared guards.

    ``notes`` share the guard's live dictionaries and already contain proposed
    arrows. All changed arrows are included in legality/acceptance checks,
    even when only one cell moves. At most eight width-eight beam layers and
    512 cell proposals are explored; failure returns ``None`` without a commit.
    """
    from map_creation.v2_playability_reach import _Search,reach_costs

    ordered = sorted(notes,key=lambda n:(n['time'],n['id']))
    index = {n['id']:i for i,n in enumerate(ordered)}
    mutable = {index[identity] for identity in mutable_ids if identity in index}
    changed = {i for i,n in enumerate(ordered)
               if any(n[key] != guard.accepted[guard.index[n['id']]][key] for key in ('hand','x','y','d'))}
    state = _Search(ordered,cap,guard.nps,False)
    if state.legal(changed) and guard.allows({ordered[i]['id'] for i in changed}):
        return {}
    if not mutable:
        return None
    first,last = min(ordered[i]['time'] for i in mutable),max(ordered[i]['time'] for i in mutable)
    times = [n['time'] for n in ordered]
    lo,hi = bisect_left(times,first-2.),bisect_right(times,last+2.)
    context = ordered[lo:hi]
    accepted = [guard.accepted[guard.index[n['id']]] for n in context]
    old = reach_costs(accepted,guard.nps)
    proposed = reach_costs(context,guard.nps)
    old_events = _event_excess(old)
    incident = {}
    for event,value in _event_excess(proposed).items():
        increase = value-old_events.get(event,0.)
        if increase > 1e-9:
            for identity in event[1]:
                i = index[identity]
                if i in mutable:
                    incident[i] = incident.get(i,0.)+increase
    if not incident:
        return None
    # Exposure incidents may need a neighboring arrival to keep speed, dots or
    # both-hand clearance feasible. Candidate count remains strictly bounded.
    selected = sorted(incident,key=lambda i:(-incident[i],i))[:8]
    support = sorted(mutable-set(selected),key=lambda i:(
        min(abs(ordered[i]['time']-ordered[j]['time']) for j in selected),i))
    selected.extend(support[:8-len(selected)])
    original = {i:(ordered[i]['x'],ordered[i]['y']) for i in selected}

    def install(edits):
        for i,(x,y) in original.items():
            ordered[i]['x'],ordered[i]['y'] = edits.get(i,(x,y))

    def rank(edits):
        report = reach_costs(context,guard.nps)
        increase = sum(max(0.,metric['excess']-old['metrics'][key]['excess'])
                       for key,metric in report['metrics'].items())
        travel = sum(report['metrics']['recovery_adjusted_travel']['severities'])
        distance = sum(abs(x-original[i][0])+abs(y-original[i][1]) for i,(x,y) in edits.items())
        physical = state.legal(changed|set(edits))
        return increase,int(not physical),report['total_excess'],travel,len(edits),distance,tuple(sorted(edits.items()))

    beam = [({},rank({}))]
    proposals = 0
    try:
        for i in selected:
            x,y = original[i]
            options = [(x,y)]+[(xx,yy) for xx,yy in ((x-1,y),(x+1,y),(x,y-1),(x,y+1))
                                if 0 <= xx < 4 and 0 <= yy < 3]
            candidates = []
            for edits,_ in beam:
                for cell in options:
                    if proposals >= 512:
                        break
                    proposal = dict(edits)
                    if cell != original[i]:
                        proposal[i] = cell
                    proposals += 1
                    install(proposal)
                    candidates.append((proposal,rank(proposal)))
            beam = sorted(candidates,key=lambda item:item[1])[:8]
            for edits,cost in beam:
                if cost[0] > 1e-8 or cost[1]:
                    continue
                install(edits)
                identities = {ordered[j]['id'] for j in changed|set(edits)}
                if guard.allows(identities):
                    return {ordered[j]['id']:cell for j,cell in edits.items()}
            if proposals >= 512:
                break
        return None
    finally:
        install({})
