"""Offline human calibration and reports for the versioned priority policy."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from map_creation.v2_playability_priority_metrics import FAST_SECONDS, STEP, VERSION, priority_metrics

BUDGETS = dict(preparation=.10, preparation_excursion=.03, inner_inward=.03,
               vertical_preparation=.03, outer_placement=.03, axis=.03,
               slow_crossing=.03, near_miss=.10)
BANDS = ('<3', '3-5', '5-7', '7+')
REFERENCE = Path(__file__).resolve().parents[1] / 'map_creation/v2_playability_priority_reference.json'
CURATED_ROOT = Path(__file__).resolve().parents[2] / 'Data/training/evaluate_beats/bs_map_input'
MIN_ELIGIBLE_MAPS = 5


def fit_cutoff(populations, budget):
    """Fit a strict upper tail meeting pooled and equal-map event budgets."""
    populations = [list(values) for values in populations if len(values)]
    values = sorted(value for group in populations for value in group)
    if not values:
        return dict(cutoff=None, eligible=0, flagged=None, raw_positive=0, maps=0, supported=False,
                    pooled_rate=None, equal_map_rate=None, budget=budget)
    if not 0 <= budget < 1 or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid calibration budget or nonfinite severity')
    allowed = math.floor(budget * len(values) + 1e-12)
    pooled = values[len(values)-allowed-1]
    weighted = sorted((value, 1. / (len(group)*len(populations)))
                      for group in populations for value in group)
    cumulative, equal = 0., weighted[-1][0]
    for value, weight in weighted:
        cumulative += weight
        if cumulative >= 1.-budget-1e-12:
            equal = value
            break
    cutoff = max(pooled, equal)
    flagged = sum(value > cutoff for value in values)
    return dict(cutoff=cutoff, eligible=len(values), flagged=flagged, raw_positive=sum(v > 0 for v in values), maps=len(populations),
                supported=True, pooled_rate=flagged/len(values),
                equal_map_rate=sum(sum(v > cutoff for v in group)/len(group)
                                   for group in populations)/len(populations), budget=budget)


def _nps(notes):
    from map_creation.v2_playability_reach import note_span_nps
    return note_span_nps(notes)


def _band(nps):
    return BANDS[sum(nps >= value for value in (3, 5, 7))]


def _observations(primary, notes):
    by_id = {note['id']: note for note in notes}
    arrows = [e for e in primary['preparation']['events'] if e['preparation_eligible']]
    placement = [e for e in primary['preparation']['events'] if e['placement_eligible']]
    def inner(event):
        note = by_id[event['id']]
        dx, _ = STEP[note['d']]
        return float((note['hand'] == 0 and note['x'] == 1 and dx == -1)
                     or (note['hand'] == 1 and note['x'] == 2 and dx == 1))
    def vertical(event):
        note = by_id[event['id']]
        _, dy = STEP[note['d']]
        return float((note['y'] == 2 and dy == 1) or (note['y'] == 0 and dy == -1))
    def weighted(event, value):
        components = event['components']
        raw = components['excursion'] + components['specific']
        return float(value) * event['total'] / raw if raw else 0.
    return dict(
        # Explicitly a coarse composite, not an individual preparation measure.
        preparation=[e['total'] for e in arrows],
        preparation_excursion=[weighted(e, e['excursion']) for e in arrows],
        inner_inward=[weighted(e, inner(e)) for e in arrows],
        vertical_preparation=[weighted(e, vertical(e)) for e in arrows],
        outer_placement=[weighted(e, e['components']['placement']) for e in placement],
        axis=[e['total'] for e in primary['axis']['events'] if e['eligible']],
        slow_crossing=[e['total'] if e['detected'] else 0. for e in primary['crossing']['events']
                       if e['eligible'] and e['dt'] > FAST_SECONDS + 1e-9],
        near_miss=[e['total'] if not e['detected'] else 0.
                   for e in primary['crossing']['events'] if e['eligible']])


def fit_reference(rows):
    """Fit each band using eligible maps; fall back independently per metric."""
    observations = [dict(name=row['name'], band=_band(_nps(row['notes'])),
                         values=_observations(priority_metrics(row['notes']), row['notes'])) for row in rows]
    def fit_metric(selected, name, budget):
        fitted = fit_cutoff([row['values'][name] for row in selected], budget)
        if fitted['maps'] < MIN_ELIGIBLE_MAPS:
            fitted.update(cutoff=None, flagged=None, supported=False, pooled_rate=None, equal_map_rate=None)
        return fitted
    global_fit = {name: fit_metric(observations, name, budget) for name, budget in BUDGETS.items()}
    bands = {}
    for band in BANDS:
        selected = [row for row in observations if row['band'] == band]
        metrics = {}
        for name, budget in BUDGETS.items():
            local = fit_metric(selected, name, budget)
            fallback = not local['supported']
            chosen = dict(global_fit[name] if fallback else local)
            chosen.update(fallback=fallback, band_eligible_maps=local['maps'])
            metrics[name] = chosen
        bands[band] = dict(maps=len(selected), metrics=metrics)
    return dict(reference_version='priority-human-events-v2-20261108', event_version=VERSION,
                budgets=BUDGETS, human_maps=len(rows), global_metrics=global_fit, bands=bands,
                metric_definitions={'preparation': 'Coarse composite of arrow preparation and placement costs.',
                                    'specific_preparation': 'Component occurrence or excursion multiplied by its local speed and repetition factor; zeros retained.',
                                    'slow_crossing': 'All eligible pairs after fast boundary; non-intersections are zero.',
                                    'near_miss': 'All eligible pairs; intersections count as zero.'},
                calibration_scope='In-sample event fit; no held-out quality claim. Strict severity > cutoff.',
                hard_policy='Hard preparation and fast segment crossings are independent of soft calibration.')


def _flags(primary, reference, nps, notes):
    profile = reference['bands'][_band(nps)]['metrics']
    result = {}
    for name, values in _observations(primary, notes).items():
        fitted = profile[name]
        flagged = sum(value > fitted['cutoff'] for value in values) if fitted['supported'] else None
        result[name] = dict(cutoff=fitted['cutoff'], budget=BUDGETS[name], eligible=len(values),
                            flagged=flagged, raw_positive=sum(v > 0 for v in values),
                            rate=flagged/len(values) if values and flagged is not None else None,
                            reference_supported=fitted['supported'], fallback=fitted['fallback'])
    return result


def score_notes(notes, reference=None):
    """Keep primary events, placement and lower-priority monitoring distinct."""
    from evaluation.playability_metrics_v2 import score_map
    from map_creation.v2_playability_directions import axis_run_events
    from map_creation.v2_playability_placement import placement_costs
    from map_creation.v2_playability_reach import reach_costs
    from map_creation.v2_playability_visibility import visibility_costs
    notes = list(notes)
    nps = _nps(notes)
    primary = priority_metrics(notes)
    if reference is None:
        reference = json.loads(REFERENCE.read_text())
    if reference['event_version'] != VERSION:
        raise ValueError('Priority reference and event definition versions differ')
    tuples = [tuple(n[key] for key in ('time', 'hand', 'x', 'y', 'd')) for n in notes]
    observed = score_map(tuples)
    monitoring = {name: dict(value=float(observed[name]['value'])) for name in ('m1', 'm2', 'm3', 'm4')}
    visibility = visibility_costs(notes, nps)
    legacy_m2 = monitoring['m2']
    monitoring['legacy_m2'] = dict(legacy_m2, version=observed['geometry_version'])
    # A whole-song rise average cannot represent a sustained local obstruction.
    # Keep its original units separate from the calibrated current M2 component.
    excess = visibility['excess']
    mean_excess = None if excess is None else excess / max(1, visibility['eligible_events'])
    monitoring['m2'] = dict(
        version=visibility['version'],
        value=mean_excess,
        definition='Mean calibrated upper-center exposure excess; local events and maxima govern visibility repair.',
        pairwise_value=legacy_m2['value'], max_severity=visibility['max_severity'],
        flagged_events=visibility['flagged_events'], eligible_events=visibility['eligible_events'])
    monitoring['m5'] = observed['m5']
    monitoring['geometry_version'] = observed['geometry_version']
    monitoring['axis_runs'] = axis_run_events(notes)
    # Historical variety/reversal counters are descriptive, never repair goals.
    monitoring['m5']['definition'] = 'Historical direction variety; fast reversals and long runs are monitoring only.'
    travel = reach_costs(notes, nps)['metrics']['recovery_adjusted_travel']
    return dict(primary=primary, nps=nps, flags=_flags(primary, reference, nps, notes),
                placement=placement_costs(notes, nps), travel=travel, visibility=visibility,
                monitoring=monitoring)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def _curated_root(human_root):
    curated = CURATED_ROOT.resolve()
    supplied = Path(human_root).resolve()
    if supplied not in (curated, curated.parent) or not curated.is_dir():
        raise ValueError('Human calibration requires the curated evaluate_beats/bs_map_input root')
    return curated


def calibrate(human_root, output, reference_output):
    from evaluation.playability_report import _package_dirs, decode_package_notes
    root = _curated_root(human_root)
    rows, skipped, hashes = [], [], {}
    for package in _package_dirs(root):
        try:
            decoded = decode_package_notes(package)
            notes = [dict(zip(('time', 'hand', 'x', 'y', 'd'), n), id=i)
                     for i, n in enumerate(decoded['notes'])]
            if len(notes) < 2:
                raise ValueError('Fewer than two colored notes')
            rows.append(dict(name=package.name, notes=notes))
            hashes[package.name] = hashlib.sha256(json.dumps(decoded, sort_keys=True).encode()).hexdigest()
        except (ValueError, OSError, KeyError) as exc:
            skipped.append(dict(package=package.name, reason=str(exc)))
    if not rows:
        raise ValueError('No accepted curated human charts')
    reference = fit_reference(rows)
    reference['human_root'] = str(root)
    reference['population_sha256'] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    reports = []
    for row in rows:
        primary = priority_metrics(row['notes'])
        reports.append(dict(package=row['name'], nps=_nps(row['notes']),
                            flags=_flags(primary, reference, _nps(row['notes']), row['notes']),
                            hard_preparation=primary['preparation']['hard'],
                            hard_fast_crossings=primary['crossing']['hard']))
    _save(reference_output, reference)
    _save(output, dict(reference=reference, accepted=len(rows), skipped=skipped, maps=reports))
    print(f'Accepted {len(rows)} human charts; skipped {len(skipped)} individually recorded packages.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('calibrate')
    fit.add_argument('--human-root', required=True)
    fit.add_argument('--json', required=True)
    fit.add_argument('--reference', default=str(REFERENCE))
    report = commands.add_parser('report')
    report.add_argument('--package', required=True)
    report.add_argument('--difficulty')
    report.add_argument('--json', required=True)
    args = parser.parse_args()
    if args.command == 'calibrate':
        calibrate(args.human_root, args.json, args.reference)
    else:
        from evaluation.playability_report import decode_package_notes
        decoded = decode_package_notes(args.package, args.difficulty)
        notes = [dict(zip(('time', 'hand', 'x', 'y', 'd'), n), id=i)
                 for i, n in enumerate(decoded['notes'])]
        _save(args.json, score_notes(notes))


if __name__ == '__main__':
    main()
