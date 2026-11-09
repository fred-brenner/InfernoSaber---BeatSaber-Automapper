"""Offline human cut-axis event calibration; production reads only frozen JSON."""
import math


BANDS = ('<3', '3-5', '5-7', '7+')
FIELDS = ('notes', 'seconds', 'incoming_seconds')


def fit_direction_reference(events, *, budget=.1, minimum_packages=5):
    """Fit conservative joint two-tail intervals, pooling adjacent NPS bands.

    Each of six strict tails receives one sixth of the event budget. Ties
    remain inside: the empirical union cannot exceed the requested budget.
    Unsupported hand/angle strata stay explicit rather than inventing data.
    """
    if not 0 < budget <= .1:
        raise ValueError('Direction event budget must be in (0, .1]')
    strata = {}
    for band_index, band in enumerate(BANDS):
        for hand in (0, 1):
            for angle in (45, 90):
                source = [e for e in events if e['hand'] == hand and e['ending_angle'] == angle]
                selected_bands = []
                for index in sorted(range(len(BANDS)), key=lambda i: (abs(i-band_index), i)):
                    selected_bands.append(BANDS[index])
                    selected = [e for e in source if e['band'] in selected_bands]
                    if len({e['package'] for e in selected}) >= minimum_packages:
                        break
                packages = len({e['package'] for e in selected})
                supported = packages >= minimum_packages
                intervals = {}
                if supported:
                    share = budget / (2 * len(FIELDS))
                    for field in FIELDS:
                        values = sorted(float(e[field]) for e in selected)
                        lower = values[math.floor(len(values) * share)]
                        upper = values[math.ceil(len(values) * (1.-share))-1]
                        intervals[field] = [lower, upper]
                flagged = sum(any(float(e[field]) < lo or float(e[field]) > hi
                                  for field, (lo, hi) in intervals.items()) for e in selected)
                strata[f'{band}|{hand}|{angle}'] = dict(
                    source_bands=[b for b in BANDS if b in selected_bands],
                    packages=packages, events=len(selected), supported=supported,
                    intervals=intervals, budget=budget,
                    empirical_flagged=flagged,
                    empirical_flagged_share=flagged / len(selected) if selected else None)
    return dict(schema_version=1, policy='joint-two-tail-axis-events-v1',
                minimum_packages=minimum_packages, budget=budget, strata=strata)


def calibrate(human_root, output, reference_output):
    """Read every curated highest Standard chart and record individual skips."""
    import hashlib
    import json
    from pathlib import Path
    from evaluation.playability_report import decode_package_notes
    from map_creation.v2_playability_directions import axis_run_events
    from map_creation.v2_playability_reach import note_span_nps
    from map_creation.v2_playability_reference import nps_band
    root = Path(human_root)
    root = root / 'bs_map_input' if (root / 'bs_map_input').is_dir() else root
    events, accepted, skipped = [], [], []
    for package in sorted(root.iterdir()):
        if not package.is_dir():
            continue
        try:
            decoded = decode_package_notes(package)
            notes = [dict(id=i, time=t, hand=h, x=x, y=y, d=d)
                     for i, (t, h, x, y, d) in enumerate(decoded['notes'])]
            nps = note_span_nps(notes)
            observed = axis_run_events(notes)
            events.extend(dict(e, package=package.name, band=nps_band(nps))
                          for e in observed['completed'])
            accepted.append(dict(package=package.name, difficulty=decoded['difficulty'],
                notes=len(notes), nps=nps, completed_runs=len(observed['completed'])))
        except (OSError, ValueError, KeyError, TypeError) as error:
            skipped.append(dict(package=package.name, reason=str(error)))
    reference = fit_direction_reference(events)
    fingerprint = hashlib.sha256(json.dumps(events, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    reference.update(source_sha256=fingerprint, source='curated bs_map_input highest Standard',
                     accepted=accepted, skipped=skipped,
                     interval_fields=list(FIELDS),
                     note='Strict outside intervals; joint notes/duration/incoming two-tail budget. '
                          'Censored runs test upper bounds only; next-change time is the next completed run duration.')
    reference_path = Path(reference_output)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps(reference, indent=2, allow_nan=False)+'\n')
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'calibration.json').write_text(json.dumps(reference, indent=2, allow_nan=False)+'\n')
    (destination / 'human-events.json').write_text(json.dumps(events, indent=2, allow_nan=False)+'\n')
    return reference


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--human-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-output', required=True)
    args = parser.parse_args()
    result = calibrate(args.human_root, args.output, args.reference_output)
    print(json.dumps(dict(accepted=len(result['accepted']), skipped=result['skipped'],
                         events=sum(r['completed_runs'] for r in result['accepted']),
                         max_fit_share=max(s['empirical_flagged_share'] or 0. for s in result['strata'].values())),
                     indent=2))
