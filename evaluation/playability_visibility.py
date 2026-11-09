"""Curated-human calibration and offline reports for sustained visibility."""
import argparse
import hashlib
import json
from pathlib import Path

from evaluation.playability_priority import fit_cutoff
from map_creation.v2_playability_visibility import (
    BANDS, GROUP_SECONDS, MIN_ONSETS, VERSION, WINDOW_ONSETS, WINDOW_SECONDS,
    visibility_costs, visibility_events,
)

BUDGET = .03
MIN_ELIGIBLE_MAPS = 5
REFERENCE_VERSION = 'visibility-human-events-v1-20261108'
REFERENCE = Path(__file__).resolve().parents[1] / 'map_creation/v2_playability_visibility_reference.json'
CURATED_ROOT = Path(__file__).resolve().parents[2] / 'Data/training/evaluate_beats/bs_map_input'


def _nps(notes):
    from map_creation.v2_playability_reach import note_span_nps
    return note_span_nps(notes)


def _band(nps):
    return BANDS[sum(nps >= bound for bound in (3, 5, 7))]


def fit_reference(rows):
    """Fit strict pooled/equal-map tails, falling back only on eligible maps."""
    observations = [dict(band=_band(_nps(row['notes'])),
                         values=[e['severity'] for e in visibility_events(row['notes'])]) for row in rows]

    def fit(selected):
        result = fit_cutoff([row['values'] for row in selected], BUDGET)
        if result['maps'] < MIN_ELIGIBLE_MAPS:
            result.update(cutoff=None, flagged=None, supported=False,
                          pooled_rate=None, equal_map_rate=None)
        return result

    global_fit = fit(observations)
    bands = {}
    for band in BANDS:
        selected = [row for row in observations if row['band'] == band]
        local = fit(selected)
        fallback = not local['supported']
        chosen = dict(global_fit if fallback else local)
        chosen.update(fallback=fallback, band_eligible_maps=local['maps'], band_maps=len(selected))
        bands[band] = chosen
    return dict({'global': global_fit}, reference_version=REFERENCE_VERSION, event_version=VERSION, budget=BUDGET,
                comparison='>', human_maps=len(observations), bands=bands,
                group_seconds=GROUP_SECONDS, window_seconds=WINDOW_SECONDS,
                window_onsets=WINDOW_ONSETS, minimum_onsets=MIN_ONSETS,
                denominator='One event per same-hand 20ms-anchored onset with at least four onsets in the trailing 1.5 seconds, capped at eight. All zero-exposure phrases included; stacks share one onset weight.',
                severity='Mean top-row inner-column occupancy across the eligible onset groups. Each note contributes 1/(onsets*stack_size) if y=2 and x is 1 or 2.',
                calibration_scope='In-sample fit only; no held-out or player-quality claim. Strict severity > cutoff; specific 3% pooled and equal-map budgets.',
                limitations=['Placement proxy, not a headset trajectory measurement or a global top-row quota.',
                             'Overlapping phrase endpoints are not independent samples.',
                             'Fewer than five eligible maps falls back to the all-band fit; unsupported global fits remain unavailable.'])


def score_notes(notes, reference=None):
    """Use exactly the shared runtime event definition and fitted thresholds."""
    return visibility_costs(notes, reference=reference)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def _curated_root(human_root):
    curated = CURATED_ROOT.resolve()
    supplied = Path(human_root).resolve()
    if supplied not in (curated, curated.parent) or not curated.is_dir():
        raise ValueError('Human visibility calibration requires the curated evaluate_beats/bs_map_input root')
    return curated


def calibrate(human_root, output, reference_output):
    """Decode highest Standard charts only, recording each rejected package."""
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
    reference.update(human_root=str(root),
                     population_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
    reports = [dict(package=row['name'], visibility=score_notes(row['notes'], reference)) for row in rows]
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
    report.add_argument('--reference', default=str(REFERENCE))
    report.add_argument('--json', required=True)
    args = parser.parse_args()
    if args.command == 'calibrate':
        calibrate(args.human_root, args.json, args.reference)
    else:
        from evaluation.playability_report import decode_package_notes
        decoded = decode_package_notes(args.package, args.difficulty)
        notes = [dict(zip(('time', 'hand', 'x', 'y', 'd'), n), id=i)
                 for i, n in enumerate(decoded['notes'])]
        reference = json.loads(Path(args.reference).read_text(encoding='utf8'))
        _save(args.json, score_notes(notes, reference))


if __name__ == '__main__':
    main()
