"""Receipt and package adapter for the offline playability benchmark CLI."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.playability_benchmark import (
    DISTANCE_METRICS, DWELL_DISTANCE_METRICS, compact_scorecard, freeze_material_harm_margins,
    load_human_reference, paired_decision_report,
)
from evaluation.playability_experiments import POLICIES, SWITCHES, case_seed
from evaluation.playability_report import DIFFICULTY_ORDER, _jsonable, decode_package_notes

_CALCULATION_FILES = ('evaluation/playability_metrics.py', 'evaluation/playability_directions.py',
                      'evaluation/playability_report.py', 'evaluation/playability_benchmark.py')


def _scorer_sources():
    return {name: _hash(name) for name in _CALCULATION_FILES}


def _numerical_hash(reference, margins):
    return hashlib.sha256(json.dumps({'reference': reference, 'margins': margins},
        allow_nan=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), allow_nan=False, indent=2) + '\n', encoding='utf-8')


def _cases(manifest, split, indices, difficulties, replicates):
    registry = json.loads(Path(manifest).read_text(encoding='utf-8'))
    songs = [row for row in registry['songs'] if row['split'] == split]
    if indices is not None and any(index < 0 or index >= len(songs) for index in indices):
        raise ValueError('Selected song index outside manifest split')
    replicates = replicates if replicates is not None else registry[split + '_replicates']
    cases = []
    for index, song in enumerate(songs):
        if indices is not None and index not in indices:
            continue
        for difficulty, variant in song['variants'].items():
            if difficulties and difficulty not in difficulties:
                continue
            for replicate in replicates:
                cases.append({'song': song['package'], 'audio': song['audio'],
                    'audio_hash': song['sha256'], 'index': index, 'difficulty': difficulty,
                    'seed': song.get('literal_seed', case_seed(song['sha256'], variant['nps'], replicate)),
                    'replicate': replicate, 'actual_settings': variant, 'split': split})
    if not cases:
        raise ValueError('Selection contains no manifest cases')
    return cases


def _case_key(case):
    return f"{case['song']}|{case['difficulty']}|{case['seed']}"


def _receipt(runs, case, arm):
    path = (Path(runs) / case['split'] / arm / f"{case['index']:02d}" /
            case['difficulty'] / str(case['replicate']) / 'result.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    expected = {key: case[key] for key in ('song', 'difficulty', 'seed', 'replicate', 'audio_hash')}
    expected['arm'] = arm
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f'{path}: mismatched {key}: {data.get(key)!r} != {value!r}')
    if data.get('actual_settings') != case['actual_settings']:
        raise ValueError(f'{path}: settings differ from the frozen manifest')
    expected_policy = {'enabled': arm != 'app1_control', **SWITCHES}
    if arm != 'app1_control':
        for component in arm.split(','):
            expected_policy.update(POLICIES[component])
    if 'policy_settings' in data and data['policy_settings'] != expected_policy:
        raise ValueError(f'{path}: policy flags do not match the declared arm')
    if 'effective_config' in data:
        for key, value in expected_policy.items():
            if data['effective_config'].get('v2_playability_' + key) != value:
                raise ValueError(f'{path}: effective policy setting differs: {key}')
        temperature = case['actual_settings'].get('temperature')
        if temperature is not None and data['effective_config'].get('v2_note_sample_temperature') != temperature:
            raise ValueError(f'{path}: effective note temperature differs from the case settings')
    if data.get('stages') is not None:
        required = {'intensity','timing','positions','arrows','multiplier','hazards','arcs','lighting'}
        missing = required - data['stages'].keys()
        if missing:
            raise ValueError(f'{path}: missing production stage: {sorted(missing)}')
        if data['stages']['lighting'].get('backend') != 'v1':
            raise ValueError(f'{path}: production lighting did not use the retained v1 backend')
    if _hash(case['audio']) != case['audio_hash']:
        raise ValueError(f"Changed source recording: {case['audio']}")
    package = (path.parent / data['package']).resolve()
    if not package.is_relative_to(path.parent.resolve()):
        raise ValueError(f'Package path escapes receipt folder: {path}')
    worker_path = (Path(runs) / data['worker']).resolve()
    if not worker_path.is_relative_to(Path(runs).resolve()):
        raise ValueError(f'Worker path escapes run root: {path}')
    worker = json.loads(worker_path.read_text(encoding='utf-8'))
    if worker.get('environment', {}).get('CUDA_VISIBLE_DEVICES', '-1') != '-1':
        raise ValueError(f'{path}: worker did not record the CPU activation profile')
    if any('GPU' in str(device) for device in worker.get('devices', [])):
        raise ValueError(f'{path}: GPU execution is outside this CPU comparison')
    notes = decode_package_notes(package, case['difficulty'])['notes']
    if data.get('package_hashes') is not None:
        actual = {file.name: _hash(file) for file in sorted(package.iterdir()) if file.is_file()}
        if actual != data['package_hashes']:
            raise ValueError(f'{path}: package hashes no longer match generation receipt')
    package_hashes = {str(file.relative_to(package)): _hash(file)
                      for file in sorted(package.rglob('*')) if file.is_file() and file.suffix.lower() == '.dat'}
    provenance = {'receipt': str(path.resolve()), 'receipt_sha256': _hash(path),
                  'worker_sha256': _hash(worker_path), 'package_dat_sha256': package_hashes}
    return {'data': data, 'worker': worker, 'notes': notes, 'provenance': provenance,
            'receipt_completed_timestamp': path.stat().st_mtime}


def freeze_run_reference(manifest, runs, human_root, *, split='development', indices=None,
                         difficulties=None, replicates=None):
    """Freeze human tables and margins from complete controls, without candidates."""
    cases = _cases(manifest, split, indices, difficulties, replicates)
    reference = load_human_reference(human_root)
    if not reference['maps']:
        raise ValueError('No eligible curated human maps')
    controls, sources = [], {}
    for case in cases:
        receipt = _receipt(runs, case, 'app1_control')
        card = compact_scorecard(receipt['notes'], reference)
        controls.append({**case, 'control': card})
        sources[_case_key(case)] = receipt['provenance']
    scorer = _scorer_sources()
    margins = freeze_material_harm_margins(controls, reference)
    return {'schema_version': 1, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
            'manifest_sha256': _hash(manifest), 'manifest': str(Path(manifest).resolve()),
            'scope': {'split': split, 'indices': indices, 'difficulties': difficulties,
                      'replicates': replicates, 'cases': len(cases),
                      'case_keys': [_case_key(case) for case in cases],
                      'replicate_values': sorted({case['replicate'] for case in cases})},
            'reference': reference, 'margins': margins,
            'numerical_sha256': _numerical_hash(reference, margins),
            'controls': sources, 'scorer_sources': scorer,
            'thresholds': {'closure': {key: 0.2 if key == 'm1' else 0.15
                                       for key in (*DISTANCE_METRICS, *DWELL_DISTANCE_METRICS)},
                           'nps_drift': 0.03, 'dot_share_increase': 0.01,
                           'm3_coverage_loss': 0.05, 'cpu_slowdown': 0.1}}


def _canonical_config(config):
    values = {key:value for key,value in config.items() if not key.startswith('v2_playability_')}
    mode = 'sport' if values.get('sporty_obstacles') else 'norm'
    # gen_obstacles.add_obstacle reassigns these two derived values before
    # use. A first-case snapshot can legitimately precede their creation.
    for name in ('obstacle_allowed_types','obstacle_positions'):
        source = mode + '_' + name
        if source in values:
            values[name] = values[source]
    return values


def _pair_audit(control, candidate):
    differences = []
    for key in ('models', 'python', 'tensorflow', 'devices', 'environment', 'model_settings'):
        a, b = control['worker'].get(key), candidate['worker'].get(key)
        if a is None or b is None:
            differences.append({'kind': 'missing_worker_provenance', 'field': key,
                                'blocking':key not in ('environment','model_settings')})
        elif a != b:
            if key == 'model_settings':
                a = {**a, 'note_sample_temperature':control['data']['actual_settings'].get('temperature')}
                b = {**b, 'note_sample_temperature':candidate['data']['actual_settings'].get('temperature')}
                if a == b:
                    continue
            differences.append({'kind': 'worker_provenance_difference', 'field': key})
    a, b = control['worker'].get('sources', {}), candidate['worker'].get('sources', {})
    for name in sorted(set(a) | set(b)):
        if a.get(name) != b.get(name):
            differences.append({'kind': 'source_hash_difference', 'file': name,
                                'control': a.get(name), 'candidate': b.get(name),
                                'blocking': (not name.startswith('evaluation/') or name in (
                                    'evaluation/playability_metrics.py',
                                    'evaluation/playability_directions.py',
                                    'evaluation/playability_report.py'))})
    for field in ('effective_config', 'model_settings', 'execution'):
        a, b = control['data'].get(field), candidate['data'].get(field)
        if a is not None and b is not None:
            if field == 'effective_config':
                a, b = _canonical_config(a), _canonical_config(b)
            if a != b:
                differences.append({'kind': 'case_settings_difference', 'field': field})
    return differences


def _pair_checks(control, candidate, arm):
    a, b = control['data'], candidate['data']
    checks = {}
    for key in ('chains_intact', 'arc_endpoints_intact', 'roundtrip_v2_valid', 'roundtrip_v3_valid'):
        values = [receipt.get('checks', {}).get(key) for receipt in (a, b)]
        checks[key] = False if False in values else True if values == [True, True] else None
    # Compare per-category residual increases; one improvement cannot hide a
    # new violation of a different hard rule.
    if a.get('residual') is not None and b.get('residual') is not None:
        increases = sum(max(0, value - a['residual'].get(key, 0)) for key, value in b['residual'].items())
        same_notes = sorted(control['notes']) == sorted(candidate['notes'])
        identities_unknown = any(value > 0 and a['residual'].get(key, 0) > 0
                                 for key, value in b['residual'].items())
        checks['new_hard_clearance_violations'] = increases if increases > 0 or not identities_unknown or same_notes else None
    for field, key in (('frozen_speed', 'frozen_speed_violations_increase'),
                       ('frozen_flow', 'frozen_flow_violations_increase')):
        checks[key] = b[field] - a[field] if field in a and field in b else None
    recorded = b.get('speed', {}).get('playability_preserved')
    invariant_evidence = 'recorded_mutation_stage'
    if recorded is None:
        components = arm.split(',')
        has_repair = any(component in ('m1', 'm2', 'm3', 'm4', 'dwell', 'isolated', 'late') for component in components)
        if not has_repair:
            recorded = True
            invariant_evidence = 'no_direction_or_geometry_repair_stage'
        else:
            # Earlier pilot workers have no pre/post repair audit. A package
            # equality is useful independent evidence; inequality is unknown,
            # since later expansion may differ even with a preserving repair.
            before = Counter((round(note[0], 8), note[1]) for note in control['notes'])
            after = Counter((round(note[0], 8), note[1]) for note in candidate['notes'])
            recorded = True if before == after else None
            invariant_evidence = 'equal_exported_time_hand_multisets' if recorded else 'missing_mutation_stage_audit'
    checks['timestamps_hands_count_preserved'] = recorded
    return checks, invariant_evidence


def _observations(receipts):
    output = {}
    for side, receipt in receipts.items():
        data, notes = receipt['data'], receipt['notes']
        details = {key: data.get(key) for key in
                   ('onset', 'intro', 'counts', 'raw', 'stages', 'residual', 'actual_settings')}
        cutoff = (data.get('intro') or {}).get('no_notes_seconds', 0)
        details['intro_notes'] = sum(note[0] < cutoff - 1e-8 for note in notes)
        raw_count = (data.get('raw') or {}).get('position_notes')
        details['net_note_drop_share'] = max(0, 1 - len(notes) / raw_count) if raw_count else None
        details['hand_note_counts'] = {str(hand): sum(note[1] == hand for note in notes) for hand in (0, 1)}
        # The receipt's onset statistic belongs to the snap stage. Without
        # final audio/onset arrays, alignment is independently established
        # only when the final exported timestamps are unchanged.
        details['silence_alignment'] = 'unverified_without_audio_activity_reference'
        output[side] = details
    times = {side: Counter(round(note[0], 8) for note in receipt['notes'])
             for side, receipt in receipts.items()}
    output['timing'] = {'exported_timestamps_preserved': times['control'] == times['candidate'],
                       'added_onsets': sum((times['candidate'] - times['control']).values()),
                       'removed_onsets': sum((times['control'] - times['candidate']).values())}
    return output


def report_run(manifest, runs, frozen, *, split='development', candidate='m1', target='m1',
               indices=None, difficulties=None, replicates=None,
               bootstrap_samples=2000, bootstrap_seed=20261107, confirmation=False, selection=None,
               control_runs=None, control_equivalence=None):
    """Score every expected manifest pair, preserving missing and failed cases."""
    if isinstance(frozen, (str, Path)):
        frozen = json.loads(Path(frozen).read_text(encoding='utf-8'))
    control_runs = Path(runs) if control_runs is None else Path(control_runs)
    equivalence = None
    if control_runs.resolve() != Path(runs).resolve():
        if control_equivalence is None:
            raise ValueError('Reusing a separate control root requires verified control equivalence')
        from evaluation.playability_control_equivalence import validate_control_equivalence
        equivalence = validate_control_equivalence(control_equivalence,control_runs,runs,manifest)
        original_frozen=json.loads(Path(equivalence['frozen_controls'][split]['path']).read_text())
        if frozen != original_frozen:
            raise ValueError('Reused numerical freeze must match the immutable original panel exactly')
    if frozen['manifest_sha256'] != _hash(manifest):
        raise ValueError('Manifest changed after reference and margins were frozen')
    if frozen.get('numerical_sha256') is not None and frozen['numerical_sha256'] != _numerical_hash(frozen['reference'],frozen['margins']):
        raise ValueError('Numerical reference or margins changed after freezing')
    cases = _cases(manifest, split, indices, difficulties, replicates)
    confirmation = confirmation or split == 'holdout'
    confirmation_audit = {'required': confirmation, 'passed': None, 'problems': []}
    if confirmation:
        selection_path = Path(runs) / 'locked-selection.json' if selection is None else selection
        if isinstance(selection_path, (str, Path)):
            selection = json.loads(Path(selection_path).read_text()) if Path(selection_path).is_file() else {}
        declaration = selection or {}
        expectations = {'decision': 'ready_for_confirmation', 'arm': candidate, 'split': split,
            'manifest_sha256': frozen['manifest_sha256'],
            'numerical_sha256': frozen.get('numerical_sha256'),
            'case_keys': [_case_key(case) for case in cases],
            'replicates': sorted({case['replicate'] for case in cases})}
        for key, value in expectations.items():
            if declaration.get(key) != value:
                confirmation_audit['problems'].append('selection_mismatch:' + key)
        if target not in declaration.get('targets', []):
            confirmation_audit['problems'].append('primary_endpoint_not_locked')
        if not declaration.get('locked_at_utc'):
            confirmation_audit['problems'].append('missing_selection_timestamp')
        if frozen['scope'].get('case_keys') != expectations['case_keys'] or frozen['scope']['split'] != split:
            confirmation_audit['problems'].append('numerical_freeze_scope_mismatch')
        confirmation_audit['passed'] = not confirmation_audit['problems']
        if not confirmation_audit['passed']:
            # Never inspect reserved candidate packages before the selected
            # policy, endpoints, scope and numerical reference are locked.
            rows = [{key: case[key] for key in ('song','difficulty','seed')} for case in cases]
            result = paired_decision_report(rows, target=target, bootstrap_samples=bootstrap_samples)
            result.update(expected_cases=len(cases), candidate_arm=candidate, scorecards=rows,
                audit={'passed':False,'confirmation':confirmation_audit,'failures':[]},
                failures=[{'error':'Confirmation selection is incomplete or changed'}],
                human_skipped=frozen['reference']['skipped'])
            return result
    rows, failures, audit_failures, evidence, predating, policy_missing = [], [], [], [], [], []
    current_scorer = _scorer_sources()
    for name, digest in current_scorer.items():
        if frozen['scorer_sources'].get(name) != digest:
            audit_failures.append({'kind': 'frozen_scorer_changed', 'file': name,
                'frozen': frozen['scorer_sources'].get(name), 'current': digest})
    for case in cases:
        pair = {key: case[key] for key in ('song', 'difficulty', 'seed', 'replicate', 'index')}
        receipts = {}
        for arm, side in (('app1_control', 'control'), (candidate, 'candidate')):
            try:
                receipts[side] = _receipt(control_runs if side == 'control' else runs, case, arm)
                control_nps = pair.get('control', {}).get('nps')
                pair[side] = compact_scorecard(receipts[side]['notes'], frozen['reference'], control_nps=control_nps)
                pair[side + '_runtime_seconds'] = receipts[side]['data']['runtime_seconds']
            except Exception as error:
                failures.append({**{key: case[key] for key in ('song', 'difficulty', 'seed')},
                                 'arm': arm, 'error': str(error)})
        if len(receipts) == 2 and pair.get('control') is not None and pair.get('candidate') is not None:
            before, after = receipts['control'], receipts['candidate']
            for side, receipt in receipts.items():
                if (any(receipt['data'].get(field) is None for field in ('effective_config','policy_settings','stages'))
                        or any(receipt['worker'].get(field) is None for field in ('environment','model_settings'))):
                    policy_missing.append({**{key: case[key] for key in ('song','difficulty','seed')},'side':side})
            if after['receipt_completed_timestamp'] < datetime.fromisoformat(frozen['frozen_at_utc']).timestamp():
                predating.append({key: case[key] for key in ('song', 'difficulty', 'seed')})
            if all('end_to_end_seconds' in receipt['data'] for receipt in (before, after)):
                pair['control_runtime_seconds'] = before['data']['end_to_end_seconds']
                pair['candidate_runtime_seconds'] = after['data']['end_to_end_seconds']
                pair['runtime_scope'] = 'end_to_end_amortized_preparation'
            else:
                pair['runtime_scope'] = 'generation_only'
            problems = _pair_audit(before, after)
            if equivalence is not None:
                from evaluation.playability_control_equivalence import control_diff_allowed
                for problem in problems:
                    if control_diff_allowed(problem,before,equivalence['allowed_difference']):
                        problem.update(blocking=False,kind='verified_unreachable_control_module',
                            equivalence_manifest=str(Path(control_equivalence).resolve()))
                if after['worker'].get('sources',{}).get(equivalence['allowed_difference']['file']) != equivalence['allowed_difference']['candidate_sha256']:
                    problems.append({'kind':'candidate_source_outside_equivalence','blocking':True})
            pinned = frozen['controls'].get(_case_key(case))
            if pinned and pinned != before['provenance']:
                problems.insert(0, {'kind': 'frozen_control_changed', 'frozen': pinned,
                                    'current': before['provenance']})
            pair['checks'], invariant = _pair_checks(before, after, candidate)
            for problem in problems:
                audit_failures.append({**{key: case[key] for key in ('song', 'difficulty', 'seed')}, **problem})
            evidence.append({**{key: case[key] for key in ('song', 'difficulty', 'seed')},
                'invariant_evidence': invariant, 'provenance': {side: value['provenance'] for side, value in receipts.items()},
                'observations': _observations(receipts)})
        rows.append(pair)
    report = paired_decision_report(rows, target=target, frozen_margins=frozen['margins'],
        bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
    blocking_audits = [problem for problem in audit_failures if problem.get('blocking', True)]
    report['guardrails']['policy_settings'] = {'passed':None if policy_missing else True,
                                               'unverified_cases':policy_missing}
    if confirmation and predating:
        confirmation_audit['passed'] = False
        confirmation_audit['problems'].append('candidate_receipts_predate_numerical_freeze')
    report.update(candidate_arm=candidate, manifest=str(Path(manifest).resolve()),
        expected_cases=len(cases), failures=failures, human_skipped=frozen['reference']['skipped'],
        audit={'passed': not blocking_audits and not failures, 'failures': blocking_audits,
               'nonblocking_changes': [problem for problem in audit_failures if not problem.get('blocking', True)],
               'candidate_receipts_predating_freeze': predating,
               'confirmation': confirmation_audit,
               'controls_reused': equivalence is not None,
               'control_source_equivalence': ({'manifest':str(Path(control_equivalence).resolve()),
                   'sha256':_hash(control_equivalence),'reason':equivalence['reason'],
                   'allowed_difference':equivalence['allowed_difference'],
                   'original_control_root':str(control_runs.resolve()),
                   'real_replay_cases':len(equivalence['replays'])} if equivalence else None),
               'frozen_at_utc': frozen['frozen_at_utc'], 'scorer_sources': frozen['scorer_sources']},
        evidence=evidence, scorecards=rows, frozen_thresholds=frozen['thresholds'])
    if blocking_audits or failures or policy_missing or predating:
        report['decision'] = 'experimental'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('freeze', 'report'):
        sub = commands.add_parser(command)
        for option in ('manifest', 'runs', 'split', 'output'):
            sub.add_argument('--' + option, required=True)
        sub.add_argument('--indices', type=int, nargs='+')
        sub.add_argument('--difficulties', nargs='+', choices=DIFFICULTY_ORDER)
        sub.add_argument('--replicates', type=int, nargs='+')
        if command == 'freeze':
            sub.add_argument('--human-root', required=True)
        else:
            sub.add_argument('--frozen', required=True)
            sub.add_argument('--candidate', required=True)
            sub.add_argument('--target', choices=(*DISTANCE_METRICS, *DWELL_DISTANCE_METRICS), default='m1')
            sub.add_argument('--bootstrap-samples', type=int, default=2000)
            sub.add_argument('--bootstrap-seed', type=int, default=20261107)
            sub.add_argument('--confirmation', action='store_true')
            sub.add_argument('--selection')
            sub.add_argument('--control-runs')
            sub.add_argument('--control-equivalence')
    args = vars(parser.parse_args(argv))
    command, output = args.pop('command'), args.pop('output')
    function = freeze_run_reference if command == 'freeze' else report_run
    try:
        result = function(**args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    _write(output, result)
    print(json.dumps({'output': str(Path(output).resolve()), 'decision': result.get('decision'),
                      'cases': result.get('expected_cases', result.get('scope', {}).get('cases'))}))
