"""Plan and run resumable CPU cohorts after a source freeze.

This wrapper uses ordinary full production generation and export. Controls,
control-only margin freezing, development candidates and held-out selection
are distinct steps. No generation or corpus access occurs for ``prepare``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from evaluation.playability_benchmark_run import _hash, _write

_SOURCE_DIRECTORIES = ('app_helper', 'beat_prediction', 'evaluation', 'lighting_prediction',
    'map_creation', 'preprocessing', 'tools', 'training')


def source_fingerprint():
    paths = {path for directory in _SOURCE_DIRECTORIES for path in Path(directory).rglob('*.py')
             if '__pycache__' not in path.parts}
    paths.update(Path('.').glob('*.py'))
    # Calculation hashes are pinned separately when numerical margins freeze.
    # The generation lock includes its runner and package decoder, so adapter
    # reviews can finish without invalidating completed production controls.
    paths = {path for path in paths if path.parts[0] != 'evaluation' or path.name in (
        'playability_experiments.py', 'playability_report.py')}
    return {str(path): _hash(path) for path in sorted(paths)}


def build_schedule(manifest, output, *, components, workers=4):
    """Create full-export stages and leave-one-component-out development arms."""
    if workers < 1 or workers > 4:
        raise ValueError('Use one to four CPU workers')
    components = list(components)
    if not components or len(set(components)) != len(components):
        raise ValueError('Expected distinct combined-policy components')
    registry = json.loads(Path(manifest).read_text(encoding='utf-8'))
    combined = ','.join(components)
    arms = [combined] + [','.join(component for component in components if component != removed)
                         for removed in components if len(components) > 1]
    stages = {}
    specifications = (
        ('development_controls', 'development', ['app1_control'], []),
        ('development_candidates', 'development', arms, ['frozen-development.json']),
        ('holdout_controls', 'holdout', ['app1_control'], []),
        ('holdout_candidate', 'holdout', ['<locked-selection>'], ['locked-selection.json']),
    )
    for name, split, stage_arms, requires in specifications:
        songs = [row for row in registry['songs'] if row['split'] == split]
        replicates = registry[split + '_replicates']
        maps = sum(len(song['variants']) for song in songs) * len(replicates) * len(stage_arms)
        shards = [list(range(worker, len(songs), workers)) for worker in range(workers)]
        stages[name] = {'split': split, 'arms': stage_arms, 'requires': requires,
                       'songs': len(songs), 'replicates': replicates, 'maps': maps,
                       'shards': [shard for shard in shards if shard]}
    return {'schema_version': 1, 'manifest': str(Path(manifest).resolve()),
            'manifest_sha256': _hash(manifest), 'output': str(Path(output).resolve()),
            'workers': workers, 'combined': combined, 'components': components,
            'stages': stages, 'checkpoint_reuse': False}


def prepare_cohort(manifest, output, *, components, workers=4):
    """Pin sources and interpreter in a fresh root; existing locks are immutable."""
    output = Path(output)
    lock = output / 'source-lock.json'
    if lock.exists():
        raise FileExistsError(f'Cohort already locked: {lock}')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output root for the locked cohort')
    schedule = build_schedule(manifest, output, components=components, workers=workers)
    output.mkdir(parents=True, exist_ok=True)
    _write(output / 'schedule.json', schedule)
    _write(lock, {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'sources': source_fingerprint(), 'python': sys.version,
                  'interpreter': str(Path(sys.executable).resolve()),
                  'manifest_sha256': schedule['manifest_sha256'],
                  'cpu_environment': {key: os.environ.get(key) for key in (
                      'CUDA_VISIBLE_DEVICES', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS',
                      'TF_USE_LEGACY_KERAS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')}})
    return schedule


def validate_stage(output, stage):
    """Refuse source drift or an undeclared candidate before launching workers."""
    output = Path(output)
    lock = json.loads((output / 'source-lock.json').read_text(encoding='utf-8'))
    schedule = json.loads((output / 'schedule.json').read_text(encoding='utf-8'))
    if lock['sources'] != source_fingerprint():
        raise ValueError('Source files changed after the cohort freeze; use a fresh cohort')
    if lock['python'] != sys.version or lock['interpreter'] != str(Path(sys.executable).resolve()):
        raise ValueError('CPU interpreter changed after the cohort freeze')
    current_environment = {key: os.environ.get(key) for key in lock['cpu_environment']}
    if lock['cpu_environment'] != current_environment or os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise ValueError('Load the same recorded CPU profile before cohort execution')
    if _hash(schedule['manifest']) != lock['manifest_sha256']:
        raise ValueError('Manifest changed after the cohort freeze')
    spec = schedule['stages'][stage]
    for requirement in spec['requires']:
        if not (output / requirement).is_file():
            raise ValueError(f'{stage} requires {requirement} before candidate generation')
    if stage == 'development_candidates':
        frozen = json.loads((output / 'frozen-development.json').read_text(encoding='utf-8'))
        if frozen['manifest_sha256'] != lock['manifest_sha256']:
            raise ValueError('Margin freeze belongs to a different manifest')
        if lock.get('control_reuse'):
            from evaluation.playability_control_equivalence import validate_control_equivalence
            validate_control_equivalence(output/'control-equivalence.json',lock['control_reuse']['original_root'],output,schedule['manifest'])
    if stage == 'holdout_candidate':
        selection = json.loads((output / 'locked-selection.json').read_text(encoding='utf-8'))
        if selection.get('decision') != 'ready_for_confirmation':
            raise ValueError('Development selection is not ready for reserved confirmation')
        if selection.get('arm') not in schedule['stages']['development_candidates']['arms']:
            raise ValueError('Selected arm was not in the declared development comparison')
        if not (output / 'frozen-development.json').is_file():
            raise ValueError('Reserved confirmation requires the development reference and margins')
        if not (output / 'frozen-holdout.json').is_file():
            raise ValueError('Freeze the reserved numerical reference before confirmation')
        frozen = json.loads((output / 'frozen-holdout.json').read_text(encoding='utf-8'))
        required = {'split':'holdout', 'manifest_sha256':lock['manifest_sha256'],
                    'numerical_sha256':frozen.get('numerical_sha256'),
                    'case_keys':frozen['scope'].get('case_keys'),
                    'replicates':frozen['scope'].get('replicate_values')}
        if (any(selection.get(key) != value for key,value in required.items())
                or not selection.get('targets') or not selection.get('locked_at_utc')):
            raise ValueError('Complete the confirmation declaration before reserved generation')
        spec = {**spec, 'arms': [selection['arm']]}
    return schedule, spec


def execute_stage(output, stage):
    """Run independent song shards; completed receipts are resumed by run_cases."""
    schedule, spec = validate_stage(output, stage)
    output = Path(output)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    children = []
    try:
        for worker, indices in enumerate(spec['shards']):
            command = [sys.executable, '-s', '-u', '-m', 'evaluation.playability_experiments', 'run',
                '--manifest', schedule['manifest'], '--output', str(output.resolve()),
                '--split', spec['split'], '--arms', *spec['arms'], '--indices', *map(str, indices)]
            stream = (logs / f'{stage}-{worker}.log').open('a', encoding='utf-8')
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            children.append((worker, process, stream))
        statuses = []
        for worker, process, stream in children:
            code = process.wait()
            stream.close()
            statuses.append({'worker': worker, 'exit_code': code})
        _write(output / f'{stage}-status.json', statuses)
        if any(row['exit_code'] for row in statuses):
            raise RuntimeError('Cohort worker failed; inspect logs and resume the same stage')
        return statuses
    finally:
        for _, process, stream in children:
            if process.poll() is None:
                process.terminate()
            if not stream.closed:
                stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--manifest', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--components', nargs='+', required=True)
    prepare.add_argument('--workers', type=int, default=4)
    run = commands.add_parser('run')
    run.add_argument('--output', required=True)
    run.add_argument('--stage', required=True, choices=('development_controls', 'development_candidates',
        'holdout_controls', 'holdout_candidate'))
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    result = prepare_cohort(**args) if command == 'prepare' else execute_stage(**args)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
