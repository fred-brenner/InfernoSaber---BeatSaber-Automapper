"""Reproducible CPU playability experiments on the production v2 pipeline.

Freeze a song manifest with prepare, then run selected arms into distinct output
directories. Each finished package has a receipt with its effective settings,
source hashes and actual seed. No export to the game's library is performed.
"""
import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from uuid import uuid4
from pathlib import Path, PureWindowsPath
import random
import subprocess
import sys
import time

import numpy as np

from evaluation.playability_report import _load_info, decode_package_notes

SWITCHES = {
    'priority': False, 'reach': False, 'm1': False, 'm2': False, 'm3': False, 'm4': False, 'dwell': False, 'isolated': False,
    'late_turns': False, 'slow90_seconds': 0., 'slow45_seconds': 0., 'recovery_ratio': 0.,
    'recovery_balanced': False,
}
POLICIES = {name: {name: True} for name in ('m1', 'm2', 'm3', 'm4', 'dwell', 'isolated')}
POLICIES['late'] = {'late_turns': True}
POLICIES['priority'] = dict(priority=True, reach=True, late_turns=True,
    slow90_seconds=.5, slow45_seconds=.65)
POLICIES['combined'] = dict(reach=True, m1=True, m2=True, m3=True, m4=True,
    dwell=True, isolated=True, late_turns=True, slow90_seconds=.5, slow45_seconds=.65,
    recovery_ratio=.9, recovery_balanced=True)
for name, field, values in [('slow90', 'slow90_seconds', (350, 500)),
                            ('slow45', 'slow45_seconds', (500, 650))]:
    for ms in values:
        POLICIES[f'{name}_{ms}'] = {field: ms / 1000}
for percent in (75, 90):
    POLICIES[f'hand_{percent}'] = {'recovery_ratio': percent / 100}


@contextmanager
def experiment_policy(arm):
    from map_creation.v2_generation import scoped_settings
    from tools.config import config
    values = SWITCHES.copy()
    if arm != 'app1_control':
        for component in arm.split(','):
            values.update(POLICIES[component])
    with scoped_settings():
        config.v2_playability_enabled = arm != 'app1_control'
        for field, value in values.items():
            setattr(config, 'v2_playability_' + field, value)
        yield


def _policy_settings():
    from tools.config import config
    fields = ('enabled', *SWITCHES)
    return {key: getattr(config, 'v2_playability_' + key) for key in fields}


def case_seed(audio_hash, requested_nps, replicate):
    key = f'{audio_hash}|{float(requested_nps):.12g}|{replicate}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ordinary(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=ordinary,
                                    allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def freeze_manifest(packages, development_list, batch_reports, output):
    from tools.utils.numpy_shorts import get_factor_from_max_speed
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    latest = {}
    for report in sorted(map(Path, batch_reports)):
        batch = json.loads(report.read_bytes())
        for package in batch['packages']:
            name = PureWindowsPath(package['path']).name
            variants = {}
            for row in package['variants']:
                options = row['report']['variants']
                # Combined labels identify the accepted source file.
                sub = next((x for x in options if x['filename'] == row.get('source_filename')), None)
                if sub is None:
                    if len(options) != 1:
                        raise ValueError(f'Ambiguous generation receipt for {name}/{row["difficulty"]}')
                    sub = options[0]
                diag = sub['diagnostics']
                internal = sub['internal_difficulty']
                # The saved applied cap is the authoritative movement budget.
                intensity = 100 * diag['speed_cap']['speed_cap_value'] / (
                    2 * internal * get_factor_from_max_speed(internal, .5, 1))
                variants[row['difficulty']] = dict(nps=internal / 4, intensity=round(intensity, 6),
                    temperature=diag['note_sample_temperature'], random_factor=diag['random_factor'],
                    intensity_shape=diag['intensity_postprocessing'])
            latest[name] = (str(report.resolve()), variants)
    population = {}
    for package in sorted(Path(packages).iterdir()):
        if not package.is_dir():
            continue
        _, info, bpm = _load_info(package)
        if info.get('_customData', {}).get('_editors', {}).get('InfernoSaber', {}).get('version') != '2.0.1.app1':
            continue
        audio = (package / info['_songFilename']).resolve()
        if not audio.is_relative_to(package.resolve()):
            raise ValueError(f'Invalid audio path in {package}')
        population[package.name] = dict(package=package.name, audio=str(audio), sha256=file_hash(audio),
                                        bpm=bpm, cover=str((package / info['_coverImageFilename']).resolve()))
    with Path(development_list).open(encoding='utf8', newline='') as stream:
        development = [r['package'] for r in csv.DictReader(stream)]
    if len(set(development)) != 30 or not set(development) <= population.keys():
        raise ValueError('Expected the original 30-song app1 sample')
    used = {population[name]['sha256'] for name in development}
    if len(used) != len(development):
        raise ValueError('Duplicate development audio')
    holdout = []
    for name in random.Random(20261108).sample(sorted(population), len(population)):
        song = population[name]
        if song['sha256'] not in used and 'break the horizon' not in name.casefold():
            holdout.append(name)
            used.add(song['sha256'])
        if len(holdout) == 30:
            break
    if len(holdout) != 30:
        raise ValueError('Insufficient distinct holdout recordings')
    output_rows = []
    for split, names in [('development', development), ('holdout', holdout)]:
        for name in names:
            receipt, variants = latest[name]
            output_rows.append({**population[name], 'split': split, 'variants': variants, 'receipt': receipt})
    save(output, dict(version='2.0.1.app1', population=len(population), holdout_draw_seed=20261108,
                      development_replicates=[11], holdout_replicates=[19, 43], songs=output_rows))


def _package_hashes(package):
    return {f.name: file_hash(f) for f in sorted(package.iterdir()) if f.is_file()}


def _completed_case(folder, previous, expected):
    for key, value in expected.items():
        if previous.get(key) != value:
            raise ValueError(f'Incompatible {key} at {folder}')
    if previous.get('package') != 'package' or not (folder / 'package').is_dir():
        raise ValueError(f'Missing package at {folder}')
    hashes = previous.get('package_hashes')
    if hashes and hashes == _package_hashes(folder / 'package'):
        return
    raise ValueError(f'Missing or changed package fingerprint at {folder}. Use a fresh output root.')


def _run_context(runtime):
    from map_creation.v2_generation import settings_snapshot
    from tools.config import config
    import tensorflow as tf
    from lighting_prediction.v1_lighting import lighting_paths
    files = {}
    for directory in ('map_creation', 'evaluation', 'tools/config', 'tools/utils', 'training/pipeline_v2', 'training/beat_models', 'lighting_prediction'):
        for f in sorted(Path(directory).glob('*.py')):
            files[str(f)] = file_hash(f)
    for reference in sorted(Path('map_creation').glob('v2_playability_*.json')):
        files[str(reference)] = file_hash(reference)
    models = {str(path): file_hash(path) for path in lighting_paths()}
    for root in (Path(runtime.root), Path(config.v2_arrows_reference_bundle) / 'arrows'):
        for f in sorted(root.rglob('*')):
            if f.is_file() and f.suffix not in ('.txt', '.md'):
                models[str(f.resolve())] = file_hash(f)
    return dict(git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                python=sys.version, tensorflow=tf.__version__, devices=[str(d) for d in tf.config.list_logical_devices()],
                sources=files, models=models, effective_settings=settings_snapshot()['config'])


def run_cases(manifest, split, output, arms, indices=None, difficulties=None, replicates=None):
    from map_creation.v2_generation import configure_inference, _difficulty_settings, settings_snapshot
    from map_creation.v2_export import write_package
    from tools.config import config, paths
    output = Path(output).resolve()
    manifest = Path(manifest)
    registry = json.loads(manifest.read_bytes())
    songs = [r for r in registry['songs'] if r['split'] == split]
    if not songs:
        raise ValueError(f'Empty or unknown split {split!r}')
    if indices is not None and any(i < 0 or i >= len(songs) for i in indices):
        raise ValueError('Song index outside split')
    if difficulties and any(d not in song['variants'] for d in difficulties for song in songs):
        raise ValueError('Unknown difficulty in selected split')
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('Choose distinct experiment arms')
    for arm in arms:
        if arm != 'app1_control' and any(k not in POLICIES for k in arm.split(',')):
            raise ValueError(f'Unknown experiment arm {arm!r}')
    replicates = replicates or registry[split + '_replicates']
    configure_inference()
    import tensorflow as tf
    from map_creation.v2_postprocessing import clearance_violations
    from map_creation.v2_pipeline import generate
    from map_creation.v2_runtime import V2Runtime
    worker_temp = output / 'scratch' / str(os.getpid())
    worker_temp.mkdir(parents=True, exist_ok=True)
    paths.temp_path = str(worker_temp) + '/'
    runtime = V2Runtime()
    context_file = output / 'workers' / f'{os.getpid()}-{uuid4().hex}.json'
    context = _run_context(runtime)
    context['model_settings'] = runtime.settings
    context['environment'] = {key: os.environ.get(key) for key in
                              ('CUDA_VISIBLE_DEVICES', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS',
                               'TF_USE_LEGACY_KERAS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    save(context_file, context)
    errors = []
    for index, song in enumerate(songs):
        if indices is not None and index not in indices:
            continue
        audio = Path(song['audio'])
        if file_hash(audio) != song['sha256']:
            raise ValueError(f'Changed input recording {audio}')
        prepared = runtime.prepare_audio(audio)
        # Production prepares a recording once for its set of difficulty variants.
        preparation_share = prepared.diagnostics['preparation_seconds'] / len(song['variants'])
        for label, variant in song['variants'].items():
            if difficulties and label not in difficulties:
                continue
            for replicate in replicates:
                seed = song.get('literal_seed', case_seed(song['sha256'], variant['nps'], replicate))
                # Each arm starts with the same app audio cache for this case.
                before_intensity = prepared.intensity
                for arm in arms:
                    folder = output / split / arm / f'{index:02d}' / label / str(replicate)
                    receipt = folder / 'result.json'
                    expected_case = dict(song=song['package'], difficulty=label, seed=seed, replicate=replicate,
                                         arm=arm, audio_hash=song['sha256'], actual_settings=variant)
                    if receipt.exists():
                        prev = json.loads(receipt.read_bytes())
                        _completed_case(folder, prev, expected_case)
                        continue
                    try:
                        with experiment_policy(arm):
                            config.add_beat_intensity_orig = variant['intensity']
                            config.v2_note_sample_temperature = variant['temperature']
                            runtime.settings['note_sample_temperature'] = variant['temperature']
                            for key in ('low_boost', 'high_reduction', 'scale', 'adjustment_strength'):
                                setattr(config, 'v2_intensity_' + key, variant['intensity_shape'][key])
                            _difficulty_settings(variant['nps'] * 4)
                            random.seed(seed)
                            np.random.seed(seed)
                            tf.random.set_seed(seed)
                            effective_config = settings_snapshot()['config']
                            policy_settings = _policy_settings()
                            started = time.perf_counter()
                            result = generate(prepared.song, variant['nps'], runtime.predictors,
                                prepared.encoder, runtime.settings, runtime.vocabularies,
                                lighting_generator=runtime.lighting, audio_path=audio,
                                random_factor=variant['random_factor'], intensity=before_intensity,
                                postprocessors=None, rng=np.random.default_rng(seed), logger=lambda message: None)
                            elapsed = time.perf_counter() - started
                            state = result.state
                            prepared.intensity = state['raw_intensity'].copy()
                            package = folder / 'package'
                            write_package(package, audio.stem, float(prepared.song.metadata['bpm']),
                                [dict(difficulty=label, internal_difficulty=variant['nps'] * 4,
                                      state=state, jump_speed_offset=float(config.jump_speed_offset))],
                                audio, cover_path=song.get('cover'), version='v3')
                            end_to_end = time.perf_counter() - started + preparation_share
                            decoded = decode_package_notes(package, label)['notes']
                            expected = sorted([tuple(n[k] for k in ('time','hand','x','y','d')) for n in state['notes']],
                                              key=lambda n:n[0])
                            # Exports may order simultaneous notes by color rather than by model ID.
                            np.testing.assert_allclose(sorted(decoded), sorted(expected), rtol=0, atol=1e-8)
                            # Use the legacy serializer/decoder too, checking its retained colored notes.
                            from map_creation.v2_export import serialize_beatmap
                            from evaluation.playability_report import _decode_notes
                            bpm = float(prepared.song.metadata['bpm'])
                            legacy, _ = serialize_beatmap(state, bpm, 'v2')
                            legacy_notes = _decode_notes(legacy, '2', bpm)
                            np.testing.assert_allclose(sorted(legacy_notes), sorted(expected), rtol=0, atol=1e-8)
                            r = result.diagnostics
                            post = r.get('speed_cap_post_expansion', {})
                            residual = clearance_violations(state['notes'])
                            # Both export paths and the pipeline validate anchor and endpoint IDs.
                            checks = dict(chains_intact=True, arc_endpoints_intact=True,
                                          roundtrip_v2_valid=True, roundtrip_v3_valid=True)
                            save(receipt, dict(song=song['package'], audio_hash=song['sha256'],
                                difficulty=label, seed=seed, replicate=replicate, arm=arm,
                                worker=str(context_file.relative_to(output)), runtime_seconds=elapsed,
                                effective_config=effective_config, policy_settings=policy_settings,
                                end_to_end_seconds=end_to_end, preparation_share_seconds=preparation_share,
                                package='package', package_hashes=_package_hashes(package), actual_settings=variant, residual=residual,
                                frozen_speed=post.get('speed_cap_retained_skipped', 0),
                                frozen_flow=post.get('speed_cap_flow_skipped', 0), checks=checks,
                                reach_final=r.get('reach_post_expansion'),
                                onset=r.get('onset_snap'), intro=r.get('intro_zones'), counts=r['final_counts'],
                                raw=r['raw_decisions'], speed=r['speed_cap'], stages=result.stages))
                            print(f'{split} {index+1}/{len(songs)} {label} {arm}: {len(expected)} notes, {elapsed:.1f}s',flush=True)
                    except Exception as exc:
                        errors.append(dict(song=song['package'], label=label, arm=arm, seed=seed, error=repr(exc)))
                        print(errors[-1], flush=True)
    if errors:
        save(output/'workers'/f'{os.getpid()}-failures.json',errors)
        raise RuntimeError(f'{len(errors)} generation or export failures')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='cmd', required=True)
    prep = commands.add_parser('prepare')
    for flag in ('packages', 'development-list', 'output'):
        prep.add_argument('--'+flag, required=True)
    prep.add_argument('--batch-reports', nargs='+', required=True)
    run = commands.add_parser('run')
    for flag in ('manifest','output','split'):
        run.add_argument('--'+flag, required=True)
    run.add_argument('--indices', nargs='+', type=int)
    run.add_argument('--replicates', nargs='+', type=int)
    run.add_argument('--arms', nargs='+', required=True)
    run.add_argument('--difficulties', nargs='+')
    args=vars(parser.parse_args())
    cmd=args.pop('cmd')
    (freeze_manifest if cmd=='prepare' else run_cases)(**args)


if __name__ == '__main__':
    main()
