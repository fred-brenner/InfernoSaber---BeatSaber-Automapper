"""Narrow evidence for reusing immutable master-off generation controls.

The only permitted production difference is one specific geometry module
revision. Every control must explicitly disable the master switch, every
other production source must match, and real replays must agree byte for byte
and preserve the captured Python/NumPy RNG states on same-source repetition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from evaluation.playability_benchmark_run import _cases, _case_key, _hash, _receipt, _pair_audit

GEOMETRY_FILE='map_creation/v2_playability_geometry.py'


def scoped_source_difference(control_sources,candidate_sources):
    names=set(control_sources)|set(candidate_sources)
    changed=[name for name in names if control_sources.get(name)!=candidate_sources.get(name)]
    if changed!=[GEOMETRY_FILE]:
        raise ValueError('Control reuse requires exactly the declared geometry-module difference')
    return {'file':GEOMETRY_FILE,'control_sha256':control_sources[GEOMETRY_FILE],
            'candidate_sha256':candidate_sources[GEOMETRY_FILE]}


def control_diff_allowed(problem,control,approval):
    return (problem.get('kind')=='source_hash_difference'
        and problem.get('file')==approval.get('file')==GEOMETRY_FILE
        and problem.get('control')==approval.get('control_sha256')
        and problem.get('candidate')==approval.get('candidate_sha256')
        and control['data'].get('effective_config',{}).get('v2_playability_enabled') is False
        and control['data'].get('policy_settings',{}).get('enabled') is False)


def _rng_record(root,case):
    initial=np.random.default_rng(case['seed']).bit_generator.state
    fingerprint=hashlib.sha256(json.dumps(initial,sort_keys=True,allow_nan=False).encode()).hexdigest()
    records=json.loads((Path(root)/'rng-replay.json').read_text())
    matches=[item for item in records if item['audio']==str(Path(case['audio']).resolve())
        and item['requested_nps']==case['actual_settings']['nps']
        and item['rng_before']['explicit_numpy']==fingerprint]
    if len(matches)!=1:
        raise ValueError('Missing or ambiguous RNG replay for '+_case_key(case))
    if matches[0].get('geometry_never_imported') is not True or matches[0].get('master_disabled') is not True:
        raise ValueError('A control replay imported the changed geometry module')
    return matches[0]


def build_control_equivalence(control_root,candidate_root,manifest,proofs):
    """Verify old/new controls and repeated new controls, returning a manifest."""
    control_root,candidate_root=Path(control_root).resolve(),Path(candidate_root).resolve()
    old_lock=json.loads((control_root/'source-lock.json').read_text())
    new_lock=json.loads((candidate_root/'source-lock.json').read_text())
    difference=scoped_source_difference(old_lock['sources'],new_lock['sources'])
    frozen_files={}
    control_receipts={}
    for split in ('development','holdout'):
        path=control_root/f'frozen-{split}.json'
        frozen=json.loads(path.read_text())
        if frozen['manifest_sha256']!=_hash(manifest):raise ValueError('Changed original manifest')
        frozen_files[split]={'path':str(path),'sha256':_hash(path),'frozen_at_utc':frozen['frozen_at_utc'],
                            'numerical_sha256':frozen['numerical_sha256']}
        for key,provenance in frozen['controls'].items():
            receipt_path=Path(provenance['receipt'])
            if _hash(receipt_path)!=provenance['receipt_sha256']:raise ValueError('Changed original control receipt')
            receipt=json.loads(receipt_path.read_text())
            if (receipt.get('effective_config',{}).get('v2_playability_enabled') is not False
                    or receipt.get('policy_settings',{}).get('enabled') is not False):
                raise ValueError('Reused control did not explicitly disable the master switch')
            control_receipts[key]={'path':str(receipt_path),'sha256':provenance['receipt_sha256']}
    replays=[]
    proof_files={}
    for proof in proofs:
        a_root,b_root=Path(proof['left_root']),Path(proof['repeat_root'])
        for root in (a_root,b_root):proof_files[str((root/'rng-replay.json').resolve())]=_hash(root/'rng-replay.json')
        cases=_cases(manifest,proof['split'],proof['indices'],proof['difficulties'],None)
        for case in cases:
            old=_receipt(control_root,case,'app1_control')
            fresh=_receipt(a_root,case,'app1_control')
            repeat=_receipt(b_root,case,'app1_control')
            expected=old['provenance']['package_dat_sha256']
            if expected!=fresh['provenance']['package_dat_sha256'] or expected!=repeat['provenance']['package_dat_sha256']:
                raise ValueError('Exported beatmap/Info bytes changed: '+_case_key(case))
            rng_a,rng_b=_rng_record(a_root,case),_rng_record(b_root,case)
            if rng_a!=rng_b:raise ValueError('Same-source RNG or intermediate-state replay changed: '+_case_key(case))
            for problem in _pair_audit(old,fresh):
                if problem.get('blocking',True) and not control_diff_allowed(problem,old,difference):
                    raise ValueError('Unapproved replay provenance difference: '+str(problem))
            for item in (fresh,repeat):
                receipt=item['provenance']['receipt'];proof_files[receipt]=_hash(receipt)
            replays.append({'case_key':_case_key(case),'song':case['song'],'difficulty':case['difficulty'],
                'seed':case['seed'],'split':case['split'],'original':old['provenance'],
                'corrected':fresh['provenance'],'same_source_repeat':repeat['provenance'],
                'beatmap_and_hazard_bytes_equal':True,'same_source_rng_and_state_equal':True,
                'geometry_never_imported':True,'rng':rng_a})
    if len(replays)<4 or len({row['difficulty'] for row in replays})<2 or len({row['seed'] for row in replays})<2:
        raise ValueError('Control reuse needs several real replays spanning difficulties and seeds')
    result={'schema_version':1,'kind':'unreachable_master_off_control_module',
        'control_root':str(control_root),'candidate_root':str(candidate_root),
        'manifest_sha256':_hash(manifest),'allowed_difference':difference,
        'original_source_lock':{'path':str(control_root/'source-lock.json'),'sha256':_hash(control_root/'source-lock.json')},
        'candidate_source_lock':{'path':str(candidate_root/'source-lock.json'),'sha256':_hash(candidate_root/'source-lock.json')},
        'frozen_controls':frozen_files,'control_receipts':control_receipts,
        'replays':replays,'proof_files':proof_files,
        'reason':'The sole changed production file is imported only after the master-enabled guard. All 360 controls explicitly disable it; fresh replays forbid its import and match original exported note/hazard bytes. Same-source repetitions match explicit NumPy, global NumPy and Python RNG states plus selected intermediate-state fingerprints.'}
    return result


def validate_control_equivalence(path,control_root,candidate_root,manifest):
    """Validate immutable references before applying any scoped report exemption."""
    path=Path(path);value=json.loads(path.read_text())
    if (value.get('kind')!='unreachable_master_off_control_module'
            or value['control_root']!=str(Path(control_root).resolve())
            or value['candidate_root']!=str(Path(candidate_root).resolve())
            or value['manifest_sha256']!=_hash(manifest)):
        raise ValueError('Control equivalence belongs to a different cohort')
    for descriptor in (value['original_source_lock'],value['candidate_source_lock'],*value['frozen_controls'].values(),*value['control_receipts'].values()):
        if _hash(descriptor['path'])!=descriptor['sha256']:raise ValueError('Changed immutable control-equivalence input')
    expected_receipts={}
    for descriptor in value['frozen_controls'].values():
        original=json.loads(Path(descriptor['path']).read_text())
        expected_receipts.update({key:{'path':item['receipt'],'sha256':item['receipt_sha256']}
                                 for key,item in original['controls'].items()})
    if expected_receipts!=value['control_receipts']:
        raise ValueError('Control equivalence must retain every original panel receipt')
    for file,digest in value['proof_files'].items():
        if _hash(file)!=digest:raise ValueError('Changed replay proof')
    old=json.loads(Path(value['original_source_lock']['path']).read_text())['sources']
    new=json.loads(Path(value['candidate_source_lock']['path']).read_text())['sources']
    if scoped_source_difference(old,new)!=value['allowed_difference']:
        raise ValueError('Control equivalence exceeds the approved source difference')
    if _hash(GEOMETRY_FILE)!=value['allowed_difference']['candidate_sha256']:
        raise ValueError('Current geometry differs from the approved corrected revision')
    return value
