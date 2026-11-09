"""Control reuse is restricted to one unreachable module and explicit hashes."""
from pathlib import Path
import sys
import pytest

ROOT_DIR=Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:sys.path.insert(0,str(ROOT_DIR))
from evaluation.playability_control_equivalence import control_diff_allowed, scoped_source_difference


def test_only_the_declared_geometry_revision_can_differ():
    path='map_creation/v2_playability_geometry.py'
    old={path:'before','map_creation/v2_pipeline.py':'same'}
    new={path:'after','map_creation/v2_pipeline.py':'same'}
    assert scoped_source_difference(old,new)=={'file':path,'control_sha256':'before','candidate_sha256':'after'}
    new['map_creation/v2_pipeline.py']='different'
    with pytest.raises(ValueError,match='exactly'):
        scoped_source_difference(old,new)


def test_equivalence_cannot_waive_enabled_control_or_unknown_hashes():
    path='map_creation/v2_playability_geometry.py'
    approval={'file':path,'control_sha256':'before','candidate_sha256':'after'}
    problem={'kind':'source_hash_difference','file':path,'control':'before','candidate':'after'}
    control={'data':{'effective_config':{'v2_playability_enabled':False},'policy_settings':{'enabled':False}}}
    assert control_diff_allowed(problem,control,approval)
    control['data']['policy_settings']['enabled']=True
    assert not control_diff_allowed(problem,control,approval)
    control['data']['policy_settings']['enabled']=False
    problem['candidate']='unapproved-next-version'
    assert not control_diff_allowed(problem,control,approval)


def test_separate_control_root_requires_explicit_verified_equivalence(tmp_path):
    from evaluation.playability_benchmark_run import report_run
    import json,hashlib
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    frozen={'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    with pytest.raises(ValueError,match='equivalence'):
        report_run(manifest,tmp_path/'new',frozen,control_runs=tmp_path/'old')
