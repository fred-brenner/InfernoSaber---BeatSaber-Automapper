"""Exact development inversion regression; deliberately loaded from /tmp."""
from copy import deepcopy

import pytest

from map_creation import v2_playability_geometry as geometry
from map_creation import v2_postprocessing as post
from tools.config import config


@pytest.fixture(autouse=True)
def movement(monkeypatch):
    monkeypatch.setattr(config, 'max_speed',26.)
    monkeypatch.setattr(config, 'v2_playability_enabled',True)
    monkeypatch.setattr(config, 'v2_playability_late_turns',False)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds',0.)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds',0.)


def passage():
    # Raw IDs remain ordered after per-note onset snapping. The opposite-hand
    # onset at211.162 moves after ID1463's onset at211.135.
    return [dict(id=1461,time=210.9734132637974,hand=1,x=1,y=0,d=1,kind='retained'),
            dict(id=1462,time=211.16226757369614,hand=0,x=2,y=0,d=0,kind='retained'),
            dict(id=1463,time=211.13470792225596,hand=1,x=3,y=1,d=0,kind='retained'),
            dict(id=1464,time=211.29600258071454,hand=1,x=1,y=0,d=6,kind='retained'),
            dict(id=1466,time=211.61859189763166,hand=1,x=1,y=0,d=3,kind='retained')]


@pytest.mark.parametrize('sightline,coordination,interference',[(False,False,True),(True,True,True)])
def test_onset_time_inversion_cannot_introduce_a_retained_clash(sightline,coordination,interference):
    notes=passage();before=deepcopy(notes)
    assert not any(post.clearance_violations(notes).values())
    geometry.repair_geometry(notes,lambda n:33.54,sightline=sightline,coordination=coordination,interference=interference)
    assert not any(post.clearance_violations(notes).values())
    assert [n['id'] for n in notes]==[n['id'] for n in before]
    assert [(n['id'],n['time'],n['hand'],n['d']) for n in notes]==[(n['id'],n['time'],n['hand'],n['d']) for n in before]
    ordered=sorted(deepcopy(before),key=lambda n:(n['time'],n['id']))
    geometry.repair_geometry(ordered,lambda n:33.54,sightline=sightline,coordination=coordination,interference=interference)
    assert {n['id']:(n['x'],n['y']) for n in notes}=={n['id']:(n['x'],n['y']) for n in ordered}
    snapshot=deepcopy(notes)
    post.enforce_note_flow_and_clearance(notes)
    assert notes==snapshot


def test_final_real_clearance_audit_rolls_back_a_local_validation_miss(monkeypatch):
    notes=[dict(id=9,time=1.,hand=0,x=0,y=0,d=1,kind='retained'),
           dict(id=17,time=1.2,hand=1,x=0,y=2,d=0,kind='retained'),*passage()]
    before=deepcopy(notes)
    original=geometry._Geometry.proposal
    def missed_commit(self,edits,*,commit=False):
        result=original(self,edits,commit=commit)
        if commit and result is not None:
            # Simulate a stale local neighborhood accepting the real invalid
            # passage cell while another legal position edit has committed.
            target=next(n for n in self.notes if n['id']==1464)
            target['x']=2
        return result
    monkeypatch.setattr(geometry._Geometry,'proposal',missed_commit)
    report=geometry.repair_geometry(notes,lambda n:33.54,sightline=True)
    assert report['geometry_rolled_back'] is True
    assert notes==before
    assert report['geometry_changed']==0
    assert report['geometry_after']==report['geometry_before']
    assert report['geometry_attempted_clearance']['swing_into_pairs']>0
