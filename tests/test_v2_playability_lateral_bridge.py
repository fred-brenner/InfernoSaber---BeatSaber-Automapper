"""Joint M1 proposals can bridge cells without spending reach/physical safety."""

from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from tools.config import config


def bridge():
    name = 'map_creation.v2_playability_lateral_bridge'
    assert importlib.util.find_spec(name) is not None, 'joint M1 cell bridge is missing'
    return importlib.import_module(name).find_lateral_bridge


@pytest.fixture
def witness(monkeypatch):
    from map_creation.v2_playability_policy import PatternGuard
    monkeypatch.setattr(config,'v2_playability_enabled',True)
    monkeypatch.setattr(config,'v2_playability_reach',True)
    monkeypatch.setattr(config,'v2_flow_min_turn_steps',3)
    passage = json.loads((Path(__file__).parent/'fixtures/v2_r7_horizontal_passage.json').read_text())
    notes = deepcopy(passage['notes'])
    guard = PatternGuard(notes,nps=4.5)
    mutable = {int(k) for k in passage['witness']} - set(passage['fixed_ids'])
    for note in notes:
        if note['id'] in mutable:
            note['d'] = passage['witness'][str(note['id'])]
    cap = lambda n: passage['arrival_caps'][str(n['id'])]
    return notes,guard,mutable,cap


def test_archived_r7_diagonal_proposal_finds_guarded_cell_bridge_without_mutation(witness):
    from map_creation.v2_playability_lateral import consecutive_horizontal_load
    from map_creation.v2_playability_reach import _Search,reach_costs
    notes,guard,_,cap = witness
    # A short opposite-diagonal pair keeps the original phrase's movement
    # variety. The old full-vertical witness is intentionally tested below:
    # it cannot spend placement/coordination safety to obtain M1 relief.
    mutable = {294,296}
    for note,old in zip(notes,guard.accepted):
        note['d'] = {294:5,296:6}.get(note['id'],old['d'])
    before = deepcopy(notes)
    accepted = deepcopy(guard.accepted)
    assert not guard.allows(mutable), 'fixture no longer exercises required joint recovery'
    edits = bridge()(notes,cap,mutable,guard)
    assert edits is not None
    assert edits and len(edits) <= 8
    assert set(edits) <= mutable
    assert notes == before
    assert guard.accepted == accepted
    assert guard.commits == 0
    for note in notes:
        if note['id'] in edits:
            x,y = edits[note['id']]
            assert abs(x-note['x'])+abs(y-note['y']) <= 1
            note['x'],note['y'] = x,y
    changed = {n['id'] for n,old in zip(notes,accepted) if n != old}
    search = _Search(notes,cap,4.5,False)
    assert search.legal({i for i,n in enumerate(notes) if n['id'] in changed})
    assert guard.allows(changed)
    old_reach,new_reach = reach_costs(accepted,4.5),reach_costs(notes,4.5)
    assert all(m['excess'] <= old_reach['metrics'][key]['excess']+1e-8
               for key,m in new_reach['metrics'].items())
    assert consecutive_horizontal_load(notes,4.5)['total_excess'] < consecutive_horizontal_load(accepted,4.5)['total_excess']
    assert [n['d'] for n in notes] == [n['d'] for n in before]


def test_exhausted_bridge_rolls_back_all_cells_and_never_commits(witness):
    notes,guard,mutable,_ = witness
    before,accepted = deepcopy(notes),deepcopy(guard.accepted)
    assert bridge()(notes,lambda n:0.,mutable,guard) is None
    assert notes == before
    assert guard.accepted == accepted
    assert guard.commits == 0


def test_unavailable_mutable_notes_cannot_be_changed_to_escape_guard(witness):
    notes,guard,_,cap = witness
    before = deepcopy(notes)
    assert bridge()(notes,cap,set(),guard) is None
    assert notes == before


def test_full_vertical_legacy_witness_cannot_spend_row_or_coordination_safety(witness):
    notes,guard,mutable,cap = witness
    before = deepcopy(notes)
    assert bridge()(notes,cap,mutable,guard) is None
    assert notes == before
    assert guard.commits == 0

