"""The repair budget must survive the primary-to-frozen ramp renumbering."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from map_creation import v2_postprocessing as post
from tools.config import config


@pytest.fixture
def alles_neu(monkeypatch):
    fixture = json.loads((Path(__file__).parent / 'fixtures/v2_alles_neu_start_ramp.json').read_text())
    for key, value in fixture['movement_settings'].items():
        monkeypatch.setattr(config, key, value)
    from evaluation.playability_experiments import SWITCHES
    monkeypatch.setattr(config, 'v2_playability_enabled', False)
    for field, value in SWITCHES.items():
        monkeypatch.setattr(config, 'v2_playability_' + field, value)
    notes = fixture['notes']
    # Supply the original passage's bridge allowance without carrying audio,
    # weights or an unrelated whole-map timeline into the test.
    notes.extend(dict(id=10000+i, time=1000.+i, hand=0, x=0, y=1,
                      d=i%2, kind='retained') for i in range(200))
    return notes


def test_exact_alles_neu_repair_does_not_add_a_frozen_ramp_violation(alles_neu, monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', False)
    control = deepcopy(alles_neu)
    post.apply_speed_cap(control)
    baseline = post.apply_speed_cap(control, drop_retained=False)
    assert baseline['speed_cap_retained_skipped'] == 1
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_m1', True)
    candidate = deepcopy(alles_neu)
    repair = post.apply_speed_cap(candidate)
    result = post.apply_speed_cap(candidate, drop_retained=False)
    assert result['speed_cap_retained_skipped'] <= 1
    assert repair['speed_cap_dropped'] == 7
    assert [(n['id'], n['time'], n['hand'], n['x'], n['y']) for n in candidate] == [
        (n['id'], n['time'], n['hand'], n['x'], n['y']) for n in control]


def test_every_closure_arm_gets_the_conservative_retained_timeline_budget(alles_neu, monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    observed = {}
    def inspect_budgets(notes, cap, stats, *args):
        observed.update({n['id']:float(cap(n)) for n in notes})
    monkeypatch.setattr(post, '_apply_playability_experiments', inspect_budgets)
    post.apply_speed_cap(alles_neu)
    assert observed[23] == pytest.approx(21.673457015457785)
    assert observed[24] == pytest.approx(22.525138644470864)


def test_mutable_earlier_notes_do_not_inflate_the_future_frozen_budget(alles_neu, monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    # An earlier heuristic can disappear before the frozen pass. Its slot is
    # not safe support for a repair's future retained-arrival ramp budget.
    alles_neu[0]['kind'] = 'heuristic'
    observed = {}
    def inspect_budgets(notes, cap, stats, *args):
        observed.update({n['id']:float(cap(n)) for n in notes})
    monkeypatch.setattr(post, '_apply_playability_experiments', inspect_budgets)
    post.apply_speed_cap(alles_neu)
    assert observed[24] == pytest.approx(21.673457015457785)
