"""Supported hand recovery changes only audited primary survivors."""
from copy import deepcopy

import numpy as np
import pytest

from tools.config import config


def test_combined_capture_preserves_original_position_and_arrow_inputs(monkeypatch):
    from test_v2_generation_pipeline import audio_song, vocabulary, Timing, processors, lighting
    from map_creation.v2_pipeline import generate
    from training.pipeline_v2.pipeline import dummy_predictors
    from training.pipeline_v2.settings import preparation_settings
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', False)
    monkeypatch.setattr(config, 'single_notes_only_flag', True)
    monkeypatch.setattr(config, 'v2_playability_recovery_balanced', True)
    monkeypatch.setattr(config, 'v2_note_sample_temperature', 0.)
    times = [.4, .5, .7, .85, 1., 1.1, 1.3, 1.5]
    vocab = vocabulary()
    def run(ratio):
        monkeypatch.setattr(config, 'v2_playability_recovery_ratio', ratio)
        predictors = dummy_predictors(vocab)
        predictors['timing'] = Timing()
        observed = {'positions': [], 'arrows': []}
        class Positions:
            status = 'dummy'
            def __call__(self, inputs):
                index = len(observed['positions'])
                observed['positions'].append(deepcopy(inputs))
                red, blue = np.zeros(13), np.zeros(13)
                red[[0, 5]] = [.9, .1] if index == 0 else [.5, .5]
                blue[[0, 8]] = [.1, .9] if index == 0 else [.52, .48]
                return {'red': red[None], 'blue': blue[None]}
        base_arrow = predictors['arrows']
        def arrows(inputs):
            observed['arrows'].append(deepcopy(inputs))
            return base_arrow.predict(inputs, verbose=0)
        predictors['positions'], predictors['arrows'] = Positions(), arrows
        post = processors()
        post.postprocess_timing = lambda *args, **kwargs: ([dict(time=t, count=1) for t in times], {})
        # Exercise the production reattachment across a reconstructing cleanup boundary.
        post.postprocess_notes = lambda notes, *args, **kwargs: (
            [{k:v for k,v in n.items() if not k.startswith('_')} for n in notes], {})
        result = generate(audio_song(), 4., predictors,
            lambda t: np.zeros((len(t), 16), np.float32), preparation_settings(), vocab,
            lighting_generator=lighting, postprocessors=post, random_factor=0,
            rng=np.random.default_rng(3), logger=lambda _:None)
        return result, observed
    original, first = run(0.)
    deferred, second = run(.9)
    np.testing.assert_array_equal(original.state['raw_model_notes'], deferred.state['raw_model_notes'])
    for stage in first:
        assert len(first[stage]) == len(second[stage])
        for a, b in zip(first[stage], second[stage]):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
    assert any('_recovery_alternatives' in n for n in deferred.state['notes'])
    assert deferred.diagnostics['raw_decisions']['recovery_hand_swaps'] == 0
    assert deferred.diagnostics['raw_decisions']['recovery_deferred_notes'] > 0


@pytest.fixture(autouse=True)
def recovery_policy(monkeypatch):
    monkeypatch.setattr(config, 'v2_playability_enabled', True)
    monkeypatch.setattr(config, 'v2_playability_reach', True)
    monkeypatch.setattr(config, 'v2_playability_slow90_seconds', .5)
    monkeypatch.setattr(config, 'v2_playability_slow45_seconds', .65)
    monkeypatch.setattr(config, 'v2_playability_recovery_ratio', .9)
    monkeypatch.setattr(config, 'v2_flow_min_turn_steps', 3)


def _endpoint_witness():
    notes = [dict(id=0, time=.3, hand=1, x=3, y=1, d=0, kind='retained')]
    notes.extend(dict(id=i+1, time=t, hand=0, x=0, y=1, d=i%2, kind='retained')
                 for i, t in enumerate((.5, .7, .9, 1.1)))
    notes[-1]['_recovery_alternatives'] = [dict(hand=1, x=3, y=1, support=1.)]
    return notes


def test_supported_local_recovery_strictly_reduces_fast_reversals_and_preserves_other_notes():
    from map_creation.v2_playability_recovery import repair_recovery, primary_signature
    from map_creation.v2_playability_directions import axis_run_events
    from map_creation.v2_playability_policy import PatternGuard
    from map_creation import v2_postprocessing as post
    notes = _endpoint_witness()
    before = deepcopy(notes)
    signature = primary_signature(notes)
    result = repair_recovery(notes, lambda _:100., guard=PatternGuard(notes))
    assert notes[:-1] == before[:-1]
    assert [(n['id'],n['time']) for n in notes] == [(n['id'],n['time']) for n in before]
    assert (notes[-1]['hand'],notes[-1]['x'],notes[-1]['y'],notes[-1]['d']) == (1,3,1,1)
    assert result['recovery_hand_swaps'] == 1
    assert result['recovery_fast_reversals_before'] == 3
    assert result['recovery_fast_reversals_after'] == 2
    assert axis_run_events(notes)['fast_reversals'] == 2
    assert result['recovery_primary_signature_sha256'] == signature
    assert result['recovery_preserved_onsets_count_doubles']
    assert not any(post.clearance_violations(notes).values())


def test_recovery_keeps_double_and_rejects_guard_veto():
    from map_creation.v2_playability_recovery import repair_recovery
    notes = _endpoint_witness()
    notes.append(dict(id=10, time=1.1, hand=1, x=3, y=1, d=1, kind='retained'))
    before = deepcopy(notes)
    result = repair_recovery(notes, lambda _:100.)
    assert notes == before
    assert result['recovery_hand_swaps'] == 0
    notes = _endpoint_witness()
    before = deepcopy(notes)
    class Reject:
        def allows(self, changed_ids): return False
        def commit(self, changed_ids): raise AssertionError('Rejected recovery committed')
    result = repair_recovery(notes, lambda _:100., guard=Reject())
    assert notes == before
    assert result['recovery_rejected_guard'] > 0


def test_recovery_rebuilds_former_hand_outgoing_speed_budget():
    from map_creation.v2_playability_recovery import repair_recovery
    notes = _endpoint_witness()
    notes[3].update(x=0, y=0, d=0)
    notes[4].update(x=1, y=1, d=7)
    notes.append(dict(id=5, time=1.5, hand=0, x=2, y=0, d=2, kind='retained'))
    class Allow:
        def allows(self, changed_ids): return True
        def commit(self, changed_ids): pass
    feasible = deepcopy(notes)
    assert repair_recovery(feasible, lambda _:100., guard=Allow())['recovery_hand_swaps'] == 1
    before = deepcopy(notes)
    result = repair_recovery(notes, lambda n:9. if n['id'] == 5 else 100., guard=Allow())
    assert notes == before
    assert result['recovery_hand_swaps'] == 0


def test_recovery_cannot_reuse_source_overspeed_id_to_hide_new_hand_overspeed():
    from map_creation.v2_playability_recovery import repair_recovery
    notes = _endpoint_witness()
    before = deepcopy(notes)
    result = repair_recovery(notes, lambda n:0. if n['id'] == 4 else 100.)
    assert notes == before
    assert result['recovery_hand_swaps'] == 0


def test_recovery_without_supported_metadata_keeps_primary_signature_and_residual():
    from map_creation.v2_playability_recovery import repair_recovery, primary_signature
    notes = _endpoint_witness()
    notes[-1].pop('_recovery_alternatives')
    before = deepcopy(notes)
    result = repair_recovery(notes, lambda _:100.)
    assert notes == before
    assert result['recovery_primary_signature_sha256'] == primary_signature(notes)
    assert result['recovery_hand_swaps'] == 0
    assert result['recovery_fast_reversals_before'] == result['recovery_fast_reversals_after'] == 3
    assert result['recovery_preserved_onsets_count_doubles']


def _interior_pair_witness():
    rows = [(.5,1,1), (.8,0,0), (1.,0,1), (1.25,1,0), (1.45,1,1), (1.8,0,0)]
    notes = [dict(id=i, time=t, hand=h, x=3*h, y=1, d=d, kind='retained')
             for i,(t,h,d) in enumerate(rows)]
    notes[2]['_recovery_alternatives'] = [dict(hand=1,x=3,y=1,support=1.)]
    notes[3]['_recovery_alternatives'] = [dict(hand=0,x=0,y=1,support=1.)]
    return notes


def test_joint_supported_interior_swap_preserves_cut_order_and_reduces_two_fast_edges():
    from map_creation.v2_playability_recovery import repair_recovery
    from map_creation.v2_playability_policy import PatternGuard
    from map_creation import v2_postprocessing as post
    notes = _interior_pair_witness()
    before = deepcopy(notes)
    result = repair_recovery(notes, lambda _:100., guard=PatternGuard(notes))
    assert [(n['id'],n['time']) for n in notes] == [(n['id'],n['time']) for n in before]
    assert [n for n in notes if n['id'] not in (2,3)] == [n for n in before if n['id'] not in (2,3)]
    assert (notes[2]['hand'],notes[3]['hand']) == (1,0)
    for hand in (0,1):
        assert [n['d'] for n in notes if n['hand']==hand] == [n['d'] for n in before if n['hand']==hand]
    assert result['recovery_fast_reversals_before'] == 2
    assert result['recovery_fast_reversals_after'] == 0
    assert result['recovery_hand_swaps'] == 2
    assert result['recovery_joint_swaps'] == 1
    assert result['recovery_joint_changed_ids'] == [2,3]
    assert result['recovery_preserved_onsets_count_doubles']
    assert not any(post.clearance_violations(notes).values())


@pytest.mark.parametrize('case', ['missing', 'weak', 'double', 'radius', 'no_gain', 'balance', 'cap'])
def test_joint_recovery_keeps_unsupported_or_unsafe_pair_unchanged(case):
    from map_creation.v2_playability_recovery import repair_recovery
    notes = _interior_pair_witness()
    cap = lambda _:100.
    if case == 'missing':
        notes[3].pop('_recovery_alternatives')
    elif case == 'weak':
        notes[3]['_recovery_alternatives'][0]['support'] = .89
    elif case == 'double':
        notes.append(dict(id=9,time=1.,hand=1,x=3,y=1,d=1,kind='retained'))
    elif case == 'radius':
        notes[0]['y'] = 0
        notes[2]['_recovery_alternatives'][0].update(x=2,y=2)
    elif case == 'no_gain':
        for note, time in zip(notes, (.5,.8,1.2,1.7,2.1,2.5)):
            note['time'] = time
    elif case == 'balance':
        notes.insert(0,dict(id=9,time=.2,hand=1,x=3,y=1,d=0,kind='retained'))
    elif case == 'cap':
        cap = lambda n:0. if n['id']==2 else 100.
    before = deepcopy(notes)
    result = repair_recovery(notes, cap)
    assert notes == before
    assert result['recovery_joint_swaps'] == 0
    assert result['recovery_hand_swaps'] == 0


def test_joint_recovery_respects_shared_guard_veto():
    from map_creation.v2_playability_recovery import repair_recovery
    notes = _interior_pair_witness()
    before = deepcopy(notes)
    class Reject:
        def allows(self, changed_ids): return False
        def commit(self, changed_ids): raise AssertionError('Rejected joint repair committed')
    result = repair_recovery(notes, lambda _:100., guard=Reject())
    assert notes == before
    assert result['recovery_rejected_guard'] > 0
    assert result['recovery_joint_swaps'] == 0


def test_joint_pairs_share_total_eight_changed_note_budget():
    from map_creation.v2_playability_recovery import repair_recovery
    notes=[]
    for phrase in range(5):
        for note in _interior_pair_witness():
            note['id'] += phrase*10
            note['time'] += phrase*3.
            note['d'] ^= phrase%2
            notes.append(note)
    result = repair_recovery(notes, lambda _:100.)
    assert result['recovery_joint_swaps'] == 4
    assert result['recovery_hand_swaps'] == 8
    assert len(set(result['recovery_joint_changed_ids'])) == 8
    assert result['recovery_fast_reversals_before'] == 10
    assert result['recovery_fast_reversals_after'] == 2
