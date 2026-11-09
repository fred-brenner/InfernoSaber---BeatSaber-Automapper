"""Human event calibration uses strict tails and equal-map support."""
import importlib
import importlib.util
import json


def api():
    name = 'evaluation.playability_priority'
    assert importlib.util.find_spec(name) is not None, 'priority calibration is missing'
    return importlib.import_module(name)


def test_equal_map_budget_prevents_dense_maps_hiding_sparse_outliers():
    result = api().fit_cutoff([[0.] * 1000, [10.]], .03)
    assert result['cutoff'] == 10.
    assert result['pooled_rate'] <= .03 and result['equal_map_rate'] <= .03


def test_tied_events_are_not_flagged_at_the_cutoff():
    result = api().fit_cutoff([[1.] * 100, [1.] * 3], .03)
    assert result['cutoff'] == 1.
    assert result['flagged'] == 0 and result['eligible'] == 103


def test_empty_populations_have_no_fabricated_calibration_support():
    result = api().fit_cutoff([[], []], .03)
    assert not result['supported'] and result['eligible'] == 0
    assert result['pooled_rate'] is None and result['equal_map_rate'] is None


def test_reference_and_report_preserve_component_eligibility_and_monitoring():
    notes = [dict(id=i, time=float(i), hand=0, x=0, y=1, d=8) for i in range(8)]
    reference = api().fit_reference([dict(name='dots', notes=notes)])
    result = api().score_notes(notes, reference)
    assert result['primary']['preparation']['preparation_eligible'] == 0
    assert result['primary']['preparation']['placement_eligible'] == 8
    assert result['flags']['preparation']['eligible'] == 0
    assert 'placement' in result and 'monitoring' in result
    assert all(key in result['monitoring'] for key in ('m1', 'm2', 'm3', 'm4', 'm5'))
    json.dumps(result, allow_nan=False)


def test_current_m2_detects_a_flat_upper_center_stream_legacy_m2_misses():
    upper = [dict(id=i, time=i*.2, hand=0, x=1, y=2, d=i % 2) for i in range(8)]
    clear = [dict(n, y=1) for n in upper]
    bad, good = api().score_notes(upper), api().score_notes(clear)
    assert 'visibility' in bad, 'Current M2 must include sustained upper-center exposure'
    assert bad['visibility']['flagged_events'] > 0
    assert good['visibility']['flagged_events'] == 0
    assert bad['monitoring']['legacy_m2']['value'] == 0.
    assert bad['monitoring']['m2']['value'] > good['monitoring']['m2']['value']
    assert bad['monitoring']['m2']['version'] == bad['visibility']['version']
    assert bad['monitoring']['m2']['pairwise_value'] == 0.
    assert bad['monitoring']['m2']['definition']
    json.dumps(bad, allow_nan=False)


def test_current_m2_reports_a_local_peak_even_with_long_clear_context():
    notes = [dict(id=i, time=i*.2, hand=0, x=1, y=1, d=i % 2) for i in range(80)]
    notes[30:38] = [dict(n, y=2) for n in notes[30:38]]
    result = api().score_notes(notes)
    assert 'visibility' in result, 'A song mean must not hide local M2 peaks'
    assert result['visibility']['max_severity'] == 1.
    assert result['visibility']['flagged_events'] > 0
    assert result['monitoring']['m2']['max_severity'] == 1.


def _note(i, time=0., hand=0, x=1, y=1, d=0):
    return dict(id=i, time=time, hand=hand, x=x, y=y, d=d)


def test_band_fallback_uses_eligible_maps_per_metric(monkeypatch):
    m=api()
    monkeypatch.setattr(m, '_nps', lambda notes: 1. if notes[0]['x']==0 else 8.)
    rows=[dict(name=f'low{i}', notes=[_note(0,x=0)] + ([_note(1,.1,1,x=1)] if i==0 else [])) for i in range(5)]
    rows += [dict(name=f'high{i}',notes=[_note(0,x=1),_note(1,.1,1,x=2)]) for i in range(5)]
    profile=m.fit_reference(rows)['bands']['<3']['metrics']['near_miss']
    assert profile['fallback'] is True and profile['supported'] is True
    assert profile['band_eligible_maps']==1 and profile['maps']==6


def test_unsupported_cutoff_and_flags_are_unavailable():
    m=api(); reference=m.fit_reference([])
    result=m._flags(m.priority_metrics([_note(0),_note(1,1,d=2)]),reference,1.,[_note(0),_note(1,1,d=2)])
    assert reference['global_metrics']['axis']['cutoff'] is None
    assert result['axis']['eligible']==1
    assert result['axis']['cutoff'] is None and result['axis']['flagged'] is None and result['axis']['rate'] is None


def test_specific_preparation_and_crossing_populations_include_eligible_zeros():
    m=api()
    notes=[_note(0,0,x=0,d=8), _note(1,1,x=1,d=3), _note(2,2,x=1,y=2,d=1),
           _note(3,3,x=2,d=0), _note(4,4,hand=1,x=3,d=8)]
    obs=m._observations(m.priority_metrics(notes),notes)
    assert len(obs['outer_placement'])==5 and obs['outer_placement'].count(0)==3
    assert len(obs['inner_inward'])==3 and obs['inner_inward'].count(0)==2
    assert len(obs['vertical_preparation'])==3 and obs['vertical_preparation'].count(0)==2
    assert len(obs['preparation_excursion'])==3
    events={'preparation':{'events':[]},'axis':{'events':[]},'crossing':{'events':[
        {'dt':.5,'eligible':True,'detected':False,'total':.4}, {'dt':.5,'eligible':True,'detected':True,'total':.8}]}}
    obs=m._observations(events,[])
    assert obs['slow_crossing']==[0.,.8] and obs['near_miss']==[.4,0.]


def test_human_root_rejects_favorites_and_unrelated(tmp_path, monkeypatch):
    import pytest
    m=api(); curated=tmp_path/'evaluate_beats'/'bs_map_input'; curated.mkdir(parents=True)
    favorites=tmp_path/'evaluate_beats'/'favorites_bs_input'; favorites.mkdir()
    monkeypatch.setattr(m,'CURATED_ROOT',curated)
    assert m._curated_root(curated.parent)==curated.resolve()
    assert m._curated_root(curated)==curated.resolve()
    with pytest.raises(ValueError): m._curated_root(favorites)
    with pytest.raises(ValueError): m._curated_root(tmp_path)


def test_empty_band_falls_back_and_sparse_global_remains_unavailable(monkeypatch):
    m=api()
    monkeypatch.setattr(m,'_nps',lambda notes: 1. if notes[0]['x']==0 else 8.)
    low=[dict(name=f'low{i}',notes=[_note(0,x=0)]) for i in range(5)]
    high=[dict(name=f'high{i}',notes=[_note(0,x=1),_note(1,.5,hand=1,x=2)]) for i in range(5)]
    profile=m.fit_reference(low+high)['bands']['<3']['metrics']['near_miss']
    assert profile['fallback'] and profile['band_eligible_maps']==0 and profile['maps']==5 and profile['supported']
    profile=m.fit_reference(low+high[:4])['bands']['<3']['metrics']['near_miss']
    assert profile['fallback'] and not profile['supported'] and profile['maps']==4 and profile['cutoff'] is None


def test_each_specific_measure_reports_both_map_and_pooled_rates():
    m=api()
    rows=[]
    for i in range(5):
        notes=[_note(0,0,x=0,d=8),_note(1,1,x=1,d=3),_note(2,2,x=1,y=2,d=1),_note(3,3,x=2,d=0)]
        rows.append({'name':str(i),'notes':notes})
    ref=m.fit_reference(rows)
    for key in ('outer_placement','inner_inward','vertical_preparation','preparation_excursion'):
        metric=ref['global_metrics'][key]
        assert metric['supported'] and metric['maps']==5
        assert metric['pooled_rate'] <= .03 and metric['equal_map_rate'] <= .03
        assert metric['eligible']==(20 if key=='outer_placement' else 15)


def test_slow_crossing_support_includes_all_non_crossing_pairs():
    m=api()
    rows=[{'name':str(i),'notes':[_note(0,0,hand=0,x=0,d=0),_note(1,.5,hand=1,x=3,d=0)]} for i in range(5)]
    result=m.fit_reference(rows)['global_metrics']['slow_crossing']
    assert result['supported'] and result['maps']==5 and result['eligible']==5 and result['cutoff']==0
    mixed=m.fit_cutoff([[0.]*99+[.4] for _ in range(5)],.03)
    assert mixed['pooled_rate'] <= .03 and mixed['equal_map_rate'] <= .03


def test_weighted_specific_severity_flags_repeated_fast_awkward_preparation():
    m=api()
    rows=[{'name':str(j),'notes':[_note(i,10.*i,x=0,d=0) for i in range(10)]} for j in range(5)]
    ref=m.fit_reference(rows)
    profile=ref['global_metrics']['outer_placement']
    assert profile['supported'] and profile['pooled_rate']<=.03 and profile['equal_map_rate']<=.03
    assert profile['raw_positive']==50
    repeated=[_note(i,.1*i,x=0,d=0) for i in range(10)]
    result=m._flags(m.priority_metrics(repeated),ref,1.,repeated)['outer_placement']
    assert result['eligible']==10 and result['flagged'] > 0
