"""Human visibility calibration keeps eligible-map and zero-event support honest."""
import importlib
import importlib.util
import json
import subprocess
import sys

import pytest


def api():
    name = 'evaluation.playability_visibility'
    assert importlib.util.find_spec(name) is not None, 'visibility calibration is missing'
    return importlib.import_module(name)


def notes(*, upper=False, sparse=False, eligible=True):
    times = (0., 10., 10.1, 10.2, 10.3, 100.) if sparse else (0., .1, .2, .3, .4, .5)
    if not eligible:
        times = (0., 3.)
    return [dict(id=i, time=t, hand=0, x=1, y=2 if upper else 1, d=0) for i, t in enumerate(times)]


def package(root, name, *, upper=False):
    path = root/name
    path.mkdir()
    info = {'_version': '2.1.0', '_beatsPerMinute': 120, '_songTimeOffset': 0,
            '_shuffle': 0, '_shufflePeriod': 0,
            '_difficultyBeatmapSets': [{'_beatmapCharacteristicName': 'Standard',
                '_difficultyBeatmaps': [{'_difficulty': 'ExpertPlus', '_beatmapFilename': 'ExpertPlus.dat'}]}]}
    chart = {'version': '3.0.0', 'colorNotes': [dict(b=.4*i, c=0, x=1, y=2 if upper else 1, d=0) for i in range(8)]}
    (path/'Info.dat').write_text(json.dumps(info))
    (path/'ExpertPlus.dat').write_text(json.dumps(chart))
    return path


def test_calibration_keeps_eligible_zero_phrases_and_counts_only_eligible_maps():
    m = api()
    rows = [dict(name=str(i), notes=notes()) for i in range(5)]
    rows.append(dict(name='empty-support', notes=notes(eligible=False)))
    ref = m.fit_reference(rows)
    fitted = ref['global']
    assert fitted['maps'] == 5 and fitted['eligible'] == 15
    assert fitted['supported'] and fitted['cutoff'] == 0. and fitted['raw_positive'] == 0
    assert fitted['flagged'] == 0 and fitted['pooled_rate'] == 0. and fitted['equal_map_rate'] == 0.
    assert ref['budget'] == .03


def test_sparse_band_falls_back_to_supported_global_fit():
    rows = [dict(name=f'high{i}', notes=notes(upper=i == 0)) for i in range(5)]
    rows += [dict(name=f'low{i}', notes=notes(sparse=True)) for i in range(4)]
    ref = api().fit_reference(rows)
    profile = ref['bands']['<3']
    assert profile['fallback'] and profile['supported']
    assert profile['band_eligible_maps'] == 4 and profile['maps'] == 9
    assert profile['cutoff'] == ref['global']['cutoff']
    for profile in (ref['global'], *ref['bands'].values()):
        assert profile['pooled_rate'] <= .03 and profile['equal_map_rate'] <= .03


def test_sparse_global_has_no_cutoff_and_empty_band_has_no_support():
    ref = api().fit_reference([dict(name=str(i), notes=notes()) for i in range(4)])
    assert ref['global']['maps'] == 4 and not ref['global']['supported']
    assert ref['global']['cutoff'] is None and ref['global']['flagged'] is None
    assert ref['bands']['<3']['band_eligible_maps'] == 0 and ref['bands']['<3']['fallback']
    result = api().score_notes(notes(upper=True), ref)
    assert result['flagged_events'] is None and result['excess'] is None
    json.dumps(result, allow_nan=False)


def test_calibration_strict_ties_respect_pooled_and_equal_map_budgets():
    ref = api().fit_reference([dict(name=str(i), notes=notes(upper=True)) for i in range(5)])
    fitted = ref['global']
    assert fitted['cutoff'] == 1. and fitted['flagged'] == 0
    assert fitted['pooled_rate'] <= .03 and fitted['equal_map_rate'] <= .03
    assert api().score_notes(notes(upper=True), ref)['flagged_events'] == 0


def test_human_root_accepts_curated_parent_and_rejects_favorites(tmp_path, monkeypatch):
    m = api()
    root = tmp_path/'evaluate_beats'/'bs_map_input'
    root.mkdir(parents=True)
    favorites = root.parent/'favorites_bs_input'
    favorites.mkdir()
    monkeypatch.setattr(m, 'CURATED_ROOT', root)
    assert m._curated_root(root) == root.resolve()
    assert m._curated_root(root.parent) == root.resolve()
    with pytest.raises(ValueError, match='curated'):
        m._curated_root(favorites)
    with pytest.raises(ValueError, match='curated'):
        m._curated_root(tmp_path)


def test_calibrate_decodes_real_packages_and_records_each_skip(tmp_path, monkeypatch):
    m = api()
    root = tmp_path/'evaluate_beats'/'bs_map_input'
    root.mkdir(parents=True)
    for i in range(5):
        package(root, f'good{i}', upper=i == 0)
    (root/'missing-info').mkdir()
    bad = package(root, 'tempo-change')
    chart = json.loads((bad/'ExpertPlus.dat').read_text())
    chart['bpmEvents'] = [{'b': 0., 'm': 125}]
    (bad/'ExpertPlus.dat').write_text(json.dumps(chart))
    monkeypatch.setattr(m, 'CURATED_ROOT', root)
    output, reference = tmp_path/'reports'/'human.json', tmp_path/'reports'/'reference.json'
    m.calibrate(root, output, reference)
    report = json.loads(output.read_text())
    fitted = json.loads(reference.read_text())
    assert report['accepted'] == 5 and len(report['maps']) == 5
    assert {row['package'] for row in report['skipped']} == {'missing-info', 'tempo-change'}
    assert all(row['reason'] for row in report['skipped'])
    assert len(fitted['population_sha256']) == 64 and fitted['human_root'] == str(root.resolve())
    assert fitted['event_version'] == m.VERSION and fitted['global']['eligible'] == 25
    assert all(row['visibility']['supported'] for row in report['maps'])


def test_report_cli_writes_json_for_selected_difficulty(tmp_path):
    m = api()
    root = tmp_path/'maps'
    root.mkdir()
    chart = package(root, 'generated', upper=True)
    ref = m.fit_reference([dict(name=str(i), notes=notes()) for i in range(5)])
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps(ref))
    output = tmp_path/'report.json'
    run = subprocess.run([sys.executable, '-s', '-m', 'evaluation.playability_visibility', 'report',
        '--package', str(chart), '--difficulty', 'ExpertPlus', '--reference', str(reference), '--json', str(output)],
        text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(output.read_text())
    assert result['eligible_events'] == 5 and result['flagged_events'] == 5
    assert result['max_severity'] == 1. and result['cutoff'] == 0.
