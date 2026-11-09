"""Offline scorecards preserve scorer semantics and pair maps at song level."""
from pathlib import Path
import json
import hashlib
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation import playability_benchmark as benchmark
from evaluation.playability_metrics import score_map
from evaluation.playability_report import verdict


def _notes(directions, *, step=0.2, hand=0):
    return [(index * step, hand, index % 4, index % 3, direction)
            for index, direction in enumerate(directions)]


def _reference_rows():
    return [{"package": f"human-{index}", "difficulty": "Expert",
             "notes": _notes([2, 3, 2, 4, 7, 0, 1, 3], step=0.25 + index * 0.01),
             "scores": score_map(_notes([2, 3, 2, 4, 7, 0, 1, 3], step=0.25 + index * 0.01))}
            for index in range(6)]


def test_scorecard_matches_official_verdict_and_freezes_control_density_band():
    rows = _reference_rows()
    reference = benchmark.prepare_human_reference(rows)
    notes = _notes([2, 3, 2, 3, 2, 3], step=0.12)
    card = benchmark.compact_scorecard(notes, reference, control_nps=4.0)
    official = verdict(score_map(notes), rows)
    assert card["band"] == "7+"
    assert card["control_band"] == "3-5"
    assert card["band_migration"] is True
    for metric in ("m1", "m2", "m3", "m4", "m5"):
        assert card["metrics"][metric]["official"] == official[metric]
    assert card["distances"]["m1"] >= 0
    assert card["metrics"]["m1"]["threshold"] == reference["official"]["m1"]["incident_threshold"]
    assert card["metrics"]["m1"]["positive_events"] == len(score_map(notes)["m1"]["units"])
    json.dumps(card, allow_nan=False)
    json.dumps(reference, allow_nan=False)


def test_axis_dwell_splits_ending_angles_hands_and_excludes_terminal_runs():
    # Left axis: H,H,H,D,D,V,V, H (runs 3/2/2; endings 45/45/90).
    # Right has a dot interruption; its first H run is unobserved.
    left = _notes([2, 3, 2, 4, 7, 0, 1, 3])
    right = _notes([2, 8, 3, 2, 0], step=0.3, hand=1)
    card = benchmark.compact_scorecard(left + right)
    l45 = card["axis_dwell"]["0"]["45"]
    l90 = card["axis_dwell"]["0"]["90"]
    assert l45["notes"]["count"] == 2
    assert l45["notes"]["median"] == 2.5
    assert l45["milliseconds"]["median"] == pytest.approx(500)
    assert l90["notes"]["median"] == 2
    assert card["axis_dwell"]["1"]["45"]["notes"]["count"] == 0
    assert card["axis_dwell"]["1"]["90"]["notes"]["median"] == 2
    for hand in ("0", "1"):
        pooled = card["axis_dwell"][hand]["pooled"]
        scorer = score_map(left + right)["m5"]["diagnostics"]["per_hand"][hand]["completed_cut_axis_runs"]
        assert pooled["notes"]["count"] == scorer["notes"]["count"]
        assert pooled["notes"]["median"] == scorer["notes"]["median"]
        assert pooled["milliseconds"]["median"] == pytest.approx(scorer["seconds"]["median"] * 1000)


def test_empty_eligibility_is_missing_instead_of_zero_distance_or_nan():
    reference = benchmark.prepare_human_reference(_reference_rows())
    card = benchmark.compact_scorecard([(0, 0, 0, 0, 8)], reference)
    assert card["coverage"]["m3_eligible_segments"] == 0
    assert card["distances"]["m3"] is None
    assert card["distances"]["m5_non_reversal"] is None
    assert card["metrics"]["m5"]["non_reversal_share"] is None
    assert card["axis_dwell"]["0"]["45"]["notes"]["median"] is None
    assert card["longest_fast_horizontal_run"] == 0
    json.dumps(card, allow_nan=False)


def test_horizontal_run_observations_break_at_dots_and_count_arrivals():
    notes = [(0, 0, 0, 0, 2), (0.25, 0, 1, 0, 3),
             (0.5, 0, 0, 0, 2), (0.7, 0, 1, 0, 3),
             (0.9, 0, 1, 0, 8), (1.0, 0, 0, 0, 2), (1.2, 0, 1, 0, 3)]
    card = benchmark.compact_scorecard(notes)
    assert card["longest_fast_horizontal_run"] == 3
    assert card["horizontal_runs"]["0"]["arrivals"] == [3, 1]
    # The >250 ms edge ends a run, even if both arrows remain horizontal.
    notes[2] = (0.501, 0, 0, 0, 2)
    assert benchmark.compact_scorecard(notes)["longest_fast_horizontal_run"] == 1


def test_reference_pools_fast_events_by_minutes_and_discloses_sparse_dwell_band():
    rows = _reference_rows()
    reference = benchmark.prepare_human_reference(rows)
    expected = sum(row["scores"]["m5"]["fast_reversals"] for row in rows) / (sum(row["scores"]["span_seconds"] for row in rows) / 60)
    assert reference["all"]["m5_fast_reversal"]["median"] == pytest.approx(expected)
    # All six maps are in 3-5; <3 dwell is pooled with the nearest band.
    dwell = reference["bands"]["<3"]["axis_dwell"]["0"]["45"]
    assert dwell["reference_maps"] == 6
    assert dwell["source_bands"] == ["<3", "3-5"]
    assert dwell["fallback"] is True


def _card(distance, *, nps=5.0, longest=0, coverage=4, dot_share=0.0):
    return {"nps": nps, "band": "5-7", "control_band": "5-7",
            "distances": {key: distance for key in (*benchmark.DISTANCE_METRICS, *benchmark.DWELL_DISTANCE_METRICS)},
            "longest_fast_horizontal_run": longest,
            "note_summary": {"dot_share": dot_share, "vertical_share": 0.4,
                             "left_share": 0.5, "exported_notes": 100},
            "coverage": {"m3_eligible_segments": coverage}}


def _checks():
    return {"timestamps_hands_count_preserved": True,
            "new_hard_clearance_violations": 0, "chains_intact": True,
            "arc_endpoints_intact": True, "roundtrip_v2_valid": True,
            "roundtrip_v3_valid": True, "frozen_speed_violations_increase": 0,
            "frozen_flow_violations_increase": 0}


def _pair(song, control=10, candidate=7, **kwargs):
    return {"song": song, "difficulty": "Expert", "seed": 3,
            "control": _card(control, longest=10),
            "candidate": _card(candidate, longest=5),
            "control_runtime_seconds": 10, "candidate_runtime_seconds": 10,
            "checks": _checks(), **kwargs}


def test_paired_comparison_weights_songs_instead_of_maps_and_reports_missing_pairs():
    rows = [_pair("short", candidate=0), *[_pair("long", candidate=10, seed=seed) for seed in range(3)],
            {"song": "missing", "difficulty": "Easy", "seed": 3, "control": _card(10)}]
    margins = {"5-7": {key: 0.5 for key in benchmark.DISTANCE_METRICS}}
    report = benchmark.paired_decision_report(rows, frozen_margins=margins, bootstrap_samples=200)
    m1 = report["metrics"]["m1"]
    assert m1["songs"] == 2
    assert m1["pairs"] == 4
    assert m1["control_mean_distance"] == 10
    assert m1["candidate_mean_distance"] == 5
    assert m1["closure"] == 0.5
    assert m1["delta_ci95"][0] == -10
    assert m1["delta_ci95"][1] == 0
    assert report["incomplete_pairs"][0]["song"] == "missing"
    assert report["decision"] == "experimental"


def test_guardrails_reject_actual_density_dot_coverage_runtime_and_geometry_harm():
    pair = _pair("bad")
    pair["candidate"]["nps"] = 4.5
    pair["candidate"]["note_summary"]["dot_share"] = 0.02
    pair["candidate"]["coverage"]["m3_eligible_segments"] = 3
    pair["candidate_runtime_seconds"] = 12
    pair["checks"]["new_hard_clearance_violations"] = 1
    margins = {"5-7": {key: 100 for key in benchmark.DISTANCE_METRICS}}
    result = benchmark.paired_decision_report([pair], frozen_margins=margins, bootstrap_samples=30)
    assert result["decision"] == "experimental"
    for key in ("density", "dots", "m3_coverage", "runtime", "hard_checks"):
        assert result["guardrails"][key]["passed"] is False
    assert result["guardrails"]["density"]["by_difficulty"]["Expert"]["median_relative_change"] == pytest.approx(-0.1)


def test_margin_freeze_uses_control_band_iqr_and_maximum_aa_difference():
    reference = benchmark.prepare_human_reference(_reference_rows())
    pair = _pair("song")
    aa = _pair("song", control=10, candidate=13)
    margins = benchmark.freeze_material_harm_margins([pair], reference, aa_rows=[aa])
    assert margins["5-7"]["m1"] == 3
    assert margins["5-7"]["m2"] == 3
    result = benchmark.paired_decision_report([pair], frozen_margins=margins, bootstrap_samples=30)
    assert result["metrics"]["m1"]["harm_margin"] == 3
    assert result["metrics"]["m1"]["closure"] == pytest.approx(0.3)
    assert result["decision"] == "advance_for_playtest"


def test_missing_frozen_margins_or_runtime_never_grants_automatic_acceptance():
    pair = _pair("song")
    pair.pop("candidate_runtime_seconds")
    result = benchmark.paired_decision_report([pair], bootstrap_samples=30)
    assert result["decision"] == "experimental"
    assert result["guardrails"]["runtime"]["passed"] is None
    assert result["margins_frozen"] is False


def test_reference_survives_json_roundtrip_and_does_not_invent_old_m5_support():
    rows = _reference_rows()
    rows[0]["scores"].pop("m5")
    reference = json.loads(json.dumps(benchmark.prepare_human_reference(rows)))
    card = benchmark.compact_scorecard(_notes([2, 3, 0, 1]), reference)
    assert card["metrics"]["m5"]["official"] == verdict(score_map(_notes([2, 3, 0, 1])), rows)["m5"]


def test_m1_band_counts_use_uncapped_observations_and_include_vertical_denominator():
    notes = _notes([2, 3, 2, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3])
    reference = benchmark.prepare_human_reference(_reference_rows())
    card = benchmark.compact_scorecard(notes, reference)
    bands = card["metrics"]["m1"]["gap_bands"]
    assert sum(band["transitions"] for band in bands.values()) == 14
    assert sum(band["positive_events"] for band in bands.values()) == 12
    assert len(card["metrics"]["m1"]["top_incidents"]) <= 10
    assert sum(band["incident_count"] for band in bands.values()) == card["metrics"]["m1"]["incident_count"]


def test_pooled_report_sums_counts_and_minutes_instead_of_averaging_map_rates():
    reference = benchmark.prepare_human_reference(_reference_rows())
    pairs = []
    for song, step in (("short", 0.1), ("long", 0.4)):
        card = benchmark.compact_scorecard(_notes([2, 3, 2, 3], step=step), reference)
        pairs.append({**_pair(song), "control": card, "candidate": card})
    report = benchmark.paired_decision_report(pairs, bootstrap_samples=30)
    pooled = report["pooled_events"]["control"]["m5"]
    assert pooled["count"] == 3
    assert pooled["minutes"] == pytest.approx(0.025)
    assert pooled["rate"] == pytest.approx(120)
    assert report["axis_dwell"]["5-7"]["0"]["45"]["control"]["notes"]["count"] == 0


def _write_package(path, notes):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'Info.dat').write_text(json.dumps({
        '_version': '2.1.0', '_beatsPerMinute': 60, '_songTimeOffset': 0,
        '_shuffle': 0, '_shufflePeriod': 0,
        '_difficultyBeatmapSets': [{'_beatmapCharacteristicName': 'Standard',
            '_difficultyBeatmaps': [{'_difficulty': 'Expert', '_beatmapFilename': 'Expert.dat'}]}]}))
    (path / 'Expert.dat').write_text(json.dumps({'version': '3.0.0',
        'colorNotes': [{'b': t, 'c': hand, 'x': x, 'y': y, 'd': d}
                       for t, hand, x, y, d in notes]}))


def _receipt_run(tmp_path):
    from evaluation.playability_experiments import case_seed, file_hash
    audio = tmp_path / 'song.egg'
    audio.write_bytes(b'example source audio; no decoder invoked')
    audio_hash = file_hash(audio)
    variant = {'nps': 5, 'intensity': 90}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'development_replicates': [11],
        'songs': [{'package': 'song', 'split': 'development', 'audio': str(audio),
                   'sha256': audio_hash, 'variants': {'Expert': variant}}]}))
    runs = tmp_path / 'runs'
    worker = runs / 'workers' / '123.json'
    worker.parent.mkdir(parents=True)
    worker.write_text(json.dumps({'git': 'abcdef', 'python': '3.10', 'tensorflow': '2.15',
        'devices': ['CPU:0'], 'sources': {'map_creation/v2_pipeline.py': 'same-source'},
        'models': {'notes/model.keras': 'same-model'}, 'effective_settings': {}}))
    for arm in ('app1_control', 'm1'):
        folder = runs / 'development' / arm / '00' / 'Expert' / '11'
        notes = _notes([2, 3, 2, 3])
        _write_package(folder / 'package', notes)
        data = {'song': 'song', 'audio_hash': audio_hash, 'difficulty': 'Expert',
            'seed': case_seed(audio_hash, 5, 11), 'replicate': 11, 'arm': arm,
            'worker': 'workers/123.json', 'runtime_seconds': 10, 'package': 'package',
            'actual_settings': variant, 'residual': {'same_cell': 0, 'swing_into': 0, 'same_dir': 0},
            'frozen_speed': 0, 'frozen_flow': 0, 'checks': _checks(), 'counts': {'notes': 4},
            'speed': {}, 'onset': {}, 'intro': {}}
        (folder / 'result.json').write_text(json.dumps(data))
    human = tmp_path / 'human' / 'bs_map_input'
    _write_package(human / 'valid', _notes([0, 1, 0, 1]))
    (human / 'invalid').mkdir()
    return manifest, runs, human.parent


def test_receipt_cli_freezes_controls_and_scores_actual_exported_packages(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    frozen_path, report_path = tmp_path / 'frozen.json', tmp_path / 'report.json'
    benchmark.main(['freeze', '--manifest', str(manifest), '--runs', str(runs),
        '--human-root', str(human), '--split', 'development', '--indices', '0',
        '--difficulties', 'Expert', '--output', str(frozen_path)])
    benchmark.main(['report', '--manifest', str(manifest), '--runs', str(runs),
        '--frozen', str(frozen_path), '--split', 'development', '--candidate', 'm1',
        '--target', 'm1', '--indices', '0', '--difficulties', 'Expert',
        '--bootstrap-samples', '30', '--output', str(report_path)])
    result = json.loads(report_path.read_text())
    assert result['pairs'] == 1
    assert result['audit']['passed'] is True
    assert result['scorecards'][0]['control']['notes'] == 4
    assert result['human_skipped'][0]['package'] == 'invalid'
    assert result['guardrails']['hard_checks']['passed'] is True


def test_report_keeps_manifest_expected_pairs_visible_when_receipts_are_missing(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    frozen = benchmark.freeze_run_reference(manifest, runs, human, split='development')
    candidate = runs / 'development' / 'm1' / '00' / 'Expert' / '11' / 'result.json'
    candidate.unlink()
    result = benchmark.report_run(manifest, runs, frozen, split='development', candidate='m1', bootstrap_samples=30)
    assert result['pairs'] == 0
    assert len(result['incomplete_pairs']) == 1
    assert result['incomplete_pairs'][0]['song'] == 'song'
    assert result['failures'][0]['arm'] == 'm1'
    assert result['decision'] == 'experimental'


def test_report_detects_control_package_changes_after_margin_freeze(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    frozen = benchmark.freeze_run_reference(manifest, runs, human, split='development')
    package = runs / 'development' / 'app1_control' / '00' / 'Expert' / '11' / 'package'
    _write_package(package, _notes([0, 1, 0, 1]))
    result = benchmark.report_run(manifest, runs, frozen, split='development', candidate='m1', bootstrap_samples=30)
    assert result['audit']['passed'] is False
    assert result['audit']['failures'][0]['kind'] == 'frozen_control_changed'
    assert result['decision'] == 'experimental'


def test_generation_only_receipts_do_not_claim_end_to_end_cpu_guardrail(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    frozen = benchmark.freeze_run_reference(manifest, runs, human, split='development')
    result = benchmark.report_run(manifest, runs, frozen, split='development', candidate='m1', bootstrap_samples=30)
    assert result['guardrails']['runtime']['passed'] is None
    assert result['guardrails']['runtime']['scope_counts'] == {'generation_only': 1}


def test_end_to_end_receipts_use_actual_duration_instead_of_generation_duration(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    for arm, duration in (('app1_control', 20), ('m1', 23)):
        receipt = runs / 'development' / arm / '00' / 'Expert' / '11' / 'result.json'
        data = json.loads(receipt.read_text())
        data['end_to_end_seconds'] = duration
        receipt.write_text(json.dumps(data))
    frozen = benchmark.freeze_run_reference(manifest, runs, human, split='development')
    result = benchmark.report_run(manifest, runs, frozen, split='development', candidate='m1', bootstrap_samples=30)
    assert result['guardrails']['runtime']['passed'] is False
    assert result['guardrails']['runtime']['median_relative_change'] == pytest.approx(.15)


def test_evaluation_adapter_hash_drift_is_disclosed_without_invalidating_generation(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    frozen = benchmark.freeze_run_reference(manifest, runs, human, split='development')
    worker = json.loads((runs / 'workers' / '123.json').read_text())
    worker['sources']['evaluation/playability_benchmark.py'] = 'built-after-control'
    (runs / 'workers' / 'candidate.json').write_text(json.dumps(worker))
    path = runs / 'development' / 'm1' / '00' / 'Expert' / '11' / 'result.json'
    data = json.loads(path.read_text())
    data['worker'] = 'workers/candidate.json'
    path.write_text(json.dumps(data))
    result = benchmark.report_run(manifest, runs, frozen, split='development', candidate='m1', bootstrap_samples=30)
    assert result['audit']['passed'] is True
    assert any(change.get('file') == 'evaluation/playability_benchmark.py'
               for change in result['audit']['nonblocking_changes'])


def test_named_manifest_literal_seed_is_respected(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    data = json.loads(manifest.read_text())
    data['songs'][0]['literal_seed'] = 3
    manifest.write_text(json.dumps(data))
    for arm in ('app1_control', 'm1'):
        path = runs / 'development' / arm / '00' / 'Expert' / '11' / 'result.json'
        record = json.loads(path.read_text())
        record['seed'] = 3
        path.write_text(json.dumps(record))
    frozen = benchmark.freeze_run_reference(manifest, runs, human, split='development')
    result = benchmark.report_run(manifest, runs, frozen, split='development', candidate='m1', bootstrap_samples=30)
    assert result['pairs'] == 1
    assert result['scorecards'][0]['seed'] == 3


def test_paired_report_bootstraps_hand_and_ending_angle_dwell_distances_separately():
    reference = benchmark.prepare_human_reference(_reference_rows())
    control = benchmark.compact_scorecard(_notes([2, 3, 2, 3, 2, 4, 7, 0]), reference)
    candidate = benchmark.compact_scorecard(_notes([2, 3, 2, 4, 7, 0, 1, 3]), reference,
                                           control_nps=control['nps'])
    pair = {**_pair('song'), 'control': control, 'candidate': candidate}
    report = benchmark.paired_decision_report([pair], bootstrap_samples=30)
    metric = report['dwell_metrics']['dwell_h0_a45_notes_median']
    assert metric['pairs'] == 1
    assert metric['candidate_mean_distance'] < metric['control_mean_distance']
    assert report['dwell_metrics']['dwell_h1_a90_milliseconds_median']['pairs'] == 0
    assert report['axis_dwell']['5-7']['0']['45']['reference']['reference_maps'] == 6


def test_exported_note_density_guard_does_not_hide_loss_inside_collapsed_stacks():
    pair = _pair('song')
    pair['control']['note_summary']['exported_nps'] = 5
    pair['candidate']['note_summary']['exported_nps'] = 4.5
    report = benchmark.paired_decision_report([pair], bootstrap_samples=30)
    assert report['guardrails']['density']['passed'] is False


def test_report_records_that_pilot_candidates_preceded_controls_only_margin_freeze(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    frozen = benchmark.freeze_run_reference(manifest, runs, human, split='development')
    result = benchmark.report_run(manifest, runs, frozen, split='development', candidate='m1', bootstrap_samples=30)
    assert len(result['audit']['candidate_receipts_predating_freeze']) == 1
    assert result['evidence'][0]['observations']['timing']['exported_timestamps_preserved'] is True
    assert result['evidence'][0]['observations']['control']['intro_notes'] == 0


def test_cohort_schedule_exhausts_development_and_holdout_without_candidate_early_start(tmp_path):
    from evaluation.playability_cohort import build_schedule
    manifest = tmp_path / 'manifest.json'
    songs = [{'package': f'{split}-{index}', 'split': split,
              'variants': {difficulty: {'nps': 5} for difficulty in ('Easy','Normal','Hard','Expert')}}
             for split in ('development','holdout') for index in range(30)]
    manifest.write_text(json.dumps({'development_replicates':[11], 'holdout_replicates':[19,43], 'songs':songs}))
    plan = build_schedule(manifest, tmp_path / 'locked', components=['m1','m2','m4'], workers=4)
    assert plan['stages']['development_controls']['maps'] == 120
    assert plan['stages']['development_candidates']['maps'] == 480
    assert plan['stages']['holdout_controls']['maps'] == 240
    assert plan['stages']['holdout_candidate']['maps'] == 240
    assert plan['stages']['development_candidates']['requires'] == ['frozen-development.json']
    assert plan['stages']['holdout_candidate']['requires'] == ['locked-selection.json']
    assert plan['stages']['development_candidates']['arms'] == ['m1,m2,m4','m2,m4','m1,m4','m1,m2']
    shards = plan['stages']['development_controls']['shards']
    assert sorted(index for shard in shards for index in shard) == list(range(30))


def test_cohort_refuses_source_drift_and_unfrozen_development_candidate(tmp_path, monkeypatch):
    from evaluation.playability_cohort import prepare_cohort, validate_stage
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'main.py').write_text('print("fixture")\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'development_replicates': [11], 'holdout_replicates': [19,43],
        'songs': [{'package':'song','split':'development','variants':{'Expert':{'nps':5}}}]}))
    root = tmp_path / 'locked'
    prepare_cohort(manifest, root, components=['m1','m4'])
    with pytest.raises(ValueError, match='requires frozen-development'):
        validate_stage(root, 'development_candidates')
    (tmp_path / 'main.py').write_text('print("changed")\n')
    with pytest.raises(ValueError, match='Source files changed'):
        validate_stage(root, 'development_controls')


def test_frozen_calculation_hash_mismatch_prevents_acceptance(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    frozen = benchmark.freeze_run_reference(manifest, runs, human, split='development')
    frozen['scorer_sources']['evaluation/playability_benchmark.py'] = 'prior-calculation-revision'
    result = benchmark.report_run(manifest, runs, frozen, split='development', candidate='m1', bootstrap_samples=30)
    assert result['audit']['passed'] is False
    assert result['audit']['failures'][0]['kind'] == 'frozen_scorer_changed'


def test_m5_advance_is_blocked_by_worsened_dwell_distributions():
    pair = _pair('song')
    for metric in benchmark.DWELL_DISTANCE_METRICS:
        pair['control']['distances'][metric] = 1
        pair['candidate']['distances'][metric] = 1000
    margins = {'5-7': {metric: 1 for metric in (*benchmark.DISTANCE_METRICS, *benchmark.DWELL_DISTANCE_METRICS)}}
    report = benchmark.paired_decision_report([pair], target='m5_non_reversal', frozen_margins=margins, bootstrap_samples=30)
    assert report['target_passed'] is True
    assert report['decision'] == 'experimental'
    assert report['guardrails']['axis_dwell']['passed'] is False


def test_late_repair_without_audit_does_not_assume_it_preserved_rhythm(tmp_path):
    from evaluation.playability_benchmark_run import _pair_checks
    receipt = {'data': {'speed':{},'checks':{}}, 'notes':[(0,0,0,0,0)]}
    changed = {'data': {'speed':{},'checks':{}}, 'notes':[(0.1,1,0,0,0)]}
    checks, evidence = _pair_checks(receipt,changed,'late,slow90_500,slow45_650')
    assert checks['timestamps_hands_count_preserved'] is None
    assert evidence == 'missing_mutation_stage_audit'


def test_nonzero_hard_residual_totals_cannot_prove_no_new_incident_identity():
    from evaluation.playability_benchmark_run import _pair_checks
    before = {'data':{'residual':{'same_cell':1}}, 'notes':[(0,0,0,0,0)]}
    after = {'data':{'residual':{'same_cell':1}}, 'notes':[(0,0,1,0,0)]}
    checks,_ = _pair_checks(before,after,'m2')
    assert checks['new_hard_clearance_violations'] is None


def test_vertical_chart_with_dots_still_counts_as_all_vertical():
    pair = _pair('song')
    pair['candidate']['note_summary'].update(vertical_share=.9,dot_share=.1,directional_vertical_share=1)
    pair['control']['note_summary']['directional_vertical_share'] = .4
    assert benchmark.package_guardrails([pair])['axis_use']['passed'] is False


def test_receipt_policy_flags_must_match_the_declared_candidate(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    frozen = benchmark.freeze_run_reference(manifest, runs, human, split='development')
    path = runs / 'development' / 'm1' / '00' / 'Expert' / '11' / 'result.json'
    data = json.loads(path.read_text())
    data['policy_settings'] = {'enabled':False,'m1':False}
    path.write_text(json.dumps(data))
    report = benchmark.report_run(manifest,runs,frozen,split='development',candidate='m1',bootstrap_samples=30)
    assert report['pairs'] == 0
    assert 'policy' in report['failures'][0]['error'].lower()


def test_mismatched_confirmation_scope_is_rejected_before_reserved_scoring(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    frozen = benchmark.freeze_run_reference(manifest,runs,human,split='development')
    result = benchmark.report_run(manifest,runs,frozen,split='development',candidate='m1',
                                  confirmation=True,bootstrap_samples=30)
    assert result['decision'] == 'experimental'
    assert result['audit']['confirmation']['passed'] is False


def test_case_temperature_mismatch_is_rejected(tmp_path):
    from evaluation.playability_experiments import SWITCHES
    manifest, runs, human = _receipt_run(tmp_path)
    frozen = benchmark.freeze_run_reference(manifest,runs,human,split='development')
    path = runs/'development'/'m1'/'00'/'Expert'/'11'/'result.json'
    data=json.loads(path.read_text())
    data['policy_settings']={'enabled':True,**SWITCHES,'m1':True}
    data['effective_config']={'v2_playability_'+key:value for key,value in data['policy_settings'].items()}
    data['effective_config']['v2_note_sample_temperature'] = .1
    data['actual_settings']['temperature'] = 1
    path.write_text(json.dumps(data))
    # Match the manifest's case settings so the meaningful failure is temperature.
    registry=json.loads(manifest.read_text())
    registry['songs'][0]['variants']['Expert']['temperature']=1
    manifest.write_text(json.dumps(registry))
    frozen['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    result=benchmark.report_run(manifest,runs,frozen,split='development',candidate='m1',bootstrap_samples=30)
    assert any('temperature' in failure['error'] for failure in result['failures'])


def test_recorded_pipeline_cannot_bypass_a_required_production_stage(tmp_path):
    manifest, runs, human = _receipt_run(tmp_path)
    frozen=benchmark.freeze_run_reference(manifest,runs,human,split='development')
    path=runs/'development'/'m1'/'00'/'Expert'/'11'/'result.json'
    data=json.loads(path.read_text())
    data['stages']={'arrows':{'directed_notes':4}}
    path.write_text(json.dumps(data))
    result=benchmark.report_run(manifest,runs,frozen,split='development',candidate='m1',bootstrap_samples=30)
    assert result['pairs'] == 0
    assert 'production stage' in result['failures'][0]['error']


def test_reserved_generation_requires_a_complete_locked_scope(tmp_path, monkeypatch):
    from evaluation.playability_cohort import prepare_cohort, validate_stage
    monkeypatch.chdir(tmp_path)
    (tmp_path/'main.py').write_text('pass\n')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'development_replicates':[11],'holdout_replicates':[19,43],
        'songs':[{'package':'song','split':'holdout','variants':{'Expert':{'nps':5}}}]}))
    root=tmp_path/'locked'
    prepare_cohort(manifest,root,components=['m1','m4'])
    (root/'locked-selection.json').write_text(json.dumps({'decision':'ready_for_confirmation','arm':'m1,m4'}))
    (root/'frozen-development.json').write_text('{}')
    with pytest.raises(ValueError,match='reserved numerical|confirmation declaration'):
        validate_stage(root,'holdout_candidate')


def test_lazy_derived_obstacle_config_uses_authoritative_mode_tables():
    from evaluation.playability_benchmark_run import _pair_audit
    common={'sporty_obstacles':False,'norm_obstacle_allowed_types':[0,1,2],
            'norm_obstacle_positions':[[0],[3]]}
    worker={'models':{},'python':'3.10','tensorflow':'2.15','devices':['CPU'],
            'sources':{},'environment':{'CUDA_VISIBLE_DEVICES':'-1'},'model_settings':{}}
    control={'worker':worker,'data':{'effective_config':common,'actual_settings':{}}}
    candidate={'worker':worker,'data':{'effective_config':{**common,
        'obstacle_allowed_types':[0,1,2],'obstacle_positions':[[0],[3]]},'actual_settings':{}}}
    assert not _pair_audit(control,candidate)
    candidate['data']['effective_config']['norm_obstacle_positions']=[[0],[2]]
    assert any(problem['kind']=='case_settings_difference' for problem in _pair_audit(control,candidate))


def test_symmetric_missing_metric_support_is_visible_without_vetoing_other_cases():
    rows=[_pair('eligible'),_pair('no-segment')]
    rows[1]['control']['distances']['m3']=None
    rows[1]['candidate']['distances']['m3']=None
    margins={'5-7':{key:1 for key in (*benchmark.DISTANCE_METRICS,*benchmark.DWELL_DISTANCE_METRICS)}}
    result=benchmark.paired_decision_report(rows,frozen_margins=margins,bootstrap_samples=30)
    assert len(result['metrics']['m3']['missing_eligibility']) == 1
    assert result['decision'] == 'advance_for_playtest'
    rows[1]['control']['distances']['m3']=10
    assert benchmark.paired_decision_report(rows,frozen_margins=margins,bootstrap_samples=30)['decision']=='experimental'


def test_longest_run_gate_uses_song_maximum_across_difficulties():
    a=_pair('song',difficulty='Easy')
    b=_pair('song',difficulty='Expert')
    a['control']['longest_fast_horizontal_run']=10
    b['control']['longest_fast_horizontal_run']=0
    a['candidate']['longest_fast_horizontal_run']=6
    b['candidate']['longest_fast_horizontal_run']=6
    margins={'5-7':{key:1 for key in (*benchmark.DISTANCE_METRICS,*benchmark.DWELL_DISTANCE_METRICS)}}
    result=benchmark.paired_decision_report([a,b],frozen_margins=margins,bootstrap_samples=30)
    assert result['longest_fast_horizontal_run']['control_song_p90'] == 10
    assert result['longest_fast_horizontal_run']['candidate_song_p90'] == 6
    assert result['longest_fast_horizontal_run']['exploratory_control_song_mean_p90'] == 5
    assert result['target_passed'] is True
