"""Lockstep generate_variants: sequential-generate parity and scope mechanics."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from map_creation.v2_generation import _difficulty_scope, _difficulty_settings, scoped_settings, settings_snapshot
from map_creation.v2_pipeline import generate, generate_variants
from tools.config import config
from training.beat_models.contracts import Song
from training.pipeline_v2.pipeline import PipelineResult, dummy_predictors
from training.pipeline_v2.settings import preparation_settings


def audio_song(frames=1100):
    duration = frames / 100
    bins = int(np.ceil(duration))
    return Song("audio-only", {"duration": duration, "bpm": 120.},
                {"features": np.zeros((frames, 82), np.float32),
                 "coarse_features": np.zeros((bins, 82), np.float32),
                 "frame_times": np.arange(frames, dtype=np.float64) / 100,
                 "bin_durations": np.minimum(1., duration - np.arange(bins)).astype(np.float32)})


def vocabulary():
    return {"multiplier": [{"kind": "none"}, {"kind": "copies", "copies": [{"dx": 0, "dy": -1, "dt": .05}]}],
            "walls": [{"kind": "none"}, {"kind": "wall", "x": 0, "y": 0, "width": 1, "height": 3}],
            "lighting": [{"type": 0, "value": 1}, {"type": 1, "value": 2}]}


def lighting(times, encoder):
    return ([{"time": float(time), "type": 0, "value": 2000000804} for time in times],
            {"backend": "v1", "queries": len(times)})


def processors():
    def timing(song, beats, audio_path, logger=None):
        return [{"time": float(time), "count": int(beats["counts"][frame])}
                for frame, time in zip(beats["event_frames"], beats["event_times"])], {}

    def notes(values, event_times, logger=None):
        return values, {}

    def validate(state, song, logger=None, **_ignored):
        return {}

    return SimpleNamespace(postprocess_timing=timing, postprocess_notes=notes,
                           validate_expansions=validate, validate_hazards=validate, validate_arcs=validate)


class RowWise:
    """Per-row deterministic one-hot heads keyed on each row's own inputs."""

    status = "dummy"

    def __init__(self, classes, shift=0):
        self.classes, self.shift = classes, shift
        self.batches = []

    def __call__(self, inputs):
        batch = len(inputs["audio"])
        self.batches.append(batch)
        pick = (np.asarray(inputs["history_mask"].sum(-1), np.int64) + self.shift) % self.classes
        result = {}
        for head in ("red", "blue"):
            values = np.zeros((batch, self.classes), np.float32)
            values[np.arange(batch), (pick + (head == "blue")) % self.classes] = 1
            result[head] = values
        return result


def time_encoder(calls=None):
    def encoder(times):
        times = np.asarray(times, np.float64)
        if calls is not None:
            calls.append(times.copy())
        return np.tile(times[:, None], (1, 16)).astype(np.float32)
    return encoder


def settings():
    resolved = preparation_settings()
    resolved["note_sample_temperature"] = 0.
    return resolved


def _assert_same(left, right, path=""):
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), f"{path}: keys differ"
        for key in left:
            if key == "elapsed_seconds":
                continue
            _assert_same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_same(a, b, f"{path}[{index}]")
    else:
        assert left == right, f"{path}: {left} != {right}"


def test_lockstep_variants_match_sequential_generate(monkeypatch):
    monkeypatch.setattr(config, "v2_playability_enabled", False)  # strict audits need real validators
    # Global-random consumers (gen_obstacles, _v2_mirroring, arc clearance)
    # share one stream in sequential stage order; interleaving reorders those
    # cosmetic draws, so parity is proven against the model decisions instead.
    monkeypatch.setattr(config, "add_obstacle_flag", False)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    positions, arrows, multiplier = RowWise(13), RowWise(9), RowWise(2)
    predictors.update(positions=positions, arrows=arrows, multiplier=multiplier)
    nps_list = [4., 8.]
    sequential = [
        generate(song, nps, predictors, time_encoder(), settings(), vocab,
                 lighting_generator=lighting, postprocessors=processors(), random_factor=0,
                 rng=np.random.default_rng(seed))
        for nps, seed in zip(nps_list, (11, 22))]
    sequential_batches = [list(predictor.batches) for predictor in (positions, arrows, multiplier)]
    for predictor in (positions, arrows, multiplier):
        predictor.batches.clear()
    batched = generate_variants(song, nps_list, predictors, time_encoder(),
                                [settings() for _ in nps_list], vocab,
                                lighting_generator=lighting, postprocessors=processors(),
                                random_factor=0, rngs=[np.random.default_rng(seed) for seed in (11, 22)])
    assert len(batched) == len(sequential)
    for expected, result in zip(sequential, batched):
        assert expected.status == result.status
        _assert_same(expected.state, result.state, "state")
        _assert_same(expected.stages, result.stages, "stages")
        _assert_same(expected.diagnostics, result.diagnostics, "diagnostics")
        assert expected.review_stops == result.review_stops
    # One predict per step and model: calls drop from the candidate sum to the max.
    for predictor, before in zip((positions, arrows, multiplier), sequential_batches):
        assert all(batch == 1 for batch in before)
        assert set(predictor.batches) <= {1, 2} and 2 in predictor.batches
        assert len(predictor.batches) < len(before)


def test_lockstep_variants_prefetch_encoder_union_once(monkeypatch):
    monkeypatch.setattr(config, "v2_playability_enabled", False)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    nps_list = [4., 8.]
    sequential_calls = []
    for nps in nps_list:
        generate(song, nps, predictors, time_encoder(sequential_calls), settings(), vocab,
                 lighting_generator=lighting, postprocessors=processors(), random_factor=0)
    batched_calls = []
    generate_variants(song, nps_list, predictors, time_encoder(batched_calls),
                      [settings() for _ in nps_list], vocab,
                      lighting_generator=lighting, postprocessors=processors(), random_factor=0)
    # Single-time calls are per-candidate cache lookups (identical multiset in
    # both paths); multi-time calls are stage prefetches — the driver issues
    # one union per stage instead of one per variant.
    sequential_prefetch = [call for call in sequential_calls if len(call) > 1]
    batched_prefetch = [call for call in batched_calls if len(call) > 1]
    assert len(sequential_prefetch) == 4 and len(batched_prefetch) == 2
    np.testing.assert_array_equal(batched_prefetch[0],
                                  np.unique(np.concatenate([sequential_prefetch[0],
                                                            sequential_prefetch[2]])))
    np.testing.assert_array_equal(batched_prefetch[1],
                                  np.unique(np.concatenate([sequential_prefetch[1],
                                                            sequential_prefetch[3]])))
    lookups = lambda calls: sorted(call[0] for call in calls if len(call) == 1)
    np.testing.assert_array_equal(lookups(sequential_calls), lookups(batched_calls))


def test_lockstep_variants_apply_each_scope_to_its_own_variant(monkeypatch):
    monkeypatch.setattr(config, "v2_playability_enabled", False)
    monkeypatch.setattr(config, "single_notes_only_flag", False)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)

    def flag_scope(flag):
        @contextmanager
        def scope():
            previous = config.single_notes_only_flag
            config.single_notes_only_flag = flag
            try:
                yield
            finally:
                config.single_notes_only_flag = previous
        return scope

    results = generate_variants(song, [4., 4.], predictors, time_encoder(),
                                [settings(), settings()], vocab, lighting_generator=lighting,
                                postprocessors=processors(), random_factor=0,
                                scopes=[flag_scope(False), flag_scope(True)])
    assert config.single_notes_only_flag is False  # scopes never leak
    assert any(note.get("kind") == "copy" for note in results[0].state["notes"])
    assert not any(note.get("kind") == "copy" for note in results[1].state["notes"])
    assert results[0].diagnostics["raw_decisions"]["multiplier_non_none"] \
        == results[1].diagnostics["raw_decisions"]["multiplier_non_none"] > 0


def test_lockstep_arrows_variety_groups_by_chosen_model(monkeypatch):
    monkeypatch.setattr(config, "v2_playability_enabled", False)
    monkeypatch.setattr(config, "v2_arrows_variety", .5)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    arrows, reference = RowWise(9), RowWise(9, shift=3)  # distinct opinions make picks observable
    predictors.update(arrows=arrows, arrows_reference=reference)
    nps_list = [4., 8.]
    sequential = [
        generate(song, nps, predictors, time_encoder(), settings(), vocab,
                 lighting_generator=lighting, postprocessors=processors(), random_factor=0,
                 rng=np.random.default_rng(seed))
        for nps, seed in zip(nps_list, (7, 9))]
    sequential_picks = [result.diagnostics["raw_decisions"]["arrow_variety"] for result in sequential]
    arrows.batches.clear()
    reference.batches.clear()
    batched = generate_variants(song, nps_list, predictors, time_encoder(),
                                [settings() for _ in nps_list], vocab,
                                lighting_generator=lighting, postprocessors=processors(),
                                random_factor=0, rngs=[np.random.default_rng(seed) for seed in (7, 9)])
    # Same per-variant draws => identical model picks and identical notes.
    assert [result.diagnostics["raw_decisions"]["arrow_variety"] for result in batched] == sequential_picks
    for expected, result in zip(sequential, batched):
        _assert_same(expected.state["notes"], result.state["notes"], "notes")
    total_steps = max(len(result.state["event_times"]) for result in batched)
    calls = arrows.batches + reference.batches
    assert len(calls) <= 2 * total_steps and all(1 <= batch <= 2 for batch in calls)


def test_difficulty_scope_replays_delta_and_restores_reentrant():
    baseline = settings_snapshot()
    scope = _difficulty_scope(20.)
    with scope():
        via_delta = settings_snapshot()
        config.scope_test_added = [1]       # added keys are removed on exit
        config.max_speed = 99.              # reassigned delta keys are restored
    np.testing.assert_equal(settings_snapshot(), baseline)
    with scoped_settings():
        _difficulty_settings(20.)
        via_full = settings_snapshot()
    np.testing.assert_equal(via_delta, via_full)  # same effective config as the full scope
    with scope():                                # reusable: second entry replays it
        np.testing.assert_equal(settings_snapshot(), via_full)
    np.testing.assert_equal(settings_snapshot(), baseline)
    assert not hasattr(config, "scope_test_added")


def test_lockstep_variants_replay_late_turns_phase(monkeypatch):
    # @_scope_playability_flow wraps generate(): batched variants replay the
    # same ContextVar phase inside each of their own scoped pieces.
    from map_creation.v2_postprocessing import _LATE_TURNS_READY

    monkeypatch.setattr(config, "v2_playability_enabled", True)
    monkeypatch.setattr(config, "v2_playability_late_turns", True)
    monkeypatch.setattr(config, "v2_playability_reach", False)
    monkeypatch.setattr(config, "v2_playability_priority", False)
    song, vocab = audio_song(), vocabulary()
    predictors = dummy_predictors(vocab)
    hooks, observed = processors(), []

    def timing(song, beats, audio_path, logger=None):
        observed.append(_LATE_TURNS_READY.get())
        return [{"time": float(time), "count": int(beats["counts"][frame])}
                for frame, time in zip(beats["event_frames"], beats["event_times"])], {}

    hooks.postprocess_timing = timing
    generate(song, 4., predictors, time_encoder(), settings(), vocab,
             lighting_generator=lighting, postprocessors=hooks, random_factor=0)
    sequential_observed = list(observed)
    generate_variants(song, [4., 8.], predictors, time_encoder(), [settings(), settings()], vocab,
                      lighting_generator=lighting, postprocessors=hooks, random_factor=0)
    assert sequential_observed == [False] and observed[1:] == [False, False]
    assert _LATE_TURNS_READY.get() is True  # no phase leaks outside segments


def test_generate_batch_scopes_settings_and_caches_intensity(monkeypatch, tmp_path):
    import threading
    from map_creation import v2_pipeline, v2_runtime
    from training.beat_models.artifacts import statistics_hash

    statistics = {"version": 1, "fit_split": "train", "song_ids": ["training-song"],
                  **{key: {"mean": [0.] * 82, "scale": [1.] * 82, "count": 1}
                     for key in ("features", "coarse_features")}}
    value = v2_runtime.V2Runtime.__new__(v2_runtime.V2Runtime)
    value.statistics, value.settings, value.vocabularies = statistics, preparation_settings(), {}
    value.predictors, value.lighting = {}, object()
    value.encoder_fingerprint, value.root = "retained", tmp_path
    value._lock = threading.RLock()
    value._cache_metrics = {"bundle_loads": 1, "prepared_audio_cache_hits": 0,
                            "audio_preparations": 0, "generation_calls": 0}
    song = Song("audio", {"duration": 1.5, "bpm": 120.,
                          "normalization_sha256": statistics_hash(statistics)},
                {"features": np.zeros((150, 82), np.float32),
                 "coarse_features": np.zeros((2, 82), np.float32),
                 "frame_times": np.arange(150, dtype=np.float64) / 100,
                 "bin_durations": np.array([1., .5], np.float32)})
    prepared = SimpleNamespace(song=song, audio_path=None, intensity=None, diagnostics={},
                               encoder=SimpleNamespace(metadata={"encoder_fingerprint": "retained"}))
    seen = {}

    def fake(song, nps_list, predictors, encoder, settings_list, vocabularies, *,
             lighting_generator, audio_path, intensity, logger, scopes):
        seen.update(nps=list(nps_list), scopes=list(scopes),
                    temps=[item["note_sample_temperature"] for item in settings_list])
        return [PipelineResult("trained", {"raw_intensity": np.array([.2, .8], np.float32)})
                for _ in nps_list]

    monkeypatch.setattr(v2_pipeline, "generate_variants", fake)

    def temp_scope(temperature):
        @contextmanager
        def scope():
            previous = config.v2_note_sample_temperature
            config.v2_note_sample_temperature = temperature
            try:
                yield
            finally:
                config.v2_note_sample_temperature = previous
        return scope

    results = value.generate_batch(prepared, [2., 4.], scopes=[temp_scope(.3), temp_scope(.7)])
    assert seen["nps"] == [2., 4.] and seen["temps"] == [.3, .7]
    np.testing.assert_allclose(prepared.intensity, [.2, .8])
    assert [result.diagnostics["runtime"]["intensity_cache_reused"] for result in results] == [False, True]
    assert all(result.diagnostics["bundle_path"] == str(tmp_path) for result in results)
    with pytest.raises(ValueError, match="variant"):
        value.generate_batch(prepared, [], scopes=[])
