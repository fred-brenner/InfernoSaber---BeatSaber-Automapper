"""Bundle preflight and per-process/per-song caching without real checkpoints."""

from copy import deepcopy
import json
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from map_creation import v2_runtime as runtime
from tools.config import config
from training.beat_models.artifacts import file_hash, statistics_hash
from training.beat_models.contracts import REPRESENTATION
from training.beat_models.settings import architecture_settings as beat_architecture
from training.pipeline_v2.artifacts import compatibility
from training.pipeline_v2.encoder import ENCODER_REPRESENTATION
from training.pipeline_v2.pipeline import PipelineResult
from training.pipeline_v2.settings import STAGES, preparation_settings


def statistics():
    return {"version": 1, "fit_split": "train", "song_ids": ["training-song"],
            **{key: {"mean": [0.] * 82, "scale": [1.] * 82, "count": 1}
               for key in ("features", "coarse_features")}}


def bundle_fixture(root):
    root.mkdir()

    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    stats = statistics()
    vocab = {"multiplier": [{"kind": "none"}], "walls": [{"kind": "none"}],
             "lighting": [{"type": 0, "value": 1}]}
    write("normalization.json", stats)
    write("vocabularies.json", vocab)
    (root / "encoder").mkdir()
    (root / "encoder/encoder.h5").write_bytes(b"retained encoder test fixture")
    encoder_hash = file_hash(root / "encoder/encoder.h5")
    identity = {"status": "retained", "checkpoint_fingerprint": encoder_hash,
                "grid_seconds": .5, "representation": deepcopy(ENCODER_REPRESENTATION)}
    write("encoder/manifest.json", {"kind": "retained_v2_audio_encoder", "checkpoint_sha256": encoder_hash,
                                    "encoder_identity": identity})
    context = {"beat_representation": REPRESENTATION, "normalization_sha256": statistics_hash(stats),
               "encoder_identity": identity, "data_settings": preparation_settings(), "vocabularies": vocab}
    stages = {stage: stage for stage in ("intensity", "timing", *STAGES)}
    for stage in stages:
        directory = root / stage
        directory.mkdir()
        (directory / "best.keras").write_bytes(f"non-executable {stage} fixture".encode())
        manifest = {"schema_version": 1, "stage": stage, "status": "trained", "checkpoint": "best.keras",
                    "checkpoint_sha256": file_hash(directory / "best.keras"), "training": {"status": "trained"}}
        if stage in ("intensity", "timing"):
            manifest.update(kind="model_checkpoint", representation=REPRESENTATION, normalization=stats,
                            normalization_sha256=statistics_hash(stats))
            manifest["training"]["architecture"] = beat_architecture(stage)
        else:
            manifest.update(kind="pipeline_v2_checkpoint", compatibility=compatibility(context, stage), vocabularies=vocab)
        write(f"{stage}/manifest.json", manifest)
    document = {"kind": "infernosaber_v2_inference_bundle", "schema_version": 1, "model_stages": stages,
                "encoder": "encoder/encoder.h5", "data_settings": context["data_settings"],
                "normalization_sha256": statistics_hash(stats), "vocabularies_sha256": statistics_hash(vocab),
                "files": {str(path.relative_to(root)): {"sha256": file_hash(path)}
                          for path in root.rglob("*") if path.is_file()}}
    write("bundle.json", document)
    return root


@pytest.fixture
def production_assets(tmp_path, monkeypatch):
    import joblib
    from lighting_prediction.v1_lighting import validate_assets

    model, vocabulary = tmp_path / "v1.h5", tmp_path / "v1-events.pkl"
    model.write_bytes(b"retained v1 fixture; model loading is injected")
    joblib.dump(SimpleNamespace(categories_=[np.array(["0;1", "1;2000000804"])]), vocabulary)
    monkeypatch.setattr(config, "v1_lighting_model_path", str(model))
    monkeypatch.setattr(config, "v1_lighting_vocabulary_path", str(vocabulary))
    return validate_assets()


def _write_bundle(root, document):
    (root / "bundle.json").write_text(json.dumps(document), encoding="utf-8")


def test_bundle_preflight_is_independent_of_tensorflow_and_original_sources(tmp_path):
    root = bundle_fixture(tmp_path / "bundle")
    validated = runtime.validate_bundle(root)
    assert validated["root"] == root
    code = "import sys; from map_creation.v2_runtime import validate_bundle; validate_bundle(sys.argv[1]); assert 'tensorflow' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_bundle_preflight_rejects_corrupt_checkpoint(tmp_path):
    root = bundle_fixture(tmp_path / "bundle")
    (root / "positions/best.keras").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        runtime.validate_bundle(root)


@pytest.mark.parametrize("validator", [runtime.validate_bundle, runtime.validate_production_bundle])
def test_bundle_preflight_rejects_incompatible_encoder_contract(tmp_path, production_assets, validator):
    root = bundle_fixture(tmp_path / "bundle")
    manifest = json.loads((root / "positions/manifest.json").read_text())
    manifest["compatibility"]["encoder_identity"]["checkpoint_fingerprint"] = "different"
    content = manifest["compatibility"].copy()
    content.pop("sha256")
    manifest["compatibility"]["sha256"] = statistics_hash(content)
    (root / "positions/manifest.json").write_text(json.dumps(manifest))
    document = json.loads((root / "bundle.json").read_text())
    document["files"]["positions/manifest.json"]["sha256"] = file_hash(root / "positions/manifest.json")
    (root / "bundle.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="encoder, vocabulary or dataset contract"):
        validator(root)


@pytest.mark.parametrize("directory", ["lighting", "experimental/lights"])
def test_production_preflight_ignores_missing_or_corrupt_experimental_lighting(tmp_path, production_assets, directory):
    root = bundle_fixture(tmp_path / "bundle")
    document = json.loads((root / "bundle.json").read_text())
    if directory != "lighting":
        destination = root / directory
        destination.parent.mkdir()
        (root / "lighting").rename(destination)
        document["model_stages"]["lighting"] = directory
        document["files"] = {name.replace("lighting/", directory + "/"): record
                             for name, record in document["files"].items()}
    (root / directory / "best.keras").write_bytes(b"corrupt experimental checkpoint")
    (root / directory / "manifest.json").unlink()
    document["files"][directory + "/report.json"] = {"sha256": "missing experimental report"}
    document["calibrations"] = {"lighting": "calibrations/lighting.json"}
    document["files"]["calibrations/lighting.json"] = {"sha256": "missing experimental sidecar"}
    _write_bundle(root, document)
    before = (root / "bundle.json").read_bytes()
    validated = runtime.validate_production_bundle(root)
    assert set(validated["manifests"]) == {"intensity", "timing", *runtime.PRODUCTION_STAGES}
    assert validated["calibrations"] == {}
    assert validated["bundle"] == document
    np.testing.assert_array_equal(validated["lighting_assets"]["categories"], [[0, 1], [1, 2000000804]])
    assert (root / "bundle.json").read_bytes() == before
    with pytest.raises(ValueError, match="checksum|Missing"):
        runtime.validate_bundle(root)


def test_production_preflight_accepts_optional_lighting_mapping_without_tensorflow(tmp_path, production_assets):
    root = bundle_fixture(tmp_path / "bundle")
    document = json.loads((root / "bundle.json").read_text())
    document["model_stages"].pop("lighting")
    shutil.rmtree(root / "lighting")
    _write_bundle(root, document)
    code = '''
import sys
from tools.config import config
config.v1_lighting_model_path, config.v1_lighting_vocabulary_path = sys.argv[2:]
from map_creation.v2_runtime import validate_production_bundle
validate_production_bundle(sys.argv[1])
assert 'tensorflow' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", code, str(root), str(production_assets["model_path"]),
                             str(production_assets["vocabulary_path"])], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_production_preflight_does_not_require_v2_lighting_categories(tmp_path, production_assets):
    root = bundle_fixture(tmp_path / "bundle")
    document = json.loads((root / "bundle.json").read_text())
    vocabulary = json.loads((root / "vocabularies.json").read_text())
    vocabulary.pop("lighting")
    (root / "vocabularies.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    document["vocabularies_sha256"] = statistics_hash(vocabulary)
    document["files"]["vocabularies.json"]["sha256"] = file_hash(root / "vocabularies.json")
    _write_bundle(root, document)
    assert "lighting" not in runtime.validate_production_bundle(root)["vocabularies"]
    with pytest.raises(ValueError, match="frozen.*lighting"):
        runtime.validate_bundle(root)


@pytest.mark.parametrize("filename", [*(stage + "/best.keras" for stage in ("intensity", "timing", *runtime.PRODUCTION_STAGES)),
                                      "encoder/encoder.h5", "normalization.json", "vocabularies.json"])
def test_production_preflight_still_rejects_corrupt_required_assets(tmp_path, production_assets, filename):
    root = bundle_fixture(tmp_path / "bundle")
    (root / filename).write_bytes(b"corrupted production asset")
    with pytest.raises(ValueError, match="checksum"):
        runtime.validate_production_bundle(root)


@pytest.mark.parametrize("stage", ["hazards", "arcs"])
def test_production_preflight_requires_other_calibration_files(tmp_path, production_assets, stage):
    root = bundle_fixture(tmp_path / "bundle")
    document = json.loads((root / "bundle.json").read_text())
    document["calibrations"] = {stage: "missing-calibration.json"}
    _write_bundle(root, document)
    with pytest.raises(ValueError, match="Missing or invalid"):
        runtime.validate_production_bundle(root)


@pytest.mark.parametrize("family,value", [("model_stages", "positions"), ("model_stages", "encoder"),
                                          ("model_stages", "../outside"), ("model_stages", "."),
                                          ("calibrations", "positions/best.keras"),
                                          ("calibrations", "encoder/encoder.h5"),
                                          ("calibrations", "vocabularies.json")])
def test_production_exclusions_cannot_hide_required_assets(tmp_path, production_assets, family, value):
    root = bundle_fixture(tmp_path / "bundle")
    document = json.loads((root / "bundle.json").read_text())
    document.setdefault(family, {})["lighting"] = value
    _write_bundle(root, document)
    with pytest.raises(ValueError, match="overlap|Invalid v2 bundle path"):
        runtime.validate_production_bundle(root)


def test_production_exclusions_reject_required_file_alias(tmp_path, production_assets):
    root = bundle_fixture(tmp_path / "bundle")
    original, target = root / "normalization.json", root / "lighting/normalization.json"
    original.rename(target)
    try:
        original.symlink_to(target)
    except OSError:
        pytest.skip("Filesystem does not support symlinks")
    with pytest.raises(ValueError, match="overlap"):
        runtime.validate_production_bundle(root)


@pytest.mark.parametrize("family", ["model_stages", "calibrations"])
def test_production_preflight_rejects_unknown_stages(tmp_path, production_assets, family):
    root = bundle_fixture(tmp_path / "bundle")
    document = json.loads((root / "bundle.json").read_text())
    document.setdefault(family, {})["unknown"] = "unused"
    _write_bundle(root, document)
    with pytest.raises(ValueError, match="seven separate|Unsupported calibrated"):
        runtime.validate_production_bundle(root)


@pytest.mark.parametrize("asset", ["model_path", "vocabulary_path"])
def test_production_preflight_requires_retained_v1_assets(tmp_path, production_assets, asset):
    root = bundle_fixture(tmp_path / "bundle")
    production_assets[asset].unlink()
    with pytest.raises(ValueError, match="Missing v1 lighting asset"):
        runtime.validate_production_bundle(root)


def test_runtime_loads_v1_lighting_once_and_never_loads_v2_lighting(tmp_path, monkeypatch, production_assets):
    import tensorflow as tf
    from evaluation import beat_pair
    from lighting_prediction import v1_lighting
    from training.pipeline_v2 import runner

    root = bundle_fixture(tmp_path / "bundle")
    shutil.rmtree(root / "lighting")
    monkeypatch.setattr(config, "v2_compile_inference", False)
    monkeypatch.setattr(runtime, "_CACHE", None)
    monkeypatch.setattr(beat_pair, "BeatPair", lambda *args, **kwargs: SimpleNamespace(
        predictors={"intensity": object(), "timing": object()}))
    stages, encoders, lighting_loads = [], [], []
    monkeypatch.setattr(runner, "load_stage_artifact", lambda path, stage: (stages.append(stage) or object(), {}))
    encoder = SimpleNamespace(input_shape=(None, 24, 20, 1), output_shape=(None, 16))
    monkeypatch.setattr(tf.keras.models, "load_model", lambda path, **kwargs: encoders.append(path) or encoder)
    lighting = SimpleNamespace(model=object())
    monkeypatch.setattr(v1_lighting, "load_lighting", lambda assets, factory: lighting_loads.append(assets) or lighting)
    first = runtime.get_runtime(root)
    assert runtime.get_runtime(root) is first
    assert stages == list(runtime.PRODUCTION_STAGES)
    assert len(encoders) == len(lighting_loads) == 1
    assert lighting_loads[0]["model_path"] == production_assets["model_path"]
    assert first.lighting is lighting and first.models["lighting"] is lighting.model
    assert "lighting" not in first.predictors


def test_runtime_cache_is_per_pid_and_explicit_bundle(monkeypatch, tmp_path):
    made = []

    def make(path):
        value = object()
        made.append((path, value))
        return value

    monkeypatch.setattr(runtime, "V2Runtime", make)
    monkeypatch.setattr(runtime, "_CACHE", None)
    monkeypatch.setattr(runtime.os, "getpid", lambda: 100)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "v2_model_bundle_path", "../Data/model/v2_expert", raising=False)
    first = runtime.get_runtime()
    assert runtime.get_runtime() is first
    assert made[0][0] == (runtime._REPOSITORY / "../Data/model/v2_expert").resolve()
    monkeypatch.setattr(runtime.os, "getpid", lambda: 101)
    assert runtime.get_runtime() is not first
    assert runtime.get_runtime(tmp_path / "another-bundle") is not made[1][1]
    monkeypatch.setattr(config, "v1_lighting_model_path", str(tmp_path / "other-v1.h5"))
    assert runtime.get_runtime(tmp_path / "another-bundle") is not made[2][1]
    monkeypatch.setattr(config, "v1_lighting_vocabulary_path", str(tmp_path / "other-events.pkl"))
    assert runtime.get_runtime(tmp_path / "another-bundle") is not made[3][1]
    assert len(made) == 5


def test_prepare_audio_caches_song_and_intensity_between_difficulties(monkeypatch, tmp_path):
    from training.beat_models.contracts import CORE

    for key, setting in {"low_boost": .1, "high_reduction": .1, "scale": 1., "adjustment_strength": .8}.items():
        monkeypatch.setattr(config, f"v2_intensity_{key}", setting)
    value = runtime.V2Runtime.__new__(runtime.V2Runtime)
    value.statistics, value.settings = statistics(), preparation_settings()
    value.vocabularies, value.predictors = {}, {}
    value.lighting = object()
    value.encoder_fingerprint = "test-retained-encoder"
    value._encode = lambda patches: np.zeros((len(patches), 16), np.float32)
    value._prepared, value._prepared_key = None, None
    value._lock = threading.RLock()
    value.root = tmp_path
    path = tmp_path / "song.ogg"
    path.write_bytes(b"not decoded by this injected test")
    calls = []
    arrays = {"features": np.zeros((150, 82), np.float32), "coarse_features": np.zeros((2, 82), np.float32),
              "frame_times": np.arange(150, dtype=np.float64) / 100,
              "bin_durations": np.array([1., .5], np.float32)}
    monkeypatch.setattr(runtime, "decode_audio", lambda path: calls.append(path) or np.zeros(24000, np.float32))
    monkeypatch.setattr(runtime, "decode_encoder_audio", lambda path: np.zeros(22200, np.float32))
    monkeypatch.setattr(runtime, "extract_features", lambda samples: (deepcopy(arrays), {"duration": 1.5}))
    from map_creation import v2_pipeline

    predictions = []
    raw = np.array([.2, .8], np.float32)
    value.predictors = {"intensity": lambda inputs: predictions.append(True) or raw[None],
                        "timing": lambda inputs: np.tile([1., 0., 0.], (len(inputs["audio"]), CORE, 1))}

    def generate(song, nps, *args, intensity=None, lighting_generator, **kwargs):
        assert lighting_generator is value.lighting
        if nps == 2:
            assert intensity is None
        else:
            np.testing.assert_array_equal(intensity, raw)
        beats = v2_pipeline.infer_beats(song, nps, value.predictors, intensity=intensity)
        return PipelineResult("dummy", beats, diagnostics={"export_ready": True})

    monkeypatch.setattr(v2_pipeline, "generate", generate)
    prepared = value.prepare_audio(path, bpm=120)
    assert value.prepare_audio(path, bpm=120) is prepared
    assert len(calls) == 1
    assert set(prepared.song.arrays) == set(arrays)
    assert prepared.encoder.metadata["silence_scale_fallback"]
    first = value.generate(prepared, 2)
    second = value.generate(prepared, 4)
    np.testing.assert_array_equal(first.state["intensity"], second.state["intensity"])
    assert not np.allclose(first.state["intensity"], raw)
    monkeypatch.setattr(config, "v2_intensity_low_boost", 1.)
    monkeypatch.setattr(config, "v2_intensity_high_reduction", 1.)
    monkeypatch.setattr(config, "v2_intensity_adjustment_strength", 1.)
    third = value.generate(prepared, 6)
    np.testing.assert_allclose(third.state["intensity"], [2 / 3, 1 / 3])
    np.testing.assert_array_equal(prepared.intensity, raw)
    assert len(predictions) == 1
    assert len(calls) == 1
    assert prepared.song.metadata["bpm"] == 120


def test_prepare_audio_rejects_nonpositive_explicit_bpm(tmp_path):
    value = runtime.V2Runtime.__new__(runtime.V2Runtime)
    value._lock = threading.RLock()
    value._prepared_key = None
    path = tmp_path / "song.ogg"
    path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="BPM"):
        value.prepare_audio(path, bpm=0)
