"""Bundle preflight and per-process/per-song caching without real checkpoints."""

from copy import deepcopy
import json
import subprocess
import sys
import threading

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


def test_bundle_preflight_rejects_incompatible_encoder_contract(tmp_path):
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
        runtime.validate_bundle(root)


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
    assert len(made) == 3


def test_prepare_audio_caches_song_and_intensity_between_difficulties(monkeypatch, tmp_path):
    value = runtime.V2Runtime.__new__(runtime.V2Runtime)
    value.statistics, value.settings = statistics(), preparation_settings()
    value.vocabularies, value.predictors = {}, {}
    value.encoder_fingerprint = "test-retained-encoder"
    value._encode = lambda patches: np.zeros((len(patches), 16), np.float32)
    value._prepared, value._prepared_key = None, None
    value._lock = threading.RLock()
    value.root = tmp_path
    path = tmp_path / "song.ogg"
    path.write_bytes(b"not decoded by this injected test")
    calls = []
    arrays = {"features": np.zeros((6, 82), np.float32), "coarse_features": np.zeros((1, 82), np.float32),
              "frame_times": np.arange(6, dtype=np.float64) / 100,
              "bin_durations": np.array([.06], np.float32)}
    monkeypatch.setattr(runtime, "decode_audio", lambda path: calls.append(path) or np.zeros(960, np.float32))
    monkeypatch.setattr(runtime, "decode_encoder_audio", lambda path: np.zeros(888, np.float32))
    monkeypatch.setattr(runtime, "extract_features", lambda samples: (deepcopy(arrays), {"duration": .06}))
    from map_creation import v2_pipeline

    def generate(song, nps, *args, intensity=None, **kwargs):
        if nps == 2:
            assert intensity is None
        else:
            np.testing.assert_array_equal(intensity, [1.])
        return PipelineResult("dummy", {"intensity": np.array([1.], np.float32)}, diagnostics={"export_ready": True})

    monkeypatch.setattr(v2_pipeline, "generate", generate)
    prepared = value.prepare_audio(path, bpm=120)
    assert value.prepare_audio(path, bpm=120) is prepared
    assert len(calls) == 1
    assert set(prepared.song.arrays) == set(arrays)
    assert prepared.encoder.metadata["silence_scale_fallback"]
    value.generate(prepared, 2)
    value.generate(prepared, 4)
    assert prepared.song.metadata["bpm"] == 120


def test_prepare_audio_rejects_nonpositive_explicit_bpm(tmp_path):
    value = runtime.V2Runtime.__new__(runtime.V2Runtime)
    value._lock = threading.RLock()
    value._prepared_key = None
    path = tmp_path / "song.ogg"
    path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="BPM"):
        value.prepare_audio(path, bpm=0)
