"""Explicit, cached CPU loading and audio preparation for the v2 generator."""

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
from time import perf_counter
import zipfile

import numpy as np

from tools.config import config
from training.beat_models.artifacts import (file_hash, load_model_manifest, read_json,
                                           json_value, statistics_hash, validate_statistics)
from training.beat_models.settings import architecture_settings as beat_architecture
from training.beat_models.contracts import FEATURES, FRAME_SECONDS, REPRESENTATION, Song
from training.beat_models.features import decode_audio, extract_features, normalize_songs
from training.pipeline_v2.artifacts import load_checkpoint_manifest
from training.pipeline_v2.encoder import ENCODER_REPRESENTATION, FixedEncoder, decode_encoder_audio
from training.pipeline_v2.settings import STAGES


_REPOSITORY = Path(__file__).resolve().parents[1]
_CACHE = None
_CACHE_HITS = 0
_CACHE_LOCK = threading.Lock()


def _bundle_path(path):
    selected = Path(getattr(config, "v2_model_bundle_path", "../Data/model/v2_expert") if path is None else path)
    return (selected if selected.is_absolute() else _REPOSITORY / selected).resolve()


def _local(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Bundle paths must be relative")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or invalid v2 bundle file: {name}")
    return path


def validate_bundle(bundle_path=None):
    """Check all required assets and frozen contracts without importing TensorFlow."""
    root = _bundle_path(bundle_path)
    bundle = read_json(root / "bundle.json")
    if bundle.get("kind") != "infernosaber_v2_inference_bundle" or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported v2 inference bundle")
    required = {"intensity", "timing", *STAGES}
    stages = bundle.get("model_stages", {})
    if set(stages) != required:
        raise ValueError("V2 bundle must contain the eight separate trained stages")
    for name, record in bundle.get("files", {}).items():
        path = _local(root, name)
        if file_hash(path) != record.get("sha256"):
            raise ValueError(f"V2 bundle checksum mismatch: {name}")
    statistics = read_json(_local(root, "normalization.json"))
    validate_statistics(statistics)
    if statistics_hash(statistics) != bundle.get("normalization_sha256"):
        raise ValueError("V2 bundle normalization fingerprint mismatch")
    vocabularies = read_json(_local(root, "vocabularies.json"))
    if statistics_hash(vocabularies) != bundle.get("vocabularies_sha256"):
        raise ValueError("V2 bundle vocabulary fingerprint mismatch")
    if any(not vocabularies.get(key) for key in ("multiplier", "walls", "lighting")):
        raise ValueError("V2 bundle requires frozen multiplier, wall and lighting vocabularies")
    encoder_path = _local(root, bundle.get("encoder"))
    encoder_manifest = read_json(encoder_path.parent / "manifest.json")
    identity = encoder_manifest.get("encoder_identity", {})
    encoder_hash = file_hash(encoder_path)
    if (encoder_manifest.get("kind") != "retained_v2_audio_encoder"
            or encoder_hash != encoder_manifest.get("checkpoint_sha256")
            or identity.get("checkpoint_fingerprint") != encoder_hash
            or identity.get("representation") != ENCODER_REPRESENTATION
            or identity.get("status") != "retained"
            or identity.get("grid_seconds") != bundle.get("data_settings", {}).get("encoder_grid_seconds")):
        raise ValueError("Incompatible retained v2 audio encoder")
    context = {"beat_representation": REPRESENTATION,
               "normalization_sha256": statistics_hash(statistics), "encoder_identity": identity,
               "data_settings": bundle.get("data_settings"), "vocabularies": vocabularies}
    manifests = {}
    for stage in required:
        path = _local(root, f"{stages[stage]}/manifest.json").parent
        if stage in ("intensity", "timing"):
            _, manifest = load_model_manifest(path, stage)
            if manifest.get("normalization_sha256") != context["normalization_sha256"]:
                raise ValueError(f"Incompatible {stage} normalization")
            if manifest.get("training", {}).get("architecture") != json_value(beat_architecture(stage)):
                raise ValueError(f"Unsupported {stage} checkpoint architecture")
        else:
            _, manifest = load_checkpoint_manifest(path, stage, context)
        if manifest.get("status") != "trained" or manifest.get("training", {}).get("status", "trained") != "trained":
            raise ValueError(f"V2 generation requires a trained {stage} checkpoint")
        manifests[stage] = manifest
    calibrations = {}
    from training.pipeline_v2.calibration import load_calibration
    for stage, filename in bundle.get("calibrations", {}).items():
        if stage not in ("hazards", "lighting", "arcs"):
            raise ValueError(f"Unsupported calibrated stage: {stage}")
        calibrations[stage] = load_calibration(_local(root, filename), stage, manifests[stage])
    return {"root": root, "bundle": bundle, "normalization": statistics,
            "vocabularies": vocabularies, "encoder_identity": identity, "manifests": manifests,
            "calibrations": calibrations}


@dataclass
class PreparedAudio:
    song: Song
    encoder: object
    audio_path: Path
    raw_features: dict = field(default_factory=dict)
    intensity: object = None
    diagnostics: dict = field(default_factory=dict)


def _validate_audio_song(song):
    """Validate only audio features and timing metadata, never training targets."""
    duration, bpm = song.metadata.get("duration", 0), song.metadata.get("bpm", 0)
    if not np.isfinite([duration, bpm]).all() or duration <= 0 or bpm <= 0:
        raise ValueError("V2 audio requires positive duration and BPM")
    frames, bins = len(song.arrays["frame_times"]), len(song.arrays["bin_durations"])
    shapes = {"features": (frames, FEATURES), "coarse_features": (bins, FEATURES),
              "frame_times": (frames,), "bin_durations": (bins,)}
    for key, shape in shapes.items():
        values = song.arrays[key]
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid v2 audio features: {key}")
    times, durations = song.arrays["frame_times"], song.arrays["bin_durations"]
    if (not frames or not bins or times[-1] >= duration
            or not np.allclose(times, np.arange(frames) * FRAME_SECONDS, atol=1e-9, rtol=0)
            or np.any(durations <= 0) or np.any(durations > 1)
            or not np.isclose(durations.sum(), duration, atol=1e-5)):
        raise ValueError("Invalid v2 audio timeline")


class V2Runtime:
    """Load one complete bundle once, retaining one song's feature/latent cache."""

    def __init__(self, bundle_path=None):
        started = perf_counter()
        validated = validate_bundle(bundle_path)
        self.root, self.bundle = validated["root"], validated["bundle"]
        self.statistics, self.vocabularies = validated["normalization"], validated["vocabularies"]
        self.settings = deepcopy(self.bundle["data_settings"])
        for stage, record in validated["calibrations"].items():
            key = {"hazards": "bomb_threshold", "lighting": "lighting_threshold", "arcs": "arc_threshold"}[stage]
            self.settings[key] = record["threshold"]
        import tensorflow as tf
        threads = getattr(config, "v2_inference_threads", 2)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("V2 inference threads must be a positive integer")
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as error:
            if tf.config.get_visible_devices("GPU"):
                raise RuntimeError("V2 application generation requires CPU execution") from error
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(threads)
        except RuntimeError:
            # Existing application imports may already have initialized CPU TF.
            pass
        from evaluation.beat_pair import BeatPair, _graph_signature
        from training.pipeline_v2.runner import load_stage_artifact
        paths = self.bundle["model_stages"]
        self.models = {}
        compile_calls = getattr(config, "v2_compile_inference", True)

        def predictor(model):
            def call(inputs):
                return model(inputs, training=False)
            return tf.function(call, reduce_retracing=True) if compile_calls else call

        def beat_loader(checkpoint, stage, manifest):
            from training.beat_models import models
            with zipfile.ZipFile(checkpoint) as archive:
                saved = json.loads(archive.read("config.json"))
            with tf.device("/CPU:0"):
                model = models.build_model(stage)
                if _graph_signature(saved) != _graph_signature(json.loads(model.to_json())):
                    raise ValueError(f"Unsupported {stage} checkpoint graph")
                model.load_weights(checkpoint)
                parameters = manifest["training"].get("parameters")
                if parameters is not None and parameters != model.count_params():
                    raise ValueError(f"{stage} checkpoint parameter count does not match its manifest")
            self.models[stage] = model
            return predictor(model)

        self.pair = BeatPair(self.root / paths["intensity"], self.root / paths["timing"], model_loader=beat_loader)
        self.predictors = dict(self.pair.predictors)
        with tf.device("/CPU:0"):
            for stage in STAGES:
                model, _ = load_stage_artifact(self.root / paths[stage], stage)
                self.models[stage] = model
                self.predictors[stage] = predictor(model)
            self.encoder_model = tf.keras.models.load_model(str(self.root / self.bundle["encoder"]), compile=False)
            if tuple(self.encoder_model.input_shape[1:]) != (24, 20, 1) or tuple(self.encoder_model.output_shape[1:]) != (16,):
                raise ValueError("The retained encoder must accept [24,20,1] and return [16]")
            self._encode = predictor(self.encoder_model)
        self.encoder_fingerprint = validated["encoder_identity"]["checkpoint_fingerprint"]
        self._prepared = None
        self._prepared_key = None
        self._lock = threading.RLock()
        self.load_seconds = perf_counter() - started
        self._cache_metrics = {"bundle_loads": 1, "prepared_audio_cache_hits": 0,
                               "audio_preparations": 0, "generation_calls": 0}

    def prepare_audio(self, audio_path, bpm=None):
        started = perf_counter()
        path = Path(audio_path).resolve()
        if not path.is_file():
            raise ValueError(f"Audio file does not exist: {path}")
        manual = getattr(config, "use_fixed_bpm", None)
        if bpm is None and manual is not None and manual > 0:
            bpm = manual
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, bpm)
        with self._lock:
            if not hasattr(self, "_cache_metrics"):
                self._cache_metrics = {"bundle_loads": 1, "prepared_audio_cache_hits": 0,
                                       "audio_preparations": 0, "generation_calls": 0}
            if key == self._prepared_key:
                self._cache_metrics["prepared_audio_cache_hits"] += 1
                return self._prepared
            if bpm is None:
                from .find_bpm import get_file_bpm
                bpm = get_file_bpm(path)[0]
                if not np.isfinite(bpm) or bpm <= 0:
                    raise ValueError("No positive tempo could be estimated; set the existing manual BPM control")
            if isinstance(bpm, bool) or not np.isfinite(bpm) or bpm <= 0:
                raise ValueError("BPM must be finite and positive")
            arrays, metadata = extract_features(decode_audio(path))
            metadata["bpm"] = float(bpm)
            song = normalize_songs([Song(path.stem, metadata, arrays)], self.statistics)[0]
            _validate_audio_song(song)
            encoder = FixedEncoder(decode_encoder_audio(path), self._encode, self.encoder_fingerprint,
                                   grid_seconds=self.settings["encoder_grid_seconds"],
                                   batch_size=self.settings["encoder_batch_size"], duration=metadata["duration"])
            self._prepared = PreparedAudio(song, encoder, path, raw_features=arrays,
                                           diagnostics={"preparation_seconds": perf_counter() - started})
            self._prepared_key = key
            self._cache_metrics["audio_preparations"] += 1
            return self._prepared

    def generate(self, prepared, requested_nps, logger=None):
        from .v2_pipeline import generate
        with self._lock:
            started = perf_counter()
            _validate_audio_song(prepared.song)
            if prepared.song.metadata.get("normalization_sha256") != statistics_hash(self.statistics):
                raise ValueError("Prepared audio belongs to a different normalization contract")
            if prepared.encoder.metadata.get("encoder_fingerprint") != self.encoder_fingerprint:
                raise ValueError("Prepared audio belongs to a different retained encoder")
            reused_intensity = prepared.intensity is not None
            result = generate(prepared.song, requested_nps, self.predictors, prepared.encoder,
                              self.settings, self.vocabularies, audio_path=prepared.audio_path,
                              intensity=prepared.intensity, logger=logger)
            prepared.intensity = result.state["intensity"].copy()
            result.diagnostics["bundle_path"] = str(self.root)
            self._cache_metrics["generation_calls"] += 1
            result.diagnostics["runtime"] = {**self._cache_metrics, **prepared.diagnostics,
                                              "bundle_load_seconds": getattr(self, "load_seconds", 0.),
                                              "runtime_cache_hits": _CACHE_HITS,
                                              "intensity_cache_reused": reused_intensity,
                                              "generation_seconds": perf_counter() - started}
            return result


def get_runtime(bundle_path=None):
    """Keep one bundle per process, rebuilding after a fork or explicit selection."""
    global _CACHE, _CACHE_HITS
    key = os.getpid(), _bundle_path(bundle_path)
    with _CACHE_LOCK:
        if _CACHE is None or _CACHE[0] != key:
            _CACHE = key, V2Runtime(key[1])
            _CACHE_HITS = 0
        else:
            _CACHE_HITS += 1
        return _CACHE[1]
