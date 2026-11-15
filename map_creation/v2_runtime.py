"""Explicit, cached CPU loading and audio preparation for the v2 generator."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
from time import perf_counter
import uuid
import zipfile

import numpy as np

from tools.config import config
from training.beat_models.artifacts import (file_hash, load_model_manifest, posix_gfile_join, read_json,
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
PRODUCTION_STAGES = tuple(stage for stage in STAGES if stage != "lighting")
# ONNX Runtime dispatch targets the hot autoregressive decode stages only;
# intensity/timing, the retained encoder and v1 lighting stay on TensorFlow.
_ONNX_STAGES = frozenset({"positions", "arrows", "multiplier", "arrows_reference"})
_ONNX_OPSET = 15


def _onnx_predictor(model, source, threads):
    """Return an ORT-backed callable for one dict-input stage model.

    The conversion result is content-addressed under ``temp_path/onnx-cache``
    so processes convert each artifact once; imports stay lazy so environments
    without the optional ONNX dependencies never touch them.
    """
    import onnxruntime
    import tensorflow as tf
    import tf2onnx
    from tools.config import paths

    if not isinstance(model.input, dict):
        raise TypeError("ONNX inference requires dict-input models")
    signature = {name: tf.TensorSpec((None,) + tuple(tensor.shape[1:]), tensor.dtype, name=name)
                 for name, tensor in model.input.items()}
    # The names/shapes/dtypes fingerprint keeps a graph converted against a
    # different input contract from loading stale bytes.
    fingerprint = hashlib.sha256(repr(sorted(
        (name, tuple(spec.shape), spec.dtype.name)
        for name, spec in signature.items())).encode("utf-8")).hexdigest()[:8]
    cache = (Path(paths.temp_path) / "onnx-cache" /
             f"{file_hash(source)[:16]}-{fingerprint}-opset{_ONNX_OPSET}-tf2onnx{tf2onnx.__version__}.onnx")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads

    def create_session(blob):
        return onnxruntime.InferenceSession(blob, options, providers=["CPUExecutionProvider"])

    try:
        blob = cache.read_bytes() if cache.is_file() else None
    except OSError:
        blob = None  # Unreadable entries degrade to a fresh conversion.
    if blob is not None:
        try:
            session = create_session(blob)
        except Exception:
            # A stale or corrupt entry self-heals once: drop it and fall
            # through to a fresh conversion below.
            cache.unlink(missing_ok=True)
            blob = None
    if blob is None:
        converted, _ = tf2onnx.convert.from_function(
            tf.function(lambda inputs: model(inputs, training=False), autograph=False),
            input_signature=[signature], opset=_ONNX_OPSET)
        blob = converted.SerializeToString()
        # Unique temp name + rename: concurrent workers either see the old
        # complete file or the new one, never a partial write. A failed
        # write is tolerated — it never costs the converted blob.
        temporary = cache.with_name(f"{cache.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(blob)
            os.replace(temporary, cache)
        except OSError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
        session = create_session(blob)
    # ORT names match the Keras input/output dict keys but not their order;
    # feed and rebuild strictly by name.
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    returns_dict = isinstance(model.output, dict)

    def predict(inputs):
        outputs = session.run(output_names, {name: np.asarray(inputs[name]) for name in input_names})
        if returns_dict:
            return dict(zip(output_names, outputs))
        return outputs[0] if len(outputs) == 1 else outputs

    predict.session = session
    return predict


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


def _relative(root, name):
    """Resolve a bundle reference without requiring an experimental asset to exist."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Bundle paths must be relative")
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Invalid v2 bundle path: {name}")
    return path


def _production_exclusions(root, bundle, required):
    stages = bundle["model_stages"]
    directories = [_relative(root, stages[stage]) for stage in required]
    lighting = _relative(root, stages.get("lighting", "lighting"))
    calibrations = bundle.get("calibrations", {})
    for stage in calibrations:
        if stage not in ("hazards", "lighting", "arcs"):
            raise ValueError(f"Unsupported calibrated stage: {stage}")
    sidecar = _relative(root, calibrations["lighting"]) if "lighting" in calibrations else None
    encoder = _relative(root, bundle.get("encoder"))
    protected = [(root / "normalization.json").resolve(), (root / "vocabularies.json").resolve(), encoder,
                 (encoder.parent / "manifest.json").resolve(),
                 *(_relative(root, filename) for stage, filename in calibrations.items() if stage != "lighting")]
    sidecar_overlap = sidecar is not None and (
        sidecar in protected or any(sidecar.is_relative_to(path) for path in directories))
    if (any(lighting.is_relative_to(path) or path.is_relative_to(lighting) for path in directories)
            or any(path.is_relative_to(lighting) for path in protected)
            or sidecar_overlap):
        raise ValueError("Experimental lighting paths overlap required production assets")
    return lighting, sidecar


def _validate_bundle(bundle_path=None, *, production=False):
    root = _bundle_path(bundle_path)
    bundle = read_json(root / "bundle.json")
    if bundle.get("kind") != "infernosaber_v2_inference_bundle" or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported v2 inference bundle")
    required = {"intensity", "timing", *(PRODUCTION_STAGES if production else STAGES)}
    stages = bundle.get("model_stages", {})
    if production:
        if not required <= set(stages) or set(stages) - required - {"lighting"}:
            raise ValueError("Production bundle requires seven separate trained v2 stages")
        lighting, sidecar = _production_exclusions(root, bundle, required)
    elif set(stages) != required:
        raise ValueError("V2 bundle must contain the eight separate trained stages")
    for name, record in bundle.get("files", {}).items():
        if production:
            path = _relative(root, name)
            if path.is_relative_to(lighting) or path == sidecar:
                continue
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
    vocabulary_keys = ("multiplier", "walls") if production else ("multiplier", "walls", "lighting")
    if any(not vocabularies.get(key) for key in vocabulary_keys):
        message = ("Production bundle requires frozen multiplier and wall vocabularies" if production else
                   "V2 bundle requires frozen multiplier, wall and lighting vocabularies")
        raise ValueError(message)
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
        if production and stage == "lighting":
            continue
        calibrations[stage] = load_calibration(_local(root, filename), stage, manifests[stage])
    return {"root": root, "bundle": bundle, "normalization": statistics,
            "vocabularies": vocabularies, "encoder_identity": identity, "manifests": manifests,
            "calibrations": calibrations}


def validate_bundle(bundle_path=None):
    """Strictly validate all eight experimental bundle stages without TensorFlow."""
    return _validate_bundle(bundle_path)


def validate_production_bundle(bundle_path=None):
    """Validate seven v2 stages and retained v1 lighting without TensorFlow."""
    from lighting_prediction.v1_lighting import validate_assets
    validated = _validate_bundle(bundle_path, production=True)
    validated["lighting_assets"] = validate_assets()
    return validated


@dataclass
class PreparedAudio:
    song: Song
    encoder: object
    audio_path: Path
    raw_features: dict = field(default_factory=dict)
    intensity: object = None  # Raw model shares; generation adjustments are never cached.
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
    """Cache seven v2 stages, retained v1 lighting, encoder and one recording."""

    def __init__(self, bundle_path=None):
        started = perf_counter()
        validated = validate_production_bundle(bundle_path)
        self.root, self.bundle = validated["root"], validated["bundle"]
        self.statistics, self.vocabularies = validated["normalization"], validated["vocabularies"]
        self.settings = deepcopy(self.bundle["data_settings"])
        for stage, record in validated["calibrations"].items():
            key = {"hazards": "bomb_threshold", "arcs": "arc_threshold"}[stage]
            self.settings[key] = record["threshold"]
        self.settings["hazard_feasibility_decode"] = bool(getattr(config, "v2_hazard_feasibility_decode", False))
        self.settings["hazard_bombs_enabled"] = bool(getattr(config, "v2_hazard_bombs_enabled", True))
        self.settings["hazard_sample_temperature"] = float(getattr(config, "v2_hazard_sample_temperature", 0.) or 0.)
        self.settings["hazard_bomb_rate_decode"] = bool(getattr(config, "v2_hazard_bomb_rate_decode", False))
        self.settings["hazard_bomb_rate_seconds"] = float(getattr(config, "v2_hazard_bomb_rate_seconds", 2.0) or 2.0)
        self.settings["bomb_score_floor"] = float(getattr(config, "v2_hazard_bomb_score_floor", .05) or .05)
        factor = float(getattr(config, "v2_hazard_bomb_rate_factor", 0.) or 0.)
        if factor:
            self.settings["bomb_rate_factor"] = factor
        self.settings["note_sample_temperature"] = float(getattr(config, "v2_note_sample_temperature", 0.) or 0.)
        import tensorflow as tf
        from map_creation.v2_generation import effective_inference_threads
        threads = effective_inference_threads()
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
        from lighting_prediction.v1_lighting import load_lighting
        from training.pipeline_v2.runner import load_stage_artifact
        paths = self.bundle["model_stages"]
        self.models = {}
        compile_calls = getattr(config, "v2_compile_inference", True)

        def _input_signature(model):
            """Batch-agnostic specs matching the model's own input structure."""
            def spec(tensor):
                return tf.TensorSpec((None,) + tuple(tensor.shape[1:]), tensor.dtype)
            nested = model.input
            if isinstance(nested, dict):
                return [{key: spec(tensor) for key, tensor in nested.items()}]
            if isinstance(nested, (list, tuple)):
                return [[spec(tensor) for tensor in nested]]
            return [spec(nested)]

        def predictor(model):
            def call(inputs):
                return model(inputs, training=False)
            if not compile_calls:
                return call
            # Microbenchmarked: calling the traced ConcreteFunction directly
            # skips the tf.function dispatch (~100-200us/call on dict inputs);
            # the signature keeps batch dims dynamic like reduce_retracing.
            return tf.function(call, input_signature=_input_signature(model)).get_concrete_function()

        onnx_stages = _ONNX_STAGES if getattr(config, "v2_onnx_inference", True) else frozenset()
        onnx_warned = []

        def stage_predictor(stage, model, source):
            """Dispatch one stage's calls through ONNX Runtime when enabled.

            The fallback decision is made once here at load — the per-call path
            never pays a dispatch or exception cost.
            """
            if stage in onnx_stages:
                try:
                    return _onnx_predictor(model, source, threads)
                except Exception as error:
                    if not onnx_warned:
                        onnx_warned.append(stage)
                        print(f"Warning: v2_onnx_inference falling back to the TensorFlow "
                              f"predictor for {stage} ({type(error).__name__}: {error}); "
                              f"remaining models may also fall back")
            return predictor(model)

        def beat_loader(checkpoint, stage, manifest):
            from training.beat_models import models
            with zipfile.ZipFile(checkpoint) as archive:
                saved = json.loads(archive.read("config.json"))
            with tf.device("/CPU:0"):
                model = models.build_model(stage)
                if _graph_signature(saved) != _graph_signature(json.loads(model.to_json())):
                    raise ValueError(f"Unsupported {stage} checkpoint graph")
                with posix_gfile_join():
                    model.load_weights(checkpoint)
                parameters = manifest["training"].get("parameters")
                if parameters is not None and parameters != model.count_params():
                    raise ValueError(f"{stage} checkpoint parameter count does not match its manifest")
            self.models[stage] = model
            return predictor(model)

        self.pair = BeatPair(self.root / paths["intensity"], self.root / paths["timing"], model_loader=beat_loader)
        self.predictors = dict(self.pair.predictors)
        with tf.device("/CPU:0"):
            for stage in PRODUCTION_STAGES:
                model, manifest = load_stage_artifact(self.root / paths[stage], stage)
                self.models[stage] = model
                checkpoint = (manifest or {}).get("checkpoint", "best.keras")
                self.predictors[stage] = stage_predictor(stage, model,
                                                         self.root / paths[stage] / checkpoint)
            # Optional second arrows opinion blended at decode by v2_arrows_variety.
            if "arrows" in self.predictors and 0 < float(getattr(config, "v2_arrows_variety", 0.) or 0.):
                reference = _validate_bundle(getattr(config, "v2_arrows_reference_bundle",
                                                     "../Data/model/v2_expert"), production=True)
                stage_dir = reference["root"] / reference["bundle"]["model_stages"]["arrows"]
                model, manifest = load_stage_artifact(stage_dir, "arrows")
                self.models["arrows_reference"] = model
                checkpoint = (manifest or {}).get("checkpoint", "best.keras")
                self.predictors["arrows_reference"] = stage_predictor(
                    "arrows_reference", model, stage_dir / checkpoint)
            self.encoder_model = tf.keras.models.load_model(str(self.root / self.bundle["encoder"]), compile=False)
            if tuple(self.encoder_model.input_shape[1:]) != (24, 20, 1) or tuple(self.encoder_model.output_shape[1:]) != (16,):
                raise ValueError("The retained encoder must accept [24,20,1] and return [16]")
            self._encode = predictor(self.encoder_model)
            self.lighting = load_lighting(validated["lighting_assets"], predictor)
            self.models["lighting"] = self.lighting.model
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
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns, bpm)
        with self._lock:
            if not hasattr(self, "_cache_metrics"):
                self._cache_metrics = {"bundle_loads": 1, "prepared_audio_cache_hits": 0,
                                       "audio_preparations": 0, "generation_calls": 0}
            if key == self._prepared_key:
                self._cache_metrics["prepared_audio_cache_hits"] += 1
                return self._prepared
            octave_repair = None
            if bpm is None:
                from .find_bpm import get_file_bpm, octave_corrected_bpm
                bpm = get_file_bpm(path)[0]
                if not np.isfinite(bpm) or bpm <= 0:
                    raise ValueError("No positive tempo could be estimated; set the existing manual BPM control")
                if getattr(config, "v2_bpm_octave_correct", True):
                    corrected = octave_corrected_bpm(
                        path, bpm,
                        float(getattr(config, "v2_bpm_octave_min_bpm", 180.)),
                        float(getattr(config, "v2_bpm_octave_subbeat_max", .15)),
                        float(getattr(config, "v2_bpm_octave_min_result", 90.)))
                    if corrected != bpm:
                        octave_repair = (float(bpm), float(corrected))
                        bpm = corrected
            if isinstance(bpm, bool) or not np.isfinite(bpm) or bpm <= 0:
                raise ValueError("BPM must be finite and positive")
            arrays, metadata = extract_features(decode_audio(path))
            metadata["bpm"] = float(bpm)
            if octave_repair is not None:
                metadata["bpm_detected"] = octave_repair[0]
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
            # Re-read at call time: a resident runtime must honor live setting changes.
            self.settings["note_sample_temperature"] = float(getattr(config, "v2_note_sample_temperature", 0.) or 0.)
            result = generate(prepared.song, requested_nps, self.predictors, prepared.encoder,
                              self.settings, self.vocabularies, audio_path=prepared.audio_path,
                              intensity=prepared.intensity, logger=logger, lighting_generator=self.lighting)
            prepared.intensity = result.state["raw_intensity"].copy()
            result.diagnostics["bundle_path"] = str(self.root)
            self._cache_metrics["generation_calls"] += 1
            result.diagnostics["runtime"] = {**self._cache_metrics, **prepared.diagnostics,
                                              "bundle_load_seconds": getattr(self, "load_seconds", 0.),
                                              "runtime_cache_hits": _CACHE_HITS,
                                              "intensity_cache_reused": reused_intensity,
                                              "generation_seconds": perf_counter() - started}
            return result

    def generate_batch(self, prepared, requested_nps_list, logger=None, scopes=None):
        """Generate several difficulties of one prepared song in a single lockstep pass.

        ``scopes`` carries one context-manager factory per variant; each
        variant's config-dependent pieces then run under its own settings
        exactly like the per-difficulty ``scoped_settings`` loop did.
        """
        from .v2_pipeline import generate_variants
        with self._lock:
            started = perf_counter()
            _validate_audio_song(prepared.song)
            if prepared.song.metadata.get("normalization_sha256") != statistics_hash(self.statistics):
                raise ValueError("Prepared audio belongs to a different normalization contract")
            if prepared.encoder.metadata.get("encoder_fingerprint") != self.encoder_fingerprint:
                raise ValueError("Prepared audio belongs to a different retained encoder")
            count = len(requested_nps_list)
            if not count or scopes is not None and len(scopes) != count:
                raise ValueError("At least one variant is required and scopes must match it")
            scopes = [nullcontext] * count if scopes is None else scopes
            reused_intensity = prepared.intensity is not None
            settings_list = []
            for index in range(count):
                with scopes[index]():
                    # Re-read at call time like generate(): a resident runtime
                    # honors live setting changes under each variant's scope.
                    # Mutating self.settings keeps the same side effect the
                    # sequential generate() loop had — it ends on the last
                    # variant's live temperature.
                    self.settings["note_sample_temperature"] = \
                        float(getattr(config, "v2_note_sample_temperature", 0.) or 0.)
                    settings_list.append({**self.settings})
            results = generate_variants(prepared.song, requested_nps_list, self.predictors,
                                        prepared.encoder, settings_list, self.vocabularies,
                                        audio_path=prepared.audio_path, intensity=prepared.intensity,
                                        logger=logger, lighting_generator=self.lighting, scopes=scopes)
            prepared.intensity = results[0].state["raw_intensity"].copy()
            seconds = perf_counter() - started
            for index, result in enumerate(results):
                result.diagnostics["bundle_path"] = str(self.root)
                self._cache_metrics["generation_calls"] += 1
                result.diagnostics["runtime"] = {**self._cache_metrics, **prepared.diagnostics,
                                                 "bundle_load_seconds": getattr(self, "load_seconds", 0.),
                                                 "runtime_cache_hits": _CACHE_HITS,
                                                 "intensity_cache_reused": reused_intensity or index > 0,
                                                 # Amortized share of the one
                                                 # shared call — summed
                                                 # diagnostics stay <= wall.
                                                 "generation_seconds": seconds / count}
            return results


def get_runtime(bundle_path=None):
    """Keep one bundle per process, rebuilding after a fork or explicit selection."""
    from lighting_prediction.v1_lighting import lighting_paths
    global _CACHE, _CACHE_HITS
    key = os.getpid(), _bundle_path(bundle_path), *lighting_paths()
    with _CACHE_LOCK:
        if _CACHE is None or _CACHE[0] != key:
            _CACHE = key, V2Runtime(key[1])
            _CACHE_HITS = 0
        else:
            _CACHE_HITS += 1
        return _CACHE[1]
