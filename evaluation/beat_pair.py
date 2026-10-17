"""Read-only CPU inference for the trained intensity and timing stages.

This adapter deliberately stops at frame counts. It never imports the generation
runner, constructs a note model, or writes into a checkpoint or input package.
TensorFlow is imported only when the default model loader is first called.
"""

from copy import deepcopy
import json
from numbers import Real
from pathlib import Path
import zipfile

import numpy as np

from training.beat_models.artifacts import (file_hash, json_value, load_model_manifest,
                                           statistics_hash)
from training.beat_models.contracts import (CORE, FEATURES, FRAME_SECONDS, REPRESENTATION,
                                           Song, frame_conditions, intensity_inputs,
                                           local_rates, timing_inputs)
from training.beat_models.data import _load_song
from training.beat_models.features import normalize_songs
from training.beat_models.settings import architecture_settings


def _probabilities(value, shape, stage):
    values = np.asarray(value, dtype=np.float32)
    if values.shape != shape or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Invalid {stage} probabilities: expected {shape}, got {values.shape}")
    if not np.allclose(values.sum(axis=-1), 1, rtol=0, atol=1e-5):
        raise ValueError(f"Unnormalized {stage} probabilities")
    return values


def _graph_signature(document):
    """Compare execution graphs across supported legacy-Keras serializations."""
    if document.get("class_name") != "Functional":
        raise ValueError("Checkpoint must contain a Functional stage model")
    graph = document["config"]
    layers = graph["layers"]
    names = {layer["name"]: index for index, layer in enumerate(layers)}
    if len(names) != len(layers):
        raise ValueError("Checkpoint layer names must be distinct")

    def connections(value):
        if isinstance(value, list):
            return [connections(item) for item in value]
        if isinstance(value, dict):
            return {key: connections(item) for key, item in value.items()}
        return names[value] if isinstance(value, str) and value in names else value

    records = []
    for layer in layers:
        settings = dict(layer["config"])
        settings.pop("name", None)
        # tf-keras 2.21 records this default; Keras 2.15 predates the field.
        if layer["class_name"] == "InputLayer" and settings.get("optional") is False:
            settings.pop("optional")
        records.append({"class_name": layer["class_name"], "config": settings,
                        "inbound_nodes": connections(layer["inbound_nodes"])})
    return {"name": graph["name"], "layers": records,
            "input_layers": connections(graph["input_layers"]),
            "output_layers": connections(graph["output_layers"])}


def _load_cpu_model(checkpoint, stage, manifest):
    """Rebuild only an exactly matching stage graph, then load its saved weights.

    GPU-trained tf-keras archives name modules unavailable in the recorded CPU
    Keras environment. Graph comparison makes weight-only restoration explicit
    and rejects unsupported graph changes rather than guessing at compatibility.
    """
    from training.beat_models import models

    tf = models.tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError("Beat-pair evaluation requires a CPU-only runtime") from error
    if Path(checkpoint).suffix != ".keras":
        raise ValueError("Beat-pair evaluation supports versioned .keras checkpoints")
    with zipfile.ZipFile(checkpoint) as archive:
        saved = json.loads(archive.read("config.json"))
    with tf.device("/CPU:0"):
        model = models.build_model(stage)
        if _graph_signature(saved) != _graph_signature(json.loads(model.to_json())):
            raise ValueError(f"Unsupported {stage} checkpoint graph; current architecture does not match")
        model.load_weights(checkpoint)
    parameters = manifest["training"].get("parameters")
    if parameters is not None and parameters != model.count_params():
        raise ValueError(f"{stage} checkpoint parameter count does not match its manifest")

    class CpuPredictor:
        def predict(self, inputs, verbose=0):
            with tf.device("/CPU:0"):
                return model(inputs, training=False).numpy()

    return CpuPredictor()


class BeatPair:
    """A compatible, explicitly trained pair; the note stage is never required.

    ``model_loader(checkpoint_path, stage, manifest)`` may supply predictors for
    focused contract tests. The default validates and restores the saved graph
    and weights using ``tensorflow.keras`` on the CPU.
    """

    def __init__(self, intensity_artifact, timing_artifact, *, model_loader=None):
        paths = {"intensity": intensity_artifact, "timing": timing_artifact}
        self.manifests, checkpoints = {}, {}
        for stage, path in paths.items():
            checkpoint, manifest = load_model_manifest(path, stage)
            if manifest["status"] != "trained" or manifest.get("training", {}).get("status") != "trained":
                raise ValueError(f"{stage} evaluation requires an explicitly trained checkpoint")
            architecture = manifest["training"].get("architecture")
            if architecture != json_value(architecture_settings(stage)):
                raise ValueError(f"Unsupported {stage} checkpoint architecture for the current configuration")
            self.manifests[stage], checkpoints[stage] = manifest, checkpoint
        intensity = self.manifests["intensity"]
        timing = self.manifests["timing"]
        if intensity["normalization_sha256"] != timing["normalization_sha256"]:
            raise ValueError("Intensity and timing checkpoints require identical normalization statistics")
        self.statistics = deepcopy(intensity["normalization"])
        self.normalization_sha256 = statistics_hash(self.statistics)
        self.metadata = {
            "kind": "intensity_timing_pair", "device": "CPU", "conditioning": "predicted_intensity",
            "decoder": "per_frame_count_argmax", "density_unit": REPRESENTATION["density_unit"],
            "representation": deepcopy(REPRESENTATION),
            "normalization": {"sha256": self.normalization_sha256, "fit_split": self.statistics["fit_split"],
                              "training_song_count": len(self.statistics["song_ids"])},
            "checkpoints": {
                stage: {"path": str(checkpoints[stage].resolve()),
                        "sha256": manifest["checkpoint_sha256"],
                        "manifest_path": str((checkpoints[stage].parent / "manifest.json").resolve()),
                        "manifest_sha256": file_hash(checkpoints[stage].parent / "manifest.json"),
                        "status": manifest["status"], "training_status": manifest["training"]["status"],
                        "architecture": manifest["training"]["architecture"],
                        "training_conditioning": manifest["training"].get("conditioning"),
                        "split_song_counts": {split: len(ids) for split, ids in manifest.get("splits", {}).items()},
                        "source_record_count": len(manifest.get("sources", [])),
                        "checkpoint_runtime": manifest.get("runtime", {})}
                for stage, manifest in self.manifests.items()},
            "overlap_evidence": {
                "available": "Checkpoint manifests record source byte hashes and song split IDs",
                "limitation": "Flat source records do not associate each audio hash with a split; absence of a byte match does not prove recording independence",
            },
            "loading": "current_graph_comparison_and_strict_weights_on_cpu" if model_loader is None else "injected_model_loader",
        }
        loader = _load_cpu_model if model_loader is None else model_loader
        self.predictors = {stage: loader(checkpoints[stage], stage, self.manifests[stage])
                           for stage in paths}
        self._intensity_cache = {}

    def prepare(self, reference):
        """Use the immutable shared ingestion/features and saved train statistics."""
        if reference.get("characteristic", "Standard") != "Standard":
            raise ValueError("Beat-pair preparation supports the trained Standard characteristic")
        info, beatmap, audio = (Path(reference[key]).resolve()
                                for key in ("info_path", "beatmap_path", "audio_path"))
        provenance = {"adapter": "evaluation_human_map", "metadata_verified": True,
                      "difficulty": reference.get("difficulty"),
                      "characteristic": reference.get("characteristic", "Standard")}
        song = _load_song(info, beatmap, audio, str(beatmap), provenance, {})
        return normalize_songs([song], self.statistics)[0]

    def _inference_song(self, song, bpm):
        """Validate audio inputs, then exclude all reference targets from inference."""
        metadata = dict(song.metadata)
        if (metadata.get("normalization_version") != self.statistics["version"]
                or metadata.get("normalization_sha256") != self.normalization_sha256):
            raise ValueError("Song normalization does not match the checkpoint pair")
        duration = metadata.get("duration", 0)
        bpm = metadata.get("bpm") if bpm is None else bpm
        if (isinstance(bpm, (bool, np.bool_)) or not isinstance(bpm, Real)
                or not np.isfinite(bpm) or bpm <= 0):
            raise ValueError("BPM must be a separate finite positive value")
        if (isinstance(duration, (bool, np.bool_)) or not isinstance(duration, Real)
                or not np.isfinite(duration) or duration <= 0):
            raise ValueError("Song duration must be finite and positive")
        metadata["bpm"] = float(bpm)
        arrays = {key: np.asarray(song.arrays[key]) for key in
                  ("features", "coarse_features", "frame_times", "bin_durations")}
        frames, bins = len(arrays["frame_times"]), len(arrays["bin_durations"])
        shapes = {"features": (frames, FEATURES), "coarse_features": (bins, FEATURES),
                  "frame_times": (frames,), "bin_durations": (bins,)}
        if not frames or not bins:
            raise ValueError("Song must contain audio frames and intensity bins")
        for key, shape in shapes.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Invalid inference array: {key}")
        if (not np.allclose(arrays["frame_times"], np.arange(frames) * FRAME_SECONDS, rtol=0, atol=1e-9)
                or arrays["frame_times"][-1] >= duration):
            raise ValueError("Inference frames must use the valid sample-derived timeline")
        # timing_inputs supplies training labels alongside windows; use zeros so
        # inference never reads reference timing, intensity, or note arrays.
        arrays["timing"] = np.zeros(frames, dtype=np.int32)
        return Song(song.song_id, metadata, arrays)

    def predict(self, song, requested_nps, bpm=None):
        """Predict every real frame once, without snapping or budget forcing."""
        if (isinstance(requested_nps, (bool, np.bool_)) or not isinstance(requested_nps, Real)
                or not np.isfinite(requested_nps) or requested_nps < 0):
            raise ValueError("Requested NPS must be finite and nonnegative")
        audio = self._inference_song(song, bpm)
        bins, frames = len(audio.arrays["bin_durations"]), len(audio.arrays["frame_times"])
        cached = self._intensity_cache.get(id(song))
        if cached is None or cached[0] is not song:
            shares = _probabilities(self.predictors["intensity"].predict(intensity_inputs(audio), verbose=0),
                                    (1, bins), "intensity")[0].copy()
            self._intensity_cache[id(song)] = (song, shares)
        else:
            shares = cached[1]
        conditions = frame_conditions(audio, shares, requested_nps)
        probabilities = np.zeros((frames, 3), dtype=np.float32)
        probabilities[:, 0] = 1
        if requested_nps > 0:
            for start in range(0, frames, CORE):
                inputs, _, _ = timing_inputs(audio, conditions, start)
                core = _probabilities(self.predictors["timing"].predict(inputs, verbose=0),
                                      (1, CORE, 3), "timing")[0]
                valid = min(CORE, frames - start)
                probabilities[start:start + valid] = core[:valid]
        counts = probabilities.argmax(axis=-1).astype(np.int32)
        events = np.flatnonzero(counts).astype(np.int64)
        return {"intensity_shares": shares.copy(),
                "local_nps": local_rates(shares, audio.arrays["bin_durations"], requested_nps,
                                         audio.metadata["duration"]),
                "frame_probabilities": probabilities, "counts": counts, "event_frames": events,
                "times_seconds": audio.arrays["frame_times"][events].copy(),
                "event_counts": counts[events].copy(), "bpm": audio.metadata["bpm"],
                "duration": audio.metadata["duration"], "requested_nps": float(requested_nps)}
