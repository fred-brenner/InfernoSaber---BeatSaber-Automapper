"""Explicit checkpoint adapters; no model is loaded at module import."""

from pathlib import Path

from training.beat_models.artifacts import file_hash
from .encoder import ENCODER_REPRESENTATION, FixedEncoder, decode_encoder_audio
from .settings import preparation_settings


class EncoderFactory:
    """Reuse one retained encoder and one recording's audio/latent cache at a time."""

    def __init__(self, checkpoint, settings=None):
        from tensorflow.keras.models import load_model
        self.path = Path(checkpoint)
        if not self.path.is_file():
            raise ValueError("An existing explicit encoder checkpoint is required")
        self.model = load_model(str(self.path), compile=False)
        if tuple(self.model.input_shape[1:]) != (24, 20, 1) or tuple(self.model.output_shape[1:]) != (16,):
            raise ValueError("The retained encoder must accept [24,20,1] and return [16]")
        self.fingerprint = file_hash(self.path)
        self.settings = preparation_settings() if settings is None else settings
        self.identity = {"status": "retained", "checkpoint_fingerprint": self.fingerprint,
                         "representation": ENCODER_REPRESENTATION,
                         "grid_seconds": self.settings["encoder_grid_seconds"]}
        self._last = None

    def __call__(self, song):
        base = song.base if hasattr(song, "base") else song
        records = [r for r in base.metadata.get("sources", []) if r.get("role") == "audio"]
        if len(records) != 1:
            raise ValueError("A single original audio source is required for retained encoding")
        record = records[0]
        duration = float(base.metadata["duration"])
        key = (record["path"], record["sha256"], duration)
        if self._last is None or self._last[0] != key:
            if file_hash(record["path"]) != record["sha256"]:
                raise ValueError("Original audio changed since beat dataset preparation")
            samples = decode_encoder_audio(record["path"])
            provider = FixedEncoder(samples, lambda x: self.model(x, training=False).numpy(), self.fingerprint,
                                    grid_seconds=self.settings["encoder_grid_seconds"],
                                    batch_size=self.settings["encoder_batch_size"], duration=duration)
            self._last = key, provider
        return self._last[1]


def trained_predictors(dataset, trained_beat_run, artifacts):
    """Load intensity/timing plus six explicitly selected downstream artifacts."""
    from evaluation.beat_pair import BeatPair
    from .runner import load_stage_artifact
    from .settings import STAGES
    if set(artifacts) != set(STAGES) or not all(artifacts.values()):
        raise ValueError("Select explicit checkpoints for all six downstream stages")
    run = Path(trained_beat_run)
    pair = BeatPair(run / "intensity", run / "timing")
    if dataset.manifest.get("normalization_sha256") != pair.normalization_sha256:
        raise ValueError("Prepared audio normalization differs from the trained intensity/timing run")
    predictors = dict(pair.predictors)
    for predictor in predictors.values():
        predictor.status = "trained"
    manifests = {"intensity": pair.manifests["intensity"], "timing": pair.manifests["timing"]}
    for stage in STAGES:
        model, manifest = load_stage_artifact(artifacts[stage], stage, dataset)
        if manifest["status"] != "trained":
            raise ValueError(f"Concept inference requires a trained {stage} artifact; use smoke for untrained models")
        predictors[stage] = lambda inputs, model=model: model(inputs, training=False)
        predictors[stage].status = "trained"
        manifests[stage] = manifest
    return predictors, manifests
