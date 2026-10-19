"""Retained v1 lighting inference, independent of mapper selection and training."""

from pathlib import Path

import numpy as np

from tools.config import config
from training.beat_models.artifacts import file_hash


BLOCK_LENGTH = 16
_REPOSITORY = Path(__file__).resolve().parents[1]


def lighting_paths():
    """Resolve the explicit model and categorical vocabulary against the repo."""
    selected = (config.v1_lighting_model_path, config.v1_lighting_vocabulary_path)
    return tuple((Path(value) if Path(value).is_absolute() else _REPOSITORY / value).resolve()
                 for value in selected)


def validate_assets():
    """Check local lighting assets and decode integer labels without TensorFlow."""
    import joblib

    model_path, vocabulary_path = lighting_paths()
    for path in (model_path, vocabulary_path):
        if not path.is_file():
            raise ValueError(f"Missing v1 lighting asset: {path}")
    try:
        vocabulary = joblib.load(vocabulary_path)
        if len(vocabulary.categories_) != 1:
            raise ValueError("Expected one categorical event vocabulary")
        labels = [str(label).split(";") for label in vocabulary.categories_[0]]
        if not labels or any(len(label) != 2 for label in labels):
            raise ValueError("Expected type;value event labels")
        categories = np.asarray([[int(part) for part in label] for label in labels], dtype=np.int64)
        if np.any(categories < 0) or np.any(categories[:, 0] == 100):
            raise ValueError("Expected nonnegative lighting commands without tempo changes")
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid v1 lighting vocabulary: {vocabulary_path}: {error}") from error
    return {"model_path": model_path, "vocabulary_path": vocabulary_path,
            "model_sha256": file_hash(model_path), "vocabulary_sha256": file_hash(vocabulary_path),
            "categories": categories}


def apply_random_mapper(y_class, rd_distribution, rd_counter):
    """Preserve the v1 block weighting shared with the historical generator."""
    c_window = 60
    c_val = config.random_note_map_factor
    if c_val == 0:
        return y_class, rd_distribution, rd_counter
    y_class_sc = y_class / np.max(y_class)
    if rd_counter <= 0:
        rd_distribution = np.random.rand(*y_class.shape) * c_val
        center = int(np.random.rand(1)[0] * y_class.shape[2])
        if center < c_window:
            center = c_window
        elif center > y_class.shape[2] - c_window - 1:
            center = y_class.shape[2] - c_window - 1
        rd_distribution[:, :, center - c_window:center + c_window] += c_val
        rd_distribution += 1
        rd_counter = config.random_note_map_change
    return y_class_sc * rd_distribution, rd_distribution, rd_counter - 1


class V1Lighting:
    """Predict legacy blocks using cached weights and exact integer categories."""

    def __init__(self, model, categories, predictor=None, metadata=None):
        self.model = model
        self.categories = np.asarray(categories, dtype=np.int64)
        classes = len(self.categories)
        inputs = [(None, BLOCK_LENGTH, 16, 1), (None, BLOCK_LENGTH, 1),
                  (None, BLOCK_LENGTH, classes)]
        if (self.categories.shape != (classes, 2) or not classes
                or [tuple(shape) for shape in model.input_shape] != inputs
                or tuple(model.output_shape) != (None, BLOCK_LENGTH, classes)):
            raise ValueError("V1 lighting model dimensions do not match the retained event vocabulary")
        self.predict = predictor if predictor is not None else lambda inputs: model(inputs, training=False)
        self.metadata = dict(metadata or {})

    def __call__(self, times, encoder):
        times = np.asarray(times, np.float64)
        if (times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0)
                or np.any(np.diff(times) <= 0)):
            raise ValueError("V1 lighting requires ordered finite gameplay timestamps")
        blocks = max(len(times) // BLOCK_LENGTH - 1, 0)
        count = blocks * BLOCK_LENGTH
        report = {**self.metadata, "backend": "v1", "queries": len(times), "blocks": blocks,
                  "block_length": BLOCK_LENGTH, "tail_queries": len(times) % BLOCK_LENGTH,
                  "truncated_queries": len(times) - count, "events": count}
        if not blocks:
            return [], report
        latents = np.asarray(encoder(times[:count]), np.float32)
        if latents.shape != (count, 16) or not np.isfinite(latents).all():
            raise ValueError("V1 lighting requires finite [N,16] retained audio latents")
        audio = latents.reshape(blocks, BLOCK_LENGTH, 16, 1)
        gaps = np.concatenate(([1.], np.diff(times)))[:count].reshape(blocks, BLOCK_LENGTH, 1)
        history = np.zeros((1, BLOCK_LENGTH, len(self.categories)), np.float32)
        distribution, counter, events = None, 0, []
        for index in range(blocks):
            scores = np.asarray(self.predict([audio[index:index + 1], gaps[index:index + 1], history]))
            if (scores.shape != history.shape or not np.isfinite(scores).all()
                    or np.any(scores < 0) or np.any(scores > 1) or not scores.max() > 0):
                raise ValueError("Invalid v1 lighting model scores")
            scores, distribution, counter = apply_random_mapper(scores, distribution, counter)
            winners = scores.argmax(axis=2)[0]
            history.fill(0)
            history[0, np.arange(BLOCK_LENGTH), winners] = 1
            for time, (event_type, value) in zip(times[index * BLOCK_LENGTH:(index + 1) * BLOCK_LENGTH],
                                                self.categories[winners]):
                events.append({"time": float(time), "type": int(event_type), "value": int(value)})
        return events, report


def load_lighting(assets, predictor_factory):
    """Load once inside the runtime's CPU device context; never compile training."""
    from tensorflow.keras.models import load_model

    model = load_model(str(assets["model_path"]), compile=False)
    metadata = {key: str(value) if isinstance(value, Path) else value
                for key, value in assets.items() if key != "categories"}
    return V1Lighting(model, assets["categories"], predictor_factory(model), metadata)
