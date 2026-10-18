"""Serializable downstream v2 networks and masked stage objectives.

The retained intensity, timing, and audio encoder are separate unchanged models.
Importing this module defines builders and losses; it never starts training.
"""

import tensorflow as tf

from .settings import STAGES, architecture_settings

keras = tf.keras


@keras.utils.register_keras_serializable(package="InfernoSaberV2")
class MaskValues(keras.layers.Layer):
    """Sanitize padding before embeddings, convolutions, and recurrent inputs."""

    def call(self, inputs):
        values, mask = inputs
        while mask.shape.rank < values.shape.rank:
            mask = mask[..., None]
        return tf.where(tf.cast(mask, tf.bool), values, tf.zeros_like(values))


@keras.utils.register_keras_serializable(package="InfernoSaberV2")
class Field(keras.layers.Layer):
    """Select one categorical/numeric field without an unserializable lambda."""

    def __init__(self, index, integer=False, **kwargs):
        super().__init__(**kwargs)
        self.index, self.integer = index, integer

    def call(self, values):
        selected = tf.gather(values, self.index, axis=-1) if isinstance(self.index, (list, tuple)) else values[..., self.index]
        return tf.cast(selected, tf.int32) if self.integer else selected

    def get_config(self):
        return {**super().get_config(), "index": self.index, "integer": self.integer}


@keras.utils.register_keras_serializable(package="InfernoSaberV2")
class FloatValues(keras.layers.Layer):
    def call(self, values):
        return tf.cast(values, tf.float32)


@keras.utils.register_keras_serializable(package="InfernoSaberV2")
class StripMask(keras.layers.Layer):
    """Keep independent current-query and preceding-history axes independent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, values):
        return tf.identity(values)

    def compute_mask(self, inputs, mask=None):
        return None


@keras.utils.register_keras_serializable(package="InfernoSaberV2")
class EndpointIdentity(keras.layers.Layer):
    """Accept endpoint identity for the contract without learning numeric note IDs."""

    def call(self, inputs):
        features, endpoint_ids = inputs
        return tf.identity(features)


def _embedding(values, classes, width, name, history=False, length=None):
    values = keras.layers.Embedding(classes, width, name=f"{name}_embedding")(values)
    if history:
        return keras.layers.Reshape((length, -1), name=f"{name}_sequence")(values)
    return keras.layers.Flatten(name=f"{name}_flatten")(values)


def _common(stage, settings, history_width, history_dtype="int32"):
    length = settings["history_length"]
    return {"audio": keras.Input((16,), name="audio"),
            "history": keras.Input((length, history_width), dtype=history_dtype, name="history"),
            "history_age": keras.Input((length, 1), name="history_age"),
            "history_mask": keras.Input((length,), dtype="bool", name="history_mask"),
            "conditions": keras.Input((6,), name="conditions")}


def _history(stage, inputs, settings, vocabularies):
    history = MaskValues(name="sanitize_history")([inputs["history"], inputs["history_mask"]])
    length, width = settings["history_length"], settings["embedding_dim"]
    if stage in ("positions", "arrows", "multiplier"):
        classes = {"positions": 15, "arrows": 12}.get(stage)
        classes = len(vocabularies["multiplier"]) + 2 if classes is None else classes
        branches = [_embedding(history, classes, width, "history", True, length)]
    elif stage == "hazards":
        wall = Field(0, integer=True, name="history_wall")(history)
        branches = [_embedding(wall, len(vocabularies["walls"]) + 2, width, "wall_history", True, length)]
        # Bomb history is multi-label occupancy, rather than one categorical ID.
        branches.append(Field(list(range(1, 13)), name="history_bombs")(history))
    else:
        branches = []
        for index, classes, name in ((0, 2, "attachment"), (3, 3, "anchor")):
            values = Field(index, integer=True, name=f"history_{name}")(history)
            branches.append(_embedding(values, classes, width, f"{name}_history", True, length))
        branches.append(Field([1, 2], name="history_controls")(history))
    age = MaskValues(name="sanitize_history_age")([inputs["history_age"], inputs["history_mask"]])
    branches.append(age)
    joined = keras.layers.Concatenate(name="history_features")(branches)
    joined = keras.layers.Dropout(settings["history_dropout"], name="history_dropout")(joined)
    return keras.layers.GRU(settings["gru_units"], name="history_gru")(joined, mask=inputs["history_mask"])


def _shared(stage, inputs, settings, vocabularies, context=()):
    audio = keras.layers.Dense(settings["audio_dense_units"], activation=settings["activation"],
                               name="audio_features")(inputs["audio"])
    history = _history(stage, inputs, settings, vocabularies)
    joined = keras.layers.Concatenate(name="current_and_history")([audio, history, inputs["conditions"], *context])
    return keras.layers.Dense(settings["shared_dense_units"], activation=settings["activation"],
                              name="shared_features")(joined)


def _hands(values, classes):
    return {hand: keras.layers.Dense(classes, activation="softmax", name=hand)(values) for hand in ("red", "blue")}


def build_positions_model(vocabularies=None):
    settings = architecture_settings("positions")
    inputs = _common("positions", settings, 2)
    joined = _shared("positions", inputs, settings, vocabularies or {})
    return keras.Model(inputs, _hands(joined, 13), name="positions")


def build_arrows_model(vocabularies=None):
    settings = architecture_settings("arrows")
    inputs = _common("arrows", settings, 2)
    inputs.update(current_positions=keras.Input((2,), dtype="int32", name="current_positions"),
                  occupied=keras.Input((2,), dtype="bool", name="occupied"),
                  neighbors=keras.Input((2, 2, 2), name="neighbors"))
    width = settings["embedding_dim"]
    context = [_embedding(inputs["current_positions"], 13, width, "current_position"),
               FloatValues(name="occupied_values")(inputs["occupied"])]
    positions = Field(0, integer=True, name="neighbor_positions")(inputs["neighbors"])
    gaps = Field(1, name="neighbor_gaps")(inputs["neighbors"])
    context.extend([_embedding(positions, 13, width, "neighbor_position"),
                    keras.layers.Flatten(name="neighbor_gap_flatten")(gaps)])
    joined = _shared("arrows", inputs, settings, vocabularies or {}, context)
    return keras.Model(inputs, _hands(joined, 9), name="arrows")


def build_multiplier_model(vocabularies):
    settings = architecture_settings("multiplier")
    inputs = _common("multiplier", settings, 2)
    inputs.update(current_notes=keras.Input((2, 2), dtype="int32", name="current_notes"),
                  neighbors=keras.Input((2, 2, 3), name="neighbors"))
    context = []
    for family, name in (("current_notes", "current"), ("neighbors", "neighbor")):
        for index, classes, attribute in ((0, 13, "position"), (1, 10, "direction")):
            values = Field(index, integer=True, name=f"{name}_{attribute}")(inputs[family])
            context.append(_embedding(values, classes, settings["embedding_dim"], f"{name}_{attribute}"))
    gaps = Field(2, name="neighbor_gaps")(inputs["neighbors"])
    context.append(keras.layers.Flatten(name="neighbor_gap_flatten")(gaps))
    joined = _shared("multiplier", inputs, settings, vocabularies, context)
    return keras.Model(inputs, _hands(joined, len(vocabularies["multiplier"])), name="multiplier")


def build_hazards_model(vocabularies):
    settings = architecture_settings("hazards")
    inputs = _common("hazards", settings, 13, "float32")
    inputs.update(occupancy=keras.Input((2, 2, 12), name="occupancy"),
                  interval=keras.Input((2,), name="interval"))
    occupancy = keras.layers.Flatten(name="occupancy_flatten")(inputs["occupancy"])
    occupancy = keras.layers.Dense(settings["audio_dense_units"], activation=settings["activation"],
                                   name="occupancy_features")(occupancy)
    joined = _shared("hazards", inputs, settings, vocabularies, [occupancy, inputs["interval"]])
    outputs = {"wall": keras.layers.Dense(len(vocabularies["walls"]), activation="softmax", name="wall")(joined),
               "bombs": keras.layers.Dense(12, activation="sigmoid", name="bombs")(joined)}
    return keras.Model(inputs, outputs, name="hazards")


def build_arcs_model(vocabularies=None):
    settings = architecture_settings("arcs")
    inputs = _common("arcs", settings, 4, "float32")
    inputs.update(endpoints=keras.Input((2,), dtype="int32", name="endpoints"),
                  endpoint_features=keras.Input((2, 3), name="endpoint_features"),
                  gap=keras.Input((1,), name="gap"))
    context = []
    for index, classes, name in ((0, 13, "position"), (1, 10, "direction"), (2, 2, "hand")):
        values = Field(index, integer=True, name=f"endpoint_{name}")(inputs["endpoint_features"])
        context.append(_embedding(values, classes, settings["embedding_dim"], f"endpoint_{name}"))
    endpoints = keras.layers.Concatenate(name="endpoint_context")(context)
    endpoints = EndpointIdentity(name="endpoint_identity")([endpoints, inputs["endpoints"]])
    joined = _shared("arcs", inputs, settings, vocabularies or {}, [endpoints, inputs["gap"]])
    shape = keras.layers.Dense(2, activation="sigmoid", name="bounded_controls")(joined)
    low, high = settings["shape_bounds"]
    outputs = {"attach": keras.layers.Dense(2, activation="softmax", name="attach")(joined),
               "shape": keras.layers.Rescaling(high - low, offset=low, name="shape")(shape),
               "anchor": keras.layers.Dense(3, activation="softmax", name="anchor")(joined)}
    return keras.Model(inputs, outputs, name="arcs")


def build_lighting_model(vocabularies):
    settings = architecture_settings("lighting")
    length, classes = settings["block_length"], len(vocabularies["lighting"])
    inputs = {"audio": keras.Input((length, 16, 1), name="audio"),
              "gaps": keras.Input((length, 1), name="gaps"),
              "history": keras.Input((length, classes), name="history"),
              "mask": keras.Input((length,), dtype="bool", name="mask"),
              "history_mask": keras.Input((length,), dtype="bool", name="history_mask")}
    audio = MaskValues(name="sanitize_audio")([inputs["audio"], inputs["mask"]])
    audio = keras.layers.Conv2D(settings["conv_filters"], 3, activation="relu", name="audio_conv")(audio)
    audio = keras.layers.Flatten(name="audio_flatten")(audio)
    gaps = MaskValues(name="sanitize_gaps")([inputs["gaps"], inputs["mask"]])
    time = keras.layers.LSTM(settings["time_lstm_units"], return_sequences=True, name="time_lstm")(
        gaps, mask=inputs["mask"])
    history = MaskValues(name="sanitize_history")([inputs["history"], inputs["history_mask"]])
    history = keras.layers.LSTM(settings["history_lstm_units"], return_sequences=True, name="history_lstm")(
        history, mask=inputs["history_mask"])
    # A short current block still needs the complete preceding history block.
    # Inherited AND masks here would erase its independent, right-aligned tail.
    time = StripMask(name="independent_time_axis")(time)
    history = StripMask(name="independent_history_axis")(history)
    combined = keras.layers.Concatenate(name="time_and_history")([time, history])
    combined = keras.layers.LSTM(settings["combined_lstm_units"], name="combined_lstm")(combined)
    joined = keras.layers.Concatenate(name="audio_and_sequence")([audio, combined])
    joined = keras.layers.Dense(settings["dense_units"], activation="relu", name="lighting_features")(joined)
    joined = keras.layers.Dropout(settings["dropout"], name="lighting_dropout")(joined)
    output = keras.layers.Dense(length * classes, activation="sigmoid", name="event_scores")(joined)
    output = keras.layers.Reshape((length, classes), name="events")(output)
    return keras.Model(inputs, {"events": output}, name="lighting")


BUILDERS = {"positions": build_positions_model, "arrows": build_arrows_model,
            "multiplier": build_multiplier_model, "hazards": build_hazards_model,
            "arcs": build_arcs_model, "lighting": build_lighting_model}


def build_model(stage, vocabularies=None):
    if stage not in BUILDERS:
        raise ValueError(f"Unknown v2 model stage: {stage}")
    vocabularies = vocabularies or {}
    required = {"multiplier": "multiplier", "hazards": "walls", "lighting": "lighting"}.get(stage)
    if required and not vocabularies.get(required):
        raise ValueError(f"{stage} requires an explicit frozen {required} vocabulary")
    return BUILDERS[stage](vocabularies)


def _weighted_terms(values, weights):
    weights = tf.cast(weights, values.dtype)
    while weights.shape.rank < values.shape.rank:
        weights = weights[..., None]
    weights = tf.broadcast_to(weights, tf.shape(values))
    values = tf.where(weights > 0, values, tf.zeros_like(values))
    return tf.reduce_sum(values * weights), tf.reduce_sum(weights)


def loss_components(stage, targets, predicted, weights, balance=None):
    """Return per-head sums/counts so validation is invariant to batch partition."""
    if stage not in STAGES or set(targets) != set(predicted) or set(targets) != set(weights):
        raise ValueError("Stage loss requires matching target, prediction, and weight heads")
    components = {}
    for head, output in predicted.items():
        target = tf.cast(targets[head], output.dtype)
        weight = tf.convert_to_tensor(weights[head])
        if (stage, head) in (("hazards", "bombs"), ("lighting", "events")):
            # Multi-label targets allow simultaneous light instructions/bombs.
            values = keras.backend.binary_crossentropy(target, output)
        elif stage == "arcs" and head == "shape":
            difference = target - output
            absolute = tf.abs(difference)
            values = tf.where(absolute <= 1., .5 * tf.square(difference), absolute - .5)
        else:
            target = tf.cast(target, tf.int32)
            target = tf.where(weight > 0, target, tf.zeros_like(target))
            values = keras.losses.sparse_categorical_crossentropy(target, output)
        if balance and head in balance:
            factors = balance[head]
            factor = tf.where(target > 0, tf.cast(factors["positive"], values.dtype),
                              tf.cast(factors["negative"], values.dtype))
            values = values * factor
        components[head] = _weighted_terms(values, weight)
    return components


def combine_loss_components(stage, components):
    settings = architecture_settings(stage)
    if set(components) != set(settings["loss_weights"]):
        raise ValueError("Configured loss weights must identify every stage output head")
    terms, active = [], []
    for head, pair in components.items():
        numerator, denominator = (tf.cast(value, tf.float32) for value in pair)
        weight = tf.cast(settings["loss_weights"][head], tf.float32)
        terms.append(weight * tf.math.divide_no_nan(numerator, denominator))
        active.append(weight * tf.cast(denominator > 0, tf.float32))
    return tf.math.divide_no_nan(tf.add_n(terms), tf.add_n(active))


def stage_loss(stage, targets, predicted, weights):
    return combine_loss_components(stage, loss_components(stage, targets, predicted, weights))
