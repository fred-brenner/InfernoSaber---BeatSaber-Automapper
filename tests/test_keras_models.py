"""Check the model-loading path shared by CPU and legacy-Keras GPU execution."""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tcn import TCN

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.helpers import load_model


def test_load_legacy_tcn_model(tmp_path):
    with tf.device("/CPU:0"):
        inputs = tf.keras.Input(shape=(8, 1))
        outputs = TCN(nb_filters=2, kernel_size=2, dilations=(1,),
                      use_weight_norm=False)(inputs)
        original = tf.keras.Model(inputs, outputs)
        model_path = tmp_path / "legacy_tcn.h5"
        original.save(model_path, include_optimizer=False)

        restored = load_model(model_path, custom_objects={"TCN": TCN}, compile=False)
        sample = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8, 1))

        np.testing.assert_allclose(
            restored(sample, training=False).numpy(),
            original(sample, training=False).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
