"""Training script for the beat intensity prediction model."""

from datetime import datetime

import numpy as np
from keras.optimizers import Adam

from helpers import load_keras_model
from preprocessing.intensity_pre import load_intensity_dataset
from tensorflow_models import create_keras_model
from tools.config import get_config, paths

config = get_config()

# Configuration
learning_rate = config.intensity_learning_rate
n_epochs = config.intensity_n_epochs
batch_size = config.intensity_batch_size
test_samples_cfg = config.intensity_test_samples
np.random.seed(config.random_seed)

# Load dataset
song_windows, intensity_targets = load_intensity_dataset()

# Shuffle samples in unison
permutation = np.random.permutation(song_windows.shape[0])
song_windows = song_windows[permutation]
intensity_targets = intensity_targets[permutation]

# Split into train/validation/test sets
test_samples = min(test_samples_cfg, max(1, song_windows.shape[0] // 10))
if test_samples >= song_windows.shape[0]:
    test_samples = song_windows.shape[0] - 1

x_test = song_windows[:test_samples]
y_test = intensity_targets[:test_samples]

train_inputs = song_windows[test_samples:]
train_targets = intensity_targets[test_samples:]

if train_inputs.shape[0] < 2:
    raise RuntimeError("Not enough samples available for training after test split.")

split_idx = int(train_inputs.shape[0] * 0.85)
if split_idx <= 0:
    split_idx = 1
if split_idx >= train_inputs.shape[0]:
    split_idx = train_inputs.shape[0] - 1

x_train = train_inputs[:split_idx]
y_train = train_targets[:split_idx]
x_val = train_inputs[split_idx:]
y_val = train_targets[split_idx:]

# Build or load model
date_time = datetime.now()
timestamp = f"{date_time.month}_{date_time.day}__{date_time.hour}_{date_time.minute}"
save_model_name = f"tf_model_intensity_{timestamp}.h5"

intensity_model, save_model_name = load_keras_model(save_model_name)
if intensity_model is None:
    input_shape = [x_train.shape[1:]]
    intensity_model = create_keras_model('intensity_cnn', input_shape, 1)
    adam = Adam(learning_rate=learning_rate, weight_decay=learning_rate / max(1, n_epochs))
    intensity_model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])

intensity_model.summary()

# Train model
intensity_model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=n_epochs,
    batch_size=batch_size,
    shuffle=True,
    verbose=1,
)

# Evaluate model
print("\nEvaluating test data...")
eval_results = intensity_model.evaluate(x_test, y_test, verbose=1)
print(f"Test loss: {eval_results[0]:.4f}, test MAE: {eval_results[1]:.4f}")

# Save model
model_path = paths.model_path + save_model_name
print(f"Saving model at: {model_path}")
intensity_model.save(model_path)

print("\nFinished Training")
