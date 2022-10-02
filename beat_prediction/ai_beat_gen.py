import numpy as np
from datetime import datetime
from tensorflow import keras
from keras.optimizers import adam_v2
from tabulate import tabulate

from training.helpers import *
from training.tensorflow_models import *
from preprocessing.bs_mapper_pre import load_ml_data, lstm_shift
from tools.config import config, paths


# Setup configuration
#####################
# Check Cuda compatible GPU
if not test_gpu_tf():
    exit()

# Load pretrained model
encoder_path = paths.model_path + config.enc_version
encoder_model = keras.models.load_model(encoder_path)

# gather input

