from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
import os
import re
import pathlib

model_dir = "/tmp/traffic_sign/"
model_dir = pathlib.Path(model_dir)
models_made = np.array([item.name for item in model_dir.glob('*')]).astype(np.float)
print(models_made)
mod_to_use = input("Which model version to use? ")
if round(float(mod_to_use),1) in models_made:
    latest_mod = mod_to_use

loaded = tf.saved_model.load(str(model_dir/str(latest_mod)))