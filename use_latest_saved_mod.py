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

# Should match image dimensions in CNN initial layer
IMG_WIDTH = 32
IMG_HEIGHT = 32

# Should be a common divisor of both sets to 
BATCH_SIZE = 70

model_dir = "/tmp/traffic_sign/"
model_dir = pathlib.Path(model_dir)
models_made = np.array([item.name for item in model_dir.glob('*')]).astype(np.float)
print(models_made)
mod_to_use = input("Which model version to use? ")
if round(float(mod_to_use),1) in models_made:
    latest_mod = mod_to_use

loaded = tf.keras.models.load_model(str(model_dir/str(latest_mod)))

data_dir = r'.'
test_data_dir = data_dir + '\TestingDataNoise'
test_data_dir = pathlib.Path(test_data_dir)

CLASS_NAMES = np.array([item.name for item in test_data_dir.glob('*')])
print (CLASS_NAMES)

test_image_count = len(list(test_data_dir.glob('*/*.jpg')))
TEST_STEPS_PER_EPOCH = np.ceil(test_image_count/BATCH_SIZE)

# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data_gen = image_generator.flow_from_directory(directory=str(test_data_dir),
                                                     batch_size= BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES),
                                                     class_mode = 'sparse')                                                     

image_test, label_test= next(test_data_gen)

test_loss, test_acc = loaded.evaluate(image_test,  label_test, verbose=2)

print(test_acc)
