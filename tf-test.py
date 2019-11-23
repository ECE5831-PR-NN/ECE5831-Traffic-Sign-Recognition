from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)
#path = r'/C:/Users/Splinter/Pictures/GTSRB/Final_Training/Images'

list_ds = tf.data.Dataset.list_files('~/Pictures/GTSRB/Final_Training/Images/*/*.ppm')

#print (list_ds)
for f in list_ds.take(5):
  print(f.values)