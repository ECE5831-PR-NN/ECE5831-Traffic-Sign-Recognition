from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
data_dir = tf.keras.utils.get_file(origin='http://cvrr.ucsd.edu/LISA/Datasets/signDatabasePublicFramesOnly.zip',
                                         fname='LISA_set', extract=True, )
import pathlib
data_dir = pathlib.Path(data_dir)

#data_dir = tf.keras.utils.get_file()
#print(tf.__version__)
#path = r'/C:/Users/Splinter/Pictures/GTSRB/Final_Training/Images'
#data_dir = pathlib.Path("C:/Users/Splinter/Pictures/")

list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))

#print (list_ds)
for f in list_ds.take(5):
  print(f.numpy())