from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
import os
import re
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Values are random right now
IMG_WIDTH = 32
IMG_HEIGHT = 32
BATCH_SIZE = 32

# TODO: 
# Change to a regular expression script to take label from the file name instead of the directory name.
def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

# TODO:
# Change to decode png and specify a img size that works
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')

#data_dir = tf.keras.utils.get_file(origin='http://cvrr.ucsd.edu/LISA/Datasets/signDatabasePublicFramesOnly.zip',
#                                         fname='LISA_set', extract=True, )
data_dir = 'TrainingSet'

data_dir = pathlib.Path(data_dir)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
#CLASS_NAMES = np.array(re.search(''))
#CLASS_NAMES = np.array(['stop','keepRight','addedLane'])
print (CLASS_NAMES)

#data_dir = tf.keras.utils.get_file()
print(tf.__version__)
#path = r'/C:/Users/Splinter/Pictures/GTSRB/Final_Training/Images'
#data_dir = pathlib.Path("C:/Users/Splinter/Pictures/")

# TODO:
# Continue load and preprocess image example to load dataset as a Tensorflow dataset (https://www.tensorflow.org/tutorials/load_data/images).
# Once data set is ready and loaded, run on CNN in tf-cnn-test.py file
# Go from there

list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))

#print (list_ds)
for f in list_ds.take(5):
  print(f.numpy())

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path)

for image, label in labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(train_ds))

show_batch(image_batch.numpy(), label_batch.numpy())