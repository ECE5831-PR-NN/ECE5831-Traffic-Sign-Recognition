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

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Values are random right now
IMG_WIDTH = 32
IMG_HEIGHT = 32
BATCH_SIZE = 54
'''
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
'''
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
  plt.show()

#data_dir = tf.keras.utils.get_file(origin='http://cvrr.ucsd.edu/LISA/Datasets/signDatabasePublicFramesOnly.zip',
#                                         fname='LISA_set', extract=True, )
data_dir = r'C:\Users\Splinter\.keras\datasets'
train_data_dir = data_dir + '\TrainingData'
test_data_dir = data_dir + '\TestingData'

train_data_dir = pathlib.Path(train_data_dir)
test_data_dir = pathlib.Path(test_data_dir)

CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*')])
print (CLASS_NAMES)

#data_dir = tf.keras.utils.get_file()
print(tf.__version__)
#path = r'/C:/Users/Splinter/Pictures/GTSRB/Final_Training/Images'
#data_dir = pathlib.Path("C:/Users/Splinter/Pictures/")
train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
TRAIN_STEPS_PER_EPOCH = np.ceil(train_image_count/BATCH_SIZE)
test_image_count = len(list(test_data_dir.glob('*/*.jpg')))
TEST_STEPS_PER_EPOCH = np.ceil(test_image_count/BATCH_SIZE)

# TODO:
# Continue load and preprocess image example to load dataset as a Tensorflow dataset (https://www.tensorflow.org/tutorials/load_data/images).
# Once data set is ready and loaded, run on CNN in tf-cnn-test.py file
# Go from there
'''
train_list_ds = tf.data.Dataset.list_files(str(train_data_dir/'*'))
test_list_ds = tf.data.Dataset.list_files(str(test_data_dir/'*'))

#print (list_ds)
for f in train_list_ds.take(5):
  print(f.numpy())

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_labeled_ds = train_list_ds.map(process_path)
test_labeled_ds = test_list_ds.map(process_path)

# for image, label in labeled_ds.take(1):
#   print("Image shape: ", image.numpy().shape)
#   print("Label: ", label.numpy())

train_ds = prepare_for_training(train_labeled_ds)

image_train, label_train = next(iter(train_ds))
image_test, label_test = next(iter(test_labeled_ds))

show_batch(image_train.numpy(), label_train.numpy())
'''
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = image_generator.flow_from_directory(directory=str(train_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES),
                                                     class_mode = 'sparse')

test_data_gen = image_generator.flow_from_directory(directory=str(test_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES),
                                                     class_mode = 'sparse')                                                     

image_train, label_train= next(train_data_gen)

image_test, label_test= next(test_data_gen)

#show_batch(image_train, label_train)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(image_train, label_train, epochs=50, 
                    validation_data=(image_test, label_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(image_test,  label_test, verbose=2)

print(test_acc)