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

# Should match image dimensions in CNN initial layer
IMG_WIDTH = 32
IMG_HEIGHT = 32

# Should be a common divisor of both sets to 
BATCH_SIZE = 70

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
#data_dir = r'C:\Users\Splinter\.keras\datasets'
data_dir = '.'
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

# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = image_generator.flow_from_directory(directory=str(train_data_dir),
                                                     batch_size= 4 * BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES),
                                                     class_mode = 'sparse')

test_data_gen = image_generator.flow_from_directory(directory=str(test_data_dir),
                                                     batch_size= BATCH_SIZE,
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
model.add(layers.Dense(14, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(image_train, label_train, epochs=40, 
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
model_dir = "/tmp/traffic_sign/"
model_dir = pathlib.Path(model_dir)
models_made = np.array([item.name for item in model_dir.glob('*')]).astype(np.float)
print(models_made)
latest_mod = np.amax(models_made)
mod_sav = input("Save model? (Y/N) ")
if mod_sav is 'Y':
  tf.saved_model.save(model, str(model_dir/str(round(latest_mod + 0.1, 1))))
