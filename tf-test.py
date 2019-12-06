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
import tkinter as tk
from tkinter import filedialog

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Should match image dimensions in CNN initial layer
IMG_WIDTH = 32
IMG_HEIGHT = 32

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
  plt.show()


def plot_metrics(hist):
  plt.subplot(2,1,1)
  plt.plot(hist.history['accuracy'], label='accuracy')
  plt.plot(hist.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0.5, 1])
  plt.legend(loc='lower right')
  
  plt.subplot(2,1,2)
  plt.plot(hist.history['loss'], label='loss')
  plt.plot(hist.history['val_loss'], label = 'val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.ylim([0, 1])
  plt.legend(loc='lower right')
  
  plt.show()

root = tk.Tk()
train_data_dir = filedialog.askdirectory(title='Pick folder with Training Data', initialdir='.')
test_data_dir = filedialog.askdirectory(title='Pick folder with Testing Data', initialdir='.')
root.destroy()

train_data_dir = pathlib.Path(train_data_dir)
test_data_dir = pathlib.Path(test_data_dir)

CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*')])
print (CLASS_NAMES)

#data_dir = tf.keras.utils.get_file()
print(tf.__version__)
#path = r'/C:/Users/Splinter/Pictures/GTSRB/Final_Training/Images'
#data_dir = pathlib.Path("C:/Users/Splinter/Pictures/")
train_image_count = len(list(train_data_dir.glob('*/*.png')))
test_image_count = len(list(test_data_dir.glob('*/*.png')))

# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = image_generator.flow_from_directory(directory=str(train_data_dir),
                                                     batch_size= train_image_count,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES),
                                                     class_mode = 'sparse')

test_data_gen = image_generator.flow_from_directory(directory=str(test_data_dir),
                                                     batch_size= test_image_count,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES),
                                                     class_mode = 'sparse')                                                     

image_train, label_train= next(train_data_gen)

image_test, label_test= next(test_data_gen)

#show_batch(image_train, label_train)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
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

history = model.fit(image_train, label_train, epochs=35, 
                    validation_data=(image_test, label_test))

plot_metrics(history)
test_loss, test_acc = model.evaluate(image_test,  label_test, verbose=2)
print(test_acc)

model_dir = "/tmp/"
model_dir = pathlib.Path(model_dir)
models_made = np.array([item.name for item in model_dir.glob('*')])
print(models_made)
mod_sav = input("Save model? (Y/N) ")
if mod_sav is 'Y':
  model_name = input("Model name: ")
  model_dir = model_dir/model_name
  models_made = np.array([item.name for item in model_dir.glob('*')]).astype(np.float)
  if models_made.size == 0:
    latest_mod = 0
  else:
    latest_mod = np.amax(models_made)
  tf.saved_model.save(model, str(model_dir/str(round(latest_mod + 0.1, 1))))
