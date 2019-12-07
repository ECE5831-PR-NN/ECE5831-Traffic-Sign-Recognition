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

# Should match image dimensions in CNN initial layer
IMG_WIDTH = 32
IMG_HEIGHT = 32

CLASS_NAMES = ['AddedLane', 'KeepRight', 'leftTurn', 'merge', 'pedestrianCrossing', 'school',
    'signalAhead', 'speedLimit25', 'speedLimit30', 'speedLimit35', 'speedLimit45',
    'stopAhead', 'Stopsign', 'Yield']

root = tk.Tk()
model_dir = filedialog.askdirectory(title='Pick saved model folder to use', initialdir=r'C:/tmp/')
loaded = tf.keras.models.load_model(str(model_dir))
loaded.summary()

img_file = filedialog.askopenfilename(title='Pick image to test against', initialdir='.', filetypes=[('Images', "*.jpg *.png *.jpeg")])
root.destroy()

img_og = tf.keras.preprocessing.image.load_img(img_file)
img = tf.keras.preprocessing.image.load_img(img_file, target_size=[IMG_HEIGHT, IMG_WIDTH])
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis,...])

predictions = loaded.predict(tf.constant(x))
predicted_label = np.argmax(predictions[0])
print("Prediction on image:\n", CLASS_NAMES[predicted_label])

plt.imshow(img_og)
plt.axis('off')
plt.title(CLASS_NAMES[predicted_label])
plt.show()