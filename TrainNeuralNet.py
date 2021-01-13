import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import keras2onnx
import onnxruntime

import os
import random
import cv2
import numpy as np

DATADIR = "DataSet/simplified/train/"
VALIDATIONDIR = "DataSet/simplified/validate/"
NAMED_CATEGORIES = ["Wave", "Peacesign", "Three", "Paper", "Fist", "Horns", "One", "Four", "Ok", "Thumb", "Pinky"]
CATEGORIES = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11"]

batch_size = 32
img_height = 480
img_width = 640

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATADIR,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'
    )

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VALIDATIONDIR,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'
    )

class_names = train_ds.class_names
print(class_names)

# Shows the first 9 images of your dataset
#plt.figure(figsize=(10, 10))
#for images, labels in train_ds.take(1):
#    for i in range(9):
#        ax = plt.subplot(3, 3, i + 1)
#        plt.imshow(images[i].numpy().astype("uint8"))
#        plt.title(class_names[labels[i]])
#        plt.axis("off")
#    plt.show()

# Optimizations to speed up the learning
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

for i in labels_batch:
    print(i)

################## Create model ####################
num_classes = 11

model = models.Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

################### Train model ######################
epochs=5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

################### Save model ######################
model.save("saved_model")

# We can now convert this model to onnx using
# python -m tf2onnx.convert --opset 12 --saved-model "\saved_model" --output "\model.onnx"