'''Transfer learning workflow:
1) define base model and load weights
2) freeze all layers of base: trainable = False
3) new model (on top of the base)
4) training new model with new dataset'''

#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout

from tensorflow.keras.applications.resnet50 import ResNet50

print(tf.__version__)
print(keras.__version__)

import time

batch_size = 32
img_height = 224
img_width = 224

EPOCHS = 25
buffer_size = 1024

init_dir = "archive/"
data_dir = init_dir + "train"
data_dir_test = init_dir + "test"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

train_ds = train_ds.shuffle(buffer_size)

base_model = ResNet50(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)

base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
# passing `training=False`. This is important for fine-tuning
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(2)(x)
model = keras.Model(inputs, outputs)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = keras.optimizers.Adam(learning_rate = 0.0001)

start = time.time()
for epoch in range(EPOCHS):
    start_time = time.time()
    for image, label in train_ds:
        with tf.GradientTape() as tape:
            logits = model(image, training = True)
            loss_value = loss_fn(label, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(label, logits)

    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    train_acc_metric.reset_states()
    for val_image, val_label in val_ds:
        val_logits = model(val_image, training=False)
        val_acc_metric.update_state(val_label, val_logits)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

print('Total time taken %.2fs' %(time.time() - start))
