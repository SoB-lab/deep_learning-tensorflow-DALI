import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout

from tensorflow.keras.applications.vgg16 import VGG16

print(tf.__version__)
print(keras.__version__)

import time
start=time.time()

batch_size = 32
img_height = 224
img_width = 224

EPOCHS = 25

init_dir = "archive/"
data_dir = init_dir + "train"
data_dir_test = init_dir + "test"

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BATCH_SIZE_PER_REPLICA = 8 
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=GLOBAL_BATCH_SIZE
    )

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=GLOBAL_BATCH_SIZE)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

buffer_size = len(train_ds)
train_ds = train_ds.shuffle(buffer_size)

train_dist_dataset = strategy.experimental_distribute_dataset(train_ds)
test_dist_dataset = strategy.experimental_distribute_dataset(val_ds)

with strategy.scope():
    base_model = keras.applications.VGG16(
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

with strategy.scope():
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True,
                                                         reduction=tf.keras.losses.Reduction.NONE)
    def compute_loss(label, prediction):
        loss_per_example = loss(label, prediction)
        return tf.nn.compute_average_loss(loss_per_example, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer)


def train_step(image, label):
    with tf.GradientTape() as tape:
        y_pred = model(image, training=True)
        loss = compute_loss(label, y_pred)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_accuracy.update_state(label, y_pred)
    return loss


def test_step(image, label):
    y_pred = model(image, training=False)
    t_loss = loss(label, y_pred)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(label, y_pred)


@tf.function
def distributed_train_step(image, label):
    per_replica_loss = strategy.run(train_step, args=(image, label,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)


@tf.function
def distributed_test_step(image, label):
    return strategy.run(test_step, args=(image, label,))


start_time = time.time()
for epoch in range(EPOCHS):
    # train loop
    total_loss = 0.0
    num_batches = 0
    start = time.time()
    for image, label in train_dist_dataset:
        total_loss += distributed_train_step(image, label)
        num_batches += 1
    train_loss = total_loss / num_batches

    for image, label in test_dist_dataset:
        distributed_test_step(image, label)

    template = ('epoch {} (in {} seconds), loss: {}, accuracy: {}, val loss {}, val accuracy {}')
    end = time.time()
    print(template.format(epoch + 1, (end - start),
                          train_loss,
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

print('Total time: %.2fs' %(time.time() - start_time))

