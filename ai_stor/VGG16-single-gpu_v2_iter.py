import sys

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
from tensorflow import keras

import time

train_directory = sys.argv[1]
valid_directory = sys.argv[2]

class Data_Pipeline(Pipeline):
    def __init__(self, batch_size, device, data_path, device_id=7, num_threads=4, seed=0):
        super(Data_Pipeline, self).__init__(
            batch_size, num_threads, device_id, seed)
        self.device = device
        self.reader = ops.FileReader(file_root =data_path, random_shuffle=True)
        self.decode = ops.ImageDecoder(
            device='mixed' if device == 'gpu' else 'cpu',
            output_type=types.RGB)
        self.cmn = ops.CropMirrorNormalize(
            device=device,
            dtype=types.FLOAT,
            std=[15.9687],
            output_layout="HWC")
        self.rotate = ops.Rotate(device = "gpu")
        self.rng = ops.random.Uniform(range = (-10.0, 10.0))
        self.coin = ops.random.CoinFlip(probability = 0.5)
        self.flip = ops.Flip(device = "gpu")
        self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224, interp_type=types.INTERP_TRIANGULAR)

    def define_graph(self):
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)
        if self.device == 'gpu':
            labels = labels.gpu()
        images = self.cmn(images)
        angle = self.rng()
        images = self.rotate(images, angle=angle)
        images = self.flip(images, horizontal = self.coin(), vertical = self.coin())
        images = self.res(images)
        return (images, labels)


BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 25
DATA_SIZE = 2637
VALIDATION_SIZE = 660
ITERATIONS_PER_EPOCH = DATA_SIZE // BATCH_SIZE
VALIDATION_STEPS = VALIDATION_SIZE // BATCH_SIZE

shapes = (
    (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3),
    (BATCH_SIZE))
dtypes = (
    tf.float32,
    tf.int32)

Pipeline_train = Data_Pipeline(BATCH_SIZE, device='gpu', data_path = train_directory, device_id=7)
Pipeline_valid = Data_Pipeline(BATCH_SIZE, device='gpu', data_path = valid_directory, device_id=7)

with tf.device('/GPU:7'):
    data_set = dali_tf.DALIDataset(
        pipeline=Pipeline_train,
        batch_size=BATCH_SIZE,
        output_shapes=shapes,
        output_dtypes=dtypes,
        device_id=7)

    valid_data_set = dali_tf.DALIDataset(
        pipeline=Pipeline_valid,
        batch_size=BATCH_SIZE,
        output_shapes=shapes,
        output_dtypes=dtypes,
        device_id=7)

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

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
optimizer = keras.optimizers.Adam(learning_rate = 0.0001)


print('starting the training')
start = time.time()
with tf.device('/GPU:7'):
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_iter = iter(data_set)
        for _ in range(ITERATIONS_PER_EPOCH):
            with tf.GradientTape() as tape:
                image, label = next(train_iter)
                logits = model(image, training=True)
                loss_value = loss_fn(label, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(label, logits)

        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        train_acc_metric.reset_states()
        val_iter = iter(valid_data_set)
        for _ in range(VALIDATION_STEPS):
            val_image, val_label = next(val_iter)
            val_logits = model(val_image, training=False)
            val_acc_metric.update_state(val_label, val_logits)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))

print('total time for training: %.fs'% (time.time() - start))
