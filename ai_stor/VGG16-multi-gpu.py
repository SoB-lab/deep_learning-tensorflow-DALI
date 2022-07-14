import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.ops as ops

import nvidia.dali.plugin.tf as dali_tf

import tensorflow as tf
from tensorflow import keras

batch_size = 16
EPOCHS = 25
data_size = 2637
valid_size = 660
ITERATIONS = data_size // batch_size
valid_iterations = valid_size // batch_size
IMAGE_SIZE = 224

image_dir = '/ai/ndaces_solene/archive/train'
image_dir_valid = '/ai/ndaces_solene/archive/test'

def sharded_pipeline(device_id, shard_id):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=device_id)
    with pipe:
        jpegs, labels = fn.readers.file(
            file_root=image_dir, random_shuffle=True, shard_id=shard_id, num_shards=2)
        images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
        images = fn.normalize(images, scale=255., dtype=types.FLOAT)
        pipe.set_outputs(images, labels.gpu())

    return pipe

def sharded_pipeline_valid(device_id, shard_id):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=device_id)
    with pipe:
        jpegs, labels = fn.readers.file(
            file_root=image_dir_valid, random_shuffle=True, shard_id=shard_id, num_shards=2)
        images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
        images = fn.normalize(images, scale=255., dtype=types.FLOAT)
        pipe.set_outputs(images, labels.gpu())

    return pipe



strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0', '/GPU:1'])

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

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=train_acc_metrics)

shapes = (
    (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3),
    (batch_size))
dtypes = (
    tf.float32,
    tf.int32)

def dataset_fn(input_context):
    with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
        device_id = input_context.input_pipeline_id
        return dali_tf.DALIDataset(
            pipeline=sharded_pipeline(
                device_id=device_id, shard_id=device_id),
            batch_size=batch_size,
            output_shapes=shapes,
            output_dtypes=dtypes,
            device_id=device_id)

def validation_fn(input_context):
    with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
        device_id = input_context.input_pipeline_id
        return dali_tf.DALIDataset(
            pipeline=sharded_pipeline_valid(
                device_id=device_id, shard_id=device_id),
            batch_size=batch_size,
            output_shapes=shapes,
            output_dtypes=dtypes,
            device_id=device_id)

input_options = tf.distribute.InputOptions(
    experimental_place_dataset_on_device=True,
    experimental_fetch_to_device=False,
    experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA)

train_dataset = strategy.distribute_datasets_from_function(dataset_fn, input_options)
valid_dataset = strategy.distribute_datasets_from_function(validation_fn, input_options)

model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=ITERATIONS,
    verbose = 2,
    validation_data = valid_dataset,
    validation_steps = valid_iterations)
