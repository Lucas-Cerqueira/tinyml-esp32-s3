import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_NAME = "cifar10"
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 64
USE_CALLBACKS = False


def model_stem(**kwargs) -> str:
    return "cifar10"


def build_model(**kwargs) -> tf.keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=INPUT_SHAPE),
        keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(NUM_CLASSES),
    ])
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def get_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return (x_train, y_train), (x_test, y_test)


def make_tf_pipeline(x, y, batch_size, shuffle=False):
    def _preprocess(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(10000)
    return ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preprocess_single(img) -> np.ndarray:
    return img.astype("float32") / 255.0


def get_representative_data(**kwargs):
    (x_train, _), _ = get_dataset()
    for i in range(1000):
        yield [preprocess_single(x_train[i])[None]]


def quantize_for_firmware(raw_img, tflite_path=None) -> np.ndarray:
    q = raw_img.astype(np.int16) - 128
    return np.clip(q, -128, 127).astype(np.int8)


def write_sample_header(f, label, q_flat):
    f.write("#pragma once\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"constexpr int test_label = {label};\n\n")
    f.write("const int8_t test_image[3072] = {\n")
    for i, value in enumerate(q_flat):
        if i % 16 == 0:
            f.write("    ")
        f.write(f"{int(value)},")
        if i % 16 == 15:
            f.write("\n")
    f.write("\n};\n")
