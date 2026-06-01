import numpy as np
import tensorflow as tf

MODEL_NAME = "mnist"
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 32
USE_CALLBACKS = False


def model_stem(**kwargs) -> str:
    return "mnist"


def build_model(**kwargs) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, kernel_size=3, activation="relu", input_shape=INPUT_SHAPE),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def make_tf_pipeline(x, y, batch_size, shuffle=False):
    def _preprocess(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        img = img[..., None]
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(10000)
    return ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preprocess_single(img) -> np.ndarray:
    return (img.astype("float32") / 255.0)[..., None]


def get_representative_data(**kwargs):
    (x_train, _), _ = get_dataset()
    for i in range(100):
        yield [preprocess_single(x_train[i])[None]]


def quantize_for_firmware(raw_img, tflite_path=None) -> np.ndarray:
    q = raw_img.astype(np.int16) - 128
    return np.clip(q, -128, 127).astype(np.int8)


def write_sample_header(f, label, q_flat):
    f.write("#pragma once\n\n")
    f.write("const int8_t test_digit[784] = {\n")
    for i, value in enumerate(q_flat):
        if i % 16 == 0:
            f.write("    ")
        f.write(f"{int(value)},")
        if i % 16 == 15:
            f.write("\n")
    f.write("\n};\n")
