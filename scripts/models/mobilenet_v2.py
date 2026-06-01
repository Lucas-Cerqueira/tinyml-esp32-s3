import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_NAME = "mobilenet_v2"
INPUT_SHAPE = (96, 96, 3)
NUM_CLASSES = 10
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
USE_CALLBACKS = True


def model_stem(**kwargs) -> str:
    return "mobilenet_v2_035"


def build_model(**kwargs) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        alpha=0.35,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
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
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, (96, 96))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(10000)
    return ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preprocess_single(img) -> np.ndarray:
    img = tf.image.resize(img, (96, 96))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img.numpy()


def get_representative_data(**kwargs):
    rng = np.random.default_rng(42)
    (x_train, _), _ = get_dataset()
    indices = rng.choice(len(x_train), size=5000, replace=False)
    for idx in indices:
        img = preprocess_single(x_train[idx])
        yield [img[None].astype(np.float32)]


def quantize_for_firmware(raw_img, tflite_path) -> np.ndarray:
    img = preprocess_single(raw_img)
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    detail = interp.get_input_details()[0]
    scale = float(detail["quantization"][0])
    zero_point = int(detail["quantization"][1])
    q = np.round(img / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def write_sample_header(f, label, q_flat):
    n = 96 * 96 * 3
    f.write("#pragma once\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"constexpr int test_label = {label};\n\n")
    f.write(f"const int8_t test_image[{n}] = {{\n")
    for i, value in enumerate(q_flat):
        if i % 16 == 0:
            f.write("    ")
        f.write(f"{int(value)},")
        if i % 16 == 15:
            f.write("\n")
    f.write("\n};\n")
