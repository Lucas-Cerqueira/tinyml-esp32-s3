import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

IMG_SIZE = (96, 96)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Keras model to TFLite with selected quantization"
    )
    parser.add_argument(
        "-m", "--model-path",
        default="models/mobilenet_v2_035.keras",
        help="Path to the input Keras model (.keras)",
    )
    parser.add_argument(
        "-q", "--quantization",
        choices=["float", "int8"],
        default="float",
        help="Quantization mode to use for conversion",
    )
    return parser.parse_args()

def preprocess_image(img):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def representative_dataset():
    rng = np.random.default_rng(42)

    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()

    indices = rng.choice(
        len(x_train),
        size=5000,
        replace=False
    )

    for idx in indices:
        img = preprocess_image(x_train[idx])
        img = np.expand_dims(img, axis=0)
        yield [img.astype(np.float32)]


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    output_path = model_path.with_name(f"{model_path.stem}_float.tflite")

    if args.quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        output_path = model_path.with_name(f"{model_path.stem}_int8.tflite")


    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()