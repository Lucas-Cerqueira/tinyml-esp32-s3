import argparse
import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Keras MobileNet V1 0.25 model to TFLite with selected quantization"
    )
    parser.add_argument(
        "-q", "--quantization",
        choices=["float", "int8"],
        default="float",
        help="Quantization mode to use for conversion",
    )
    return parser.parse_args()


def representative_dataset():
    rng = np.random.default_rng(42)

    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()

    indices = rng.choice(
        len(x_train),
        size=1000,
        replace=False
    )

    for idx in indices:
        img = tf.image.resize(x_train[idx], (96, 96))

        img = tf.keras.applications.mobilenet.preprocess_input(img)

        img = np.expand_dims(img, axis=0)

        yield [img.astype(np.float32)]


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model("models/mobilenet_v1_025.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    output_path = "models/mobilenet_v1_025_float.tflite"

    if args.quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        output_path = "models/mobilenet_v1_025_int8.tflite"

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()