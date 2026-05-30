import argparse

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Keras MNIST model to TFLite with selected quantization"
    )
    parser.add_argument(
        "-q", "--quantization",
        choices=["float", "int8"],
        default="float",
        help="Quantization mode to use for conversion",
    )
    return parser.parse_args()


def representative_dataset():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = x_train[..., None]

    for i in range(100):
        yield [x_train[i:i + 1]]


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model("models/mnist.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    output_path = "models/mnist_float.tflite"

    if args.quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        output_path = "models/mnist_int8.tflite"

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()