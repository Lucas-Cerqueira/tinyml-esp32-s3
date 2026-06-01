import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Keras model to TFLite")
    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=["mnist", "cifar10", "mobilenet_v1", "mobilenet_v2"],
        help="Model to convert",
    )
    parser.add_argument(
        "-q", "--quantization",
        choices=["float", "int8"],
        default="float",
        help="Quantization mode (default: float)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="MobileNetV1 width multiplier (used to resolve default model path)",
    )
    parser.add_argument(
        "-k", "--keras-path",
        default=None,
        help="Explicit path to the .keras model file (overrides default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mod = importlib.import_module(f"models.{args.model}")

    stem = mod.model_stem(alpha=args.alpha)
    model_dir = Path("models") / args.model

    if args.keras_path:
        model_path = Path(args.keras_path)
        stem = model_path.stem
    else:
        model_path = model_dir / f"{stem}.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: mod.get_representative_data(alpha=args.alpha)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        output_path = model_dir / f"{stem}_int8.tflite"
    else:
        output_path = model_dir / f"{stem}_float.tflite"

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
