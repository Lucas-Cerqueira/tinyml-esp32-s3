import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TFLite model(s) on the test set")
    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=["mnist", "cifar10", "mobilenet_v1", "mobilenet_v2"],
        help="Model family (determines dataset and preprocessing)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="MobileNetV1 width multiplier (used to resolve default .tflite paths)",
    )
    parser.add_argument(
        "--tflite",
        nargs="+",
        default=None,
        help="Explicit .tflite path(s) to evaluate; if omitted, evaluates all in models/{model}/",
    )
    return parser.parse_args()


def build_interpreter_context(model_path: Path) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]
    input_scale, input_zero_point = input_details[0]["quantization"]

    print(f"\nEvaluating: {model_path}")
    print(f"Input dtype : {input_dtype}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")

    return {
        "path": model_path,
        "interpreter": interpreter,
        "input_index": input_details[0]["index"],
        "output_index": output_details[0]["index"],
        "input_dtype": input_dtype,
        "input_scale": input_scale,
        "input_zero_point": input_zero_point,
        "correct": 0,
    }


def evaluate_models(model_paths: list[Path], mod) -> dict:
    (_, _), (x_test, y_test) = mod.get_dataset()
    y_test = np.array(y_test).flatten()

    contexts = [build_interpreter_context(p) for p in model_paths]
    total = len(x_test)

    for i in range(total):
        img = mod.preprocess_single(x_test[i])
        target = int(y_test[i])

        for ctx in contexts:
            sample = np.expand_dims(img, axis=0)

            if ctx["input_dtype"] == np.int8:
                sample = sample / ctx["input_scale"] + ctx["input_zero_point"]
                sample = np.clip(np.round(sample), -128, 127).astype(np.int8)
            else:
                sample = sample.astype(np.float32)

            ctx["interpreter"].set_tensor(ctx["input_index"], sample)
            ctx["interpreter"].invoke()
            output = ctx["interpreter"].get_tensor(ctx["output_index"])

            if int(np.argmax(output)) == target:
                ctx["correct"] += 1

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{total}")

    results = {}
    for ctx in contexts:
        accuracy = ctx["correct"] / total
        results[str(ctx["path"])] = accuracy

    return results


def main() -> None:
    args = parse_args()
    mod = importlib.import_module(f"models.{args.model}")

    if args.tflite:
        model_paths = [Path(p) for p in args.tflite]
    else:
        stem = mod.model_stem(alpha=args.alpha)
        model_dir = Path("models") / args.model
        model_paths = sorted(model_dir.glob("*.tflite"))
        if not model_paths:
            raise FileNotFoundError(f"No .tflite files found in {model_dir}")

    results = evaluate_models(model_paths, mod)

    print("\n========== SUMMARY ==========")
    for path, accuracy in results.items():
        print(f"{Path(path).name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()
