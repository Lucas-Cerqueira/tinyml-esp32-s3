import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a test image as a C header for firmware")
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
        help="MobileNetV1 width multiplier (used to locate the int8 .tflite for quantization params)",
    )
    parser.add_argument(
        "-p", "--idf-project",
        default=None,
        help="IDF project subfolder name (defaults to --model value)",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Index of the test sample to export (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mod = importlib.import_module(f"models.{args.model}")

    idf_project = args.idf_project or args.model
    output_dir = REPO_ROOT / idf_project / "main" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = mod.model_stem(alpha=args.alpha)
    output_path = output_dir / f"{stem}_sample.h"

    (_, _), (x_test, y_test) = mod.get_dataset()
    raw_img = x_test[args.idx]
    label = int(y_test[args.idx] if hasattr(y_test[args.idx], "__len__") is False else y_test[args.idx][0])
    print(f"Label: {label}")

    tflite_path = REPO_ROOT / "models" / args.model / f"{stem}_int8.tflite"
    q = mod.quantize_for_firmware(raw_img, tflite_path=tflite_path if tflite_path.exists() else None)
    print(f"Tensor shape: {q.shape}")
    print(f"Range: [{q.min()}, {q.max()}]")

    with open(output_path, "w") as f:
        mod.write_sample_header(f, label, q.flatten())

    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
