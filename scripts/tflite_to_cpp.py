import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a .tflite model file to a C array source (.cc)")
    parser.add_argument("model_path", help="Path to the input .tflite model")
    parser.add_argument(
        "-p", "--idf-project",
        required=True,
        help="IDF project subfolder name (e.g. mnist, cifar10, mobilenet_v1)",
    )
    return parser.parse_args()


def to_c_identifier(name: str) -> str:
    return re.sub(r"\W|^(?=\d)", "_", name)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    output_dir = REPO_ROOT / args.idf_project / "main" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = output_dir / f"{model_path.stem}.cc"
    header_path = output_dir / f"{model_path.stem}.h"

    with open(model_path, "rb") as f:
        data = f.read()

    symbol = f"{to_c_identifier(model_path.stem)}_tflite"

    with open(header_path, "w") as f:
        f.write("#pragma once\n\n")
        f.write(f"extern const unsigned char {symbol}[];\n")
        f.write(f"extern const unsigned int {symbol}_len;\n")

    with open(source_path, "w") as f:
        f.write(f"#include \"{header_path.name}\"\n\n")
        f.write(f"const unsigned char {symbol}[] = {{\n")

        for i, b in enumerate(data):
            if i % 12 == 0:
                f.write("    ")
            f.write(f"0x{b:02x}, ")
            if i % 12 == 11:
                f.write("\n")

        f.write("\n};\n")
        f.write(f"const unsigned int {symbol}_len = {len(data)};\n")

    print(f"Generated {source_path}")
    print(f"Generated {header_path}")


if __name__ == "__main__":
    main()
