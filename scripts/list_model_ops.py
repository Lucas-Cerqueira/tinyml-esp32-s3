import argparse
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List the operations used in a .tflite model")
    parser.add_argument("model_path", help="Path to the input .tflite model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    ops = interpreter._get_ops_details()
    for op_name in sorted({op["op_name"] for op in ops}):
        print(op_name)


if __name__ == "__main__":
    main()
