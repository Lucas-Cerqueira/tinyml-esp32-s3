import argparse

import tensorflow as tf

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List the operations used in a .tflite model"
    )
    parser.add_argument("model_path", help="Path to the input .tflite model")
    return parser.parse_args()

args = parse_args()

interpreter = tf.lite.Interpreter(
    model_path=args.model_path
)

interpreter.allocate_tensors()

ops = interpreter._get_ops_details()
ops_names = set(op["op_name"] for op in ops)

for op_name in ops_names:
    print(op_name)