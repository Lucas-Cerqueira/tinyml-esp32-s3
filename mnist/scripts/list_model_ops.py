import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="models/mnist_int8.tflite"
)

interpreter.allocate_tensors()

ops = interpreter._get_ops_details()
ops_names = set(op["op_name"] for op in ops)

for op_name in ops_names:
    print(op_name)