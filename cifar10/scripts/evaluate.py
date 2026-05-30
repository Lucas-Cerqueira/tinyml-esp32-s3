import numpy as np
import tensorflow as tf


def evaluate_model(model_path):
    print(f"\nEvaluating: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    print(f"Input dtype : {input_dtype}")
    print(f"Output dtype: {output_dtype}")

    # Load CIFAR-10
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_test = x_test.astype(np.float32) / 255.0

    correct = 0

    input_scale, input_zero_point = input_details[0]["quantization"]

    for i in range(len(x_test)):
        sample = x_test[i:i + 1]

        if input_dtype == np.int8:
            sample = sample / input_scale + input_zero_point
            sample = np.round(sample).astype(np.int8)
        else:
            sample = sample.astype(np.float32)

        interpreter.set_tensor(
            input_details[0]["index"],
            sample
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )

        prediction = np.argmax(output)

        if prediction == y_test[i]:
            correct += 1

    accuracy = correct / len(y_test)

    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":
    float_acc = evaluate_model("models/cifar10_float.tflite")
    int8_acc = evaluate_model("models/cifar10_int8.tflite")

    print("\n========== SUMMARY ==========")
    print(f"Float TFLite Accuracy : {float_acc:.4f}")
    print(f"INT8 TFLite Accuracy  : {int8_acc:.4f}")
    print(f"Accuracy Loss         : {(float_acc - int8_acc):.4f}")