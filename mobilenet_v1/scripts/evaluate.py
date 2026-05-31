import numpy as np
import tensorflow as tf

    
def preprocess_image(img):
    img = tf.image.resize(img, (96, 96))
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    return img.numpy()


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

    input_scale, input_zero_point = \
        input_details[0]["quantization"]

    print(
        f"Input quantization: "
        f"scale={input_scale}, "
        f"zero_point={input_zero_point}"
    )

    # --------------------------------------------------
    # Load CIFAR-10
    # --------------------------------------------------

    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    y_test = y_test.flatten()
    correct = 0

    for i in range(len(x_test)):

        img = preprocess_image(x_test[i])

        sample = np.expand_dims(img, axis=0)

        if input_dtype == np.int8:

            sample = (
                sample / input_scale
                + input_zero_point
            )

            sample = np.round(sample)

            sample = np.clip(
                sample,
                -128,
                127
            ).astype(np.int8)

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

        if prediction == int(y_test[i]):
            correct += 1

        if (i + 1) % 1000 == 0:
            print(
                f"Processed {i + 1}/{len(x_test)}"
            )

    accuracy = correct / len(x_test)

    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":

    float_acc = evaluate_model(
        "models/mobilenet_v1_025_float.tflite"
    )

    int8_acc = evaluate_model(
        "models/mobilenet_v1_025_int8.tflite"
    )

    print("\n========== SUMMARY ==========")
    print(f"Float TFLite Accuracy : {float_acc:.4f}")
    print(f"INT8 TFLite Accuracy  : {int8_acc:.4f}")
    print(
        f"Accuracy Loss         : "
        f"{(float_acc - int8_acc):.4f}"
    )
