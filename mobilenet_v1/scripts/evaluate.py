from pathlib import Path

import numpy as np
import tensorflow as tf


def preprocess_image(img):
    img = tf.image.resize(img, (96, 96))
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    return img.numpy()


def build_model_context(model_path: Path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]
    input_scale, input_zero_point = input_details[0]["quantization"]

    print(f"\nEvaluating: {model_path}")
    print(f"Input dtype : {input_dtype}")
    print(f"Output dtype: {output_dtype}")
    print(
        f"Input quantization: "
        f"scale={input_scale}, "
        f"zero_point={input_zero_point}"
    )

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


def prepare_input_sample(img, context):
    sample = np.expand_dims(img, axis=0)

    if context["input_dtype"] == np.int8:
        scale = context["input_scale"]
        zero_point = context["input_zero_point"]

        sample = (sample / scale) + zero_point
        sample = np.round(sample)
        sample = np.clip(sample, -128, 127).astype(np.int8)
        return sample

    return sample.astype(np.float32)


def evaluate_models(model_paths):
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.flatten()

    contexts = [build_model_context(model_path) for model_path in model_paths]

    total = len(x_test)
    for i in range(total):
        img = preprocess_image(x_test[i])
        target = int(y_test[i])

        for context in contexts:
            sample = prepare_input_sample(img, context)

            context["interpreter"].set_tensor(context["input_index"], sample)
            context["interpreter"].invoke()

            output = context["interpreter"].get_tensor(context["output_index"])
            prediction = int(np.argmax(output))

            if prediction == target:
                context["correct"] += 1

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{total}")

    results = {}
    for context in contexts:
        accuracy = context["correct"] / total
        results[str(context["path"])] = accuracy
        print(f"Accuracy ({context['path'].name}): {accuracy:.4f}")

    return results


if __name__ == "__main__":
    models_dir = Path("models")
    model_paths = sorted(models_dir.glob("*.tflite"))

    if not model_paths:
        raise FileNotFoundError("No .tflite models found in models/")

    results = evaluate_models(model_paths)

    print("\n========== SUMMARY ==========")
    for model_path, accuracy in results.items():
        print(f"{Path(model_path).name}: {accuracy:.4f}")
