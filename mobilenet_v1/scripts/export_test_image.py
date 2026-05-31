import numpy as np
import tensorflow as tf

IMG_SIZE = 96

(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
idx = 0
img = x_test[idx]
label = int(y_test[idx][0])
print(f"Label: {label}")


img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
img = tf.keras.applications.mobilenet.preprocess_input(img)
img = img.numpy()

# Read quantization params from the model
interp = tf.lite.Interpreter(model_path="models/mobilenet_v1_025_int8.tflite")
interp.allocate_tensors()
inp_detail = interp.get_input_details()[0]
SCALE = float(inp_detail["quantization"][0])
ZERO_POINT = int(inp_detail["quantization"][1])
print(f"Quant: scale={SCALE}, zero_point={ZERO_POINT}")

q = np.round(img / SCALE + ZERO_POINT)
q = np.clip(q, -128, 127).astype(np.int8)
print(f"Tensor shape: {q.shape}")
print(f"Range: [{q.min()}, {q.max()}]")


with open("main/models/mobilenet_v1_sample.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"constexpr int test_label = {label};\n\n")
    f.write(f"const int8_t test_image[{IMG_SIZE} * {IMG_SIZE} * 3] = {{\n")

    flat = q.flatten()
    for i, value in enumerate(flat):
        if i % 16 == 0:
            f.write("    ")
        f.write(f"{int(value)},")
        if i % 16 == 15:
            f.write("\n")
    f.write("\n};\n")

print("Generated mobilenet_sample.h")