import numpy as np
from tensorflow import keras

# ============================================================
# Export one CIFAR-10 test image for ESP32
# ============================================================

(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()

idx = 0

img = x_test[idx]
label = y_test[idx][0]

print(f"Label: {label}")

# Quantization parameters from TFLite
SCALE = 1.0 / 255.0
ZERO_POINT = -128

# Quantize exactly like the input tensor
q = img.astype(np.int16) - 128

with open("main/models/cifar10_sample.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("#include <stdint.h>\n\n")

    f.write(f"constexpr int test_label = {label};\n\n")

    f.write("const int8_t test_image[3072] = {\n")

    flat = q.flatten()

    for i, value in enumerate(flat):
        if i % 16 == 0:
            f.write("    ")

        f.write(f"{int(value)},")

        if i % 16 == 15:
            f.write("\n")

    f.write("\n};\n")

print("Generated cifar10_sample.h")