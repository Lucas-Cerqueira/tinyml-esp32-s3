import tensorflow as tf
import numpy as np

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

idx = 0

img = x_test[idx]

print("Label:", y_test[idx])

q = img.astype(np.int16) - 128

print("Shape:", q.shape)

with open("main/models/mnist_sample.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("const int8_t test_digit[784] = {\n")

    flat = q.flatten()

    for i, value in enumerate(flat):
        if i % 16 == 0:
            f.write("    ")

        f.write(f"{value},")

        if i % 16 == 15:
            f.write("\n")

    f.write("\n};\n")