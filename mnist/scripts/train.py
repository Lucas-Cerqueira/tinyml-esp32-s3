import tensorflow as tf

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = x_train[..., None]
x_test = x_test[..., None]

# Tiny CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        8,
        kernel_size=3,
        activation="relu",
        input_shape=(28, 28, 1)
    ),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    epochs=3,
    validation_split=0.1
)

print(model.summary())

loss, acc = model.evaluate(x_test, y_test)

print(f"Accuracy: {acc:.4f}")

model.save("models/mnist.keras")