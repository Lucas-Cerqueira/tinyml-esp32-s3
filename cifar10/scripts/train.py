import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64
EPOCHS = 15
MODEL_PATH = "models/cifar10.keras"

# ============================================================
# Load Dataset
# ============================================================

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print("Train:", x_train.shape)
print("Test :", x_test.shape)

# Normalize to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ============================================================
# Model
# ============================================================

model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(
        16,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"),

    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"),

    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),

    layers.Dense(
        32,
        activation="relu"),

    layers.Dense(10)
])

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
    metrics=["accuracy"])

model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE)

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0)

print("\n==============================")
print(f"Test Accuracy: {accuracy:.4f}")
print("==============================")

model.save(MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")