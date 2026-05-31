import tensorflow as tf
from tensorflow import keras
from pathlib import Path

IMG_SIZE = 96

BATCH_SIZE = 64
EPOCHS = 10

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "mobilenet_v1_025.keras"

(x_train, y_train), (x_test, y_test) = \
    keras.datasets.cifar10.load_data()

print("Train:", x_train.shape)
print("Test :", x_test.shape)

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    return image, label

train_ds = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset
    .from_tensor_slices((x_test, y_test))
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

base_model = tf.keras.applications.MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=0.25,
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model(inputs, training=False)

x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.25)(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [

    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

loss, accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\n================================")
print(" MobileNetV1 0.25 Transfer Learning")
print("================================")
print(f"Test Accuracy: {accuracy:.4f}")
print("================================")

model.save(MODEL_PATH)

print(f"\nSaved model to: {MODEL_PATH}")