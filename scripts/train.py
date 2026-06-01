import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TinyML model")
    parser.add_argument(
        "-m", "--model",
        required=True,
        choices=["mnist", "cifar10", "mobilenet_v1", "mobilenet_v2"],
        help="Model to train",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="MobileNetV1 width multiplier: 0.25, 0.5, 0.75, or 1.0",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the default epoch count",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the default batch size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mod = importlib.import_module(f"models.{args.model}")

    stem = mod.model_stem(alpha=args.alpha)
    output_dir = Path("models") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{stem}.keras"

    epochs = args.epochs or mod.DEFAULT_EPOCHS
    batch_size = args.batch_size or mod.DEFAULT_BATCH_SIZE

    model = mod.build_model(alpha=args.alpha)
    model.summary()

    (x_train, y_train), (x_test, y_test) = mod.get_dataset()
    train_ds = mod.make_tf_pipeline(x_train, y_train, batch_size=batch_size, shuffle=True)
    test_ds = mod.make_tf_pipeline(x_test, y_test, batch_size=batch_size)

    callbacks = []
    if mod.USE_CALLBACKS:
        import tensorflow as tf
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=3,
                restore_best_weights=True,
            ),
        ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks or None,
    )

    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {accuracy:.4f}")

    model.save(model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
