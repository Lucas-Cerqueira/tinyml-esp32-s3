# MNIST Training Setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

## 1. Install uv

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal and verify:

```powershell
uv --version
```

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

## 2. Create and activate the virtual environment

From the project root:

```powershell
cd C:\Users\Lucas\Documents\Pessoal\Mestrado\IoT\tinyml-esp32-s3\mnist
uv venv
```

Activate it:

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (.\.venv\Scripts\Activate.ps1)
```

macOS / Linux activation:

```bash
source .venv/bin/activate
```

## 3. Install project dependencies

With the venv active, sync dependencies from `pyproject.toml` / `uv.lock`:

```powershell
uv sync
```

## 4. Run training

```powershell
python .\scripts\train.py
```

The script will:

- download MNIST,
- train a small CNN for 3 epochs,
- print test accuracy,
- save the model to `models/mnist.keras`.

## 5. Convert to TensorFlow Lite

After training, convert the model to TFLite.

Float model:

```powershell
python .\scripts\convert_tflite.py -q float
```

INT8 quantized model:

```powershell
python .\scripts\convert_tflite.py -q int8
```

Generated files:

- `models/mnist_float.tflite` for `-q float`
- `models/mnist_int8.tflite` for `-q int8`

## Notes

- Required Python version is `>=3.13` (see `pyproject.toml`).
- If activation is blocked on Windows, run the PowerShell activation command shown above (process scope only).

# Results

## Objective

The objective of this experiment was to validate the complete TinyML deployment pipeline on an ESP32-S3 microcontroller, including model training, quantization, conversion to TensorFlow Lite, deployment using TensorFlow Lite Micro (TFLM), and on-device inference.

## Hardware Platform

* Microcontroller: ESP32-S3-N16R8
* CPU: Dual-core Xtensa LX7 @ 160 MHz
* Flash Memory: 16 MB
* PSRAM: 8 MB
* Framework: ESP-IDF v6.0.1
* Inference Engine: TensorFlow Lite Micro (Espressif port)

## Model Architecture

A lightweight Convolutional Neural Network (CNN) was trained using the MNIST handwritten digit dataset.

Architecture:

* Conv2D (8 filters, 3×3 kernel, ReLU)
* MaxPooling2D
* Flatten
* Dense (10 outputs, Softmax)

Model parameters:

* Trainable parameters: 13,610

## Quantization Results

The trained Keras model was converted to TensorFlow Lite format and subsequently quantized to INT8.

| Metric   | Float Model | INT8 Model |
| -------- | ----------: | ---------: |
| Size     |     56.3 KB |    17.2 KB |
| Accuracy |      97.32% |     97.32% |

Quantization reduced the model size by approximately 69.5% while preserving classification accuracy.

## Deployment Results

The quantized model was converted into a C array and embedded directly into the ESP32 firmware.

The model was successfully loaded by TensorFlow Lite Micro and executed on the microcontroller.

### Tensor Information

Input tensor:

* Type: INT8
* Shape: [1, 28, 28, 1]

Output tensor:

* Type: INT8
* Shape: [1, 10]

### Memory Requirements

The tensor arena was allocated dynamically during runtime.

| Metric                |       Value |
| --------------------- | ----------: |
| Reported arena usage  | 9,164 bytes |
| Minimum working arena |        9 KB |
| 8 KB arena            |      Failed |

When the arena size was reduced below 9 KB, TensorFlow Lite Micro failed tensor allocation due to insufficient memory.

### Inference Validation

A test image from the MNIST test dataset was exported, quantized, embedded in firmware, and used as input for inference.

Expected digit:

7

Predicted digit:

7

The prediction matched the ground-truth label, confirming correct end-to-end deployment.

### Latency Measurements

Average inference latency was measured over 100 executions.

| Arena Location | Average Inference Time |
| -------------- | ---------------------: |
| Internal RAM   |           14,928.56 µs |
| PSRAM          |           14,929.67 µs |

The observed difference between internal RAM and PSRAM was negligible (<0.01%), indicating that memory bandwidth is not a limiting factor for such a small model.

## Discussion

This experiment demonstrates that a quantized convolutional neural network can be successfully deployed and executed on the ESP32-S3 using TensorFlow Lite Micro.

The model required only 17.2 KB of storage and approximately 9 KB of runtime tensor memory while maintaining 97.32% classification accuracy. Inference execution time was approximately 15 ms, corresponding to roughly 67 inferences per second.

The results indicate that the ESP32-S3 is capable of executing lightweight neural networks efficiently, making it a suitable platform for TinyML applications. Furthermore, the negligible performance difference between internal RAM and PSRAM suggests that memory placement is unlikely to impact the execution of very small models, although this effect may become more significant for larger architectures.

## Conclusion

The complete TinyML deployment workflow was successfully validated on the ESP32-S3 platform. The experiment established a reproducible methodology for future evaluation of more complex architectures, including CIFAR-10 CNNs, MobileNet variants, and MCUNet models. The collected metrics provide a baseline for future comparisons involving model size, memory consumption, inference latency, and classification accuracy.
