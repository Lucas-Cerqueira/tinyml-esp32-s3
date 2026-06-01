# TinyML on ESP32-S3

Experimental evaluation of image classification models deployed on the **ESP32-S3-N16R8** microcontroller using TensorFlow Lite Micro. Part of the graduate course CPE716 — Tópicos Especiais em Sistemas Embarcados (UFRJ, 2026).

---

## Results Summary

All models were trained on CIFAR-10 (or MNIST) and quantized to INT8 for deployment. The tensor arena is allocated in PSRAM. Latency is the average of 100 consecutive inferences measured with `esp_timer_get_time()`.

| Model | Accuracy (INT8) | Model size | Peak RAM | Latency |
|---|---|---|---|---|
| CNN MNIST | 97.32% | 17.2 KB | 9 KB | 15 ms |
| CNN CIFAR-10 | 68.22% | 76.2 KB | 34 KB | 140 ms |
| MobileNetV1 α=0.25 @ 64×64 | 62.20% | 249 KB | 95 KB | 180 ms |
| MobileNetV1 α=0.25 @ 96×96 | 73.41% | 319 KB | 123 KB | 397 ms |
| MobileNetV1 α=0.50 @ 96×96 | 79.05% | 998 KB | 329 KB | 1 411 ms |
| MobileNetV1 α=0.75 @ 96×96 | 81.67% | 2 118 KB | 687 KB | 3 424 ms |
| MobileNetV1 α=1.00 @ 96×96 | 81.84% | 3 614 KB | 1 176 KB | 5 981 ms |
| MobileNetV2 α=0.35 @ 96×96 (INT8) | 44.55%* | 654 KB | 225 KB | 708 ms |
| MobileNetV2 α=0.35 @ 96×96 (Float32) | 83.24% | 1 700 KB | 619 KB | 2 122 ms |

\* MobileNetV2 suffered severe INT8 quantization degradation (−38.69 p.p.) due to the inverted residual skip connections making activation distributions hard to calibrate with post-training quantization. The float32 model was deployed instead.

### Key findings

- **Practical ceiling:** the ESP32-S3 runs models up to ~220 000 INT8 parameters with latency under 400 ms, suitable for on-demand classification.
- **Optimal sweet spot:** MobileNetV1 α=0.25 at 96×96 — 73.41% accuracy at 397 ms and 123 KB RAM.
- **Input resolution:** latency scales super-linearly with resolution; accuracy scales logarithmically. 96×96 is the optimum for MobileNetV1.
- **Quantization robustness:** MobileNetV1 (no skip connections) is far more robust to INT8 post-training quantization than MobileNetV2. Float accuracy alone is not a reliable predictor of deployed accuracy.
- **Offloading threshold:** models with latency > 1 s (α ≥ 0.50 or MobileNetV2 float) become candidates for inference offloading to a remote server.

---

## Repository Structure

```
tinyml-esp32-s3/
├── pyproject.toml          # shared Python dependencies (tensorflow)
├── uv.lock                 # locked dependency versions
├── .python-version         # Python 3.13
│
├── scripts/                # shared Python pipeline
│   ├── train.py
│   ├── convert_tflite.py
│   ├── evaluate.py
│   ├── export_test_image.py
│   ├── tflite_to_cpp.py
│   ├── list_model_ops.py
│   └── models/             # model definitions (one module per architecture)
│       ├── mnist.py
│       ├── cifar10.py
│       ├── mobilenet_v1.py
│       └── mobilenet_v2.py
│
├── models/                 # generated artifacts — gitignored, populated by scripts
│   ├── mnist/              #   *.keras, *_float.tflite, *_int8.tflite
│   ├── cifar10/
│   ├── mobilenet_v1/
│   └── mobilenet_v2/
│
├── hello_world/            # ESP-IDF hello world (reference)
├── mnist/                  # ESP-IDF firmware project
├── cifar10/                # ESP-IDF firmware project
├── mobilenet_v1/           # ESP-IDF firmware project
└── mobilenet_v2/           # ESP-IDF firmware project
```

Each IDF project (`mnist/`, `cifar10/`, `mobilenet_v1/`, `mobilenet_v2/`) contains only the firmware sources. The `main/models/` subdirectory inside each project is also gitignored and populated by the Python scripts.

---

## Setup

### 1. Python environment — install `uv`

`uv` is a fast Python package manager. Install it once:

```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the shared virtualenv from the repo root:

```bash
uv sync
```

This creates `.venv/` at the repo root with Python 3.13 and TensorFlow installed. Activate it when needed:

```powershell
# Windows
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Alternatively, prefix every script invocation with `uv run`:

```bash
uv run python scripts/train.py -m mnist
```

### 2. ESP-IDF

Install ESP-IDF v6.0.1 following the [official guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/index.html). The quickest path:

```bash
# Clone ESP-IDF
git clone --recursive https://github.com/espressif/esp-idf.git ~/esp/esp-idf
cd ~/esp/esp-idf && git checkout v6.0.1

# Run the installer (sets up xtensa toolchain + Python deps)
./install.sh esp32s3        # macOS / Linux
./install.bat esp32s3       # Windows
```

Source the environment before building any firmware:

```bash
. ~/esp/esp-idf/export.sh   # macOS / Linux
```

```powershell
. $HOME\esp\esp-idf\export.ps1   # Windows
```

---

## Python Pipeline

All scripts are run from the **repo root**. The `-m` / `--model` flag selects the model. `--alpha` applies only to MobileNetV1.

### Train

```bash
python scripts/train.py -m mnist
python scripts/train.py -m cifar10
python scripts/train.py -m mobilenet_v1 --alpha 0.25
python scripts/train.py -m mobilenet_v1 --alpha 0.5
python scripts/train.py -m mobilenet_v2
```

Saved to `models/{model}/`.

### Convert to TFLite

```bash
python scripts/convert_tflite.py -m mnist -q float
python scripts/convert_tflite.py -m mnist -q int8
python scripts/convert_tflite.py -m mobilenet_v1 --alpha 0.25 -q int8

# Use an explicit .keras file with -k
python scripts/convert_tflite.py -m mobilenet_v1 -k models/mobilenet_v1/mobilenet_v1_050.keras -q int8
```

### Evaluate TFLite accuracy

```bash
python scripts/evaluate.py -m mnist
python scripts/evaluate.py -m mobilenet_v1 --alpha 0.25

# Evaluate specific files
python scripts/evaluate.py -m mobilenet_v1 \
  --tflite models/mobilenet_v1/mobilenet_v1_025_float.tflite \
           models/mobilenet_v1/mobilenet_v1_025_int8.tflite
```

### Export test image header (for firmware)

```bash
python scripts/export_test_image.py -m mnist        -p mnist
python scripts/export_test_image.py -m cifar10      -p cifar10
python scripts/export_test_image.py -m mobilenet_v1 --alpha 0.25 -p mobilenet_v1
python scripts/export_test_image.py -m mobilenet_v2 -p mobilenet_v2
```

Writes `{idf-project}/main/models/{stem}_sample.h`.

### Convert TFLite to C array

```bash
python scripts/tflite_to_cpp.py models/mnist/mnist_int8.tflite          -p mnist
python scripts/tflite_to_cpp.py models/cifar10/cifar10_int8.tflite      -p cifar10
python scripts/tflite_to_cpp.py models/mobilenet_v1/mobilenet_v1_025_int8.tflite -p mobilenet_v1
```

Writes `{idf-project}/main/models/{stem}.cc` and `{stem}.h`.

### Inspect TFLite operations

```bash
python scripts/list_model_ops.py models/mnist/mnist_int8.tflite
```

---

## Building and Flashing Firmware

From inside any IDF project directory (e.g. `mnist/`):

```bash
cd mnist

# Configure target (first time only)
idf.py set-target esp32s3

# Build
idf.py build

# Flash and monitor (replace PORT with your serial port)
idf.py -p PORT flash monitor
```

On Windows, `PORT` is typically `COM3`, `COM4`, etc. On Linux/macOS it is `/dev/ttyUSB0` or `/dev/cu.usbserial-*`.

### Full workflow example — MNIST

```bash
# 1. Train
python scripts/train.py -m mnist

# 2. Convert to INT8
python scripts/convert_tflite.py -m mnist -q int8

# 3. Evaluate accuracy
python scripts/evaluate.py -m mnist

# 4. Generate C array and test image header
python scripts/tflite_to_cpp.py models/mnist/mnist_int8.tflite -p mnist
python scripts/export_test_image.py -m mnist -p mnist

# 5. Build and flash
cd mnist && idf.py build && idf.py -p PORT flash monitor
```

---

## Hardware

| Resource | Specification |
|---|---|
| Microcontroller | ESP32-S3-N16R8 |
| CPU | Dual-core Xtensa LX7 |
| Clock | 240 MHz |
| Flash | 16 MB |
| PSRAM | 8 MB |
| Framework | ESP-IDF v6.0.1 |
| Inference engine | TensorFlow Lite Micro (Espressif port) |
