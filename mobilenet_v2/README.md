# MobileNetV2 0.35 Transfer Learning on CIFAR-10 (ESP32-S3)

## Experimental Setup

- Base model: MobileNetV2
- Width multiplier (alpha): 0.35
- Pretrained weights: ImageNet
- Input resolution: 96×96
- Dataset: CIFAR-10
- Hardware: ESP32-S3 @ 240 MHz
- Runtime: TensorFlow Lite Micro + ESP-NN

---

## Quantized Model Results

### Accuracy

| Metric | Value |
|----------|----------:|
| Float TFLite Accuracy | 83.24% |
| INT8 TFLite Accuracy | 44.55% |
| Accuracy Loss | 38.69% |

### Model Sizes

| Artifact | Size |
|----------|----------:|
| Keras Model | 2.39 MB |
| Float TFLite | 1.66 MB |
| INT8 TFLite | 654 KB |

### Quantization Parameters

#### Float Model

- Input dtype: float32
- Output dtype: float32

#### INT8 Model

- Input dtype: int8
- Input scale: 0.0078431377
- Input zero point: -1

- Output dtype: int8
- Output scale: 0.00390625
- Output zero point: -128

### ESP32-S3 Deployment

| Metric | Value |
|----------|----------:|
| Model Size | 654 KB |
| Arena Used | 224.9 KB |
| Latency | 708 ms |
| Prediction | Correct |

### Observations

The MobileNetV2 INT8 model suffered severe degradation after post-training quantization.

- Float accuracy: 83.24%
- INT8 accuracy: 44.55%
- Accuracy loss: 38.69 percentage points

This behavior persisted after:

- Changing the classifier head
- Using a large representative dataset
- Generating a hybrid quantized model

The hybrid model achieved:

| Metric | Value |
|----------|----------:|
| Float Accuracy | 83.24% |
| Hybrid Accuracy | 44.23% |
| Accuracy Loss | 39.01% |

The similarity between hybrid and full INT8 results suggests the issue is not limited to activation quantization.

---

## Float32 Deployment Results

To determine whether the degradation was caused by quantization or the architecture itself, the float32 model was deployed directly on the ESP32-S3.

### ESP32-S3 Deployment

| Metric | Value |
|----------|----------:|
| Model Size | 1.66 MB |
| Arena Used | 619 KB |
| Latency | 2122 ms |
| Prediction | Correct |

### Sample Output

```text
0 -> 0.000441
1 -> 0.000154
2 -> 0.000456
3 -> 0.916795
4 -> 0.000725
5 -> 0.064969
6 -> 0.011047
7 -> 0.000045
8 -> 0.005339
9 -> 0.000030

Predicted class: 3
Expected class: 3
```

The float32 deployment reproduced the desktop accuracy while executing successfully on the ESP32-S3.

---

## Comparison Against MobileNetV1 0.25 @ 96×96

| Metric | MobileNetV1 INT8 | MobileNetV2 Float |
|----------|----------:|----------:|
| Accuracy | 73.41% | 83.24% |
| Model Size | 319 KB | 1.66 MB |
| Arena | 123 KB | 619 KB |
| Latency | 397 ms | 2122 ms |

Resource increase:

- Model size: ~5.2×
- Arena: ~5.0×
- Latency: ~5.3×

Accuracy improvement:

- +9.83 percentage points

---

## Key Findings

### MobileNetV2 Achieved the Highest Float Accuracy

The MobileNetV2 0.35 model achieved:

- 83.24% float accuracy

This was the highest accuracy observed among all evaluated models.

### Post-Training Quantization Failed

Full INT8 quantization reduced accuracy to:

- 44.55%

representing a loss of:

- 38.69 percentage points

The degradation remained after testing alternative classifier heads and hybrid quantization.

### Float32 Deployment Was Successful

Despite its large size and latency, the float32 model:

- Loaded successfully on the ESP32-S3
- Executed correctly using TensorFlow Lite Micro
- Produced correct predictions
- Preserved desktop accuracy

---

## Conclusions

### Highest Accuracy

**MobileNetV2 0.35 Float32**

- 83.24% accuracy
- 619 KB arena
- 2122 ms latency
- 1.66 MB model

### Quantized Deployment

The MobileNetV2 0.35 INT8 model was not considered a viable deployment candidate due to severe accuracy degradation after quantization.

### Project Insight

This experiment demonstrated that models with excellent floating-point performance may become unsuitable for TinyML deployment when post-training quantization introduces excessive accuracy loss.

It also showed that the ESP32-S3 is capable of executing relatively large float32 neural networks when sufficient PSRAM is available, albeit with substantial memory and latency costs.
