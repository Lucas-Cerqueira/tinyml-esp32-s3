# Experiment: MobileNetV1 0.25 Transfer Learning on CIFAR-10 (ESP32-S3)

## Objective

Evaluate the impact of input resolution on:

- Classification accuracy
- INT8 quantization robustness
- Memory consumption
- Inference latency

for a MobileNetV1 0.25 model pretrained on ImageNet and fine-tuned on CIFAR-10.

---

## Experimental Setup

### Hardware

- ESP32-S3
- 240 MHz CPU
- 8 MB PSRAM
- 16 MB Flash

### Software

- TensorFlow 2.x
- TensorFlow Lite
- TensorFlow Lite Micro
- Full INT8 quantization

### Training Strategy

- MobileNetV1 (α = 0.25)
- ImageNet pretrained weights
- Frozen backbone
- GlobalAveragePooling2D
- Dense(10) classifier
- CIFAR-10 dataset

---

## Accuracy Results

| Input Size | Float Accuracy | INT8 Accuracy | Accuracy Loss |
|------------|---------------:|--------------:|--------------:|
| 32×32 | 27.80% | - | - |
| 64×64 | 64.99% | 62.20% | 2.79% |
| 96×96 | **76.69%** | **73.41%** | **3.28%** |
| 128×128 | 79.20% | 69.84% | 9.36% |
| 224×224 | 80.29% | 69.16% | 11.13% |

---

## ESP32-S3 Deployment Results

| Input Size | Arena | Latency |
|------------|-------:|---------:|
| 64×64 | 94.7 KB | 180 ms |
| 96×96 | **122.5 KB** | **397 ms** |
| 128×128 | 158.3 KB | 709 ms |
| 224×224 | 480 KB | 2367 ms |

---

## Key Findings

### 1. Highest Float Accuracy Does Not Yield Highest INT8 Accuracy

The 224×224 model achieved the highest floating-point accuracy:

```text
80.29%
```

However, after full INT8 quantization its accuracy dropped to:

```text
69.16%
```

The 96×96 model achieved a lower floating-point accuracy:

```text
76.69%
```

but obtained the highest deployed accuracy:

```text
73.41%
```

demonstrating that the model with the best floating-point performance is not necessarily the best embedded deployment candidate.

---

### 2. Quantization Loss Increased with Input Resolution

| Resolution | Quantization Loss |
|------------|------------------:|
| 64×64 | 2.79% |
| 96×96 | 3.28% |
| 128×128 | 9.36% |
| 224×224 | 11.13% |

Lower-resolution variants were substantially more robust to full integer quantization.

---

### 3. Inference Time Scaled Strongly with Input Resolution

| Resolution | Latency |
|------------|---------:|
| 64×64 | 180 ms |
| 96×96 | 397 ms |
| 128×128 | 709 ms |
| 224×224 | 2367 ms |

Reducing the input resolution from 224×224 to 96×96 reduced inference latency by approximately:

```text
2367 / 397 ≈ 6×
```

while improving deployed accuracy.

---

### 4. Memory Requirements Increased Significantly with Resolution

| Resolution | Arena |
|------------|-------:|
| 64×64 | 94.7 KB |
| 96×96 | 122.5 KB |
| 128×128 | 158.3 KB |
| 224×224 | 480 KB |

The 224×224 model required almost four times more memory than the 96×96 model while delivering lower deployed accuracy.

---

### 5. Transfer Learning Becomes Ineffective at 32×32

Using the original CIFAR-10 resolution (32×32) produced only:

```text
27.8%
```

test accuracy.

This indicates that MobileNetV1 pretrained on ImageNet is unable to effectively transfer its learned feature representations when the input resolution is reduced to the original CIFAR-10 dimensions.

---

## Comparison with Custom CIFAR CNN

| Model | INT8 Accuracy | Arena | Latency |
|---------|-------------:|-------:|---------:|
| Custom CIFAR CNN | 68.22% | 34 KB | 140 ms |
| MobileNetV1 0.25 (64×64) | 62.20% | 95 KB | 180 ms |
| MobileNetV1 0.25 (96×96) | **73.41%** | **123 KB** | **397 ms** |
| MobileNetV1 0.25 (128×128) | 69.84% | 158 KB | 709 ms |
| MobileNetV1 0.25 (224×224) | 69.16% | 480 KB | 2367 ms |

---

## Conclusion

Among all evaluated MobileNetV1 0.25 configurations, the **96×96 input resolution provided the best overall deployment tradeoff**.

The model achieved:

- **73.41% INT8 accuracy**
- **122.5 KB tensor arena**
- **397 ms inference time**
- **3.28% quantization loss**

Although larger input resolutions improved floating-point accuracy, they suffered substantially greater quantization degradation, higher memory requirements, and significantly longer inference times.

These results demonstrate that evaluating models directly on the target hardware after quantization is essential, as the configuration with the highest floating-point accuracy was not the best deployment candidate for the ESP32-S3.

For the evaluated TinyML workload, **MobileNetV1 0.25 with 96×96 inputs represents the optimal balance between accuracy, memory usage, and inference latency.**