# MobileNetV1 Transfer Learning on CIFAR-10 (ESP32-S3)

## Experimental Setup

- Base model: MobileNetV1
- Pretrained weights: ImageNet
- Input resolution: 96×96
- Dataset: CIFAR-10
- Quantization: Full INT8
- Hardware: ESP32-S3 @ 240 MHz
- Runtime: TensorFlow Lite Micro + ESP-NN

---

## Accuracy Results

| Width Multiplier | Float Accuracy | INT8 Accuracy | Accuracy Loss |
|-----------------|---------------:|--------------:|--------------:|
| 0.25 | 76.00% | 73.64% | 2.36% |
| 0.50 | 80.31% | 79.05% | 1.26% |
| 0.75 | 82.78% | 81.67% | 1.11% |
| 1.00 | 84.40% | 81.84% | 2.56% |

---

## Model Sizes

| Width Multiplier | INT8 Model Size |
|-----------------|----------------:|
| 0.25 | 319 KB |
| 0.50 | 998 KB |
| 0.75 | 2.07 MB |
| 1.00 | 3.53 MB |

---

## ESP32-S3 Deployment Results

| Width Multiplier | INT8 Accuracy | Arena | Latency |
|-----------------|--------------:|------:|---------:|
| 0.25 | 73.64% | 123 KB | 397 ms |
| 0.50 | 79.05% | 329 KB | 1411 ms |
| 0.75 | 81.67% | 687 KB | 3424 ms |
| 1.00 | 81.84% | 1176 KB | 5981 ms |

---

## Key Findings

### Accuracy Saturation

| Width | INT8 Accuracy |
|--------|-------------:|
| 0.25 | 73.64% |
| 0.50 | 79.05% |
| 0.75 | 81.67% |
| 1.00 | 81.84% |

Accuracy gains:

- 0.25 → 0.50: +5.41%
- 0.50 → 0.75: +2.62%
- 0.75 → 1.00: +0.17%

The gain from α=0.75 to α=1.00 is negligible despite substantially higher resource requirements.

### Resource Scaling

Compared to α=0.25:

- α=1.00 model is approximately 11× larger.
- Arena usage increases by approximately 9.6×.
- Inference latency increases by approximately 15×.

### Watchdog Event

The α=1.00 model triggered the ESP-IDF task watchdog during inference execution, although inference completed successfully and produced the correct prediction. This establishes a practical upper bound for MobileNetV1 deployment on the ESP32-S3.

---

## Conclusions

### Best Efficiency

**MobileNetV1 α=0.25**

- 73.64% INT8 accuracy
- 123 KB arena
- 397 ms latency

### Best Accuracy / Cost Tradeoff

**MobileNetV1 α=0.50**

- 79.05% INT8 accuracy
- 329 KB arena
- 1411 ms latency
- 998 KB model size

### Best Accuracy

**MobileNetV1 α=1.00**

- 81.84% INT8 accuracy
- 1176 KB arena
- 5981 ms latency
- 3.53 MB model size

While α=1.00 achieved the highest accuracy, the marginal gain over α=0.75 was only 0.17%, making it difficult to justify the additional memory and latency costs.

For deployment on the ESP32-S3, α=0.50 provides the most balanced tradeoff between accuracy and resource consumption.
