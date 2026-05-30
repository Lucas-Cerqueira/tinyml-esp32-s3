# Experiment 2 – Deployment of a Quantized CIFAR-10 Classifier on ESP32-S3

## Objective

The objective of this experiment was to evaluate the deployment of a more complex image classification model on the ESP32-S3 platform and compare its resource requirements against the previously evaluated MNIST classifier.

The CIFAR-10 dataset was selected because it represents a more realistic computer vision problem, containing color images of real-world objects distributed across ten classes.

## Hardware Platform

* Microcontroller: ESP32-S3-N16R8
* CPU: Dual-core Xtensa LX7 @ 160 MHz
* Flash Memory: 16 MB
* PSRAM: 8 MB
* Framework: ESP-IDF v6.0.1
* Inference Engine: TensorFlow Lite Micro (Espressif port)

## Dataset

CIFAR-10 consists of 60,000 RGB images with dimensions of 32×32 pixels distributed among ten classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

The dataset contains:

* Training images: 50,000
* Test images: 10,000

## Model Architecture

A lightweight convolutional neural network was designed to remain compatible with resource-constrained devices.

Architecture:

* Conv2D (16 filters, 3×3, ReLU)
* MaxPooling2D
* Conv2D (32 filters, 3×3, ReLU)
* MaxPooling2D
* Flatten
* Dense (32 neurons, ReLU)
* Dense (10 outputs)

Model parameters:

* Total parameters: 70,986

## Training Results

The model was trained for 15 epochs using the Adam optimizer and Sparse Categorical Cross-Entropy loss.

Final test accuracy:

* Float model accuracy: 68.22%

Although significantly lower than state-of-the-art CIFAR-10 classifiers, the model was intentionally designed to prioritize deployability on microcontroller-class hardware.

## Quantization Results

The trained model was converted to TensorFlow Lite format and fully quantized to INT8.

| Metric   | Float Model | INT8 Model |
| -------- | ----------: | ---------: |
| Size     |    281.4 KB |    76.2 KB |
| Accuracy |      68.22% |     68.22% |

Quantization reduced the model size by approximately 72.9% while preserving classification accuracy.

Accuracy loss:

* 0.00%

## TensorFlow Lite Micro Deployment

The quantized model was converted into a C array and embedded directly into the ESP32 firmware.

The model was successfully loaded and executed using TensorFlow Lite Micro.

### Tensor Information

Input tensor:

* Type: INT8
* Shape: [1, 32, 32, 3]

Output tensor:

* Type: INT8
* Shape: [1, 10]

### Operator Set

The deployed model required the following TensorFlow Lite operators:

* CONV_2D
* MAX_POOL_2D
* FULLY_CONNECTED
* PACK
* SHAPE
* RESHAPE
* STRIDED_SLICE

The absence of SOFTMAX reduced inference complexity compared to the MNIST model.

## Memory Requirements

Tensor allocation completed successfully using TensorFlow Lite Micro.

| Metric     |        Value |
| ---------- | -----------: |
| Model size | 78,048 bytes |
| Arena used | 34,284 bytes |

The model fit comfortably within the available memory resources of the ESP32-S3.

### Comparison with MNIST

| Metric          |   MNIST | CIFAR-10 |
| --------------- | ------: | -------: |
| Parameters      |  13,610 |   70,986 |
| INT8 model size | 17.2 KB |  76.2 KB |
| Arena usage     |  9.2 KB |  33.5 KB |

The CIFAR-10 model contains approximately 5.2× more parameters than the MNIST model, while runtime memory requirements increased by approximately 3.7×.

## Inference Validation

A CIFAR-10 test image was exported, quantized, embedded in firmware, and used as input for on-device inference.

Example output:

Expected class: 3

Predicted class: 3

The prediction produced by the ESP32 matched the ground-truth label, confirming successful end-to-end deployment.

Output logits:

* Class 0: -17
* Class 1: -27
* Class 2: -13
* Class 3: 30
* Class 4: -13
* Class 5: 22
* Class 6: -5
* Class 7: -29
* Class 8: 30
* Class 9: -9

The model correctly identified the target class despite the relatively low confidence margin between competing classes.

## Latency Measurements

Average inference latency was measured over 100 executions.

| Metric                 |         Value |
| ---------------------- | ------------: |
| Average inference time | 139,555.92 µs |
| Average inference time |     139.56 ms |

This corresponds to approximately:

* 7.16 inferences per second

### Comparison with MNIST

| Metric      |   MNIST | CIFAR-10 |
| ----------- | ------: | -------: |
| Arena usage |  9.2 KB |  33.5 KB |
| Latency     | 14.9 ms | 139.6 ms |

While runtime memory increased by approximately 3.7×, inference latency increased by approximately 9.4×.

This indicates that computational complexity grows more rapidly than memory consumption when moving from simple grayscale digit recognition to color image classification.

## Discussion

The CIFAR-10 experiment demonstrates that the ESP32-S3 is capable of executing moderately sized convolutional neural networks with approximately 71,000 parameters while maintaining acceptable inference times.

Several observations emerged:

1. Full INT8 quantization preserved model accuracy.
2. Runtime memory requirements remained modest relative to available PSRAM.
3. Inference latency increased substantially compared to the MNIST model.
4. Computational cost scaled faster than memory consumption.

These results suggest that memory capacity is unlikely to be the primary limiting factor for medium-sized CNNs on the ESP32-S3. Instead, inference latency is expected to become the dominant constraint as model complexity increases.

## Conclusion

A quantized CIFAR-10 classifier containing 70,986 parameters was successfully deployed and executed on the ESP32-S3. The model achieved 68.22% classification accuracy, required 76.2 KB of storage, consumed 33.5 KB of runtime tensor memory, and achieved an average inference latency of 139.6 ms.

The experiment establishes a second benchmark point for evaluating the trade-offs between model complexity, memory consumption, accuracy, and execution time. These results provide a foundation for the subsequent evaluation of MobileNet-based architectures specifically designed for efficient embedded inference.

# Step-by-step
TODO, analogous to MNIST