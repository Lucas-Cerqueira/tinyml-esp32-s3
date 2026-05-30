#include <cstdio>
#include <inttypes.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "cifar10_int8.h"
#include "cifar10_sample.h"

constexpr size_t kTensorArenaSize = 64 * 1024;

extern "C" void app_main()
{
    printf("\n");
    printf("=================================\n");
    printf(" CIFAR10 TensorFlow Lite Micro Test\n");
    printf("=================================\n");

    printf("Model size: %u bytes\n",
           cifar10_int8_tflite_len);

    printf("PSRAM free before arena: %u bytes\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    printf("Largest PSRAM block: %u bytes\n",
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));

    // Load TFLite model
    const tflite::Model *model =
        tflite::GetModel(cifar10_int8_tflite);

    if (!model)
    {
        printf("Failed to load model\n");
        return;
    }

    printf("Model loaded successfully\n");

    printf("Model schema version: %lu\n",
           (unsigned long)model->version());

    // Allocate tensor arena in PSRAM
    uint8_t *tensor_arena =
        static_cast<uint8_t *>(
            heap_caps_malloc(
                kTensorArenaSize,
                MALLOC_CAP_SPIRAM)); // MALLOC_CAP_INTERNAL

    if (!tensor_arena)
    {
        printf("Failed to allocate tensor arena\n");
        return;
    }

    printf("Tensor arena allocated successfully\n");
    printf("Arena size: %u bytes\n",
           (unsigned)kTensorArenaSize);

    printf("PSRAM free after arena: %u bytes\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    printf("Largest PSRAM block after arena: %u bytes\n",
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));

    printf("\nReady for interpreter creation.\n");

    tflite::MicroMutableOpResolver<7> resolver;

    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddPack();
    resolver.AddShape();
    resolver.AddReshape();
    resolver.AddStridedSlice();
    resolver.AddFullyConnected();

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    TfLiteStatus allocation_status =
        interpreter.AllocateTensors();

    if (allocation_status != kTfLiteOk)
    {
        printf("AllocateTensors FAILED\n");
        return;
    }

    printf("AllocateTensors SUCCESS\n");

    printf("Arena used: %u bytes\n",
           (unsigned)interpreter.arena_used_bytes());

    TfLiteTensor *input =
        interpreter.input(0);

    memcpy(
        input->data.int8,
        test_image,
        3072);

    const int runs = 50;

    int64_t start = esp_timer_get_time();

    for (int i = 0; i < runs; i++)
    {
        interpreter.Invoke();
    }

    int64_t end = esp_timer_get_time();

    printf("Average inference: %.2f us\n",
           (double)(end - start) / runs);

    TfLiteTensor *output =
        interpreter.output(0);

    for (int i = 0; i < 10; i++)
    {
        printf("%d -> %d\n",
               i,
               output->data.int8[i]);
    }

    int best_idx = 0;
    int8_t best_score = output->data.int8[0];

    for (int i = 1; i < 10; i++)
    {
        if (output->data.int8[i] > best_score)
        {
            best_score = output->data.int8[i];
            best_idx = i;
        }
    }

    printf("Predicted class: %d  |  Expected class: %d\n",
           best_idx, test_label);
}