#include <cstdio>
#include <inttypes.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "mobilenet_v1_sample.h"

#ifndef MOBILENET_V1_MODEL
#define MOBILENET_V1_MODEL 025
#endif

#if MOBILENET_V1_MODEL == 025
#include "mobilenet_v1_025_int8.h"
#define MOBILENET_MODEL_DATA mobilenet_v1_025_int8_tflite
#define MOBILENET_MODEL_LEN mobilenet_v1_025_int8_tflite_len
#define MOBILENET_MODEL_LABEL "025"
#elif MOBILENET_V1_MODEL == 050
#include "mobilenet_v1_050_int8.h"
#define MOBILENET_MODEL_DATA mobilenet_v1_050_int8_tflite
#define MOBILENET_MODEL_LEN mobilenet_v1_050_int8_tflite_len
#define MOBILENET_MODEL_LABEL "050"
#elif MOBILENET_V1_MODEL == 075
#include "mobilenet_v1_075_int8.h"
#define MOBILENET_MODEL_DATA mobilenet_v1_075_int8_tflite
#define MOBILENET_MODEL_LEN mobilenet_v1_075_int8_tflite_len
#define MOBILENET_MODEL_LABEL "075"
#elif MOBILENET_V1_MODEL == 100
#include "mobilenet_v1_100_int8.h"
#define MOBILENET_MODEL_DATA mobilenet_v1_100_int8_tflite
#define MOBILENET_MODEL_LEN mobilenet_v1_100_int8_tflite_len
#define MOBILENET_MODEL_LABEL "100"
#else
#error "Invalid MOBILENET_V1_MODEL value. Use: 025, 050, 075, or 100."
#endif

constexpr size_t kTensorArenaSize = 1200 * 1024;

extern "C" void app_main()
{
    printf("\n");
    printf("=================================\n");
    printf(" MobileNetV1 %s TensorFlow Lite Micro Test\n",
           MOBILENET_MODEL_LABEL);
    printf("=================================\n");

    printf("Model size: %u bytes\n",
           MOBILENET_MODEL_LEN);

    printf("PSRAM free before arena: %u bytes\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    printf("Largest PSRAM block: %u bytes\n",
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));

    // Load TFLite model
    const tflite::Model *model =
        tflite::GetModel(MOBILENET_MODEL_DATA);

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

    tflite::MicroMutableOpResolver<5> resolver;

    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddMean();
    resolver.AddSoftmax();

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
        96 * 96 * 3);

    const int runs = 1;

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