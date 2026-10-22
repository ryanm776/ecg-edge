#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <algorithm> // For std::clamp and std::min
#include <cmath>     // For roundf
#include "model_data.h"
#include "threshold.h"

// Explicit forward declaration for g_model array
extern const unsigned char g_model[];

// 1. Force 16-byte alignment to prevent ESP32 memory access panics
constexpr int kTensorArenaSize = 16 * 1024; // 16 KB safety margin
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Global Pointers
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::ErrorReporter* error_reporter = nullptr;
TfLiteTensor* spec_input_tensor = nullptr;
TfLiteTensor* rr_input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// 2. Safe Quantization Routine
int8_t quantize_float_to_int8(float value, float scale, int32_t zero_point) {
    if (scale <= 0.0f) return static_cast<int8_t>(zero_point); 
    int32_t quantized = static_cast<int32_t>(roundf(value / scale)) + zero_point;
    // Explicitly pass <int32_t> to force template type alignment
    return static_cast<int8_t>(std::clamp<int32_t>(quantized, -128, 127));
}

// 3. Safe Dequantization Routine
float dequantize_int8_to_float(int8_t value, float scale, int32_t zero_point) {
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

// 4. Tensor size helper, used to bind inputs by shape rather than by index
constexpr int kRrFeatureCount = 5;

int tensor_element_count(const TfLiteTensor* tensor) {
    int count = 1;
    for (int d = 0; d < tensor->dims->size; d++) {
        count *= tensor->dims->data[d];
    }
    return count;
}

void setup() {
    Serial.begin(115200);
    // Wait up to 3 seconds for Native USB CDC Serial on ESP32-S3
    uint32_t start_ms = millis();
    while (!Serial && (millis() - start_ms < 3000));
    
    pinMode(LED_BUILTIN, OUTPUT);

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter, "Model schema mismatch!");
        return;
    }

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed! Increase kTensorArenaSize.");
        return;
    }

    // Bind inputs by element count, not index: the converter does not preserve the Keras
    // input order (the current flatbuffer has rr_input at index 0, spectrogram at index 1).
    for (size_t i = 0; i < interpreter->inputs_size(); i++) {
        TfLiteTensor* t = interpreter->input(i);
        if (t && tensor_element_count(t) == kRrFeatureCount) {
            rr_input_tensor = t;
        } else if (t) {
            spec_input_tensor = t;
        }
    }
    output_tensor = interpreter->output(0);

    if (interpreter->inputs_size() != 2 || !spec_input_tensor || !rr_input_tensor || !output_tensor) {
        Serial.println("Error: Failed to bind input/output tensors!");
        spec_input_tensor = rr_input_tensor = output_tensor = nullptr;
        return;
    }

    Serial.println("TFLite Engine initialized safely on ESP32-S3!");
}

void loop() {
    if (!interpreter || !output_tensor) {
        delay(1000);
        return;
    }

    // --- DEMO DATA ---
    static float dummy_rr_features[5] = {0.80f, 0.82f, 0.81f, 0.98f, 0.97f};

    // --- STEP A: FEED R-R INPUTS (WITH BOUNDS CHECK) ---
    int rr_elements = std::min(static_cast<int>(rr_input_tensor->bytes), 5);
    for (int i = 0; i < rr_elements; i++) {
        rr_input_tensor->data.int8[i] = quantize_float_to_int8(
            dummy_rr_features[i], 
            rr_input_tensor->params.scale, 
            rr_input_tensor->params.zero_point
        );
    }

    // --- STEP B: FEED SPECTROGRAM INPUTS ---
    int total_spec_elements = spec_input_tensor->bytes; 
    for (int i = 0; i < total_spec_elements; i++) {
        spec_input_tensor->data.int8[i] = quantize_float_to_int8(
            0.5f, 
            spec_input_tensor->params.scale, 
            spec_input_tensor->params.zero_point
        );
    }

    // --- STEP C: RUN INFERENCE WITH BENCHMARK TIMING ---
    uint32_t start_time = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    uint32_t latency_us = micros() - start_time;

    if (invoke_status != kTfLiteOk) {
        Serial.println("Error: Inference execution failed!");
        delay(1000);
        return;
    }

    // --- STEP D: DEQUANTIZE OUTPUT ---
    int8_t raw_output = output_tensor->data.int8[0];
    float probability = dequantize_int8_to_float(
        raw_output, 
        output_tensor->params.scale, 
        output_tensor->params.zero_point
    );

    Serial.printf("Latency: %.2f ms | Abnormal Beat Probability: %.4f\n", latency_us / 1000.0f, probability);

    if (probability > kAbnormalThreshold) {
        Serial.println("ALERT: Abnormal Beat Detected!");
        digitalWrite(LED_BUILTIN, HIGH); 
    } else {
        Serial.println("Status: Normal Beat");
        digitalWrite(LED_BUILTIN, LOW);  
    }

    delay(1000); 
}