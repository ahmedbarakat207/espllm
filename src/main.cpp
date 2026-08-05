#include <Arduino.h>
#include "inference.hpp"
#include "model_weights.hpp"

MemoryArena* arena = nullptr;

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("Starting ESP-GPT Inference...");
    Serial.printf("Model weights size: %d bytes\n", model_weights_len);
    
    // Initialize memory arena (16 KB for activations)
    arena = new MemoryArena(16 * 1024);
    if (!arena->is_valid()) {
        Serial.println("Failed to allocate memory arena!");
        return;
    }
    Serial.println("Memory Arena Allocated: 16KB");
    
    int in_features = 128;
    int out_features = 128;
    int8_t* X = (int8_t*)arena->alloc(in_features * sizeof(int8_t));
    int32_t* Y = (int32_t*)arena->alloc(out_features * sizeof(int32_t));
    
    if (!X || !Y) {
        Serial.println("Failed to allocate activations in arena!");
        return;
    }
    
    for (int i = 0; i < in_features; i++) {
        X[i] = (i % 20) - 10;
    }
    
    Serial.println("Running inference...");
    unsigned long start_time = micros();
    
    matmul_4bit_x_8bit(model_weights, X, Y, out_features, in_features);
    
    unsigned long end_time = micros();
    
    Serial.printf("Inference time for %dx%d layer: %lu microseconds\n", 
                  in_features, out_features, (end_time - start_time));
    
    Serial.print("Output sample [0..4]: ");
    for (int i = 0; i < 5; i++) {
        Serial.printf("%d ", Y[i]);
    }
    Serial.println();
    
    arena->reset();
}

void loop() {
    delay(1000);
}
