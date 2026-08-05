#ifndef ESP32_INFERENCE_HPP
#define ESP32_INFERENCE_HPP

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

class MemoryArena {
private:
    uint8_t* buffer;
    size_t size;
    size_t offset;

public:
    MemoryArena(size_t total_size) {
        buffer = (uint8_t*)malloc(total_size);
        size = total_size;
        offset = 0;
    }

    ~MemoryArena() {
        if (buffer) free(buffer);
    }

    void* alloc(size_t alloc_size) {
        alloc_size = (alloc_size + 3) & ~3;
        if (offset + alloc_size > size) {
            return nullptr; // Out of memory
        }
        void* ptr = buffer + offset;
        offset += alloc_size;
        return ptr;
    }

    void reset() {
        offset = 0;
    }
    
    bool is_valid() const { return buffer != nullptr; }
};
static const int8_t lut_4bit_to_8bit[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 
    -8, -7, -6, -5, -4, -3, -2, -1
};

inline void unpack_byte(uint8_t packed, int8_t& w0, int8_t& w1) {
    w0 = lut_4bit_to_8bit[packed & 0x0F];
    w1 = lut_4bit_to_8bit[(packed >> 4) & 0x0F];
}
void matmul_4bit_x_8bit(
    const uint8_t* W_packed, 
    const int8_t* X, 
    int32_t* Y, 
    int out_features, 
    int in_features
) {
    int k_bytes = in_features / 2;

    for (int i = 0; i < out_features; ++i) {
        int32_t sum = 0;
        const uint8_t* w_row = W_packed + (i * k_bytes);
        
        for (int j = 0; j < k_bytes; ++j) {
            uint8_t packed_val = w_row[j];
            
            int8_t w0 = lut_4bit_to_8bit[packed_val & 0x0F];
            int8_t w1 = lut_4bit_to_8bit[(packed_val >> 4) & 0x0F];
            
            int x_idx = j * 2;
            sum += w0 * X[x_idx];
            sum += w1 * X[x_idx + 1];
        }
        
        Y[i] = sum;
    }
}

void run_inference_layer(
    const uint8_t* layer_weights, 
    const int8_t* input_activations, 
    int32_t* output_activations,
    int in_features,
    int out_features
) {
    matmul_4bit_x_8bit(layer_weights, input_activations, output_activations, out_features, in_features);
}

#endif
