#ifndef ESP32_INFERENCE_HPP
#define ESP32_INFERENCE_HPP
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

class MemoryArena {
private:
    uint8_t* buffer;
    size_t   cap;
    size_t   offset;
    bool     owns_memory;
public:
    // Static buffer constructor - zero dynamic heap allocations
    MemoryArena(uint8_t* static_buf, size_t total_size) {
        buffer      = static_buf;
        cap         = total_size;
        offset      = 0;
        owns_memory = false;
    }
    // Dynamic buffer constructor (fallback)
    explicit MemoryArena(size_t total_size) {
        buffer      = (uint8_t*)malloc(total_size);
        cap         = total_size;
        offset      = 0;
        owns_memory = true;
    }
    ~MemoryArena() {
        if (owns_memory && buffer) free(buffer);
    }
    void* alloc(size_t sz) {
        sz = (sz + 3u) & ~3u;
        if (offset + sz > cap) return nullptr;
        void* ptr = buffer + offset;
        offset   += sz;
        return ptr;
    }
    void   reset()       { offset = 0; }
    bool   is_valid()    const { return buffer != nullptr; }
    size_t used()        const { return offset; }
    size_t capacity()    const { return cap; }
    size_t remaining()   const { return cap - offset; }
};

#if defined(ESP8266) || defined(ESP8266_BOARD)
#include <Arduino.h>
static inline void llm_optimistic_yield(uint32_t every_n_ops = 256) {
    static uint32_t counter = 0;
    if (++counter >= every_n_ops) {
        counter = 0;
        ESP.wdtFeed();
        optimistic_yield(1000);
        yield();
    }
}
#elif defined(ESP32)
#include <Arduino.h>
static inline void llm_optimistic_yield(uint32_t every_n_ops = 512) {
    static uint32_t counter = 0;
    if (++counter >= every_n_ops) {
        counter = 0;
        yield();
    }
}
#else
static inline void llm_optimistic_yield(uint32_t every_n_ops = 1000) {}
#endif

static inline float dot_f32(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void matmul_bitnet_ternary(
    const uint8_t* W_packed,
    const float*   scales,
    const float*   x,
    float*         y,
    int            out_features,
    int            in_features,
    int            group_size
) {
    const int n_groups = (in_features + group_size - 1) / group_size;
    const int k_bytes  = (in_features + 3) / 4;
    
    // 1. Quantize x to int8
    int8_t x_q[1024]; // Safe for our max in_features of ~128
    float max_abs = 1e-5f;
    for (int i = 0; i < in_features; ++i) {
        float ax = fabsf(x[i]);
        if (ax > max_abs) max_abs = ax;
    }
    float x_scale = max_abs / 127.0f;
    float inv_x_scale = 1.0f / x_scale;
    for (int i = 0; i < in_features; ++i) {
        float v = roundf(x[i] * inv_x_scale);
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        x_q[i] = (int8_t)v;
    }

    // 2. Matrix multiplication with ternary weights
    // Mapping: 2 -> -1, 0 -> 0, 1 -> 1
    const int8_t ternary_map[4] = {0, 1, -1, 0};

    for (int i = 0; i < out_features; ++i) {
        llm_optimistic_yield(64);
        const uint8_t* w_row = W_packed + (size_t)i * k_bytes;
        const float*   s_row = scales   + (size_t)i * n_groups;
        
        float acc_float = 0.0f;
        
        for (int g = 0; g < n_groups; ++g) {
            int group_start = g * group_size;
            int group_end = group_start + group_size;
            if (group_end > in_features) group_end = in_features;
            
            int32_t group_acc = 0;
            for (int j = group_start; j < group_end; ++j) {
                int byte_idx = j / 4;
                int bit_shift = (j % 4) * 2;
                uint8_t packed_val = (w_row[byte_idx] >> bit_shift) & 0x03;
                int8_t w = ternary_map[packed_val];
                group_acc += w * x_q[j];
            }
            acc_float += (float)group_acc * s_row[g];
        }
        y[i] = acc_float * x_scale;
    }
}

void rms_norm(const float* x, float* y, const float* gamma, int n, float eps = 1e-5f) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv_rms * gamma[i];
}

void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; ++i) x[i] /= sum;
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }
void swiglu(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = silu(gate[i]) * up[i];
}

void apply_rope_row(float* x, const float* cos_row, const float* sin_row, int head_dim) {
    int half = head_dim / 2;
    float tmp[64];
    for (int i = 0; i < half; ++i) {
        float x0 = x[2 * i];
        float x1 = x[2 * i + 1];
        float c = cos_row[i];
        float s = sin_row[i];
        tmp[i]        = x0 * c - x1 * s;
        tmp[half + i] = x0 * s + x1 * c;
    }
    memcpy(x, tmp, head_dim * sizeof(float));
}

int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}
#endif
