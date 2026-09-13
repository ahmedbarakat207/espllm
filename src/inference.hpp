#ifndef ESP_LLM_INFERENCE_HPP
#define ESP_LLM_INFERENCE_HPP
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
#elif defined(ESP32) || defined(ESP32S3_BOARD)
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

// IEEE-754 binary16 -> float32 (no FPU support needed). Used for group scales.
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else { // subnormal: normalize
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

// INT8 embedding row (2's complement in a uint8_t array) -> float.
static inline void dequant_emb_row(const uint8_t* qrow, float scale, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = (float)((int8_t)qrow[i]) * scale;
}

static inline float dot_emb_q(const float* x, const uint8_t* qrow, float scale, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i] * (float)((int8_t)qrow[i]);
    return s * scale;
}

#if defined(ESP8266) || defined(ESP8266_BOARD)
static const int8_t base3_lut[243][5] __attribute__((aligned(4))) = {
#else
static const int8_t base3_lut[243][5] PROGMEM __attribute__((aligned(4))) = {
#endif
    { 0,  0,  0,  0,  0}, { 1,  0,  0,  0,  0}, {-1,  0,  0,  0,  0},
    { 0,  1,  0,  0,  0}, { 1,  1,  0,  0,  0}, {-1,  1,  0,  0,  0},
    { 0, -1,  0,  0,  0}, { 1, -1,  0,  0,  0}, {-1, -1,  0,  0,  0},
    { 0,  0,  1,  0,  0}, { 1,  0,  1,  0,  0}, {-1,  0,  1,  0,  0},
    { 0,  1,  1,  0,  0}, { 1,  1,  1,  0,  0}, {-1,  1,  1,  0,  0},
    { 0, -1,  1,  0,  0}, { 1, -1,  1,  0,  0}, {-1, -1,  1,  0,  0},
    { 0,  0, -1,  0,  0}, { 1,  0, -1,  0,  0}, {-1,  0, -1,  0,  0},
    { 0,  1, -1,  0,  0}, { 1,  1, -1,  0,  0}, {-1,  1, -1,  0,  0},
    { 0, -1, -1,  0,  0}, { 1, -1, -1,  0,  0}, {-1, -1, -1,  0,  0},
    { 0,  0,  0,  1,  0}, { 1,  0,  0,  1,  0}, {-1,  0,  0,  1,  0},
    { 0,  1,  0,  1,  0}, { 1,  1,  0,  1,  0}, {-1,  1,  0,  1,  0},
    { 0, -1,  0,  1,  0}, { 1, -1,  0,  1,  0}, {-1, -1,  0,  1,  0},
    { 0,  0,  1,  1,  0}, { 1,  0,  1,  1,  0}, {-1,  0,  1,  1,  0},
    { 0,  1,  1,  1,  0}, { 1,  1,  1,  1,  0}, {-1,  1,  1,  1,  0},
    { 0, -1,  1,  1,  0}, { 1, -1,  1,  1,  0}, {-1, -1,  1,  1,  0},
    { 0,  0, -1,  1,  0}, { 1,  0, -1,  1,  0}, {-1,  0, -1,  1,  0},
    { 0,  1, -1,  1,  0}, { 1,  1, -1,  1,  0}, {-1,  1, -1,  1,  0},
    { 0, -1, -1,  1,  0}, { 1, -1, -1,  1,  0}, {-1, -1, -1,  1,  0},
    { 0,  0,  0, -1,  0}, { 1,  0,  0, -1,  0}, {-1,  0,  0, -1,  0},
    { 0,  1,  0, -1,  0}, { 1,  1,  0, -1,  0}, {-1,  1,  0, -1,  0},
    { 0, -1,  0, -1,  0}, { 1, -1,  0, -1,  0}, {-1, -1,  0, -1,  0},
    { 0,  0,  1, -1,  0}, { 1,  0,  1, -1,  0}, {-1,  0,  1, -1,  0},
    { 0,  1,  1, -1,  0}, { 1,  1,  1, -1,  0}, {-1,  1,  1, -1,  0},
    { 0, -1,  1, -1,  0}, { 1, -1,  1, -1,  0}, {-1, -1,  1, -1,  0},
    { 0,  0, -1, -1,  0}, { 1,  0, -1, -1,  0}, {-1,  0, -1, -1,  0},
    { 0,  1, -1, -1,  0}, { 1,  1, -1, -1,  0}, {-1,  1, -1, -1,  0},
    { 0, -1, -1, -1,  0}, { 1, -1, -1, -1,  0}, {-1, -1, -1, -1,  0},
    { 0,  0,  0,  0,  1}, { 1,  0,  0,  0,  1}, {-1,  0,  0,  0,  1},
    { 0,  1,  0,  0,  1}, { 1,  1,  0,  0,  1}, {-1,  1,  0,  0,  1},
    { 0, -1,  0,  0,  1}, { 1, -1,  0,  0,  1}, {-1, -1,  0,  0,  1},
    { 0,  0,  1,  0,  1}, { 1,  0,  1,  0,  1}, {-1,  0,  1,  0,  1},
    { 0,  1,  1,  0,  1}, { 1,  1,  1,  0,  1}, {-1,  1,  1,  0,  1},
    { 0, -1,  1,  0,  1}, { 1, -1,  1,  0,  1}, {-1, -1,  1,  0,  1},
    { 0,  0, -1,  0,  1}, { 1,  0, -1,  0,  1}, {-1,  0, -1,  0,  1},
    { 0,  1, -1,  0,  1}, { 1,  1, -1,  0,  1}, {-1,  1, -1,  0,  1},
    { 0, -1, -1,  0,  1}, { 1, -1, -1,  0,  1}, {-1, -1, -1,  0,  1},
    { 0,  0,  0,  1,  1}, { 1,  0,  0,  1,  1}, {-1,  0,  0,  1,  1},
    { 0,  1,  0,  1,  1}, { 1,  1,  0,  1,  1}, {-1,  1,  0,  1,  1},
    { 0, -1,  0,  1,  1}, { 1, -1,  0,  1,  1}, {-1, -1,  0,  1,  1},
    { 0,  0,  1,  1,  1}, { 1,  0,  1,  1,  1}, {-1,  0,  1,  1,  1},
    { 0,  1,  1,  1,  1}, { 1,  1,  1,  1,  1}, {-1,  1,  1,  1,  1},
    { 0, -1,  1,  1,  1}, { 1, -1,  1,  1,  1}, {-1, -1,  1,  1,  1},
    { 0,  0, -1,  1,  1}, { 1,  0, -1,  1,  1}, {-1,  0, -1,  1,  1},
    { 0,  1, -1,  1,  1}, { 1,  1, -1,  1,  1}, {-1,  1, -1,  1,  1},
    { 0, -1, -1,  1,  1}, { 1, -1, -1,  1,  1}, {-1, -1, -1,  1,  1},
    { 0,  0,  0, -1,  1}, { 1,  0,  0, -1,  1}, {-1,  0,  0, -1,  1},
    { 0,  1,  0, -1,  1}, { 1,  1,  0, -1,  1}, {-1,  1,  0, -1,  1},
    { 0, -1,  0, -1,  1}, { 1, -1,  0, -1,  1}, {-1, -1,  0, -1,  1},
    { 0,  0,  1, -1,  1}, { 1,  0,  1, -1,  1}, {-1,  0,  1, -1,  1},
    { 0,  1,  1, -1,  1}, { 1,  1,  1, -1,  1}, {-1,  1,  1, -1,  1},
    { 0, -1,  1, -1,  1}, { 1, -1,  1, -1,  1}, {-1, -1,  1, -1,  1},
    { 0,  0, -1, -1,  1}, { 1,  0, -1, -1,  1}, {-1,  0, -1, -1,  1},
    { 0,  1, -1, -1,  1}, { 1,  1, -1, -1,  1}, {-1,  1, -1, -1,  1},
    { 0, -1, -1, -1,  1}, { 1, -1, -1, -1,  1}, {-1, -1, -1, -1,  1},
    { 0,  0,  0,  0, -1}, { 1,  0,  0,  0, -1}, {-1,  0,  0,  0, -1},
    { 0,  1,  0,  0, -1}, { 1,  1,  0,  0, -1}, {-1,  1,  0,  0, -1},
    { 0, -1,  0,  0, -1}, { 1, -1,  0,  0, -1}, {-1, -1,  0,  0, -1},
    { 0,  0,  1,  0, -1}, { 1,  0,  1,  0, -1}, {-1,  0,  1,  0, -1},
    { 0,  1,  1,  0, -1}, { 1,  1,  1,  0, -1}, {-1,  1,  1,  0, -1},
    { 0, -1,  1,  0, -1}, { 1, -1,  1,  0, -1}, {-1, -1,  1,  0, -1},
    { 0,  0, -1,  0, -1}, { 1,  0, -1,  0, -1}, {-1,  0, -1,  0, -1},
    { 0,  1, -1,  0, -1}, { 1,  1, -1,  0, -1}, {-1,  1, -1,  0, -1},
    { 0, -1, -1,  0, -1}, { 1, -1, -1,  0, -1}, {-1, -1, -1,  0, -1},
    { 0,  0,  0,  1, -1}, { 1,  0,  0,  1, -1}, {-1,  0,  0,  1, -1},
    { 0,  1,  0,  1, -1}, { 1,  1,  0,  1, -1}, {-1,  1,  0,  1, -1},
    { 0, -1,  0,  1, -1}, { 1, -1,  0,  1, -1}, {-1, -1,  0,  1, -1},
    { 0,  0,  1,  1, -1}, { 1,  0,  1,  1, -1}, {-1,  0,  1,  1, -1},
    { 0,  1,  1,  1, -1}, { 1,  1,  1,  1, -1}, {-1,  1,  1,  1, -1},
    { 0, -1,  1,  1, -1}, { 1, -1,  1,  1, -1}, {-1, -1,  1,  1, -1},
    { 0,  0, -1,  1, -1}, { 1,  0, -1,  1, -1}, {-1,  0, -1,  1, -1},
    { 0,  1, -1,  1, -1}, { 1,  1, -1,  1, -1}, {-1,  1, -1,  1, -1},
    { 0, -1, -1,  1, -1}, { 1, -1, -1,  1, -1}, {-1, -1, -1,  1, -1},
    { 0,  0,  0, -1, -1}, { 1,  0,  0, -1, -1}, {-1,  0,  0, -1, -1},
    { 0,  1,  0, -1, -1}, { 1,  1,  0, -1, -1}, {-1,  1,  0, -1, -1},
    { 0, -1,  0, -1, -1}, { 1, -1,  0, -1, -1}, {-1, -1,  0, -1, -1},
    { 0,  0,  1, -1, -1}, { 1,  0,  1, -1, -1}, {-1,  0,  1, -1, -1},
    { 0,  1,  1, -1, -1}, { 1,  1,  1, -1, -1}, {-1,  1,  1, -1, -1},
    { 0, -1,  1, -1, -1}, { 1, -1,  1, -1, -1}, {-1, -1,  1, -1, -1},
    { 0,  0, -1, -1, -1}, { 1,  0, -1, -1, -1}, {-1,  0, -1, -1, -1},
    { 0,  1, -1, -1, -1}, { 1,  1, -1, -1, -1}, {-1,  1, -1, -1, -1},
    { 0, -1, -1, -1, -1}, { 1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1}
};

void matmul_bitnet_ternary(
    const uint8_t* W_packed,
    const uint16_t* scales,
    const float*   x,
    float*         y,
    int            out_features,
    int            in_features,
    int            group_size
) {
    const int n_groups = (in_features + group_size - 1) / group_size;
    const int bytes_per_group = (group_size + 4) / 5;
    const int k_bytes  = n_groups * bytes_per_group;
    
    // 1. Quantize x to int8 (stack fast-path, heap fallback for wide layers)
    int8_t x_q_stack[1024];
    int8_t* x_q = x_q_stack;
    bool x_q_heap = false;
    if (in_features > 1024) {
        x_q = (int8_t*)malloc((size_t)in_features);
        if (!x_q) return; // OOM: leave y untouched, caller fails loudly downstream
        x_q_heap = true;
    }
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
    for (int i = 0; i < out_features; ++i) {
        llm_optimistic_yield(64);
        const uint8_t* w_row = W_packed + (size_t)i * k_bytes;
        const uint16_t* s_row = scales   + (size_t)i * n_groups;
        
        float acc_float = 0.0f;
        
        for (int g = 0; g < n_groups; ++g) {
            int group_start = g * group_size;
            int group_end = group_start + group_size;
            if (group_end > in_features) group_end = in_features;
            int group_len = group_end - group_start;

            const uint8_t* g_bytes = w_row + (size_t)g * bytes_per_group;
            const int8_t* g_x = x_q + group_start;
            
            int32_t group_acc = 0;
            int full_chunks = group_len / 5;
            for (int c = 0; c < full_chunks; ++c) {
                const int8_t* w5 = base3_lut[g_bytes[c]];
                const int8_t* x5 = g_x + c * 5;
                group_acc += (int32_t)w5[0] * x5[0]
                           + (int32_t)w5[1] * x5[1]
                           + (int32_t)w5[2] * x5[2]
                           + (int32_t)w5[3] * x5[3]
                           + (int32_t)w5[4] * x5[4];
            }
            int rem = group_len % 5;
            if (rem > 0) {
                const int8_t* w5 = base3_lut[g_bytes[full_chunks]];
                const int8_t* x_rem = g_x + full_chunks * 5;
                for (int r = 0; r < rem; ++r) {
                    group_acc += (int32_t)w5[r] * x_rem[r];
                }
            }
            acc_float += (float)group_acc * half_to_float(s_row[g]);
        }
        y[i] = acc_float * x_scale;
    }
    if (x_q_heap) free(x_q);
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
    float tmp[128]; // supports head_dim up to 128
    if (head_dim > 128) return;
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
