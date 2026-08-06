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
public:
    explicit MemoryArena(size_t total_size) {
        buffer = (uint8_t*)malloc(total_size);
        cap    = total_size;
        offset = 0;
    }
    ~MemoryArena() { if (buffer) free(buffer); }

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

static inline float dot_f32(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static inline void unpack_nibbles(uint8_t packed, uint8_t& lo, uint8_t& hi) {
    lo = packed & 0x0Fu;
    hi = (packed >> 4) & 0x0Fu;
}

void matmul_int4_f32(
    const uint8_t* W_packed,
    const float*   scales,
    const float*   zp,
    const float*   x,
    float*         y,
    int            out_features,
    int            in_features,
    int            group_size
) {
    const int n_groups = (in_features + group_size - 1) / group_size;
    const int k_bytes  = (in_features + 1) / 2;

    for (int i = 0; i < out_features; ++i) {
        float acc = 0.0f;
        const uint8_t* w_row = W_packed + (size_t)i * k_bytes;
        const float*   s_row = scales   + (size_t)i * n_groups;
        const float*   z_row = zp       + (size_t)i * n_groups;

        for (int j = 0; j < k_bytes; ++j) {
            uint8_t lo, hi;
            unpack_nibbles(w_row[j], lo, hi);
            int col0 = j * 2;
            int col1 = col0 + 1;
            int g0   = col0 / group_size;
            acc += ((float)lo - z_row[g0]) * s_row[g0] * x[col0];
            if (col1 < in_features) {
                int g1 = col1 / group_size;
                acc += ((float)hi - z_row[g1]) * s_row[g1] * x[col1];
            }
        }
        y[i] = acc;
    }
}

void layer_norm(const float* x, float* y, const float* gamma, const float* beta, int n, float eps = 1e-5f) {
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; ++i) y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
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
    for (int i = 0; i < half; ++i) {
        float x0 = x[2 * i];
        float x1 = x[2 * i + 1];
        x[2 * i]     = x0 * cos_row[i] - x1 * sin_row[i];
        x[2 * i + 1] = x0 * sin_row[i] + x1 * cos_row[i];
    }
}

int argmax(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}

#endif
