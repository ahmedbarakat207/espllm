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

struct NibblePair { float lo; float hi; };

static NibblePair unpack_lut[256];
static bool lut_init = false;
static void init_unpack_lut() {
    if (lut_init) return;
    for (int i = 0; i < 256; ++i) {
        unpack_lut[i].lo = (float)(i & 0x0F);
        unpack_lut[i].hi = (float)(i >> 4);
    }
    lut_init = true;
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
        init_unpack_lut();
        for (int j = 0; j < k_bytes; j += 4) {
            uint8_t p0 = w_row[j];
            uint8_t p1 = w_row[j+1];
            uint8_t p2 = w_row[j+2];
            uint8_t p3 = w_row[j+3];
            int col0 = j * 2;
            int g0 = col0 / group_size;
            int g1 = (col0 + 1) / group_size;
            int g2 = (col0 + 2) / group_size;
            int g3 = (col0 + 3) / group_size;
            int g4 = (col0 + 4) / group_size;
            int g5 = (col0 + 5) / group_size;
            int g6 = (col0 + 6) / group_size;
            int g7 = (col0 + 7) / group_size;
            NibblePair u0 = unpack_lut[p0];
            NibblePair u1 = unpack_lut[p1];
            NibblePair u2 = unpack_lut[p2];
            NibblePair u3 = unpack_lut[p3];
            float w0 = u0.lo - z_row[g0];
            float w1 = u0.hi - z_row[g1];
            float w2 = u1.lo - z_row[g2];
            float w3 = u1.hi - z_row[g3];
            float w4 = u2.lo - z_row[g4];
            float w5 = u2.hi - z_row[g5];
            float w6 = u3.lo - z_row[g6];
            float w7 = u3.hi - z_row[g7];
            acc += w0 * s_row[g0] * x[col0]     + w1 * s_row[g1] * x[col0 + 1]
                 + w2 * s_row[g2] * x[col0 + 2] + w3 * s_row[g3] * x[col0 + 3]
                 + w4 * s_row[g4] * x[col0 + 4] + w5 * s_row[g5] * x[col0 + 5]
                 + w6 * s_row[g6] * x[col0 + 6] + w7 * s_row[g7] * x[col0 + 7];
        }
        y[i] = acc;
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
