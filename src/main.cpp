#include <Arduino.h>
#include <string.h>
#include "inference.hpp"
#include "model_weights.hpp"

// ── Architecture constants (must match training config) ────────────────────
static constexpr int N_EMBD          = 128;
static constexpr int N_HEAD          = 4;
static constexpr int N_LAYER         = 2;
static constexpr int HEAD_DIM        = N_EMBD / N_HEAD;   // 32
static constexpr int MLP_HIDDEN      = 344;               // int(2/3*4*128) rounded up to mult-of-4
static constexpr int GRP             = 64;                // INT4 group size
static constexpr int INFER_CTX       = 32;                // context window for on-device inference
static constexpr int MAX_GEN_TOKENS  = 80;
static constexpr float TEMPERATURE   = 0.4f;

// ── Arena ──────────────────────────────────────────────────────────────────
// Memory breakdown for a full forward pass at INFER_CTX=32:
//   x         : INFER_CTX * N_EMBD * 4 = 16 384 B
//   kbuf/vbuf : INFER_CTX * N_EMBD * 4 = 16 384 B each
//   xnorm     : N_EMBD * 4             =    512 B
//   qkv_out   : 3*N_EMBD * 4           =  1 536 B
//   attn_out  : N_EMBD * 4             =    512 B
//   proj_out  : N_EMBD * 4             =    512 B
//   att       : INFER_CTX * 4          =    128 B
//   mlp_gate/up/hidden/out             =  5 504 B
//   logits    : 128 * 4                =    512 B
//   ───────────────────────────────────────────
//   Total                              ≈ 58 KB  →  budget 64 KB
static constexpr size_t ARENA_SIZE = 64 * 1024;

static MemoryArena* arena = nullptr;

// ── Working buffers (allocated once from arena at setup) ───────────────────
static float* g_x;          // [INFER_CTX * N_EMBD]
static float* g_kbuf;       // [INFER_CTX * N_EMBD]
static float* g_vbuf;       // [INFER_CTX * N_EMBD]
static float* g_xnorm;      // [N_EMBD]
static float* g_qkv_out;    // [3 * N_EMBD]
static float* g_attn_out;   // [N_EMBD]
static float* g_proj_out;   // [N_EMBD]
static float* g_att;        // [INFER_CTX]
static float* g_mlp_gate;   // [MLP_HIDDEN]
static float* g_mlp_up;     // [MLP_HIDDEN]
static float* g_mlp_hidden; // [MLP_HIDDEN]
static float* g_mlp_out;    // [N_EMBD]
static float* g_logits;     // [VOCAB_SIZE_MAX=128]

// ── Context buffer ─────────────────────────────────────────────────────────
static uint8_t ctx_ids[INFER_CTX];
static int     ctx_len = 0;

// ── Layer weight pointer table ─────────────────────────────────────────────
struct LayerW {
    const uint8_t* qkv_w;  const float* qkv_s;  const float* qkv_z;
    const uint8_t* proj_w; const float* proj_s; const float* proj_z;
    const uint8_t* gate_w; const float* gate_s; const float* gate_z;
    const uint8_t* up_w;   const float* up_s;   const float* up_z;
    const uint8_t* down_w; const float* down_s; const float* down_z;
    const float* ln1_g; const float* ln1_b;
    const float* ln2_g; const float* ln2_b;
};

static const LayerW g_layers[N_LAYER] = {
    {
        l0_attn_qkv_weights,      l0_attn_qkv_scales,      l0_attn_qkv_zp,
        l0_attn_proj_weights,     l0_attn_proj_scales,     l0_attn_proj_zp,
        l0_mlp_gate_proj_weights, l0_mlp_gate_proj_scales, l0_mlp_gate_proj_zp,
        l0_mlp_up_proj_weights,   l0_mlp_up_proj_scales,   l0_mlp_up_proj_zp,
        l0_mlp_down_proj_weights, l0_mlp_down_proj_scales, l0_mlp_down_proj_zp,
        l0_ln1_gamma, l0_ln1_beta,
        l0_ln2_gamma, l0_ln2_beta,
    },
    {
        l1_attn_qkv_weights,      l1_attn_qkv_scales,      l1_attn_qkv_zp,
        l1_attn_proj_weights,     l1_attn_proj_scales,     l1_attn_proj_zp,
        l1_mlp_gate_proj_weights, l1_mlp_gate_proj_scales, l1_mlp_gate_proj_zp,
        l1_mlp_up_proj_weights,   l1_mlp_up_proj_scales,   l1_mlp_up_proj_zp,
        l1_mlp_down_proj_weights, l1_mlp_down_proj_scales, l1_mlp_down_proj_zp,
        l1_ln1_gamma, l1_ln1_beta,
        l1_ln2_gamma, l1_ln2_beta,
    },
};

// ── Transformer forward pass ───────────────────────────────────────────────
// Runs the full causal transformer on `tokens[0..T-1]`.
// Writes logits for the LAST position into g_logits[0..vocab_size-1].
// Strategy: two passes per layer for attention to minimise SRAM:
//   Pass 1 — compute & cache K and V for all positions (apply RoPE to K).
//   Pass 2 — for each position compute Q (apply RoPE), then causal attention.
// The MLP runs for every position so the residual stream stays correct for
// the next layer's K/V projection.
static void transformer_forward(const uint8_t* tokens, int T) {
    // Step 1: embedding lookup → g_x[t*N_EMBD .. ]
    for (int t = 0; t < T; ++t) {
        const float* emb = tok_emb + (int)tokens[t] * N_EMBD;
        memcpy(g_x + t * N_EMBD, emb, N_EMBD * sizeof(float));
    }

    // Step 2: transformer layers
    for (int l = 0; l < N_LAYER; ++l) {
        const LayerW& lw = g_layers[l];

        // ── Attention pass 1: fill K and V for all positions ──────────────
        for (int t = 0; t < T; ++t) {
            layer_norm(g_x + t * N_EMBD, g_xnorm, lw.ln1_g, lw.ln1_b, N_EMBD);
            matmul_int4_f32(lw.qkv_w, lw.qkv_s, lw.qkv_z, g_xnorm, g_qkv_out, 3 * N_EMBD, N_EMBD, GRP);

            float* k_t = g_kbuf + t * N_EMBD;
            float* v_t = g_vbuf + t * N_EMBD;
            memcpy(k_t, g_qkv_out + N_EMBD,     N_EMBD * sizeof(float));
            memcpy(v_t, g_qkv_out + 2 * N_EMBD, N_EMBD * sizeof(float));

            // Apply RoPE to K (position t)
            const float* cos_t = rope_cos + t * (HEAD_DIM / 2);
            const float* sin_t = rope_sin + t * (HEAD_DIM / 2);
            for (int h = 0; h < N_HEAD; ++h)
                apply_rope_row(k_t + h * HEAD_DIM, cos_t, sin_t, HEAD_DIM);
        }

        // ── Attention pass 2: compute Q + causal attention per position ───
        for (int t = 0; t < T; ++t) {
            // Recompute xnorm and extract Q (re-running LN + QKV is cheaper than
            // storing a full T×N_EMBD Q buffer)
            layer_norm(g_x + t * N_EMBD, g_xnorm, lw.ln1_g, lw.ln1_b, N_EMBD);
            matmul_int4_f32(lw.qkv_w, lw.qkv_s, lw.qkv_z, g_xnorm, g_qkv_out, 3 * N_EMBD, N_EMBD, GRP);

            // Apply RoPE to Q (position t)
            const float* cos_t = rope_cos + t * (HEAD_DIM / 2);
            const float* sin_t = rope_sin + t * (HEAD_DIM / 2);
            for (int h = 0; h < N_HEAD; ++h)
                apply_rope_row(g_qkv_out + h * HEAD_DIM, cos_t, sin_t, HEAD_DIM);

            // Multi-head causal attention (sees positions 0..t)
            memset(g_attn_out, 0, N_EMBD * sizeof(float));
            const float scale = 1.0f / sqrtf((float)HEAD_DIM);

            for (int h = 0; h < N_HEAD; ++h) {
                const float* q_h = g_qkv_out + h * HEAD_DIM;

                for (int s = 0; s <= t; ++s)
                    g_att[s] = dot_f32(q_h, g_kbuf + s * N_EMBD + h * HEAD_DIM, HEAD_DIM) * scale;

                softmax(g_att, t + 1);

                float* out_h = g_attn_out + h * HEAD_DIM;
                for (int s = 0; s <= t; ++s) {
                    const float a = g_att[s];
                    const float* v_h = g_vbuf + s * N_EMBD + h * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; ++d) out_h[d] += a * v_h[d];
                }
            }

            // Output projection + residual
            matmul_int4_f32(lw.proj_w, lw.proj_s, lw.proj_z, g_attn_out, g_proj_out, N_EMBD, N_EMBD, GRP);
            for (int d = 0; d < N_EMBD; ++d) g_x[t * N_EMBD + d] += g_proj_out[d];
        }

        // ── MLP for all positions ─────────────────────────────────────────
        for (int t = 0; t < T; ++t) {
            layer_norm(g_x + t * N_EMBD, g_xnorm, lw.ln2_g, lw.ln2_b, N_EMBD);
            matmul_int4_f32(lw.gate_w, lw.gate_s, lw.gate_z, g_xnorm, g_mlp_gate, MLP_HIDDEN, N_EMBD, GRP);
            matmul_int4_f32(lw.up_w,   lw.up_s,   lw.up_z,   g_xnorm, g_mlp_up,   MLP_HIDDEN, N_EMBD, GRP);
            swiglu(g_mlp_gate, g_mlp_up, g_mlp_hidden, MLP_HIDDEN);
            matmul_int4_f32(lw.down_w, lw.down_s, lw.down_z, g_mlp_hidden, g_mlp_out, N_EMBD, MLP_HIDDEN, GRP);
            for (int d = 0; d < N_EMBD; ++d) g_x[t * N_EMBD + d] += g_mlp_out[d];
        }
    }

    // Step 3: final layernorm on last position only
    layer_norm(g_x + (T - 1) * N_EMBD, g_xnorm, ln_f_gamma, ln_f_beta, N_EMBD);

    // Step 4: head (tied embedding weights) — dot product with every vocab row
    for (int v = 0; v < (int)model_vocab_size; ++v)
        g_logits[v] = dot_f32(g_xnorm, tok_emb + v * N_EMBD, N_EMBD);
}

// ── Tokeniser ──────────────────────────────────────────────────────────────
static int encode_char(char c) {
    return (int)char_to_id[(uint8_t)c];   // 255 = unknown
}

static char decode_token(int id) {
    if (id < 0 || id >= (int)model_vocab_size) return '?';
    return (char)vocab_chars[id];
}

// ── Context management ─────────────────────────────────────────────────────
static void ctx_push(uint8_t id) {
    if (ctx_len >= INFER_CTX) {
        memmove(ctx_ids, ctx_ids + 1, (INFER_CTX - 1) * sizeof(uint8_t));
        ctx_len = INFER_CTX - 1;
    }
    ctx_ids[ctx_len++] = id;
}

static void ctx_push_str(const char* s) {
    for (; *s; ++s) {
        int id = encode_char(*s);
        if (id != 255) ctx_push((uint8_t)id);
    }
}

// ── Token sampling ─────────────────────────────────────────────────────────
static int sample_next(float temp) {
    if (ctx_len == 0) return 0;
    transformer_forward(ctx_ids, ctx_len);

    if (temp <= 0.0f) return argmax(g_logits, (int)model_vocab_size);

    // Temperature scaling + softmax + greedy
    // (greedy after softmax == argmax of raw logits, but temp changes ranking)
    int best = 0;
    float best_scaled = g_logits[0] / temp;
    for (int i = 1; i < (int)model_vocab_size; ++i) {
        float s = g_logits[i] / temp;
        if (s > best_scaled) { best_scaled = s; best = i; }
    }
    return best;
}

// ── Chat ───────────────────────────────────────────────────────────────────
static void run_chat() {
    Serial.print("\nUser: ");

    // Read until newline (handles \r\n from any terminal)
    char input[INFER_CTX + 1];
    int  ilen = 0;
    while (true) {
        if (!Serial.available()) continue;
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') { if (ilen > 0) break; continue; }
        if (ilen < (int)sizeof(input) - 1) { input[ilen++] = c; Serial.print(c); }
    }
    input[ilen] = '\0';
    Serial.println();

    if (ilen == 0) return;

    // Build prompt in context
    ctx_len = 0;
    char prompt[INFER_CTX * 2];
    snprintf(prompt, sizeof(prompt), "User: %s\nBot:", input);
    ctx_push_str(prompt);

    // Generate
    Serial.print("Bot: ");
    unsigned long t_start = millis();

    for (int i = 0; i < MAX_GEN_TOKENS; ++i) {
        unsigned long step_t = micros();
        int next_id  = sample_next(TEMPERATURE);
        char next_ch = decode_token(next_id);
        unsigned long step_ms = (micros() - step_t) / 1000UL;

        if (next_ch == '\n' && i > 3) break;
        if (next_ch == '\0' || next_ch == '\x01') break;

        Serial.print(next_ch);
        ctx_push((uint8_t)next_id);

        // Yield to watchdog every token
        yield();
    }

    unsigned long elapsed = millis() - t_start;
    Serial.println();
    Serial.printf("[%lu ms total]\n", elapsed);
}

// ── Helpers ────────────────────────────────────────────────────────────────
static void print_model_info() {
    Serial.printf("  Vocab  : %u chars\n",   (unsigned)model_vocab_size);
    Serial.printf("  n_embd : %d\n",          N_EMBD);
    Serial.printf("  n_head : %d\n",          N_HEAD);
    Serial.printf("  n_layer: %d\n",          N_LAYER);
    Serial.printf("  ctx    : %d tokens\n",   INFER_CTX);
    Serial.printf("  mlp_h  : %d\n",          MLP_HIDDEN);
    Serial.printf("  flash  : %u bytes\n",    (unsigned)model_weights_len);
}

static bool alloc_buffers() {
    g_x          = (float*)arena->alloc(INFER_CTX * N_EMBD     * sizeof(float));
    g_kbuf       = (float*)arena->alloc(INFER_CTX * N_EMBD     * sizeof(float));
    g_vbuf       = (float*)arena->alloc(INFER_CTX * N_EMBD     * sizeof(float));
    g_xnorm      = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_qkv_out    = (float*)arena->alloc(3 * N_EMBD             * sizeof(float));
    g_attn_out   = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_proj_out   = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_att        = (float*)arena->alloc(INFER_CTX              * sizeof(float));
    g_mlp_gate   = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_up     = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_hidden = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_out    = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_logits     = (float*)arena->alloc(128                    * sizeof(float));
    return g_x && g_kbuf && g_vbuf && g_xnorm && g_qkv_out &&
           g_attn_out && g_proj_out && g_att && g_mlp_gate &&
           g_mlp_up && g_mlp_hidden && g_mlp_out && g_logits;
}

// ── Setup ──────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(2000);

    Serial.println("\n╔══════════════════════════╗");
    Serial.println("║      ESP-GPT  v1.0       ║");
    Serial.println("║  INT4 Chatbot on ESP32   ║");
    Serial.println("╚══════════════════════════╝");

    Serial.printf("Free heap: %u B\n", (unsigned)ESP.getFreeHeap());

    arena = new MemoryArena(ARENA_SIZE);
    if (!arena->is_valid()) {
        Serial.printf("[FAIL] Arena alloc failed (need %u KB, free %u B)\n",
                      ARENA_SIZE / 1024, (unsigned)ESP.getFreeHeap());
        return;
    }
    Serial.printf("[OK] Arena: %u KB allocated\n", ARENA_SIZE / 1024);

    if (!alloc_buffers()) {
        Serial.println("[FAIL] Working buffer alloc failed");
        return;
    }
    Serial.printf("[OK] Buffers: %u / %u B used\n",
                  (unsigned)arena->used(), (unsigned)arena->capacity());

    print_model_info();
    Serial.printf("Free heap after init: %u B\n\n", (unsigned)ESP.getFreeHeap());
    Serial.println("Type your message and press Enter.");
    Serial.println("──────────────────────────────────");
}

// ── Loop ───────────────────────────────────────────────────────────────────
void loop() {
    if (!arena || !arena->is_valid() || !g_x) {
        delay(5000);
        return;
    }
    run_chat();
}
