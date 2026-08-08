#include <Arduino.h>
#include <string.h>
#include "inference.hpp"
#include "model_weights.hpp"

static constexpr int N_EMBD          = 128;
static constexpr int N_HEAD          = 4;
static constexpr int N_KV_HEAD       = 1;
static constexpr int N_LAYER         = 6;
static constexpr int HEAD_DIM        = N_EMBD / N_HEAD;  
static constexpr int MLP_HIDDEN      = 128;              
static constexpr int N_EXPERTS       = 16;                
static constexpr int GRP             = 64;                
#if defined(ESP8266) || defined(ESP8266_BOARD)
static constexpr int INFER_CTX       = 24;                
static constexpr int MAX_GEN_TOKENS  = 50;
static constexpr float TEMPERATURE   = 0.0f;
static constexpr size_t ARENA_SIZE   = 46 * 1024;
#else
static constexpr int INFER_CTX       = 48;                
static constexpr int MAX_GEN_TOKENS  = 80;
static constexpr float TEMPERATURE   = 0.0f;
static constexpr size_t ARENA_SIZE   = 88 * 1024;
#endif

static MemoryArena* arena = nullptr;
static float* g_x;          
static float* g_kbuf;       
static float* g_vbuf;       
static float* g_xnorm;      
static float* g_qkv_out;   
static float* g_attn_out;   
static float* g_proj_out;   
static float* g_att;        
static float* g_mlp_gate;   
static float* g_mlp_up;     
static float* g_mlp_hidden; 
static float* g_mlp_out;    
static float* g_logits;     
static uint16_t ctx_ids[INFER_CTX];
static int     ctx_len = 0;
static int     ctx_pos = 0;

struct LayerW {
    const uint8_t* qkv_w;  const float* qkv_s;  const float* qkv_z;
    const uint8_t* proj_w; const float* proj_s; const float* proj_z;
    const float* router_w;
    const uint8_t* experts_gate_q;
    const float*   experts_gate_s;
    const float*   experts_gate_z;
    const uint8_t* experts_up_q;
    const float*   experts_up_s;
    const float*   experts_up_z;
    const uint8_t* experts_down_q;
    const float*   experts_down_s;
    const float*   experts_down_z;
    const float* ln1_g;
    const float* ln2_g;
};

#define LAYER_INIT(li) \
    { \
        l##li##_attn_qkv_weights,      l##li##_attn_qkv_scales,      l##li##_attn_qkv_zp, \
        l##li##_attn_proj_weights,     l##li##_attn_proj_scales,     l##li##_attn_proj_zp, \
        l##li##_router, \
        l##li##_experts_gate_weights,  l##li##_experts_gate_scales,  l##li##_experts_gate_zp, \
        l##li##_experts_up_weights,    l##li##_experts_up_scales,    l##li##_experts_up_zp, \
        l##li##_experts_down_weights,  l##li##_experts_down_scales,  l##li##_experts_down_zp, \
        l##li##_ln1_gamma, \
        l##li##_ln2_gamma, \
    }

static const LayerW g_layers[N_LAYER] = {
    LAYER_INIT(0), LAYER_INIT(1), LAYER_INIT(2),
    LAYER_INIT(3), LAYER_INIT(4), LAYER_INIT(5),
};

static void transformer_forward(int token, int pos) {
    const float* emb = tok_emb + token * N_EMBD;
    memcpy(g_x, emb, N_EMBD * sizeof(float));
    for (int l = 0; l < N_LAYER; ++l) {
        const LayerW& lw = g_layers[l];
        rms_norm(g_x, g_xnorm, lw.ln1_g, N_EMBD);
        matmul_int4_f32(lw.qkv_w, lw.qkv_s, lw.qkv_z, g_xnorm, g_qkv_out, N_EMBD + 2 * N_KV_HEAD * HEAD_DIM, N_EMBD, GRP);
        float* layer_kbuf = g_kbuf + l * (INFER_CTX * N_KV_HEAD * HEAD_DIM);
        float* layer_vbuf = g_vbuf + l * (INFER_CTX * N_KV_HEAD * HEAD_DIM);
        float* k_pos = layer_kbuf + pos * (N_KV_HEAD * HEAD_DIM);
        float* v_pos = layer_vbuf + pos * (N_KV_HEAD * HEAD_DIM);
        memcpy(k_pos, g_qkv_out + N_EMBD,                        N_KV_HEAD * HEAD_DIM * sizeof(float));
        memcpy(v_pos, g_qkv_out + N_EMBD + N_KV_HEAD * HEAD_DIM, N_KV_HEAD * HEAD_DIM * sizeof(float));
        const float* cos_t = rope_cos + pos * (HEAD_DIM / 2);
        const float* sin_t = rope_sin + pos * (HEAD_DIM / 2);
        for (int h = 0; h < N_HEAD; ++h) {
            apply_rope_row(g_qkv_out + h * HEAD_DIM, cos_t, sin_t, HEAD_DIM);
        }
        for (int h = 0; h < N_KV_HEAD; ++h) {
            apply_rope_row(k_pos + h * HEAD_DIM, cos_t, sin_t, HEAD_DIM);
        }
        memset(g_attn_out, 0, N_EMBD * sizeof(float));
        const float scale = 1.0f / sqrtf((float)HEAD_DIM);
        for (int h = 0; h < N_HEAD; ++h) {
            const float* q_h = g_qkv_out + h * HEAD_DIM;
            int kv_h = h / (N_HEAD / N_KV_HEAD);
            for (int s = 0; s <= pos; ++s)
                g_att[s] = dot_f32(q_h, layer_kbuf + s * (N_KV_HEAD * HEAD_DIM) + kv_h * HEAD_DIM, HEAD_DIM) * scale;
            softmax(g_att, pos + 1);
            float* out_h = g_attn_out + h * HEAD_DIM;
            for (int s = 0; s <= pos; ++s) {
                const float a = g_att[s];
                const float* v_h_ptr = layer_vbuf + s * (N_KV_HEAD * HEAD_DIM) + kv_h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) out_h[d] += a * v_h_ptr[d];
            }
        }
        matmul_int4_f32(lw.proj_w, lw.proj_s, lw.proj_z, g_attn_out, g_proj_out, N_EMBD, N_EMBD, GRP);
        for (int d = 0; d < N_EMBD; ++d) g_x[d] += g_proj_out[d];
        rms_norm(g_x, g_xnorm, lw.ln2_g, N_EMBD);
        int best_expert = 0;
        float best_score = -1e9f;
        for (int e = 0; e < N_EXPERTS; ++e) {
            float score = dot_f32(g_xnorm, lw.router_w + e * N_EMBD, N_EMBD);
            if (score > best_score) {
                best_score = score;
                best_expert = e;
            }
        }
        int gate_q_off = best_expert * MLP_HIDDEN * ((N_EMBD + 1) / 2);
        int gate_s_off = best_expert * MLP_HIDDEN * ((N_EMBD + GRP - 1) / GRP);
        int down_q_off = best_expert * N_EMBD * ((MLP_HIDDEN + 1) / 2);
        int down_s_off = best_expert * N_EMBD * ((MLP_HIDDEN + GRP - 1) / GRP);
        matmul_int4_f32(lw.experts_gate_q + gate_q_off, lw.experts_gate_s + gate_s_off, lw.experts_gate_z + gate_s_off, 
                        g_xnorm, g_mlp_gate, MLP_HIDDEN, N_EMBD, GRP);
        matmul_int4_f32(lw.experts_up_q + gate_q_off, lw.experts_up_s + gate_s_off, lw.experts_up_z + gate_s_off,   
                        g_xnorm, g_mlp_up, MLP_HIDDEN, N_EMBD, GRP);
        swiglu(g_mlp_gate, g_mlp_up, g_mlp_hidden, MLP_HIDDEN);
        matmul_int4_f32(lw.experts_down_q + down_q_off, lw.experts_down_s + down_s_off, lw.experts_down_z + down_s_off, 
                        g_mlp_hidden, g_mlp_out, N_EMBD, MLP_HIDDEN, GRP);
        for (int d = 0; d < N_EMBD; ++d) g_x[d] += g_mlp_out[d];
        yield();
    }
    rms_norm(g_x, g_xnorm, ln_f_gamma, N_EMBD);
    for (int v = 0; v < (int)model_vocab_size; ++v)
        g_logits[v] = dot_f32(g_xnorm, tok_emb + v * N_EMBD, N_EMBD);
}

static bool print_token(int id) {
    bool has_newline = false;
    if (id < 0 || id >= (int)model_vocab_size) return false;
    uint32_t start = pgm_read_dword(&model_vocab_offsets[id]);
    uint32_t end   = pgm_read_dword(&model_vocab_offsets[id + 1]);
    for (uint32_t i = start; i < end; ++i) {
        char c = (char)pgm_read_byte(&model_vocab_bytes[i]);
        Serial.print(c);
        if (c == '\n') has_newline = true;
    }
    return has_newline;
}

static void ctx_push(uint16_t id) {
    if (ctx_len >= INFER_CTX) {
        memmove(ctx_ids, ctx_ids + 1, (INFER_CTX - 1) * sizeof(uint16_t));
        ctx_len = INFER_CTX - 1;
        for (int l = 0; l < N_LAYER; ++l) {
            float* layer_kbuf = g_kbuf + l * (INFER_CTX * N_KV_HEAD * HEAD_DIM);
            float* layer_vbuf = g_vbuf + l * (INFER_CTX * N_KV_HEAD * HEAD_DIM);
            memmove(layer_kbuf, layer_kbuf + (N_KV_HEAD * HEAD_DIM), (INFER_CTX - 1) * (N_KV_HEAD * HEAD_DIM) * sizeof(float));
            memmove(layer_vbuf, layer_vbuf + (N_KV_HEAD * HEAD_DIM), (INFER_CTX - 1) * (N_KV_HEAD * HEAD_DIM) * sizeof(float));
        }
        if (ctx_pos > ctx_len) ctx_pos = ctx_len;
    }
    ctx_ids[ctx_len++] = id;
}

static void ctx_push_str(const char* text) {
    const char* p = text;
    while (*p) {
        int best_id = -1;
        int best_len = 0;
        for (int i = 0; i < (int)model_vocab_size; ++i) {
            uint32_t start = pgm_read_dword(&model_vocab_offsets[i]);
            uint32_t end   = pgm_read_dword(&model_vocab_offsets[i+1]);
            int len = end - start;
            if (len > best_len) {
                bool match = true;
                for (int j = 0; j < len; ++j) {
                    if (p[j] != (char)pgm_read_byte(&model_vocab_bytes[start + j])) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    best_len = len;
                    best_id = i;
                }
            }
        }
        if (best_id != -1) {
            ctx_push((uint16_t)best_id);
            p += best_len;
        } else {
            p++;
        }
    }
}

static int sample_next(float temp) {
    if (ctx_len == 0) return 0;
    while (ctx_pos < ctx_len) {
        transformer_forward(ctx_ids[ctx_pos], ctx_pos);
        ctx_pos++;
    }
    for (int c = 0; c < ctx_len; ++c) {
        uint16_t id = ctx_ids[c];
        if (id < model_vocab_size) {
            if (g_logits[id] > 0.0f) g_logits[id] /= 1.15f;
            else g_logits[id] *= 1.15f;
        }
    }
    if (temp <= 0.0f) return argmax(g_logits, (int)model_vocab_size);
    for (int i = 0; i < (int)model_vocab_size; ++i) g_logits[i] /= temp;
    softmax(g_logits, (int)model_vocab_size);
#if defined(ESP8266) || defined(ESP8266_BOARD)
    float r = (float)random(0, 1000000) / 1000000.0f;
#else
    float r = (float)esp_random() / (float)UINT32_MAX;
#endif
    float cdf = 0.0f;
    for (int i = 0; i < (int)model_vocab_size; ++i) {
        cdf += g_logits[i];
        if (r <= cdf) return i;
    }
    return model_vocab_size - 1;
}

static void init_few_shot() {
    ctx_len = 0;
    ctx_pos = 0;
    ctx_push_str("User: hi\nBot: Hello\n");
}

static void run_chat() {
    Serial.print("\nUser: ");
    char input[INFER_CTX + 1];
    int  ilen = 0;
    while (true) {
        if (!Serial.available()) continue;
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') { if (ilen > 0) break; continue; }
        if (c == '\b' || c == 0x7F) {
            if (ilen > 0) {
                ilen--;
                Serial.print("\b \b");
            }
            continue;
        }
        if (ilen < (int)sizeof(input) - 1) { input[ilen++] = c; Serial.print(c); }
    }
    input[ilen] = '\0';
    Serial.println();
    if (ilen == 0) return;

    if (ctx_len == 0) {
        init_few_shot();
    }

    char prompt[INFER_CTX * 2];
    snprintf(prompt, sizeof(prompt), "User: %s\nBot:", input);
    ctx_push_str(prompt);
    Serial.print("Bot: ");
    unsigned long t_start = millis();
    for (int i = 0; i < MAX_GEN_TOKENS; ++i) {
        unsigned long step_t = micros();
        int next_id  = sample_next(TEMPERATURE);
        if (next_id == 0) break; 
        bool hit_newline = print_token(next_id);
        ctx_push((uint16_t)next_id);
        if (hit_newline && i > 3) break;
        yield();
    }
    ctx_push_str("\n");
    unsigned long elapsed = millis() - t_start;
    Serial.println();
    Serial.printf("[%lu ms total]\n", elapsed);
}

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
    g_x          = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_kbuf       = (float*)arena->alloc(N_LAYER * INFER_CTX * N_KV_HEAD * HEAD_DIM * sizeof(float));
    g_vbuf       = (float*)arena->alloc(N_LAYER * INFER_CTX * N_KV_HEAD * HEAD_DIM * sizeof(float));
    g_xnorm      = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_qkv_out    = (float*)arena->alloc((N_EMBD + 2 * N_KV_HEAD * HEAD_DIM) * sizeof(float));
    g_attn_out   = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_proj_out   = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_att        = (float*)arena->alloc(INFER_CTX              * sizeof(float));
    g_mlp_gate   = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_up     = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_hidden = (float*)arena->alloc(MLP_HIDDEN             * sizeof(float));
    g_mlp_out    = (float*)arena->alloc(N_EMBD                 * sizeof(float));
    g_logits     = (float*)arena->alloc(1024                   * sizeof(float));
    return g_x && g_kbuf && g_vbuf && g_xnorm && g_qkv_out &&
           g_attn_out && g_proj_out && g_att && g_mlp_gate &&
           g_mlp_up && g_mlp_hidden && g_mlp_out && g_logits;
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n╔══════════════════════════╗");
    Serial.println("║      ESP-LLM  v1.0       ║");
#if defined(ESP8266) || defined(ESP8266_BOARD)
    Serial.println("║  INT4 Chatbot on ESP8266 ║");
#else
    Serial.println("║  INT4 Chatbot on ESP32   ║");
#endif
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
    init_few_shot();
    Serial.println("Type your message and press Enter.");
    Serial.println("──────────────────────────────────");
}

void loop() {
    if (!arena || !arena->is_valid() || !g_x) {
        delay(5000);
        return;
    }
    run_chat();
}
