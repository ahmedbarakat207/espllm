#include <Arduino.h>
#include <string.h>
#if defined(ESP8266) || defined(ESP8266_BOARD)
#  include <ESP8266WiFi.h>
#endif
#include "inference.hpp"
#include "model_weights.hpp"

static constexpr int N_EMBD          = model_n_embd;
static constexpr int N_HEAD          = model_n_head;
static constexpr int N_KV_HEAD       = model_n_kv_head;
static constexpr int N_LAYER         = model_n_layer;
static constexpr int HEAD_DIM        = N_EMBD / N_HEAD;  
static constexpr int MLP_HIDDEN      = model_mlp_hidden;              
static constexpr int N_EXPERTS       = model_n_experts;                
static constexpr int GRP             = model_group_size;                
#if defined(ESP8266) || defined(ESP8266_BOARD)
static constexpr int INFER_CTX       = 32;                
static constexpr int MAX_GEN_TOKENS  = 60;
static constexpr float TEMPERATURE   = 0.5f;
static constexpr size_t ARENA_SIZE   = 40 * 1024;

static uint8_t s_arena_mem[ARENA_SIZE] __attribute__((aligned(4)));
static MemoryArena s_arena(s_arena_mem, ARENA_SIZE);
static MemoryArena* arena = &s_arena;
#else
static constexpr int INFER_CTX       = 48;                
static constexpr int MAX_GEN_TOKENS  = 80;
static constexpr float TEMPERATURE   = 0.5f;
static constexpr size_t ARENA_SIZE   = 88 * 1024;
static MemoryArena* arena = nullptr;
#endif

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

static int few_shot_len = 0;

static void ctx_push(uint16_t id) {
    if (ctx_len >= INFER_CTX) {
        int evict_idx = (few_shot_len < INFER_CTX - 2) ? few_shot_len : 0;
        int shift_count = (INFER_CTX - 1) - evict_idx;
        if (shift_count > 0) {
            memmove(ctx_ids + evict_idx, ctx_ids + evict_idx + 1, shift_count * sizeof(uint16_t));
        }
        ctx_len = INFER_CTX - 1;
        if (ctx_pos > evict_idx) ctx_pos = evict_idx;
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
        llm_optimistic_yield(1);
    }
    int rep_window = (ctx_len < 20) ? ctx_len : 20;
    for (int c = ctx_len - rep_window; c < ctx_len; ++c) {
        uint16_t id = ctx_ids[c];
        if (id < model_vocab_size) {
            g_logits[id] -= 1.6f;
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

static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return p;
}

static bool parse_math_expr(const char*& p, double& val, int& op_count);

static bool parse_math_factor(const char*& p, double& val, int& op_count) {
    p = skip_ws(p);
    if (*p == '+') { p++; return parse_math_factor(p, val, op_count); }
    if (*p == '-') {
        p++;
        double sub = 0.0;
        if (!parse_math_factor(p, sub, op_count)) return false;
        val = -sub;
        return true;
    }
    if (*p == '(') {
        p++;
        if (!parse_math_expr(p, val, op_count)) return false;
        p = skip_ws(p);
        if (*p != ')') return false;
        p++;
        return true;
    }
    char* endp = nullptr;
    val = strtod(p, &endp);
    if (endp == p) return false;
    p = endp;

    p = skip_ws(p);
    if (*p == '^') {
        p++;
        op_count++;
        double exp = 0.0;
        if (!parse_math_factor(p, exp, op_count)) return false;
        val = pow(val, exp);
    }
    return true;
}

static bool parse_math_term(const char*& p, double& val, int& op_count) {
    if (!parse_math_factor(p, val, op_count)) return false;
    while (true) {
        p = skip_ws(p);
        char op = *p;
        if (op == '*' || op == '/' || op == '%') {
            p++;
            op_count++;
            double rhs = 0.0;
            if (!parse_math_factor(p, rhs, op_count)) return false;
            if (op == '*') val *= rhs;
            else if (op == '/') {
                if (fabs(rhs) < 1e-12) return false;
                val /= rhs;
            }
            else if (op == '%') {
                if (fabs(rhs) < 1e-12) return false;
                val = fmod(val, rhs);
            }
        } else {
            break;
        }
    }
    return true;
}

static bool parse_math_expr(const char*& p, double& val, int& op_count) {
    if (!parse_math_term(p, val, op_count)) return false;
    while (true) {
        p = skip_ws(p);
        char op = *p;
        if (op == '+' || op == '-') {
            p++;
            op_count++;
            double rhs = 0.0;
            if (!parse_math_term(p, rhs, op_count)) return false;
            if (op == '+') val += rhs;
            else val -= rhs;
        } else {
            break;
        }
    }
    return true;
}

static bool try_evaluate_math(const char* raw_input, char* output, size_t out_len) {
    if (!raw_input || !*raw_input) return false;

    char clean[128];
    size_t in_len = strlen(raw_input);
    if (in_len >= sizeof(clean)) in_len = sizeof(clean) - 1;
    memcpy(clean, raw_input, in_len);
    clean[in_len] = '\0';

    char lower[128];
    for (size_t i = 0; i <= in_len; ++i) {
        lower[i] = (char)tolower((unsigned char)clean[i]);
    }

    const char* start_ptr = clean;
    const char* lower_ptr = lower;

    const char* prefixes[] = {
        "what is", "what's", "whats", "calculate", "calc", "solve",
        "how much is", "evaluate", "compute", "tell me", "please calculate", "math:"
    };
    for (size_t k = 0; k < sizeof(prefixes)/sizeof(prefixes[0]); ++k) {
        const char* pfx = prefixes[k];
        size_t plen = strlen(pfx);
        if (strncmp(lower_ptr, pfx, plen) == 0) {
            start_ptr += plen;
            lower_ptr += plen;
            break;
        }
    }

    while (*start_ptr == ' ' || *start_ptr == ':' || *start_ptr == '\t') {
        start_ptr++;
    }

    char expr_buf[128];
    size_t elen = strlen(start_ptr);
    if (elen >= sizeof(expr_buf)) elen = sizeof(expr_buf) - 1;
    memcpy(expr_buf, start_ptr, elen);
    expr_buf[elen] = '\0';

    while (elen > 0 && (expr_buf[elen - 1] == '?' || expr_buf[elen - 1] == '=' ||
                        expr_buf[elen - 1] == ' ' || expr_buf[elen - 1] == '.' ||
                        expr_buf[elen - 1] == '!' || expr_buf[elen - 1] == '\r' ||
                        expr_buf[elen - 1] == '\n')) {
        expr_buf[--elen] = '\0';
    }

    if (elen == 0) return false;

    const char* p = expr_buf;
    double result = 0.0;
    int op_count = 0;
    if (!parse_math_expr(p, result, op_count)) return false;

    p = skip_ws(p);
    if (*p != '\0') return false; // Contains trailing non-math characters

    if (op_count == 0) return false; // Must contain at least one operator (+, -, *, /, %, ^)

    if (fabs(result - round(result)) < 1e-7 && fabs(result) < 1e14) {
        snprintf(output, out_len, "%lld", (long long)round(result));
    } else {
        snprintf(output, out_len, "%.6g", result);
    }
    return true;
}

static void init_few_shot() {
    ctx_len = 0;
    ctx_pos = 0;
    few_shot_len = 0;
    ctx_push_str("User: hi\nBot: Hello\n");
    few_shot_len = ctx_len;
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

    char math_response[64];
    if (try_evaluate_math(input, math_response, sizeof(math_response))) {
        Serial.print("Bot: ");
        Serial.println(math_response);
        Serial.println("[0 ms - Math Harness]");
        
        char turn[INFER_CTX * 2];
        snprintf(turn, sizeof(turn), "User: %s\nBot: %s\n", input, math_response);
        ctx_push_str(turn);
        return;
    }

    char prompt[INFER_CTX * 2];
    snprintf(prompt, sizeof(prompt), "User: %s\nBot:", input);
    ctx_push_str(prompt);
    Serial.print("Bot: ");
    unsigned long t_start = millis();
    for (int i = 0; i < MAX_GEN_TOKENS; ++i) {
        llm_optimistic_yield(1);
        int next_id  = sample_next(TEMPERATURE);
        if (next_id == 0) break; 
        bool hit_newline = print_token(next_id);
        ctx_push((uint16_t)next_id);
        if (hit_newline && i > 1) break;
        llm_optimistic_yield(1);
    }
    if (ctx_len == 0 || ctx_ids[ctx_len - 1] != 199) { 
        ctx_push_str("\n");
    }
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
    arena->reset();
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
    g_logits     = (float*)arena->alloc(model_vocab_size       * sizeof(float));
    return g_x && g_kbuf && g_vbuf && g_xnorm && g_qkv_out &&
           g_attn_out && g_proj_out && g_att && g_mlp_gate &&
           g_mlp_up && g_mlp_hidden && g_mlp_out && g_logits;
}

void setup() {
    Serial.begin(115200);
    delay(2000);

#if defined(ESP8266) || defined(ESP8266_BOARD)
    WiFi.mode(WIFI_OFF);
    WiFi.forceSleepBegin();
    delay(1);
    ESP.wdtEnable(5000);
    ESP.wdtFeed();
#else
    if (!arena) {
        arena = new MemoryArena(ARENA_SIZE);
    }
#endif

    Serial.println("\n╔══════════════════════════╗");
    Serial.println("║      ESP-LLM  v1.0       ║");
#if defined(ESP8266) || defined(ESP8266_BOARD)
    Serial.println("║  INT4 Chatbot on ESP8266 ║");
#else
    Serial.println("║  INT4 Chatbot on ESP32   ║");
#endif
    Serial.println("╚══════════════════════════╝");
    Serial.printf("Free heap at boot (Wi-Fi OFF): %u B\n", (unsigned)ESP.getFreeHeap());

    if (!arena || !arena->is_valid() || !alloc_buffers()) {
        Serial.printf("[FAIL] Memory arena allocation failed (need %u KB, free %u B)\n",
                      (unsigned)(ARENA_SIZE / 1024), (unsigned)ESP.getFreeHeap());
        return;
    }
    Serial.printf("[OK] Memory Arena: %u KB (%u / %u B mapped)\n",
                  (unsigned)(ARENA_SIZE / 1024),
                  (unsigned)arena->used(), (unsigned)arena->capacity());
    print_model_info();
    Serial.printf("Free heap remaining: %u B\n\n", (unsigned)ESP.getFreeHeap());
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
