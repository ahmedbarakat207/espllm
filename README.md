# ESP-LLM: Technical Architecture & Inference Specification

ESP-LLM is a bare-metal C++ inference engine designed to execute quantized Mixture-of-Experts (MoE) autoregressive transformers on resource-constrained microcontrollers (Espressif ESP32 and ESP8266). The system implements group-wise INT4 weight quantization, Multi-Query Attention (MQA), Rotary Position Embeddings (RoPE), SwiGLU non-linearities, zero-heap-allocation memory arenas, NonOS/FreeRTOS watchdog execution slicing, and a deterministic arithmetic coprocessor.

---

## 1. System Architecture & Hardware Profiles

The engine supports two distinct target configurations tailored to microcontroller SRAM and flash constraints:

| Architectural Parameter | ESP32 (esp32dev) | ESP8266 (nodemcuv2 / d1_mini) |
|---|---|---|
| Core Architecture | Xtensa Dual-Core LX6 @ 240 MHz | Tensilica L106 Single-Core @ 80/160 MHz |
| Embedding Dimension ($N_{embd}$) | 128 | 64 |
| Transformer Layers ($N_{layer}$) | 6 | 4 |
| Query Attention Heads ($N_{head}$) | 4 | 2 |
| Key/Value Attention Heads ($N_{kv\_head}$) | 1 (Multi-Query Attention) | 1 (Multi-Query Attention) |
| Head Dimension ($HeadDim$) | 32 ($N_{embd} / N_{head}$) | 32 ($N_{embd} / N_{head}$) |
| MLP Hidden Dimension ($MLP_{hidden}$) | 128 | 64 |
| Mixture-of-Experts ($N_{experts}$) | 16 experts per layer | 20 experts per layer |
| Expert Routing Policy | Top-1 Hard Routing ($K=1$) | Top-1 Hard Routing ($K=1$) |
| Context Window Capacity ($CTX$) | 48 tokens | 32 tokens |
| Vocabulary Dimension ($Vocab$) | 1,024 BPE tokens | 1,024 BPE tokens |
| Quantization Scheme | Asymmetric Group-wise INT4 ($G=64$) | Asymmetric Group-wise INT4 ($G=64$) |
| Weights Storage | SPI Flash (`PROGMEM`, 4-byte aligned) | SPI Flash (`PROGMEM`, 4-byte aligned) |
| Memory Management | 88 KB Dynamic Arena (from 275 KB Heap) | 40 KB Static BSS Buffer (Zero Heap) |
| Static System RAM | ~23.7 KB | ~12.2 KB |
| Active Inference SRAM | 82,904 Bytes (~81.0 KB) | 38,544 Bytes (~37.6 KB) |
| Binary Flash Consumption | ~3.0 MB | ~1.2 MB |

---

## 2. Transformer Mathematical Formulation

### 2.1 Rotary Position Embeddings (RoPE)
Positional encoding is applied to Query ($Q$) and Key ($K$) vectors prior to dot-product attention. For a token at sequence index $m \in [0, CTX-1]$ and dimension pair index $i \in [0, HeadDim/2 - 1]$:

$$\theta_i = 10000^{-2i / HeadDim}$$

The 2D coordinate transformation is computed as:

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

Trigonometric tables (`rope_cos`, `rope_sin`) are precomputed at compile time and stored in 32-bit aligned flash memory.

### 2.2 Multi-Query Attention (MQA)
To minimize Key-Value cache memory overhead in SRAM, a single Key-Value head ($N_{kv\_head}=1$) is projected and shared across all Query heads ($N_{head}$):

$$Q = \text{Linear}(x; W_q) \in \mathbb{R}^{N_{head} \times HeadDim}$$
$$K = \text{Linear}(x; W_k) \in \mathbb{R}^{1 \times HeadDim}$$
$$V = \text{Linear}(x; W_v) \in \mathbb{R}^{1 \times HeadDim}$$

For Query head $h \in [0, N_{head}-1]$, the attention distribution across context positions $s \in [0, pos]$ is computed with scaling and stabilized softmax:

$$A_{h, s} = \frac{Q_h \cdot K_s^\top}{\sqrt{HeadDim}}$$
$$S_{h, s} = \frac{\exp(A_{h, s} - \max_j A_{h, j})}{\sum_{j=0}^{pos} \exp(A_{h, j} - \max_k A_{h, k})}$$
$$\text{AttnOut}_h = \sum_{s=0}^{pos} S_{h, s} V_s$$

### 2.3 SwiGLU Gated Feed-Forward Networks
Each expert layer implements a SwiGLU non-linear projection:

$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$
$$\text{SwiGLU}(x) = (\text{Linear}(x; W_{\text{gate}}) \odot \text{SiLU}(\text{Linear}(x; W_{\text{gate}}))) \odot \text{Linear}(x; W_{\text{up}})$$
$$\text{ExpertOut}(x) = \text{Linear}(\text{SwiGLU}(x); W_{\text{down}})$$

### 2.4 Sparse Mixture of Experts (MoE) Top-1 Routing
At each transformer layer $l$, a normalized linear router selects the active expert:

$$e^* = \arg\max_{e \in [0, N_{experts}-1]} (x_{\text{norm}} \cdot W_{\text{router}, e}^\top)$$

Only the weights corresponding to expert $e^*$ are read from flash and computed, maintaining constant execution time per token regardless of total expert count.

### 2.5 Root Mean Square Normalization (RMSNorm)
Pre-attention, pre-MLP, and final normalization use RMSNorm with learnable scaling parameter $\gamma$:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{N_{embd}} \sum_{i=1}^{N_{embd}} x_i^2 + \epsilon}} \odot \gamma \quad (\epsilon = 10^{-5})$$

---

## 3. INT4 Quantization Engine & GEMM Kernel

### 3.1 Quantization Representation
Weights are quantized per channel in blocks of $G = 64$ elements using asymmetric affine mapping:

$$scale = \frac{\max(W_g) - \min(W_g)}{15}, \quad zp = \text{round}\left(-\frac{\min(W_g)}{scale}\right)$$
$$W_{q} = \text{clamp}\left(\text{round}\left(\frac{W}{scale}\right) + zp, 0, 15\right)$$

Dequantization during matrix multiplication evaluates as:

$$\hat{W} = (W_q - zp) \cdot scale$$

### 3.2 Storage Layout & 32-Bit Memory Alignment
* Two 4-bit nibbles are packed into each `uint8_t`: `byte = (q0 & 0x0F) | ((q1 << 4) & 0xF0)`.
* All weight tensors (`W_packed`), scale vectors, and zero-point offsets in flash are annotated with `__attribute__((aligned(4)))` and stored in `PROGMEM`. This avoids hardware load-store alignment faults (`LoadStoreAlignmentCause`) on 32-bit Tensilica buses.

### 3.3 Optimized General Matrix-Vector Multiplication (`matmul_int4_f32`)
* **Lookup Table Unpacking**: A 256-entry constant lookup table (`unpack_lut`) unpacks byte values into floating-point pairs without bit-shift instructions.
* **4-Byte Unrolling**: Inner accumulation loops process 4 packed bytes (8 INT4 weights) per iteration:

```cpp
for (int j = 0; j < k_bytes; j += 4) {
    uint8_t p0 = w_row[j], p1 = w_row[j+1], p2 = w_row[j+2], p3 = w_row[j+3];
    NibblePair u0 = unpack_lut[p0], u1 = unpack_lut[p1];
    NibblePair u2 = unpack_lut[p2], u3 = unpack_lut[p3];
}
```

---

## 4. Memory Architecture & Buffer Layout

### 4.1 Contiguous Memory Arena
The inference engine avoids heap fragmentation by allocating a single static or startup arena (`MemoryArena`). All tensor activation buffers are sliced contiguously:

```
+-------------------------------------------------------------+
| MemoryArena Pool (88 KB on ESP32 / 40 KB on ESP8266)        |
+-------------------------------------------------------------+
| Offset | Tensor Buffer | Size   | Description               |
+--------+---------------+--------+---------------------------+
| 0x0000 | g_x           | 512 B  | Token hidden state vector |
| 0x0200 | g_kbuf        | 36 KB  | Multi-layer Key cache     |
| 0x9200 | g_vbuf        | 36 KB  | Multi-layer Value cache   |
| 0x12200| g_xnorm       | 512 B  | RMSNorm normalized vector |
| 0x12400| g_qkv_out     | 768 B  | Fused QKV projection      |
| 0x12700| g_attn_out    | 512 B  | Multi-head accumulator    |
| 0x12900| g_proj_out    | 512 B  | Attention dense output    |
| 0x12B00| g_att         | 192 B  | Attention softmax scores  |
| 0x12BC0| g_mlp_gate    | 512 B  | Active expert gate        |
| 0x12DC0| g_mlp_up      | 512 B  | Active expert up          |
| 0x12FC0| g_mlp_hidden  | 512 B  | SwiGLU activation vector  |
| 0x131C0| g_mlp_out     | 512 B  | Expert down projection    |
| 0x133C0| g_logits      | 4 KB   | Output vocabulary logits  |
+-------------------------------------------------------------+
| Headroom: ~5.1 KB reserved for alignment and safety buffer  |
+-------------------------------------------------------------+
```

### 4.2 SRAM Allocation Matrix

#### ESP32 Target ($CTX=48, N_{embd}=128, N_{layer}=6$)
| Allocation Target | Dimension / Calculation | Bytes |
|---|---|---|
| Key Cache (`g_kbuf`) | $6 \times 48 \times 1 \times 32 \times 4\text{ B}$ (6 layers, 48 ctx, 1 KV head, 32 dim) | 36,864 |
| Value Cache (`g_vbuf`) | $6 \times 48 \times 1 \times 32 \times 4\text{ B}$ (6 layers, 48 ctx, 1 KV head, 32 dim) | 36,864 |
| Vocabulary Logits (`g_logits`) | $1024 \times 4\text{ B}$ (1024 vocabulary tokens) | 4,096 |
| QKV Projection Buffer (`g_qkv_out`) | $(128 + 2 \times 32) \times 4\text{ B}$ | 768 |
| Hidden State Vector (`g_x`) | $128 \times 4\text{ B}$ | 512 |
| Normalized State (`g_xnorm`) | $128 \times 4\text{ B}$ | 512 |
| Attention Projection (`g_attn_out`) | $128 \times 4\text{ B}$ | 512 |
| Dense Projection (`g_proj_out`) | $128 \times 4\text{ B}$ | 512 |
| MLP Gate Projection (`g_mlp_gate`) | $128 \times 4\text{ B}$ | 512 |
| MLP Up Projection (`g_mlp_up`) | $128 \times 4\text{ B}$ | 512 |
| SwiGLU Intermediate (`g_mlp_hidden`) | $128 \times 4\text{ B}$ | 512 |
| MLP Down Projection (`g_mlp_out`) | $128 \times 4\text{ B}$ | 512 |
| Attention Score Buffer (`g_att`) | $48 \times 4\text{ B}$ | 192 |
| **Total Arena Footprint** | **Mapped in 88 KB Pool** | **82,904 B (~81.0 KB)** |

#### ESP8266 Target ($CTX=32, N_{embd}=64, N_{layer}=4$)
| Allocation Target | Dimension / Calculation | Bytes |
|---|---|---|
| Key Cache (`g_kbuf`) | $4 \times 32 \times 1 \times 32 \times 4\text{ B}$ (4 layers, 32 ctx, 1 KV head, 32 dim) | 16,384 |
| Value Cache (`g_vbuf`) | $4 \times 32 \times 1 \times 32 \times 4\text{ B}$ (4 layers, 32 ctx, 1 KV head, 32 dim) | 16,384 |
| Vocabulary Logits (`g_logits`) | $1024 \times 4\text{ B}$ (1024 vocabulary tokens) | 4,096 |
| Intermediate Activation Buffers | Sum of activation vectors | 1,680 |
| **Total Arena Footprint** | **Static BSS Buffer (Zero Heap Allocation)** | **38,544 B (~37.6 KB)** |

### 4.3 Pinned Few-Shot Sliding Window Management
When conversation context reaches capacity ($ctx\_len \ge INFER\_CTX$):
1. Pinned few-shot prompt tokens ($[0 \dots few\_shot\_len - 1]$) remain fixed at the head of `ctx_ids`.
2. Oldest conversational turns starting at index $few\_shot\_len$ are shifted left using `memmove`.
3. The evaluation index is invalidated: `if (ctx_pos > evict_idx) ctx_pos = evict_idx;`.
4. Subsequent forward passes recompute exact RoPE coordinates and KV cache activations for the shifted positions, preventing positional drift or numerical degradation across infinite turns.

---

## 5. Execution Safeguards & Coprocessors

### 5.1 Watchdog Feeding & Microsecond Time Slicing
To comply with ESP8266 NonOS SDK watchdog limits (~1.5 s maximum blocking duration):
* Wi-Fi hardware is shut down at startup (`WiFi.mode(WIFI_OFF); WiFi.forceSleepBegin();`) to reclaim ~15 KB of internal SRAM and suspend RF background interrupts.
* Hardware watchdog timers are configured via `ESP.wdtEnable(5000)`.
* Inner GEMM loops call `llm_optimistic_yield(64)`, which executes `ESP.wdtFeed()`, `optimistic_yield(1000)`, and standard `yield()` to service SDK task queues.

### 5.2 Deterministic Arithmetic Harness
Before triggering transformer prefill, user input is passed through an integrated recursive-descent math parser (`try_evaluate_math`):
* **Supported Operations**: Addition (`+`), Subtraction (`-`), Multiplication (`*`), Division (`/`), Modulo (`%`), Exponentiation (`^`), Unary Negation (`-`), Nested Parentheses (`(...)`), and floating-point literals.
* **Natural Language Stripping**: Automatically strips leading query patterns (`what is`, `calculate`, `calc`, `solve`, `evaluate`, `compute`, `how much is`) and trailing symbols (`?`, `=`).
* **Execution**: Evaluates with exact precision and 0 ms inference latency, immediately inserting the turn into `ctx_ids` to maintain contextual history.

### 5.3 Token Sampling & Repetition Suppression
* **Temperature Scaling**: Logits are scaled by temperature ($T = 0.5$) before softmax and cumulative distribution function (CDF) sampling:
  $$P(v_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
* **Recency-Weighted Repetition Penalty**: Recent token IDs within a sliding window of the last 20 generated tokens receive a direct logit subtraction:
  $$z_k \leftarrow z_k - 1.6 \quad (\forall k \in \mathcal{W}_{\text{recent}})$$
* **Turn Termination**: Generation terminates upon producing token ID `0` (`<|endoftext|>`), a newline token, or reaching `MAX_GEN_TOKENS`.

## 6. Build, Flash & Interface Workflow

### Prerequisites
* Python 3.9+ with `torch`, `numpy`, and `pyserial`.
* PlatformIO Core CLI (`pio`).

### 6.1 Compilation and Flashing

To export weights, compile firmware, and flash to an ESP32:
```bash
python flash.py esp32
```

To compile and flash an ESP8266:
```bash
python flash.py esp8266
```

To specify an explicit serial port or baud rate:
```bash
python flash.py esp32 -p COM4 -b 115200
```

To verify compilation without flashing:
```bash
python flash.py esp32 --build-only
```

### 6.2 Serial Monitor

To connect directly to the microcontroller serial stream:
```bash
python run.py
```

Explicit port override:
```bash
python run.py -p COM4 -b 115200
```

### 6.3 Standalone Weight Conversion

To manually regenerate `src/model_weights.hpp` from a model checkpoint:
```bash
python convert_model_to_c.py esp32
python convert_model_to_c.py esp8266
```
