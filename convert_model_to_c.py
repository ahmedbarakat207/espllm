import gzip, math, os, sys
import torch
import torch.nn.functional as F
import main

sys.modules["__main__"] = main
QUANTIZED_PATH = None
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    arg_p = sys.argv[1].lower()
    candidates = [
        arg_p,
        f"model/model_{arg_p}.pt.quantized",
        f"model/{arg_p}.pt.quantized",
        f"model/{arg_p}.pt",
    ]
    if "esp32s3" in arg_p:
        candidates.extend(
            [
                "model/model_esp32s3.pt.quantized",
                "model/model_esp32s3.pt.best",
                "model/model_esp32s3.pt",
            ]
        )
    elif "esp32" in arg_p:
        candidates.extend(
            [
                "model/model.pt.quantized",
                "model/model.pt",
                "model/model_esp32.pt.quantized",
            ]
        )
    elif "esp8266" in arg_p:
        candidates.extend(
            [
                "model/model_esp8266.pt.quantized",
                "model/model_esp8266.pt.best",
                "model/model_esp8266.pt",
            ]
        )
    for c in candidates:
        if os.path.exists(c):
            QUANTIZED_PATH = c
            break

if not QUANTIZED_PATH:
    default_candidates = [
        "model/model_esp32s3.pt.quantized",
        "model/model.pt.quantized",
        "model/model_esp8266.pt.quantized",
        "model/model.pt",
        "model/model_esp8266.pt.best",
    ]
    for c in default_candidates:
        if os.path.exists(c):
            QUANTIZED_PATH = c
            break

if not QUANTIZED_PATH:
    print("Error: No model checkpoint found in model/ directory.")
    sys.exit(1)

OUTPUT_PATH = "src/model_weights.hpp"
GROUP_SIZE = 64
print(f"Loading {QUANTIZED_PATH} ...")
if QUANTIZED_PATH.endswith(".quantized"):
    try:
        with gzip.open(QUANTIZED_PATH, "rb") as f:
            model = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error: could not load {QUANTIZED_PATH} ({e}).")
        print(
            "It may be a stale checkpoint from an older code version — retrain first."
        )
        sys.exit(1)
else:
    # Raw (unquantized) state dict: infer dims from tensors, point main.py's
    # globals at them, rebuild the model, then quantize. The profile is only
    # used to pick a matching checkpoint; dims always come from the file.
    state_dict = torch.load(QUANTIZED_PATH, map_location="cpu", weights_only=False)
    if hasattr(state_dict, "blocks"):
        model = state_dict
    else:
        sd = state_dict
        _tok = sd["tok_emb.weight"]
        _vocab, _embd = _tok.shape
        _nl = len(
            [
                k
                for k in sd
                if k.startswith("blocks.") and k.endswith(".attn.qkv.weight")
            ]
        )
        _qkv_out, _ = sd["blocks.0.attn.qkv.weight"].shape
        _nh = main.n_head
        for cand_nh in (2, 4, 6, 8, 12):
            if _embd % cand_nh == 0:
                _hd = _embd // cand_nh
                if _qkv_out - _embd > 0 and (_qkv_out - _embd) % (2 * _hd) == 0:
                    _nh = cand_nh
                    break
        _hd = _embd // _nh
        _nkv = (_qkv_out - _embd) // (2 * _hd)
        _gate_out, _ = sd["blocks.0.mlp.experts.0.gate_proj.weight"].shape
        _ne = len(
            [
                k
                for k in sd
                if k.startswith("blocks.0.mlp.experts.")
                and k.endswith(".gate_proj.weight")
            ]
        )
        main.n_embd, main.n_head, main.n_kv_head = _embd, _nh, _nkv
        main.n_layer, main.n_experts, main.moe_hidden = _nl, _ne, _gate_out
        main.dropout = 0.0
        if "rope_cos" in sd:
            main.block_size = sd["rope_cos"].shape[0]
        model = main.Transformer(group_size=main.qat_group_size)
        model.load_state_dict(sd)
    model.eval()
    model = main.convert_to_bitlinear(model)
print("Model loaded.")
for _m in model.modules():
    if hasattr(_m, "group_size") and isinstance(getattr(_m, "group_size"), int):
        GROUP_SIZE = int(_m.group_size)
        break
print(f"Using quantization group size: {GROUP_SIZE}")


def unicode_to_bytes():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(cs, bs))


u2b = unicode_to_bytes()

import json

with open("bpe-vocab.json", "r", encoding="utf-8") as f:
    vocab_dict = json.load(f)

vocab_size = len(vocab_dict)
v2i = {v: k for k, v in vocab_dict.items()}
vocab_bytes_flat = bytearray()
vocab_offsets = []

for i in range(vocab_size):
    vocab_offsets.append(len(vocab_bytes_flat))
    token_str = v2i[i]
    b = bytes([u2b[c] for c in token_str])
    vocab_bytes_flat.extend(b)

vocab_offsets.append(len(vocab_bytes_flat))


def get_quantized(mod):
    if hasattr(mod, "qweight"):
        q = mod.qweight
        scale = mod.scale.float()
    else:
        w = mod.weight
        group_size = getattr(mod, "group_size", GROUP_SIZE)
        out, n = w.shape
        n_groups = math.ceil(n / group_size)
        padded_n = n_groups * group_size
        w_pad = F.pad(w, (0, padded_n - n)) if padded_n != n else w
        wg = w_pad.view(out, n_groups, group_size)
        scale = wg.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        q = torch.clamp(torch.round(wg / scale), -1, 1).to(torch.int8)
        q = q.view(out, padded_n)[:, :n]
        scale = scale.squeeze(-1)

    q_mapped = torch.where(
        q == -1, torch.tensor(2, dtype=torch.uint8, device=q.device), q.to(torch.uint8)
    )
    out, n = q_mapped.shape
    pad_len = (4 - (n % 4)) % 4
    if pad_len > 0:
        q_mapped = F.pad(q_mapped, (0, pad_len))

    packed = (
        (q_mapped[:, 0::4] & 0x03)
        | ((q_mapped[:, 1::4] & 0x03) << 2)
        | ((q_mapped[:, 2::4] & 0x03) << 4)
        | ((q_mapped[:, 3::4] & 0x03) << 6)
    )

    return packed.to(torch.uint8), scale


def emit_quantized(prefix, packed, scale):
    # Scales are stored as FP16 (2 bytes) — firmware converts to float on the fly.
    scale_h = scale.detach().cpu().float().to(torch.float16).view(torch.int16)
    parts = [
        emit_u8(packed, f"{prefix}_weights"),
        emit_u16(scale_h, f"{prefix}_scales"),
    ]
    return "\n".join(parts)


def emit_f32(arr, name):
    flat = arr.detach().cpu().float().numpy().flatten()
    lines = [
        f"static const float {name }[{len (flat )}] PROGMEM __attribute__((aligned(4))) = {{"
    ]
    row = []
    for i, v in enumerate(flat):
        row.append(f"{v :.6f}f")
        if len(row) == 8 or i == len(flat) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};\n")
    return "\n".join(lines)


def emit_u8(arr, name):
    flat = arr.detach().cpu().numpy().flatten()
    lines = [
        f"static const uint8_t {name }[{len (flat )}] PROGMEM __attribute__((aligned(4))) = {{"
    ]
    row = []
    for i, v in enumerate(flat):
        row.append(f"0x{int (v ):02x}")
        if len(row) == 12 or i == len(flat) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};\n")
    return "\n".join(lines)


def emit_u16(arr, name):
    flat = arr.detach().cpu().numpy().flatten()
    lines = [
        f"static const uint16_t {name }[{len (flat )}] PROGMEM __attribute__((aligned(4))) = {{"
    ]
    row = []
    for i, v in enumerate(flat):
        row.append(f"0x{int (v ) & 0xffff :04x}")
        if len(row) == 12 or i == len(flat) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};\n")
    return "\n".join(lines)


tok_emb_w = None

for name, mod in model.named_modules():
    if name == "tok_emb":
        tok_emb_w = mod.weight.detach().float()

n_embd = tok_emb_w.shape[1]
n_layer = len(model.blocks)
n_head = model.blocks[0].attn.n_head
n_kv_head = model.blocks[0].attn.qkv.out_features
n_kv_head = int((n_kv_head - n_embd) / 2 / (n_embd // n_head))
head_dim = n_embd // n_head
block_size = model.rope_cos.shape[0]
mlp = model.blocks[0].mlp
first_gate_q, _ = get_quantized(mlp.experts[0].gate_proj)
mlp_hidden = first_gate_q.shape[0]
n_experts = len(mlp.experts)

print(
    f"  n_embd={n_embd }  n_layer={n_layer }  n_head={n_head }  "
    f"head_dim={head_dim }  mlp_hidden={mlp_hidden }  block_size={block_size }"
)
theta = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
t = torch.arange(block_size).float()
freqs = torch.outer(t, theta)
rope_cos = freqs.cos()
rope_sin = freqs.sin()
sections = []
total_bytes = 0

sections.append(f"""\
// Auto-generated by convert_model_to_c.py — DO NOT EDIT
#ifndef MODEL_WEIGHTS_HPP
#define MODEL_WEIGHTS_HPP
#include <stdint.h>
#include <stddef.h>
#if defined(__AVR__)
#  include <avr/pgmspace.h>
#elif defined(ESP8266) || defined(ESP32)
#  include <pgmspace.h>
#else
#  define PROGMEM
#endif
static const uint16_t model_vocab_size = {vocab_size};
static const uint16_t model_n_embd     = {n_embd};
static const uint8_t  model_n_layer    = {n_layer};
static const uint8_t  model_n_head     = {n_head};
static const uint16_t model_block_size = {block_size};
static const uint8_t  model_group_size = {GROUP_SIZE};
static const uint16_t model_mlp_hidden = {mlp_hidden};
static const uint8_t  model_n_experts  = {n_experts};
static const uint8_t  model_n_kv_head  = {n_kv_head};
""")
sections.append(
    emit_u8(
        torch.tensor(list(vocab_bytes_flat), dtype=torch.uint8), "model_vocab_bytes"
    )
)
sections.append(
    f"static const uint32_t model_vocab_offsets[{vocab_size + 1}] PROGMEM __attribute__((aligned(4))) = {{"
)
sections.append("    " + ", ".join(str(o) for o in vocab_offsets) + "\n};\n")
total_bytes += len(vocab_bytes_flat) + (vocab_size + 1) * 4
# Token embeddings: per-token INT8 + FP32 row scale (4x smaller than FP32).
emb_scale = tok_emb_w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5) / 127.0
emb_q = torch.clamp(torch.round(tok_emb_w / emb_scale), -128, 127).to(torch.int16)
emb_q_u8 = (emb_q & 0xFF).to(torch.uint8)
sections.append(emit_u8(emb_q_u8.flatten(), "tok_emb_q"))
sections.append(emit_f32(emb_scale.squeeze(-1), "tok_emb_scales"))
total_bytes += emb_q_u8.numel() + emb_scale.numel() * 4
sections.append(emit_f32(rope_cos.flatten(), "rope_cos"))
sections.append(emit_f32(rope_sin.flatten(), "rope_sin"))
total_bytes += rope_cos.numel() * 4 * 2

for li, block in enumerate(model.blocks):
    attn = block.attn
    qkv_q, qkv_s = get_quantized(attn.qkv)
    proj_q, proj_s = get_quantized(attn.proj)
    sections.append(emit_quantized(f"l{li}_attn_qkv", qkv_q, qkv_s))
    sections.append(emit_quantized(f"l{li}_attn_proj", proj_q, proj_s))
    sections.append(emit_f32(block.ln1.weight.detach().float(), f"l{li}_ln1_gamma"))
    sections.append(emit_f32(block.ln2.weight.detach().float(), f"l{li}_ln2_gamma"))
    router_w = block.mlp.router.weight.detach().float().flatten()
    sections.append(emit_f32(router_w, f"l{li}_router"))
    total_bytes += router_w.numel() * 4
    gate_qs, gate_ss = zip(
        *[get_quantized(expert.gate_proj) for expert in block.mlp.experts]
    )
    up_qs, up_ss = zip(*[get_quantized(expert.up_proj) for expert in block.mlp.experts])
    down_qs, down_ss = zip(
        *[get_quantized(expert.down_proj) for expert in block.mlp.experts]
    )
    gate_q_concat = torch.cat(gate_qs, dim=0)
    gate_s_concat = torch.cat(gate_ss, dim=0)
    up_q_concat = torch.cat(up_qs, dim=0)
    up_s_concat = torch.cat(up_ss, dim=0)
    down_q_concat = torch.cat(down_qs, dim=0)
    down_s_concat = torch.cat(down_ss, dim=0)
    sections.append(emit_quantized(f"l{li}_experts_gate", gate_q_concat, gate_s_concat))
    sections.append(emit_quantized(f"l{li}_experts_up", up_q_concat, up_s_concat))
    sections.append(emit_quantized(f"l{li}_experts_down", down_q_concat, down_s_concat))
    layer_bytes = (
        qkv_q.numel()
        + proj_q.numel()
        + gate_q_concat.numel()
        + up_q_concat.numel()
        + down_q_concat.numel()
    )
    layer_bytes += 2 * (
        qkv_s.numel()
        + proj_s.numel()
        + gate_s_concat.numel()
        + up_s_concat.numel()
        + down_s_concat.numel()
    )  # FP16 scales
    layer_bytes += (block.ln1.weight.numel() + block.ln2.weight.numel()) * 4
    total_bytes += layer_bytes
sections.append(emit_f32(model.ln_f.weight.detach().float(), "ln_f_gamma"))
total_bytes += model.ln_f.weight.numel() * 4
first_qkv_q, _ = get_quantized(model.blocks[0].attn.qkv)
sections.append(emit_u8(first_qkv_q, "model_weights"))
sections.append(
    f"static const unsigned int model_weights_len = {first_qkv_q.numel()};\n"
)

layer_inits = []
for li in range(n_layer):
    layer_inits.append(f"""    {{
        l{li}_attn_qkv_weights,      l{li}_attn_qkv_scales,
        l{li}_attn_proj_weights,     l{li}_attn_proj_scales,
        l{li}_router,
        l{li}_experts_gate_weights,  l{li}_experts_gate_scales,
        l{li}_experts_up_weights,    l{li}_experts_up_scales,
        l{li}_experts_down_weights,  l{li}_experts_down_scales,
        l{li}_ln1_gamma,
        l{li}_ln2_gamma,
    }}""")

layers_str = ",\n".join(layer_inits)
sections.append(f"""
struct LayerW {{
    const uint8_t* qkv_w;  const uint16_t* qkv_s; // scales: FP16, half_to_float on load
    const uint8_t* proj_w; const uint16_t* proj_s;
    const float* router_w;
    const uint8_t* experts_gate_q;
    const uint16_t* experts_gate_s;
    const uint8_t* experts_up_q;
    const uint16_t* experts_up_s;
    const uint8_t* experts_down_q;
    const uint16_t* experts_down_s;
    const float* ln1_g;
    const float* ln2_g;
}};

static const LayerW g_layers[{n_layer}] __attribute__((aligned(4))) = {{
{layers_str}
}};

#endif // MODEL_WEIGHTS_HPP
""")

os.makedirs("src", exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    f.write("\n".join(sections))
file_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"\nWrote {OUTPUT_PATH }  ({file_kb :.0f} KB source)")
print(f"Estimated binary flash usage: ~{total_bytes //1024 } KB")
print("\nArrays exported:")
print(
    f"  model_vocab_bytes[{len (vocab_bytes_flat )}], model_vocab_offsets[{vocab_size +1 }]"
)
print(
    f"  tok_emb_q[{vocab_size }×{n_embd }] int8 + scales, rope_cos/sin[{block_size }×{head_dim //2 }]"
)
for li in range(n_layer):
    print(f"  l{li }: qkv, proj, gate_proj, up_proj, down_proj, ln1, ln2")
print("  ln_f_gamma")
