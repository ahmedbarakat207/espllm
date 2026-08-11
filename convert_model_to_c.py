import gzip ,math ,os ,struct ,sys 
import torch 
import torch .nn .functional as F 
import main 
sys .modules ['__main__']=main 
QUANTIZED_PATH = None
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    arg_p = sys.argv[1].lower()
    candidates = [
        arg_p,
        f"model/model_{arg_p}.pt.quantized",
        f"model/{arg_p}.pt.quantized",
        f"model/{arg_p}.pt",
    ]
    if "esp32" in arg_p:
        candidates.extend(["model/model.pt.quantized", "model/model.pt", "model/model_esp32.pt.quantized"])
    elif "esp8266" in arg_p:
        candidates.extend(["model/model_esp8266.pt.quantized", "model/model_esp8266.pt.best", "model/model_esp8266.pt"])
    for c in candidates:
        if os.path.exists(c):
            QUANTIZED_PATH = c
            break

if not QUANTIZED_PATH:
    default_candidates = [
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

DATASET_PATH = "dataset.txt"
OUTPUT_PATH = "src/model_weights.hpp"
GROUP_SIZE = 64
print(f"Loading {QUANTIZED_PATH} ...")
if QUANTIZED_PATH.endswith(".quantized"):
    with gzip.open(QUANTIZED_PATH, "rb") as f:
        model = torch.load(f, map_location="cpu", weights_only=False)
else:
    state_dict = torch.load(QUANTIZED_PATH, map_location="cpu", weights_only=False)
    if "esp8266" in QUANTIZED_PATH:
        main.set_target_profile("esp8266")
    else:
        main.set_target_profile("esp32")
    model = main.GPT(main.GPTConfig())
    if isinstance(state_dict, dict) and "tok_emb.weight" in state_dict:
        model.load_state_dict(state_dict)
    elif hasattr(state_dict, "blocks"):
        model = state_dict
    model.eval()
    model = main.convert_to_bitlinear(model)
print("Model loaded.")
def unicode_to_bytes ():
    bs =list (range (ord ('!'),ord ('~')+1 ))+list (range (ord ('¡'),ord ('¬')+1 ))+list (range (ord ('®'),ord ('ÿ')+1 ))
    cs =bs [:]
    n =0 
    for b in range (2 **8 ):
        if b not in bs :
            bs .append (b )
            cs .append (2 **8 +n )
            n +=1 
    cs =[chr (n )for n in cs ]
    return dict (zip (cs ,bs ))

u2b =unicode_to_bytes ()

import json 
with open("bpe-vocab.json", "r", encoding="utf-8") as f:
    vocab_dict = json.load(f)

vocab_size =len (vocab_dict )
v2i ={v :k for k ,v in vocab_dict .items ()}
vocab_bytes_flat =bytearray ()
vocab_offsets =[]

for i in range (vocab_size ):
    vocab_offsets .append (len (vocab_bytes_flat ))
    token_str =v2i [i ]
    b =bytes ([u2b [c ]for c in token_str ])
    vocab_bytes_flat .extend (b )

vocab_offsets .append (len (vocab_bytes_flat ))

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
        
    q_mapped = torch.where(q == -1, torch.tensor(2, dtype=torch.uint8, device=q.device), q.to(torch.uint8))
    out, n = q_mapped.shape
    pad_len = (4 - (n % 4)) % 4
    if pad_len > 0:
        q_mapped = F.pad(q_mapped, (0, pad_len))
    
    packed = (q_mapped[:, 0::4] & 0x03) | \
             ((q_mapped[:, 1::4] & 0x03) << 2) | \
             ((q_mapped[:, 2::4] & 0x03) << 4) | \
             ((q_mapped[:, 3::4] & 0x03) << 6)
             
    return packed.to(torch.uint8), scale

def emit_quantized(prefix, packed, scale):
    parts = [
        emit_u8(packed, f"{prefix}_weights"),
        emit_f32(scale, f"{prefix}_scales"),
    ]
    return "\n".join(parts)

def emit_f32 (arr ,name ):
    flat =arr .detach ().cpu ().float ().numpy ().flatten ()
    lines =[f"static const float {name }[{len (flat )}] PROGMEM __attribute__((aligned(4))) = {{"]
    row =[]
    for i ,v in enumerate (flat ):
        row .append (f"{v :.6f}f")
        if len (row )==8 or i ==len (flat )-1 :
            lines .append ("    "+", ".join (row )+",")
            row =[]
    lines .append ("};\n")
    return "\n".join (lines )

def emit_u8 (arr ,name ):
    flat =arr .detach ().cpu ().numpy ().flatten ()
    lines =[f"static const uint8_t {name }[{len (flat )}] PROGMEM __attribute__((aligned(4))) = {{"]
    row =[]
    for i ,v in enumerate (flat ):
        row .append (f"0x{int (v ):02x}")
        if len (row )==12 or i ==len (flat )-1 :
            lines .append ("    "+", ".join (row )+",")
            row =[]
    lines .append ("};\n")
    return "\n".join (lines )



tok_emb_w =None 

for name ,mod in model .named_modules ():
    if name =="tok_emb":
        tok_emb_w =mod .weight .detach ().float ()

n_embd =tok_emb_w .shape [1 ]
n_layer =len (model .blocks )
n_head =model .blocks [0 ].attn .n_head 
n_kv_head =model .blocks [0 ].attn .qkv .out_features 
n_kv_head =int ((n_kv_head -n_embd )/2 /(n_embd //n_head ))
head_dim =n_embd //n_head 
block_size =model .rope_cos .shape [0 ]
mlp =model .blocks [0 ].mlp 
first_gate_q ,_ =get_quantized (mlp .experts [0 ].gate_proj )
mlp_hidden =first_gate_q .shape [0 ]
n_experts =len (mlp .experts )

print (f"  n_embd={n_embd }  n_layer={n_layer }  n_head={n_head }  "
f"head_dim={head_dim }  mlp_hidden={mlp_hidden }  block_size={block_size }")
theta =10000.0 **(-torch .arange (0 ,head_dim ,2 ).float ()/head_dim )
t =torch .arange (block_size ).float ()
freqs =torch .outer (t ,theta )
rope_cos =freqs .cos ()
rope_sin =freqs .sin ()
sections =[]
total_bytes =0 

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
sections.append(emit_u8(torch.tensor(list(vocab_bytes_flat), dtype=torch.uint8), "model_vocab_bytes"))
sections.append(f"static const uint32_t model_vocab_offsets[{vocab_size + 1}] PROGMEM __attribute__((aligned(4))) = {{")
sections.append("    " + ", ".join(str(o) for o in vocab_offsets) + "\n};\n")
total_bytes += len(vocab_bytes_flat) + (vocab_size + 1) * 4
emb_bytes = tok_emb_w.numel() * 4
sections.append(emit_f32(tok_emb_w.flatten(), "tok_emb"))
total_bytes += emb_bytes
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
    gate_qs, gate_ss = zip(*[get_quantized(expert.gate_proj) for expert in block.mlp.experts])
    up_qs, up_ss = zip(*[get_quantized(expert.up_proj) for expert in block.mlp.experts])
    down_qs, down_ss = zip(*[get_quantized(expert.down_proj) for expert in block.mlp.experts])
    gate_q_concat = torch.cat(gate_qs, dim=0)
    gate_s_concat = torch.cat(gate_ss, dim=0)
    up_q_concat = torch.cat(up_qs, dim=0)
    up_s_concat = torch.cat(up_ss, dim=0)
    down_q_concat = torch.cat(down_qs, dim=0)
    down_s_concat = torch.cat(down_ss, dim=0)
    sections.append(emit_quantized(f"l{li}_experts_gate", gate_q_concat, gate_s_concat))
    sections.append(emit_quantized(f"l{li}_experts_up", up_q_concat, up_s_concat))
    sections.append(emit_quantized(f"l{li}_experts_down", down_q_concat, down_s_concat))
    layer_bytes = qkv_q.numel() + proj_q.numel() + gate_q_concat.numel() + up_q_concat.numel() + down_q_concat.numel()
    total_bytes += layer_bytes
sections.append(emit_f32(model.ln_f.weight.detach().float(), "ln_f_gamma"))
first_qkv_q, _ = get_quantized(model.blocks[0].attn.qkv)
sections.append(emit_u8(first_qkv_q, "model_weights"))
sections.append(f"static const unsigned int model_weights_len = {first_qkv_q.numel()};\n")

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
    const uint8_t* qkv_w;  const float* qkv_s;
    const uint8_t* proj_w; const float* proj_s;
    const float* router_w;
    const uint8_t* experts_gate_q;
    const float*   experts_gate_s;
    const uint8_t* experts_up_q;
    const float*   experts_up_s;
    const uint8_t* experts_down_q;
    const float*   experts_down_s;
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
file_kb =os .path .getsize (OUTPUT_PATH )/1024 
print (f"\nWrote {OUTPUT_PATH }  ({file_kb :.0f} KB source)")
print (f"Estimated binary flash usage: ~{total_bytes //1024 } KB")
print ("\nArrays exported:")
print (f"  model_vocab_bytes[{len (vocab_bytes_flat )}], model_vocab_offsets[{vocab_size +1 }]")
print (f"  tok_emb[{vocab_size }×{n_embd }], rope_cos/sin[{block_size }×{head_dim //2 }]")
for li in range (n_layer ):
    print (f"  l{li }: qkv, proj, gate_proj, up_proj, down_proj, ln1, ln2")
print ("  ln_f_gamma")
