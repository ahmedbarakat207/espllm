import math, torch, torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy
from tokenizers import ByteLevelBPETokenizer
import os, gzip
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torchao.optim import AdamW4bit
except ImportError:
    AdamW4bit = None

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

import sys

TARGET = "esp8266"
for arg in sys.argv:
    if arg.startswith("--target="):
        TARGET = arg.split("=")[1].strip()
    elif arg in ("--esp8266", "-esp8266"):
        TARGET = "esp8266"
    elif arg in ("--esp32s3", "-esp32s3"):
        TARGET = "esp32s3"
    elif arg in ("--esp32", "-esp32"):
        TARGET = "esp32"

# DeepSpeed ZeRO (multi-GPU data-parallel) flags. Single-GPU default is unchanged.
# Launch with: deepspeed --num_gpus=2 main.py --target=esp32s3 --train --deepspeed --deepspeed_config=ds_config_zero2.json
USE_DEEPSPEED = "--deepspeed" in sys.argv
DS_CONFIG = "./ds_config_zero2.json"
for _i, _arg in enumerate(sys.argv):
    if _arg.startswith("--deepspeed_config="):
        DS_CONFIG = _arg.split("=", 1)[1].strip()
    elif _arg == "--deepspeed_config" and _i + 1 < len(sys.argv):
        DS_CONFIG = sys.argv[_i + 1].strip()
DIST = False  # True once torch.distributed is initialized (deepspeed path only)
RANK = 0
WORLD = 1

if TARGET == "esp8266":
    # Max config fitting 1MB irom + 40KB static arena (INT8 emb + FP16 scales).
    # block 32 > infer 24: trains on fuller answers, inference uses first 24 rows.
    checkpoint = "./model/model_esp8266.pt"
    block_size = 32
    batch_size = 32
    n_layer = 4
    n_head = 2
    n_kv_head = 1
    n_embd = 64
    n_experts = 32
    moe_hidden = 64
    dropout = 0.2
    max_iters = 10000
    eval_interval = 100
    lr = 2e-3
    lr_min = 1e-5
    warmup_iters = 200
    eval_iters = 10
    temperature = 0.6
    start_iter = 0
    patience = 10
    label_smoothing = 0.1
    qat_group_size = 64
elif TARGET == "esp32s3":
    # Largest config fitting ESP32-S3 N16R8 (16MB flash + 8MB PSRAM).
    # INT8 embeddings + FP16 scales: ~14.6MB binary in 15.9MB app partition.
    # PSRAM arena ~1.2MB.
    checkpoint = "./model/model_esp32s3.pt"
    block_size = 192
    batch_size = 32
    n_layer = 12
    n_head = 6
    n_kv_head = 2
    n_embd = 192
    n_experts = 28
    moe_hidden = 256
    dropout = 0.1
    max_iters = 25000
    eval_interval = 200
    lr = 2e-3
    lr_min = 1e-5
    warmup_iters = 400
    eval_iters = 10
    temperature = 0.6
    start_iter = 0
    patience = 12
    label_smoothing = 0.05
    qat_group_size = 64
else:
    # Max config fitting 3.9MB app + 160KB heap arena (INT8 emb + FP16 scales).
    checkpoint = "./model/model_esp32.pt"
    block_size = 64
    batch_size = 32
    n_layer = 8
    n_head = 4
    n_kv_head = 1
    n_embd = 128
    n_experts = 16
    moe_hidden = 192
    dropout = 0.0
    max_iters = 20000
    eval_interval = 100
    lr = 2e-3
    lr_min = 1e-5
    warmup_iters = 300
    eval_iters = 10
    temperature = 0.6
    start_iter = 0
    patience = 10
    label_smoothing = 0.0
    qat_group_size = 64

# Optional env overrides (handy for smoke tests, e.g. ESPGPT_MAX_ITERS=4).
# ESPGPT_DEVICE forces the device (cpu/cuda/mps); ESPGPT_CHECKPOINT redirects saves.
if os.environ.get("ESPGPT_MAX_ITERS"):
    max_iters = int(os.environ["ESPGPT_MAX_ITERS"])
if os.environ.get("ESPGPT_BATCH_SIZE"):
    batch_size = int(os.environ["ESPGPT_BATCH_SIZE"])
if os.environ.get("ESPGPT_EVAL_ITERS"):
    eval_iters = int(os.environ["ESPGPT_EVAL_ITERS"])
if os.environ.get("ESPGPT_EVAL_INTERVAL"):
    eval_interval = int(os.environ["ESPGPT_EVAL_INTERVAL"])
if os.environ.get("ESPGPT_WARMUP_ITERS"):
    warmup_iters = int(os.environ["ESPGPT_WARMUP_ITERS"])
if os.environ.get("ESPGPT_PATIENCE"):
    patience = int(os.environ["ESPGPT_PATIENCE"])
if os.environ.get("ESPGPT_CHECKPOINT"):
    checkpoint = os.environ["ESPGPT_CHECKPOINT"]
if os.environ.get("ESPGPT_DEVICE"):
    device = os.environ["ESPGPT_DEVICE"]

torch.manual_seed(1337)

dataset_path = "dataset.txt"
qa_pairs = []
with open(dataset_path, "r") as f:
    current_q, current_a = "", ""
    for line in f:
        if line.startswith("User:"):
            if current_q:
                qa_pairs.append(current_q + current_a)
            current_q = line
            current_a = ""
        else:
            current_a += line
    if current_q:
        qa_pairs.append(current_q + current_a)

tokenizer = ByteLevelBPETokenizer(
    "bpe-vocab.json",
    "bpe-merges.txt",
)

vocab_size = tokenizer.get_vocab_size()
print(f"BPE Vocab Size: {vocab_size }")
encode = lambda s: torch.tensor(tokenizer.encode(s).ids, dtype=torch.long)
decode = lambda t: tokenizer.decode(t.tolist() if hasattr(t, "tolist") else t)

import random

random.seed(1337)
random.shuffle(qa_pairs)
data = [encode(qa) for qa in qa_pairs]

n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    dataset = train_data if split == "train" else val_data
    x_batch = []
    y_batch = []
    for _ in range(batch_size):
        idx = random.randint(0, len(dataset) - 1)
        seq = dataset[idx]
        if len(seq) > block_size + 1:
            seq = seq[: block_size + 1]
        x_seq = seq[:-1]
        y_seq = seq[1:]

        pad_len = block_size - len(x_seq)
        if pad_len > 0:
            x_pad = torch.cat([x_seq, torch.zeros(pad_len, dtype=torch.long)])
            y_pad = torch.cat([y_seq, torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            x_pad = x_seq
            y_pad = y_seq

        x_batch.append(x_pad)
        y_batch.append(y_pad)

    x = torch.stack(x_batch).to(device)
    y = torch.stack(y_batch).to(device)
    return x, y


class BitLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _quantize_weight(self, w: torch.Tensor):
        out, n = w.shape
        n_groups = math.ceil(n / self.group_size)
        padded_n = n_groups * self.group_size
        w_pad = F.pad(w, (0, padded_n - n)) if padded_n != n else w
        wg = w_pad.view(out, n_groups, self.group_size)
        scale = wg.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        w_q = torch.clamp(torch.round(wg / scale), -1, 1)
        w_deq = (w_q * scale).view(out, padded_n)[:, :n]
        return w + (w_deq - w).detach()

    def _quantize_activation(self, x: torch.Tensor):
        scale = (x.abs().max(dim=-1, keepdim=True).values / 127.0).clamp(min=1e-5)
        x_q = torch.clamp(torch.round(x / scale), -128, 127)
        x_deq = x_q * scale
        return x + (x_deq - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fake = self._quantize_weight(self.weight)
        x_fake = self._quantize_activation(x)
        return F.linear(x_fake, w_fake, self.bias)

    def to_inference(self):
        return BitLinearInference(
            self.in_features,
            self.out_features,
            self.weight.detach(),
            self.bias.detach() if self.bias is not None else None,
            self.group_size,
        )


class BitLinearInference(nn.Module):
    def __init__(self, in_features, out_features, weight, bias=None, group_size=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        out, n = weight.shape
        n_groups = math.ceil(n / group_size)
        padded_n = n_groups * group_size
        w_pad = F.pad(weight, (0, padded_n - n)) if padded_n != n else weight
        wg = w_pad.view(out, n_groups, group_size)
        scale = wg.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        w_q = torch.clamp(torch.round(wg / scale), -1, 1).to(torch.int8)
        w_q = w_q.view(out, padded_n)[:, :n]
        scale = scale.squeeze(-1)

        self.register_buffer("qweight", w_q)
        self.register_buffer("scale", scale.half())
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().half())
        else:
            self.register_parameter("bias", None)

    def _dequantize_weight(self):
        w = self.qweight.float()
        groups = torch.arange(self.in_features, device=w.device) // self.group_size
        return w * self.scale.float()[:, groups]

    def forward(self, x):
        x_scale = (x.abs().max(dim=-1, keepdim=True).values / 127.0).clamp(min=1e-5)
        x_q = torch.clamp(torch.round(x / x_scale), -128, 127)
        x_deq = x_q * x_scale
        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x_deq, self._dequantize_weight(), bias)


def precompute_freqs(head_dim: int, max_seq_len: int, device):
    theta = 10000.0 ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    B, H, T, D = x.shape
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLUMLP(nn.Module):
    def __init__(self, n_embd: int, hidden: int, dropout: float, group_size: int = 64):
        super().__init__()
        self.gate_proj = BitLinear(n_embd, hidden, bias=False, group_size=group_size)
        self.up_proj = BitLinear(n_embd, hidden, bias=False, group_size=group_size)
        self.down_proj = BitLinear(hidden, n_embd, bias=False, group_size=group_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.drop(self.down_proj(gate * up))


class SparseMoEBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_experts: int,
        hidden: int,
        dropout: float,
        group_size: int = 64,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUMLP(n_embd, hidden, dropout, group_size) for _ in range(n_experts)]
        )

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)
        router_logits = self.router(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1)
        top1_probs, top1_indices = torch.max(routing_probs, dim=-1)
        out_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = top1_indices == i
            if mask.any():
                expert_in = x_flat[mask]
                expert_out = expert(expert_in)
                scale = top1_probs[mask].unsqueeze(-1)
                expert_out = expert_out * (scale - scale.detach() + 1.0)
                out_flat[mask] = expert_out
        out = out_flat.view(B, T, C)
        route_frac = torch.bincount(
            top1_indices, minlength=self.n_experts
        ).float() / top1_indices.size(0)
        prob_mean = routing_probs.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(route_frac * prob_mean)
        return out, aux_loss


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, group_size: int = 64):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = BitLinear(
            n_embd,
            n_embd + 2 * n_kv_head * self.head_dim,
            bias=False,
            group_size=group_size,
        )
        self.proj = BitLinear(n_embd, n_embd, bias=False, group_size=group_size)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
            persistent=False,
        )

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(
            [
                self.n_head * self.head_dim,
                n_kv_head * self.head_dim,
                n_kv_head * self.head_dim,
            ],
            dim=2,
        )
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.n_head // n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_head // n_kv_head, dim=1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_experts: int,
        hidden: int,
        dropout: float,
        group_size: int = 64,
    ):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, group_size)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = SparseMoEBlock(n_embd, n_experts, hidden, dropout, group_size)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        mlp_out, aux_loss = self.mlp(self.ln2(x))
        x = x + mlp_out
        return x, aux_loss


class Transformer(nn.Module):
    def __init__(self, group_size: int = 64):
        super().__init__()
        self.group_size = group_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd,
                    n_head,
                    n_experts,
                    moe_hidden,
                    dropout,
                    group_size=group_size,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = BitLinear(n_embd, vocab_size, bias=False, group_size=group_size)
        self.lm_head.weight = self.tok_emb.weight
        head_dim = n_embd // n_head
        cos, sin = precompute_freqs(head_dim, block_size, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=True)
        self.register_buffer("rope_sin", sin, persistent=True)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, BitLinear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.tok_emb(idx)
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)
        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x, cos, sin)
            total_aux_loss += aux_loss
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                label_smoothing=label_smoothing,
            )
            loss = ce_loss + 0.01 * total_aux_loss
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temp=temperature, top_k=1, rep_penalty=1.0):
        prompt_len = len(decode(idx[0].tolist()))
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].clone()
            for token_id in set(idx_cond[0].tolist()):
                score = logits[0, token_id]
                logits[0, token_id] = (
                    score / rep_penalty if score > 0 else score * rep_penalty
                )
            logits = logits / temp
            if top_k > 0 and top_k < logits.size(-1):
                thresh = torch.topk(logits, top_k).values[:, -1, None]
                logits[logits < thresh] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() == 0:
                break
            idx = torch.cat((idx, next_id), dim=1)
            current_text = decode(idx[0].tolist())
            new_text = current_text[prompt_len:]
            if "\ufffd" not in new_text:
                print(new_text, end="", flush=True)
                prompt_len = len(current_text)
            if "\n" in new_text:
                break
        print()
        return idx


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    if DIST:
        # Average the local estimates across ranks so every rank logs the
        # same global numbers (keeps early-stopping decisions in sync).
        import torch.distributed as dist

        # NCCL needs CUDA tensors, gloo needs CPU tensors.
        _dev = device if torch.cuda.is_available() else "cpu"
        t = torch.tensor([out["train"], out["val"]], device=_dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= WORLD
        out = {"train": t[0].item(), "val": t[1].item()}
    model.train()
    return out


def convert_to_bitlinear(model: nn.Module) -> nn.Module:
    for name, child in model.named_children():
        if isinstance(child, BitLinear):
            setattr(model, name, child.to_inference())
        else:
            convert_to_bitlinear(child)
    return model


def train():
    global model
    engine = globals().get("engine", None)
    is_main = RANK == 0
    if DIST:
        # Model weights were initialized identically on all ranks (torch seed
        # 1337 at import). Now decorrelate per-rank stochasticity (dropout
        # masks) now that init is done; `random` drives per-rank batch sampling.
        torch.manual_seed(1337 + RANK)
        random.seed(1337 + RANK)
    best_val_loss = float("inf")
    patience_counter = 0
    # In the deepspeed path `optimizer` is the BASE torch optimizer (see
    # __main__: the DeepSpeedZeroOptimizer wrapper is not a torch Optimizer,
    # so the torch scheduler + manual warmup keep driving the base optimizer,
    # whose param_groups the engine reads at step time). Grad clipping comes
    # from the ds config.
    opt = optimizer
    scheduler = CosineAnnealingLR(opt, T_max=20000, eta_min=lr_min)
    for it in range(start_iter, max_iters + 1):
        if it < warmup_iters:
            warmup_lr = lr * (it + 1) / warmup_iters
            for pg in opt.param_groups:
                pg["lr"] = warmup_lr
        if it % eval_interval == 0:
            losses = estimate_loss()
            # NOTE: losses are identical on all ranks (all-reduced), so every
            # rank updates counters and hits `break` together. Only rank 0
            # prints/saves to avoid duplicate logs and file races.
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                patience_counter = 0
                if is_main:
                    cur_lr = opt.param_groups[0]["lr"]
                    print(
                        f"iter {it :5d} | train {losses ['train']:.4f} | val {losses ['val']:.4f} "
                        f"| lr {cur_lr :.2e}"
                    )
                    torch.save(model.state_dict(), checkpoint + ".best")
                    print(f"  * new best ({best_val_loss:.4f}) saved")
            else:
                patience_counter += 1
                if is_main:
                    cur_lr = opt.param_groups[0]["lr"]
                    print(
                        f"iter {it :5d} | train {losses ['train']:.4f} | val {losses ['val']:.4f} "
                        f"| lr {cur_lr :.2e}"
                    )
                if patience_counter >= patience:
                    if is_main:
                        print("Early stopping triggered.")
                    break
        xb, yb = get_batch("train")
        if engine is not None:
            logits, loss = engine(xb, yb)
            engine.backward(loss)
            engine.step()
        else:
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if it >= warmup_iters:
            scheduler.step()
    if not is_main:
        if DIST:
            import torch.distributed as dist

            dist.barrier()
        return
    torch.save(model.state_dict(), checkpoint)
    print("Full-precision model saved to", checkpoint)
    if os.path.isfile(checkpoint + ".best"):
        model.load_state_dict(torch.load(checkpoint + ".best", map_location=device))
        model.eval()
        print("Loaded best checkpoint for quantization.")
    q_model = convert_to_bitlinear(deepcopy(model))
    with gzip.open(checkpoint + ".quantized", "wb") as f:
        torch.save(q_model, f)
    quant_size = os.path.getsize(checkpoint + ".quantized")
    print(
        f"Quantized model saved → {checkpoint }.quantized  ({quant_size /1024 :.1f} KB)"
    )
    if DIST:
        import torch.distributed as dist

        dist.barrier()


def load_model():
    global model
    if os.path.isfile(checkpoint + ".quantized"):
        print("Loading quantized model (primary)...")
        with gzip.open(checkpoint + ".quantized", "rb") as f:
            model = torch.load(f, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        print("Quantized model ready.")
        return
    src = None
    if os.path.isfile(checkpoint + ".best"):
        src = checkpoint + ".best"
    elif os.path.isfile(checkpoint):
        src = checkpoint
    if src:
        print(f"Loading unquantized checkpoint from {src } and quantizing...")
        model.load_state_dict(torch.load(src, map_location=device))
        model.eval()
        q_model = convert_to_bitlinear(deepcopy(model))
        with gzip.open(checkpoint + ".quantized", "wb") as f:
            torch.save(q_model, f)
        print(f"Quantized and saved → {checkpoint }.quantized")
        with gzip.open(checkpoint + ".quantized", "rb") as f:
            model = torch.load(f, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        return
    print("No checkpoint found — training from scratch...")
    train()
    load_model()


conversation_history = "User: hi\nBot: Hello! How can I help you today?\n"


def generate_reply(max_token_length):
    global conversation_history
    conversation_history = ""
    user_input = input("User: ")
    if not user_input.strip():
        return
    user_input = user_input.strip()
    clean_input = user_input.lower().strip("?!.")
    if clean_input.startswith("i'm"):
        clean_input = "I'm" + clean_input[3:]
    elif clean_input.startswith("i "):
        clean_input = "I " + clean_input[2:]

    conversation_history += f"User: {clean_input}\nBot:"
    while len(conversation_history) > 1000:
        idx = conversation_history.find("\n")
        if idx == -1:
            conversation_history = conversation_history[-1000:]
            break
        conversation_history = conversation_history[idx + 1 :]
    model.to("cpu")
    context = encode(conversation_history).unsqueeze(0).to("cpu")
    print("Bot:", end="", flush=True)
    prompt_len = len(decode(context[0].tolist()))
    output_ids = model.generate(context, max_new_tokens=max_token_length)
    full_output = decode(output_ids[0].tolist())
    bot_reply = full_output[prompt_len:].strip()
    if "\nUser:" in bot_reply:
        bot_reply = bot_reply.split("\nUser:")[0]
    conversation_history += bot_reply + "\n"
    model.to(device)


if __name__ == "__main__":
    import sys

    force_train = "--train" in sys.argv
    use_fp_adam = "--fp-adam" in sys.argv or "--no-4bit" in sys.argv
    engine = None
    ds_mode = USE_DEEPSPEED and force_train
    if USE_DEEPSPEED and not force_train:
        print("NOTE: --deepspeed only affects --train; running single-device mode.")
    if ds_mode:
        # Multi-GPU data-parallel via DeepSpeed ZeRO. The launcher
        # (`deepspeed --num_gpus=N ...`) injects --local_rank and NCCL env vars.
        try:
            import deepspeed
        except ImportError:
            print("ERROR: --deepspeed needs the `deepspeed` package: pip install deepspeed")
            sys.exit(1)
        deepspeed.init_distributed()
        import torch.distributed as dist

        DIST = True
        RANK = dist.get_rank()
        WORLD = dist.get_world_size()
        _local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(_local_rank)
            device = f"cuda:{_local_rank}"
        # Different DATA per rank: get_batch() samples via `random`, so each
        # rank must draw DIFFERENT batches (else both GPUs redo the same work).
        # The shared initial shuffle at import (seed 1337) stays identical, and
        # the torch seed is deliberately left at 1337 so every rank
        # initializes IDENTICAL weights (required for data-parallel: averaged
        # grads are only valid on identical models). Dropout is decorrelated
        # per-rank at the top of train(), i.e. after init.
        random.seed(1337 + RANK)
        if RANK == 0:
            print(f"DeepSpeed ZeRO mode: world_size={WORLD}, config={DS_CONFIG}")
    model = Transformer(group_size=qat_group_size).to(device)
    if ds_mode:
        # ZeRO shards the optimizer states itself, so the 4-bit torchao AdamW
        # (single-GPU only) is intentionally bypassed here.
        print("Using standard 32-bit AdamW optimizer (required for DeepSpeed ZeRO)...")
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        # NOTE: deepspeed.initialize wraps base_optimizer in a
        # DeepSpeedZeroOptimizer (which is NOT a torch Optimizer and must not
        # be passed to torch LR schedulers). The engine reads LR from the BASE
        # optimizer's param_groups at step time, so warmup + CosineAnnealingLR
        # in train() keep driving `base_optimizer` — same schedule as 1-GPU.
        model_engine, _, _, _ = deepspeed.initialize(
            model=model, optimizer=base_optimizer, config=DS_CONFIG
        )
        optimizer = base_optimizer
        engine = model_engine
    elif AdamW4bit is not None and not use_fp_adam:
        print("Using 4-bit AdamW optimizer (torchao.optim.AdamW4bit)...")
        optimizer = AdamW4bit(model.parameters(), lr=lr, weight_decay=0.05)
    else:
        print("Using standard 32-bit AdamW optimizer (torch.optim.AdamW)...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    if force_train:
        print("Forcing training from scratch...")
        train()
        if ds_mode:
            # Rank 0 already saved .pt / .best / .quantized inside train().
            import torch.distributed as dist

            dist.destroy_process_group()
            sys.exit(0)
        load_model()
        print("Training and quantization complete. Exiting.")
        sys.exit(0)
    load_model()
    while True:
        generate_reply(50)
