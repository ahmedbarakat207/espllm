import math, torch, torch.nn as nn
from tabnanny import check
from torch.nn import functional as F
import torch.quantization
import os.path

if torch.cuda.is_available():
    device = "cuda"
#elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #device = "mps"
else:
    device = "cpu"
checkpoint   = "./model/model.pt"
block_size   = 128
batch_size   = 32
n_layer      = 4
n_head       = 4
n_embd       = 256 
dropout      = 0.1
max_iters    = 2000
eval_interval= 200
lr           = 3e-3 
eval_iters   = 200
generate_tokens = 400 
tempreature  = 0.7      
start_iter   = 0     
patience     = 5

torch.manual_seed(1337)  

dataset = open("dataset.txt", "r+")
text = dataset.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)                   
stoi = {ch:i for i,ch in enumerate(chars)} 
itos = {i:ch for ch,i in stoi.items()} 
encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
decode = lambda t: "".join(itos[int(i)] for i in t)

# Prepare train/validation split
data = encode(text)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]
def get_batch(split):
    d = train_data if split == "train" else val_data
    cur_block = min(block_size, max(2, len(d) - 2))
    hi = len(d) - cur_block - 1
    if hi <= 0:
        x = d[:cur_block].unsqueeze(0)
        y = d[1:cur_block+1].unsqueeze(0)
        return x.to(device), y.to(device)

    ix = torch.randint(hi, (batch_size,))
    x = torch.stack([d[i:i+cur_block] for i in ix])
    y = torch.stack([d[i+1:i+cur_block+1] for i in ix])
    return x.to(device), y.to(device)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1,1,block_size,block_size))

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        q = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf")) 
        att = F.softmax(att, dim=-1)                               
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x)) 
        x = x + self.mlp(self.ln2(x))   
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temp = tempreature):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train","val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses)/len(losses)
    model.train()
    return out

def quantize_model(model, quantization):
    model.eval()
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=quantization)
    return quantized
    
def train():
    best_val_loss = float('inf')
    patience_counter = 0
    for it in range(start_iter, max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss()
            print(f"iter {it:4d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint + ".best")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping.")
                    break
    
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.backends.quantized.engine = 'qnnpack'
    quantized_model = quantize_model(model, torch.qint8)
    torch.save(quantized_model.state_dict(), checkpoint)
    print("Model Trained Successfully, Saved Checkpoint in ../model/model.pt")

model = Transformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

if (os.path.isfile(checkpoint)):
    print("Checkpoint Found, Starting the model")
    checkpoint_load = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_load)
else:
    print("No Checkpoint Found, Training the Model")
    train()

def generate_reply(max_token_length):
    user_input = input("User: ")
    prompt = f"User: {user_input}\nBot:"
    context = encode(prompt).unsqueeze(0).to(device)
    output_ids = model.generate(context, max_new_tokens=max_token_length)
    full_response = decode(output_ids[0].tolist())

    bot_idx = full_response.find("Bot:")
    if bot_idx != -1:
        reply = full_response[bot_idx + 4:].strip()
        if "\nUser:" in reply:
            reply = reply.split("\nUser:")[0]
        print("Bot:", reply)
    else:
        print(full_response) 

while True:
    generate_reply(20)