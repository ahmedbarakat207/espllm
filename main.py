import math ,torch ,torch .nn as nn 
from torch .nn import functional as F 
from copy import deepcopy 
from tokenizers import ByteLevelBPETokenizer 
import os ,gzip 
from torch .optim .lr_scheduler import CosineAnnealingLR 

if torch .cuda .is_available ():
    device ="cuda"
elif torch .backends .mps .is_available ():
    device ="mps"
else :
    device ="cpu"
    
checkpoint ="./model/model.pt"
block_size =128 
batch_size =32 
n_layer =6 
n_head =4 
n_kv_head =1 
n_embd =128 
n_experts =16 
moe_hidden =128 
dropout =0.25 
max_iters =20000 
eval_interval =100 
lr =2e-3 
lr_min =1e-5 
warmup_iters =300 
eval_iters =10 
generate_tokens =400 
temperature =0.6 
start_iter =0 
patience =10 
label_smoothing =0.0 
qat_group_size =64 

torch .manual_seed (1337 )

dataset =open ("dataset.txt","r+")
text =dataset .read ()

tokenizer =ByteLevelBPETokenizer (
"bpe-vocab.json",
"bpe-merges.txt",
)

vocab_size =tokenizer .get_vocab_size ()
print (f"BPE Vocab Size: {vocab_size }")
encode =lambda s :torch .tensor (tokenizer .encode (s ).ids ,dtype =torch .long )
decode =lambda t :tokenizer .decode (t .tolist ()if hasattr (t ,'tolist')else t )
data =encode (text )
n =int (0.9 *len (data ))
train_data ,val_data =data [:n ],data [n :]

def get_batch (split ):
    d =train_data if split =="train"else val_data 
    cur_block =min (block_size ,max (2 ,len (d )-2 ))
    hi =len (d )-cur_block -1 
    if hi <=0 :
        x =d [:cur_block ].unsqueeze (0 )
        y =d [1 :cur_block +1 ].unsqueeze (0 )
        return x .to (device ),y .to (device )
    ix =torch .randint (hi ,(batch_size ,))
    x =torch .stack ([d [i :i +cur_block ]for i in ix ])
    y =torch .stack ([d [i +1 :i +cur_block +1 ]for i in ix ])
    return x .to (device ),y .to (device )

def _int4_pack (q :torch .Tensor )->torch .Tensor :
    out ,n =q .shape 
    if n %2 :
        q =F .pad (q ,(0 ,1 ))
        n +=1 
    lo =q [:,0 ::2 ]&0x0F 
    hi =(q [:,1 ::2 ]<<4 )&0xF0 
    return (lo |hi ).to (torch .uint8 )

def quantize_tensor_int4 (w :torch .Tensor ,group_size :int =64 ):
    out ,n =w .shape 
    n_groups =math .ceil (n /group_size )
    padded_n =n_groups *group_size 
    if padded_n !=n :
        w =F .pad (w ,(0 ,padded_n -n ))
    wg =w .view (out ,n_groups ,group_size )
    wmin =wg .min (dim =-1 ,keepdim =True ).values 
    wmax =wg .max (dim =-1 ,keepdim =True ).values 
    scale =(wmax -wmin )/15.0 
    scale =torch .where (scale ==0 ,torch .ones_like (scale ),scale )
    zp =torch .round (-wmin /scale )
    q =torch .clamp (torch .round (wg /scale )+zp ,0 ,15 ).to (torch .uint8 )
    q =q .view (out ,padded_n )[:,:n ]
    return _int4_pack (q ),scale .squeeze (-1 ),zp .squeeze (-1 )

def dequantize_tensor_int4 (q_packed :torch .Tensor ,scale :torch .Tensor ,
zp :torch .Tensor ,n :int ,group_size :int =64 ):
    m =q_packed .shape [1 ]
    q =torch .empty (q_packed .shape [0 ],m *2 ,dtype =torch .uint8 ,device =q_packed .device )
    q [:,0 ::2 ]=q_packed &0x0F 
    q [:,1 ::2 ]=(q_packed >>4 )&0x0F 
    q =q [:,:n ].float ()
    groups =torch .arange (n ,device =q .device )//group_size 
    return (q -zp .float ()[:,groups ])*scale .float ()[:,groups ]

class QATLinear (nn .Module ):
    def __init__ (self ,in_features :int ,out_features :int ,bias :bool =True ,
    group_size :int =64 ):
        super ().__init__ ()
        self .in_features =in_features 
        self .out_features =out_features 
        self .group_size =group_size 
        self .weight =nn .Parameter (torch .empty (out_features ,in_features ))
        if bias :
            self .bias =nn .Parameter (torch .zeros (out_features ))
        else :
            self .register_parameter ("bias",None )
        nn .init .kaiming_uniform_ (self .weight ,a =math .sqrt (5 ))

    def _fake_quantize (self ,w :torch .Tensor )->torch .Tensor :
        out ,n =w .shape 
        n_groups =math .ceil (n /self .group_size )
        padded_n =n_groups *self .group_size 
        w_pad =F .pad (w ,(0 ,padded_n -n ))if padded_n !=n else w 
        wg =w_pad .view (out ,n_groups ,self .group_size )
        wmin =wg .min (dim =-1 ,keepdim =True ).values 
        wmax =wg .max (dim =-1 ,keepdim =True ).values 
        scale =(wmax -wmin )/15.0 
        scale =torch .where (scale ==0 ,torch .ones_like (scale ),scale )
        zp =torch .round (-wmin /scale )
        q =torch .clamp (torch .round (wg /scale )+zp ,0 ,15 )
        w_deq =(q -zp )*scale 
        w_deq =w_deq .view (out ,padded_n )[:,:n ]
        return w +(w_deq -w ).detach ()

    def forward (self ,x :torch .Tensor )->torch .Tensor :
        w =self ._fake_quantize (self .weight )
        return F .linear (x ,w ,self .bias )
    def to_int4 (self ):
        return Int4Linear (self .in_features ,self .out_features ,
        self .weight .detach (),
        self .bias .detach ()if self .bias is not None else None ,
        self .group_size )

class Int4Linear (nn .Module ):
    def __init__ (self ,in_features ,out_features ,weight ,bias =None ,group_size =64 ):
        super ().__init__ ()
        self .in_features =in_features 
        self .out_features =out_features 
        self .group_size =group_size 
        q ,scale ,zp =quantize_tensor_int4 (weight .detach (),group_size )
        self .register_buffer ("qweight",q )
        self .register_buffer ("scale",scale .half ())
        self .register_buffer ("zero_point",zp .half ())
        if bias is not None :
            self .bias =nn .Parameter (bias .detach ().half ())
        else :
            self .register_parameter ("bias",None )

    def _dequantize_weight (self ):
        if hasattr (self ,"_cached_weight")and self ._cached_weight is not None :
            return self ._cached_weight 
        w =dequantize_tensor_int4 (
        self .qweight ,self .scale ,self .zero_point ,
        self .in_features ,self .group_size 
        )
        if not self .training :
            self ._cached_weight =w 
        return w

    def forward (self ,x ):
        bias =self .bias .float ()if self .bias is not None else None 
        return F .linear (x ,self ._dequantize_weight (),bias )

def precompute_freqs (head_dim :int ,max_seq_len :int ,device ):
    theta =10000.0 **(-torch .arange (0 ,head_dim ,2 ,device =device ).float ()/head_dim )
    t =torch .arange (max_seq_len ,device =device ).float ()
    freqs =torch .outer (t ,theta )
    return freqs .cos (),freqs .sin ()

def apply_rope (x :torch .Tensor ,cos :torch .Tensor ,sin :torch .Tensor ):
    x1 =x [...,0 ::2 ]
    x2 =x [...,1 ::2 ]
    B ,H ,T ,D =x .shape 
    cos =cos [:T ].unsqueeze (0 ).unsqueeze (0 )
    sin =sin [:T ].unsqueeze (0 ).unsqueeze (0 )
    return torch .cat ([x1 *cos -x2 *sin ,
    x1 *sin +x2 *cos ],dim =-1 )

class SwiGLUMLP (nn .Module ):
    def __init__ (self ,n_embd :int ,hidden :int ,dropout :float ,group_size :int =64 ):
        super ().__init__ ()
        self .gate_proj =QATLinear (n_embd ,hidden ,bias =False ,group_size =group_size )
        self .up_proj =QATLinear (n_embd ,hidden ,bias =False ,group_size =group_size )
        self .down_proj =QATLinear (hidden ,n_embd ,bias =False ,group_size =group_size )
        self .drop =nn .Dropout (dropout )

    def forward (self ,x ):
        gate =F .silu (self .gate_proj (x ))
        up =self .up_proj (x )
        return self .drop (self .down_proj (gate *up ))

class SparseMoEBlock (nn .Module ):
    def __init__ (self ,n_embd :int ,n_experts :int ,hidden :int ,dropout :float ,group_size :int =64 ):
        super ().__init__ ()
        self .n_experts =n_experts 
        self .router =nn .Linear (n_embd ,n_experts ,bias =False )
        self .experts =nn .ModuleList ([
        SwiGLUMLP (n_embd ,hidden ,dropout ,group_size )
        for _ in range (n_experts )
        ])
    def forward (self ,x ):
        B ,T ,C =x .size ()
        x_flat =x .view (-1 ,C )
        router_logits =self .router (x_flat )
        routing_probs =F .softmax (router_logits ,dim =-1 )
        top1_probs ,top1_indices =torch .max (routing_probs ,dim =-1 )
        out_flat =torch .zeros_like (x_flat )
        for i ,expert in enumerate (self .experts ):
            mask =(top1_indices ==i )
            if mask .any ():
                expert_in =x_flat [mask ]
                expert_out =expert (expert_in )
                scale =top1_probs [mask ].unsqueeze (-1 )
                expert_out =expert_out *(scale -scale .detach ()+1.0 )
                out_flat [mask ]=expert_out 
        out =out_flat .view (B ,T ,C )
        route_frac =torch .bincount (top1_indices ,minlength =self .n_experts ).float ()/top1_indices .size (0 )
        prob_mean =routing_probs .mean (dim =0 )
        aux_loss =self .n_experts *torch .sum (route_frac *prob_mean )
        return out ,aux_loss 

class CausalSelfAttention (nn .Module ):
    def __init__ (self ,n_embd :int ,n_head :int ,dropout :float ,group_size :int =64 ):
        super ().__init__ ()
        assert n_embd %n_head ==0 
        self .n_head =n_head 
        self .head_dim =n_embd //n_head 
        self .qkv =QATLinear (n_embd ,n_embd +2 *n_kv_head *self .head_dim ,bias =False ,group_size =group_size )
        self .proj =QATLinear (n_embd ,n_embd ,bias =False ,group_size =group_size )
        self .attn_drop =nn .Dropout (dropout )
        self .resid_drop =nn .Dropout (dropout )
        self .register_buffer (
        "mask",
        torch .tril (torch .ones (block_size ,block_size )).view (1 ,1 ,block_size ,block_size ),
        persistent =False 
        )

    def forward (self ,x ,cos ,sin ):
        B ,T ,C =x .size ()
        qkv =self .qkv (x )
        q ,k ,v =qkv .split ([self .n_head *self .head_dim ,n_kv_head *self .head_dim ,n_kv_head *self .head_dim ],dim =2 )
        q =q .view (B ,T ,self .n_head ,self .head_dim ).transpose (1 ,2 )
        k =k .view (B ,T ,n_kv_head ,self .head_dim ).transpose (1 ,2 )
        v =v .view (B ,T ,n_kv_head ,self .head_dim ).transpose (1 ,2 )
        q =apply_rope (q ,cos ,sin )
        k =apply_rope (k ,cos ,sin )
        k =k .repeat_interleave (self .n_head //n_kv_head ,dim =1 )
        v =v .repeat_interleave (self .n_head //n_kv_head ,dim =1 )
        att =(q @k .transpose (-2 ,-1 ))/math .sqrt (self .head_dim )
        att =att .masked_fill (self .mask [:,:,:T ,:T ]==0 ,float ("-inf"))
        att =F .softmax (att ,dim =-1 )
        att =self .attn_drop (att )
        y =att @v 
        y =y .transpose (1 ,2 ).contiguous ().view (B ,T ,C )
        y =self .resid_drop (self .proj (y ))
        return y 

class RMSNorm (nn .Module ):
    def __init__ (self ,dim :int ,eps :float =1e-5 ):
        super ().__init__ ()
        self .eps =eps 
        self .weight =nn .Parameter (torch .ones (dim ))
    def forward (self ,x :torch .Tensor ):
        norm_x =x *torch .rsqrt (x .pow (2 ).mean (-1 ,keepdim =True )+self .eps )
        return norm_x *self .weight 

class Block (nn .Module ):
    def __init__ (self ,n_embd :int ,n_head :int ,n_experts :int ,hidden :int ,dropout :float ,group_size :int =64 ):
        super ().__init__ ()
        self .ln1 =RMSNorm (n_embd )
        self .attn =CausalSelfAttention (n_embd ,n_head ,dropout ,group_size )
        self .ln2 =RMSNorm (n_embd )
        self .mlp =SparseMoEBlock (n_embd ,n_experts ,hidden ,dropout ,group_size )

    def forward (self ,x ,cos ,sin ):
        x =x +self .attn (self .ln1 (x ),cos ,sin )
        mlp_out ,aux_loss =self .mlp (self .ln2 (x ))
        x =x +mlp_out 
        return x ,aux_loss 

class Transformer (nn .Module ):
    def __init__ (self ,group_size :int =64 ):
        super ().__init__ ()
        self .group_size =group_size 
        self .tok_emb =nn .Embedding (vocab_size ,n_embd )
        self .drop =nn .Dropout (dropout )
        self .blocks =nn .ModuleList ([Block (n_embd ,n_head ,n_experts ,moe_hidden ,dropout ,group_size =group_size )for _ in range (n_layer )])
        self .ln_f =RMSNorm (n_embd )
        self .lm_head =QATLinear (n_embd ,vocab_size ,bias =False ,group_size =group_size )
        self .lm_head .weight =self .tok_emb .weight 
        head_dim =n_embd //n_head 
        cos ,sin =precompute_freqs (head_dim ,block_size ,device ="cpu")
        self .register_buffer ("rope_cos",cos ,persistent =True )
        self .register_buffer ("rope_sin",sin ,persistent =True )
        self .apply (self ._init_weights )

    def _init_weights (self ,module ):
        if isinstance (module ,nn .Linear ):
            torch .nn .init .normal_ (module .weight ,mean =0.0 ,std =0.02 )
            if module .bias is not None :
                torch .nn .init .zeros_ (module .bias )
        elif isinstance (module ,QATLinear ):
            torch .nn .init .normal_ (module .weight ,mean =0.0 ,std =0.02 )

    def forward (self ,idx ,targets =None ):
        B ,T =idx .size ()
        x =self .tok_emb (idx )
        cos =self .rope_cos [:T ].to (x .device )
        sin =self .rope_sin [:T ].to (x .device )
        total_aux_loss =0.0 
        for block in self .blocks :
            x ,aux_loss =block (x ,cos ,sin )
            total_aux_loss +=aux_loss 
        x =self .ln_f (x )
        logits =self .lm_head (x )
        loss =None 
        if targets is not None :
            ce_loss =F .cross_entropy (
            logits .view (-1 ,logits .size (-1 )),
            targets .view (-1 ),
            label_smoothing =label_smoothing 
            )
            loss =ce_loss +0.01 *total_aux_loss 
        return logits ,loss 
    @torch .no_grad ()

    def generate (self ,idx ,max_new_tokens ,temp =temperature ,top_k =1 ,rep_penalty =1.0 ):
        prompt_len =len (decode (idx [0 ].tolist ()))
        for _ in range (max_new_tokens ):
            idx_cond =idx [:,-block_size :]
            logits ,_ =self (idx_cond )
            logits =logits [:,-1 ,:].clone ()
            for token_id in set (idx_cond [0 ].tolist ()):
                score =logits [0 ,token_id ]
                logits [0 ,token_id ]=score /rep_penalty if score >0 else score *rep_penalty 
            logits =logits /temp 
            if top_k >0 and top_k <logits .size (-1 ):
                thresh =torch .topk (logits ,top_k ).values [:,-1 ,None ]
                logits [logits <thresh ]=float ("-inf")
            probs =F .softmax (logits ,dim =-1 )
            next_id =torch .multinomial (probs ,num_samples =1 )
            if next_id .item ()==0 :
                break 
            idx =torch .cat ((idx ,next_id ),dim =1 )
            current_text =decode (idx [0 ].tolist ())
            new_text =current_text [prompt_len :]
            if '\ufffd'not in new_text :
                print (new_text ,end ="",flush =True )
                prompt_len =len (current_text )
            if '\n'in new_text :
                break 
        print ()
        return idx 

@torch .no_grad ()
def estimate_loss ():
    model .eval ()
    out ={}
    for split in ["train","val"]:
        losses =[]
        for _ in range (eval_iters ):
            xb ,yb =get_batch (split )
            _ ,loss =model (xb ,yb )
            losses .append (loss .item ())
        out [split ]=sum (losses )/len (losses )
    model .train ()
    return out 
def convert_to_int4 (model :nn .Module )->nn .Module :
    for name ,child in model .named_children ():
        if isinstance (child ,QATLinear ):
            setattr (model ,name ,child .to_int4 ())
        else :
            convert_to_int4 (child )
    return model 

def train ():
    global model 
    best_val_loss =float ("inf")
    patience_counter =0 
    scheduler =CosineAnnealingLR (optimizer ,T_max =20000 ,eta_min =lr_min )
    for it in range (start_iter ,max_iters +1 ):
        if it <warmup_iters :
            warmup_lr =lr *(it +1 )/warmup_iters 
            for pg in optimizer .param_groups :
                pg ["lr"]=warmup_lr 
        if it %eval_interval ==0 :
            losses =estimate_loss ()
            cur_lr =optimizer .param_groups [0 ]["lr"]
            print (f"iter {it :5d} | train {losses ['train']:.4f} | val {losses ['val']:.4f} "
            f"| lr {cur_lr :.2e}")
            if losses ["val"]<best_val_loss :
                best_val_loss =losses ["val"]
                patience_counter =0 
                torch .save (model .state_dict (),checkpoint +".best")
                print (f"  ✓ new best ({best_val_loss :.4f}) saved")
            else :
                patience_counter +=1 
                if patience_counter >=patience :
                    print ("Early stopping triggered.")
                    break 
        xb ,yb =get_batch ("train")
        logits ,loss =model (xb ,yb )
        optimizer .zero_grad (set_to_none =True )
        loss .backward ()
        torch .nn .utils .clip_grad_norm_ (model .parameters (),1.0 )
        optimizer .step ()
        if it >=warmup_iters :
            scheduler .step ()
    torch .save (model .state_dict (),checkpoint )
    print ("Full-precision model saved to",checkpoint )
    if os .path .isfile (checkpoint +".best"):
        model .load_state_dict (torch .load (checkpoint +".best",map_location =device ))
        model .eval ()
        print ("Loaded best checkpoint for quantization.")
    q_model =convert_to_int4 (deepcopy (model ))
    with gzip .open (checkpoint +".quantized","wb")as f :
        torch .save (q_model ,f )
    quant_size =os .path .getsize (checkpoint +".quantized")
    print (f"Quantized model saved → {checkpoint }.quantized  ({quant_size /1024 :.1f} KB)")

def load_model ():
    global model 
    if os .path .isfile (checkpoint +".quantized"):
        print ("Loading quantized model (primary)...")
        with gzip .open (checkpoint +".quantized","rb")as f :
            model =torch .load (f ,map_location =device ,weights_only =False )
        model .to (device )
        model .eval ()
        print ("Quantized model ready.")
        return 
    src =None 
    if os .path .isfile (checkpoint +".best"):
        src =checkpoint +".best"
    elif os .path .isfile (checkpoint ):
        src =checkpoint 
    if src :
        print (f"Loading unquantized checkpoint from {src } and quantizing...")
        model .load_state_dict (torch .load (src ,map_location =device ))
        model .eval ()
        q_model =convert_to_int4 (deepcopy (model ))
        with gzip .open (checkpoint +".quantized","wb")as f :
            torch .save (q_model ,f )
        print (f"Quantized and saved → {checkpoint }.quantized")
        with gzip .open (checkpoint +".quantized","rb")as f :
            model =torch .load (f ,map_location =device ,weights_only =False )
        model .to (device )
        model .eval ()
        return 
    print ("No checkpoint found — training from scratch...")
    train ()
    load_model ()

conversation_history ="User: hi\n Bot:Hello"
def generate_reply (max_token_length ):
    global conversation_history 
    user_input =input ("User: ")
    if not user_input .strip ():
        return 
    user_input =user_input .strip ().lower ()
    conversation_history +=f"User: {user_input }\nBot:"
    while len(conversation_history) > 1000:
        idx = conversation_history.find('\n')
        if idx == -1:
            conversation_history = conversation_history[-1000:]
            break
        conversation_history = conversation_history[idx+1:]
    model .to ("cpu")
    context =encode (conversation_history ).unsqueeze (0 ).to ("cpu")
    print ("Bot:",end ="",flush =True )
    prompt_len =len (decode (context [0 ].tolist ()))
    output_ids =model .generate (context ,max_new_tokens =max_token_length )
    full_output =decode (output_ids [0 ].tolist ())
    bot_reply =full_output [prompt_len :].strip ()
    if "\nUser:"in bot_reply :
        bot_reply =bot_reply .split ("\nUser:")[0 ]
    conversation_history +=bot_reply +"\n"
    model .to (device )

if __name__ =="__main__":
    import sys 
    force_train ="--train"in sys .argv 
    model =Transformer (group_size =qat_group_size ).to (device )
    optimizer =torch .optim .AdamW (model .parameters (),lr =lr ,weight_decay =0.05 )
    if force_train :
        print ("Forcing training from scratch...")
        train ()
        load_model ()
        print ("Training and quantization complete. Exiting.")
        sys .exit (0 )
    load_model ()
    while True :
        generate_reply (50 )
