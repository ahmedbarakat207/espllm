from tokenizers import ByteLevelBPETokenizer 
def main ():
    print ("Training BPE Tokenizer on dataset.txt...")
    tokenizer =ByteLevelBPETokenizer ()
    tokenizer .train (
    files =["dataset.txt"],
    vocab_size =2048 ,
    min_frequency =2 ,
    special_tokens =[
    "<|endoftext|>"
    ]
    )
    tokenizer .save_model (".","bpe")
    print ("Saved bpe-vocab.json and bpe-merges.txt")
if __name__ =="__main__":
    main ()
