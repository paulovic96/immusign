import torch
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel
from sys import platform
import numpy as np
import pandas as pd
import os
import argparse

if platform == "darwin":
    device = "mps" if torch.backends.mps.is_available() else "cpu"  
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
model.eval().to(device)


def main():
    parser = argparse.ArgumentParser(description="Embed AS-Sequences stored sorted in a .txt file")
    # Add arguments using add_argument method
    parser.add_argument('--i', type=str, help='Input file')
    parser.add_argument('--o', type=str, help='Output file')
    parser.add_argument('--max_chunk_length', type=int, help='Maximum length to chunk AS')


    args = parser.parse_args()
    if args.i:
        AAS_PATH = args.i
    else:
        AAS_PATH = "aa_cdr3.txt"
    if args.o:
        EMBEDDINGS_PATH = args.o
    else:
        EMBEDDINGS_PATH = "aa_cdr3_embeddings.npy" 
    if args.max_chunk_length:
        MAX_CHUNK_LENGTH = args.max_chunk_length
    else:
        MAX_CHUNK_LENGTH = 500

    
        
   
    
    if not os.path.exists(EMBEDDINGS_PATH):
        with open (AAS_PATH, "r") as file:
            aa_list = [line.strip().replace("_","-").replace("*", "X") for line in file] # ESM gap, deletion = "-" or ".", stop_codon = "X"

        l, c = np.unique(pd.Series(aa_list).apply(len), return_counts=True)
        len_endpoints = np.cumsum(c)    


        # chunk same size as to avoid padding 
        aa_chunks = []
        for i, p_end in enumerate(len_endpoints):    
            if i == 0:
                p_start = 0
            else:
                p_start = len_endpoints[i-1]    
            if c[i] <= MAX_CHUNK_LENGTH:
                aa_chunks.append(aa_list[p_start:p_end])
            else:
                n_max_chunks = c[i]//MAX_CHUNK_LENGTH
                for _ in range(n_max_chunks):
                    aa_chunks.append(aa_list[p_start:(p_start + MAX_CHUNK_LENGTH)])
                    p_start += MAX_CHUNK_LENGTH
                aa_chunks.append(aa_list[p_start:p_end])

        store_embeddings = np.zeros((len(aa_list),320)) # hidden dim = 320
        
        i_start = 0
        for aas in tqdm(aa_chunks):
            inputs = tokenizer(aas, return_tensors="pt", padding=False, truncation=False).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            x = last_hidden_states.detach()     
            embedding = x[:,1:-1,:].mean(axis=1) # drop the initial beginning of sentence token and average to get embedding per protein
            i_end = i_start + len(embedding)
            store_embeddings[i_start:i_end,:] = embedding.to("cpu")
            i_start = i_end

        np.save(EMBEDDINGS_PATH, store_embeddings)
    
    else:
        print("Already existing Embedding: ",EMBEDDINGS_PATH)
    

if __name__ == "__main__":
    main()