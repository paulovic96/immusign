import subprocess
import multiprocessing
import os 
import h5py
from tqdm import tqdm

from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import numpy as np

model_dir = Path('./seqVec/uniref50_v2')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
embedder = ElmoEmbedder(options,weights, cuda_device=-1) # cuda_device=-1 for CPU

def process_chunk(input_file_path):
    output_file_path = input_file_path.replace("input", "output").replace(".txt", ".h5")  
    input_file_path = os.path.join("input_seqvec",input_file_path)
   
    with open (input_file_path, "r") as file:
            aa_list = [list(line.strip()) for line in file]
            embedding = embedder.embed_sentences(aa_list)

    if not os.path.exists(output_file_path):
        try:
            with h5py.File(output_file_path, "w") as h5file:    
                group = h5file.create_group("arrays")
                # Iterate and generate your 3D arrays
                with open(output_file_path.replace(".h5", ".txt")  , "w") as txt_file:
                    i=0
                    for emb in embedding:
                        # Create a dataset for the current array
                        dataset = group.create_dataset(f"array_{i}", data=emb, dtype="float32")
                        txt_file.write("%d\n" %i)  # Add a separator line
                        i+=1
        except Exception as e:
            print("Error executing command:", e)
            print("Command output:", e.stdout)
            print("Command error:", e.stderr)
    else:
        print("Output File already exists, skip file...")

def main():
    input_files = os.listdir("input_seqvec")
    input_files = [x for x in os.listdir("input_seqvec") if ".txt" in x]
    input_files.sort(key = lambda x: int(x.split(".")[0].split("_")[-1]))

    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_chunk, input_files)

if __name__ == "__main__":
    main()