import subprocess
import multiprocessing
import os 
import argparse
import pandas as pd
import numpy as np

def process_chunk(input_file_path, model):
    
    output_file_path = input_file_path.replace("input", "output")
    command = [
        "olga-compute_pgen",
        model,
        "-i", input_file_path,
        "-o", output_file_path
    ]

    if not os.path.exists(output_file_path):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            # Process the result if needed
            #print("Command output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error executing command:", e)
            print("Command output:", e.stdout)
            print("Command error:", e.stderr)
    else:
        print("Output File already exists, skip file...")

class Copier(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, input_file_path):
        process_chunk(input_file_path, self.model)

def compute_generation_probabilities_from_clone_files(clones_df, model):
    olga_input_file = "Olga_input.tsv"
    olga_df = clones_df[["CDR3.nucleotide.sequence", "bestVGene", "bestJGene"]].copy()
    olga_df.rename(columns = {"CDR3.nucleotide.sequence": "V1", "bestVGene": "V2", "bestJGene": "V3"}, inplace = True)
    olga_df.to_csv(olga_input_file, sep="\t", header=False, index=False)

    run_olga(olga_input_file, model)

    results = pd.read_csv(olga_input_file.replace("input", "output"), sep="\t", header=None)
    df = clones_df.copy()
    df.loc[olga_df.index, "olga_pgen_cdr3"] = list(results[1])
    df.loc[olga_df.index, "olga_pgen_aa"] = list(results[3])

    return df
    
def run_olga(input_file, model):
    df_to_chunk = pd.read_csv(input_file, sep='\t', header=None)
    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    start_point = 0
    chunk_length = int(np.ceil(len(df_to_chunk.iloc[start_point:])/num_processes))
    end_points = np.cumsum([start_point] + [chunk_length for i in range(num_processes)])

    chunk_file_name = input_file.split(".")[0]+"_%d.tsv"

    input_files = []
    for i, end_point in enumerate(end_points):
        if i == 0:
            continue
        elif i == len(end_points)-1:
            df_to_chunk.iloc[end_points[i-1]:].to_csv(chunk_file_name % i, sep="\t", header=False, index=False)
        else:
            df_to_chunk.iloc[end_points[i-1]:end_point].to_csv(chunk_file_name % i, sep="\t", header=False, index=False)
        input_files.append(chunk_file_name % i)

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(Copier(model), input_files)

    if "input" in input_file:
        output_files = [i.replace("input", "output") for i in input_files]
        output_file = input_file.replace("input", "output")
    else:
        output_files = [i.split(".")[0]+ "_output.tsv"]
        output_file = input_file.split(".")[0]+ "_output.tsv"

    results = []
    for o in output_files:
        olga_results_o = pd.read_csv(o, sep="\t", header=None)
        results.append(olga_results_o)

    results = pd.concat(results)
    results.to_csv(output_file, sep="\t", header=False, index=False)

    for i,f in enumerate(input_files):
        os.remove(f)
        os.remove(output_files[i])

def main():
    parser = argparse.ArgumentParser(description="Calculate generation probability for CDR3 sequences according to a generative V(D)J model. Input Table with 3 columns: sequence, Vgen, Jgen")
    # Add arguments using add_argument method
    parser.add_argument('--i', type=str, help='Input file')
    parser.add_argument('--model', type=str, help='Model to use with Olga (e.g. humanIGH)')


    args = parser.parse_args()
    if args.i:
        input_file = args.i
    else:
        print("Please provide a valid input file...")
    if args.model:
        model = "--" + args.model
    else:
        print("Please prov a valid model specification...")

    run_olga(input_file, model)

if __name__ == "__main__":
    main()