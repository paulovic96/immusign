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



    #input_files = ["Olga_input_igh_1.tsv", "Olga_input_igh_2.tsv", "Olga_input_igh_3.tsv", 
    #               "Olga_input_igh_4.tsv", "Olga_input_igh_5.tsv", "Olga_input_igh_6.tsv",
    #               "Olga_input_igh_7.tsv", "Olga_input_igh_8.tsv"]


    df_to_chunk = pd.read_csv(input_file,sep='\t', header=None)
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

    output_files = [i.replace("input", "output") for i in input_files]

    results = []
    for o in output_files:
        olga_results_o = pd.read_csv(o, sep="\t", header=None)
        results.append(olga_results_o)

    results = pd.concat(results)
    results.to_csv(input_file.replace("input", "output"), sep="\t", header=False, index=False)

    #input_files = ["Olga_input_trb_1.tsv", "Olga_input_trb_2.tsv", "Olga_input_trb_3.tsv", 
    #               "Olga_input_trb_4.tsv", "Olga_input_trb_5.tsv", "Olga_input_trb_6.tsv",
    #               "Olga_input_trb_7.tsv", "Olga_input_trb_8.tsv"]

    #num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    #with multiprocessing.Pool(processes=num_processes) as pool:
    #    pool.map(Copier("--humanTRB"), input_files)
    for i,f in enumerate(input_files):
        os.remove(f)
        os.remove(output_files[i])

if __name__ == "__main__":
    main()