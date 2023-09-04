import pandas as pd 
import numpy as np
import os
import utils
import argparse
import subprocess

def main():
    #parser = argparse.ArgumentParser(description="Load Data from clones file folder")
    # Add arguments using add_argument method
    #parser.add_argument('--path', type=str, help='Path to clones folder')
    #parser.add_argument('--output_file', type=str, help='Output file name')
    #parser.add_argument('--normalize_read_count', type=int, help='Normalize for repertoires with lower reads')
    #parser.add_argument('--olga', type=str, help='calculate generation probability using olga model [humanIGH, humanTRA, humanTRB, humanIGH]')

    #args = parser.parse_args()
    #if args.path and os.path.isdir(args.path):
    #    input_path = args.path
    #else:
    #    raise Exception("Please provide a valid input path...")
    #if 
    #else:
    #    raise Exception("Please provide a valid output file name...")
    #if args.normalize_read_count:
    #    normalize_read_count = args.normalize_read_count
    #else:
    #    normalize_read_count = None
    #if args.olga:
    #    olga_model =  args.olga
    #    calculate_olga = True
    #else:
    #    calculate_olga = False
    
    while True:
        input_path = input("Please provide the path to the clones.file.txt folder...")
        if os.path.isdir(input_path):
            break
        else:
            print("Unknown path..")
            continue
    
    while True:
        output_file = input("Please provide an output filename...")
        if not output_file.endswith(".pkl"):
            output_file = output_file + ".pkl"

        if os.path.exists(os.path.join(os.getcwd(), output_file)):
            print("Outputfile already exists...")
            print("Loading stored data... ")
            df = pd.read_pickle(output_file)
            continue
        else:
            break
    if not 'df' in locals():
        while True:
            normalize = input("Do you want to normalize ReadCounts? [y/n]")
            if normalize.lower() == "y" or normalize.lower() == "yes":
                try:
                    normalize_read_count = int(input("Please provide the number of reads to normalize (default: 40000)..."))
                    break
                except ValueError:
                    print("Please, provide a valid number.")
                    continue
            else:
                normalize_read_count = None
                break
        
        if normalize_read_count == None:
            print("No normalization will be applied...")
        else:
            print("Normalization of ReadCounts %d" % normalize_read_count)
        
        print("Start loading data...")
        df = utils.load_clone_files_data(input_path, normalize_read_count)
        print("Finished...")

        df.to_pickle(output_file)
    
    calculate_olga = input("Do you want to compute generation probabilities of CDR3 amino acid and nucleotide sequences? (This might take a while...) [y/n]")
    if calculate_olga.lower() == "y" or calculate_olga.lower() == "yes":
        mixed_data = input("Do you have mixed data (TRB vs. IGH, human vs. mouse)? [y/n]")
        if mixed_data.lower() == "y" or mixed_data.lower() == "yes":
            while True:
                olga_model = input("Please specity all models to use with Olga (e.g. humanIGH humanTRB)")
                olga_model = [str(item) for item in olga_model.split()]
                if sum([m in["humanTRA", "humanTRB", "humanIGH", "mouseTRB"] for m in olga_model]) == len(olga_model):
                    break
                else:
                    print("Unkown Olga Models found in input...")
                    continue 
        else:
            while True:
                olga_model = input("Please specity a model to use with Olga (e.g. humanIGH)")
                if olga_model in ["humanTRA", "humanTRB", "humanIGH", "mouseTRB"]:
                    break
                else:
                    print("Unkown Olga Model...")
                    continue 
        if isinstance(olga_model, list):
            olga_input_files = []
            olga_dfs = []
            for m in olga_model:
                if "mouse" in m:
                    raise Exception("Sorry, automatic calculation with olga for mixed human vs. mouse data not implemented")
                if "TRB" in m:
                    olga_df_i = df[df.bestVGene.apply(lambda x: "TRB" in x)].copy()
                    olga_df_i = olga_df_i[["CDR3.nucleotide.sequence", "bestVGene", "bestJGene"]]
                    olga_df_i.rename(columns = {"CDR3.nucleotide.sequence": "V1", "bestVGene": "V2", "bestJGene": "V3"}, inplace = True)
                    olga_df_i.to_csv('Olga_input_trb.tsv', sep="\t", header=False, index=False)
                    olga_input_files.append("Olga_input_trb.tsv")
                if "IGH" in m:
                    olga_df_i = df[df.bestVGene.apply(lambda x: "IGH" in x)].copy()
                    olga_df_i = olga_df_i[["CDR3.nucleotide.sequence", "bestVGene", "bestJGene"]]
                    olga_df_i.rename(columns = {"CDR3.nucleotide.sequence": "V1", "bestVGene": "V2", "bestJGene": "V3"}, inplace = True)
                    olga_df_i.to_csv('Olga_input_igh.tsv', sep="\t", header=False, index=False)
                    olga_input_files.append("Olga_input_igh.tsv")
                if "TRA" in m:
                    olga_df_i = df[df.bestVGene.apply(lambda x: "TRA" in x)].copy()
                    olga_df_i = olga_df_i[["CDR3.nucleotide.sequence", "bestVGene", "bestJGene"]]
                    olga_df_i.rename(columns = {"CDR3.nucleotide.sequence": "V1", "bestVGene": "V2", "bestJGene": "V3"}, inplace = True)
                    olga_df_i.to_csv('Olga_input_tra.tsv', sep="\t", header=False, index=False)
                    olga_input_files.append("Olga_input_tra.tsv")
                olga_dfs.append(olga_df_i)
            
            
        else:
            olga_input_files = "Olga_input.tsv"
            olga_df = df[["CDR3.nucleotide.sequence", "bestVGene", "bestJGene"]].copy()
            olga_df.rename(columns = {"CDR3.nucleotide.sequence": "V1", "bestVGene": "V2", "bestJGene": "V3"}, inplace = True)
            olga_df.to_csv('Olga_input.tsv', sep="\t", header=False, index=False)

        print("Start calculation of generation probabilities using OLGA...")
        # Replace 'your_script.py' with the path to your Python script and
        # 'arg1', 'arg2', ... with the arguments you want to pass.
        script_path = 'parallel_olga.py'
        
        if isinstance(olga_input_files, list):
            for i, file in enumerate(olga_input_files):
                arguments = ['--i %s' % file, '--model %s' % olga_model[i]]
                command = ['python', script_path] + arguments
                subprocess.run(command)
            
                results_i = pd.read_csv(file.replace("input", "output"), sep="\t", header=None)
                df.loc[olga_dfs[i].index, "olga_pgen_cdr3"] = list(results_i[1])
                df.loc[olga_dfs[i].index, "olga_pgen_aa"] = list(results_i[3])


        else:
            arguments = ['--i %s' % olga_input_files, '--model %s' % olga_model]
            command = ['python', script_path] + arguments
            subprocess.run(command)
            
            results = pd.read_csv(olga_input_files.replace("input", "output"), sep="\t", header=None)
            df.loc[olga_df.index, "olga_pgen_cdr3"] = list(results[1])
            df.loc[olga_df.index, "olga_pgen_aa"] = list(results[3])
        
        df.to_pickle(output_file)
       
    else:
        print("Skipping calculation of generation probabilities...")

    calcuate_embedding = input("Do you want to use the pretrained ESM to embed AA-sequences=? [y/n]")
    if calcuate_embedding.lower() == "y" or calculate_embedding.lower() == "yes":
        file_path_aa = "aa_cdr3.txt"
        aas = list(df["aaSeqCDR3"])
        with open(file_path_aa, "w") as file:
            for aa in aas:
                file.write(aa + "\n")
        
        script_path = 'embed_with_esm.py'
        arguments = ['--i %s' % "aa_cdr3.txt"]
        command = ['python', script_path] + arguments
        subprocess.run(command)
            
        results = pd.read_csv("aa_cdr3_embeddings.pkl")
        df["esm_embedding"] = list(results["esm_embedding"])
        df.to_pickle(output_file)
    else:
        print("Skipping calculation of AA embeddings...")


if __name__ == "__main__":
    main()

