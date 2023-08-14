import subprocess
import multiprocessing
import os 

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
    input_files = ["Olga_input_igh_1.tsv", "Olga_input_igh_2.tsv", "Olga_input_igh_3.tsv", 
                   "Olga_input_igh_4.tsv", "Olga_input_igh_5.tsv", "Olga_input_igh_6.tsv",
                   "Olga_input_igh_7.tsv", "Olga_input_igh_8.tsv"]

    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(Copier("--humanIGH"), input_files)

    input_files = ["Olga_input_trb_1.tsv", "Olga_input_trb_2.tsv", "Olga_input_trb_3.tsv", 
                   "Olga_input_trb_4.tsv", "Olga_input_trb_5.tsv", "Olga_input_trb_6.tsv",
                   "Olga_input_trb_7.tsv", "Olga_input_trb_8.tsv"]

    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(Copier("--humanTRB"), input_files)

if __name__ == "__main__":
    main()