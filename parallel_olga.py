import subprocess
import multiprocessing

def process_chunk(input_file_path):
    output_file_path = input_file_path.replace("input", "output")
    
    command = [
        "olga-compute_pgen",
        "--humanIGH",
        "-i", input_file_path,
        "-o", output_file_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # Process the result if needed
        print("Command output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing command:", e)
        print("Command output:", e.stdout)
        print("Command error:", e.stderr)

def main():
    # Assuming input_files contains a list of input file paths
    input_files = ["Olga_input_igh_1.tsv", "Olga_input_igh_2.tsv", "Olga_input_igh_3.tsv", 
                   "Olga_input_igh_4.tsv", "Olga_input_igh_5.tsv", "Olga_input_igh_6.tsv",
                   "Olga_input_igh_7.tsv", "Olga_input_igh_8.tsv"]

    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_chunk, input_files)

if __name__ == "__main__":
    main()