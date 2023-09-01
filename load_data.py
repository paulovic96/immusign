import pandas as pd 
import numpy as np
import os
import utils

def main():
    parser = argparse.ArgumentParser(description="Load Data from clones file folder")
    # Add arguments using add_argument method
    parser.add_argument('--path', type=str, help='Path to clones folder')
    parser.add_argument('--output_file', type=str, help='Output file name')
    parser.add_argument('--normalize_read_count', type=int, help='Normalize for repertoires with lower reads')

    args = parser.parse_args()
    if args.i and os.path.isdir(args.i):
        input_path = args.path
    else:
        raise Exception("Please provide a valid input path...")
    if args.output_file:
        output_file = args.output_file
        if os.path.exists(os.path.join(os.getcwd(), output_file)):
            raise Exception("Output already exists...")
    else:
        raise Exception("Please provide a valid output file name...")
    if args.normalize_read_count:
        normalize_read_count = args.normalize_read_count
    else:
        normalize_read_count = None
    

    df = utils.load_clone_files_data(input_path, normalize_read_count)

    if output_file.endswith(".pkl"):
        df.to_pickle(output_file)
    else:
        df.to_pickle(output_file + ".pkl")




