import numpy as np
import pandas as pd

nucleotid_encoding = {"A": [1,0,0,0], "C" : [0,1,0,0], "G" : [0,0,1,0], "T" : [0,0,0,1]}
armino_encoding = {"A": [1,0,0,0], "C" : [0,1,0,0], "G" : [0,0,1,0], "T" : [0,0,0,1]}

amino_acids = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C',
                    'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']

def normalize_values(array_to_norm):
    stds = np.std(array_to_norm)
    means = np.mean(array_to_norm)
    return (array_to_norm - means) / stds

def one_hot_from_label(array):
    max_in_array = np.amax(array)
    min_in_array = np.amin(array)
    array_starting_zero = array - min_in_array #starting from zero
    one_hot_labels = np.zeros((len(array_starting_zero),max_in_array - min_in_array))
    one_hot_labels = one_hot_labels[np.arange(len(one_hot_labels)), array_starting_zero] = 1
    return one_hot_labels


def convert_rtwb_to_pdtwb(r_twb):
    pd_twb = pd.DataFrame()
    for i, sample_name in enumerate(r_twb.names):
        r_sample = r_twb[i]
        pd_sample = pd.DataFrame()
        for j, col_name in enumerate(r_sample.names):
            pd_sample[col_name] = np.asarray(r_sample[j])
        pd_sample["sample"] = sample_name
        pd_sample["cloneId"] = np.arange(len(r_sample[j]))
        pd_twb = pd.concat([pd_twb, pd_sample], ignore_index=True)
    return pd_twb            