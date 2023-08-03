import numpy as np
import pandas as pd
import os
import re
import random

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm




nucleotid_encoding = {"A": [1,0,0,0], "C" : [0,1,0,0], "G" : [0,0,1,0], "T" : [0,0,0,1]}
armino_encoding = {"A": [1,0,0,0], "C" : [0,1,0,0], "G" : [0,0,1,0], "T" : [0,0,0,1]}

amino_acids = ['R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C',
                    'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']

nSeq_look_up_dict = {"A": [1,0,0,0], "C" : [0,1,0,0], "G" : [0,0,1,0], "T" : [0,0,0,1]}

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

def encode_nucleotides(nSeq):
    return np.stack(list(map(lambda x: nSeq_look_up_dict[x], nSeq)))

def read_clones_txt(files, clones_txt_dict):
    df_raw = []
    for file in tqdm(files):
        df_file = pd.read_csv(os.path.join(clones_txt_dict, file), sep="\t")
        df_file["clones.txt.name"] = file
        df_raw.append(df_file)
    df_raw = pd.concat(df_raw)
    return df_raw


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


def tryconvert(value, default, *types):
    for t in types:
        try:
            return t(value)
        except (ValueError, TypeError):
            continue
    return default


def get_stripped_pat_no(id_string):
    if pd.isna(id_string):
        return None
    elif len(id_string.split("-")) == 2:
        numbers = id_string.split("-")
        stripped_pat_number = "-".join([str(int(numbers[0])), str(int(numbers[1]))])
    else:   #file  
        numbers = re.findall(r'(?<=-)\d+',id_string)
        valid_pat_numbers = [len(i) >= 2 for i in numbers]
        pat_number = np.asarray(numbers)[valid_pat_numbers]
        stripped_pat_number = "-".join([str(int(pat_number[0])), str(int(pat_number[1]))])
    return stripped_pat_number

def get_top_n_clones(df, top_n_clones, sample_id = "clones.txt.name", cloneId = "cloneId"):
    df_sorted = df.sort_values(by=["clones.txt.name","cloneId"])
    if top_n_clones == "all":
        df_top_n = df_sorted
    else:
        df_top_n = df_sorted.groupby(sample_id).head(top_n_clones)
    return df_top_n


def grouped_patientwise_k_folds(df, group_id, patient_id, n_folds=5, random_state = 42):
    group_patient_df = df.groupby([group_id, patient_id]).size().reset_index()
    
    group_fold_dict = {} # folds per group

    
    for key, group in group_patient_df.groupby(group_id):
        patients = group[patient_id].sample(frac=1, random_state=random_state) #shuffle within group
        fold_len = round(len(patients)/5)
        
        group_fold_dict[key] = []
        for i in range(n_folds):
            
            if i == (n_folds-1):
                fold = [list(patients[i*fold_len:])]
            else:
                fold = [list(patients[i*fold_len:i*fold_len+fold_len])]            
            group_fold_dict[key] += fold
    

    train_fold_patients = [[] for i in range(n_folds)] #  groups in folds 
    test_fold_patients = [[] for i in range(n_folds)]
    patients_in_folds = group_fold_dict.values() 
    
    for i in range(n_folds):
        for patient_group in patients_in_folds:
            test_fold_patients[i] += patient_group[i]
            train_fold_patients[i] += sum(patient_group[:i],[]) + sum(patient_group[i+1:],[])
    
    return group_fold_dict, train_fold_patients, test_fold_patients

            

def get_averaged_classification_report(classification_report_dicts, output_dict = False, digits = 2): 
    
    averaged_classification_report = classification_report_dicts[0]
    
    for group in averaged_classification_report.keys():
        if type(averaged_classification_report[group]) == float:
            for report in classification_report_dicts[1:]:
                averaged_classification_report[group] += report[group]
        elif type(averaged_classification_report[group]) == dict:
            for metric in averaged_classification_report[group].keys():
                for report in classification_report_dicts[1:]:
                    averaged_classification_report[group][metric] += report[group][metric]
        else:
            raise ValueError("unkown format in classification report")
    
    for group in averaged_classification_report.keys():
        if type(averaged_classification_report[group]) == float:
            averaged_classification_report[group] /= len(classification_report_dicts)
        else:
            for metric in averaged_classification_report[group].keys():
                averaged_classification_report[group][metric] /= len(classification_report_dicts)
    
    if output_dict:
        report = averaged_classification_report    
    else:
        headers = ["precision", "recall", "f1-score", "support"]
        longest_last_line_heading = "weighted avg"
        name_width = max(len(cn) for cn in list(averaged_classification_report.keys())[:-3])
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9.{digits}f}\n"

        rows = []
        avg = []
        summed_support = 0
        for k, v in averaged_classification_report.items():
            if "accuracy" in k:
                avg.append(v)
            else:
                row = [k] + list(v.values())
                if "avg" in k:  
                    rows.append(row)
                else:
                    rows.append(row)
                    summed_support += row[-1]
        avg.append(summed_support)
            

        for row in rows[:-2]:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += "\n"

        row_fmt_accuracy = (
                    "{:>{width}s} "
                    + " {:>9.{digits}}" * 2
                    + " {:>9.{digits}f}"
                    + " {:>9.{digits}f}\n"
                )
        report += row_fmt_accuracy.format("accuracy", "", "", *avg, width=width, digits=digits)
        for row in rows[-2:]:
            report += row_fmt.format(*row, width=width, digits=digits)
    
    return report


def from_dosc_and_dob_to_age(dob, dosc):
    if pd.isnull(dob) or pd.isnull(dosc):
        return None
    else:
        dosc = str(dosc).split(",")[-1]
        dob = str(dob).split(",")[-1]
        if len(dosc) == 2:
            if int(dosc) > 20:
                dosc = "19" + dosc
            else:
                dosc = "20" + dosc
        else:
            dosc = dosc[-4:]
        
        if len(dob) == 2:
            if int(dob) > 20:
                dob = "19" + dob
            else:
                dob = "20" + dob
        else:
            dob = dob[-4:]

        age = int(dosc) - int(dob)
        if age < 0:
            age = None
        return age      
    

def get_top_n_features_wide(df, fixed_feature_list, clone_feature_list, top_n_clones, keep_remaining_columns, file_id = "clones.txt.name", clone_id = "cloneId"):
    df_wide = pd.DataFrame()
    df = df.sort_values(by=[file_id, clone_id])

    for i, key_group in enumerate(df.groupby(file_id)):
        key = key_group[0]
        group = key_group[1]
        
        for col in list(df.columns):
            if col == clone_id:
                continue
            elif col == file_id:
                df_wide.loc[i, file_id] = key
            else:
                if col in fixed_feature_list:
                    df_wide.loc[i, col] = group[col].iloc[0]
                elif col in clone_feature_list:
                    top_n_clone_features = list(group[col])
                    for j in range(top_n_clones):
                        if j < len(top_n_clone_features):
                            df_wide.loc[i,col +"_%d" % (j+1)] = top_n_clone_features[j]
                        else:
                            df_wide.loc[i,col +"_%d" % (j+1)] = None
                else:
                    if keep_remaining_columns:
                        df_wide[i, col] = group[col].iloc[0]
    
    return df_wide



def apply_padding(xx, max_len, style="zero"):
    if len(xx.shape)>=1:
        seq_length = xx.shape[0]
    else:
        seq_length = 1
    padding_size = max_len - seq_length
    if style == "same":
        padding_size = tuple([padding_size] + [1 for i in range(len(xx.shape) - 1)])
        xx = np.concatenate((xx, np.tile(xx[-1:], padding_size)), axis=0)
    elif style == "zero":
        padding_size = tuple([padding_size] + list(xx.shape[1:]))
        xx = np.concatenate((xx, np.zeros(padding_size)), axis=0)
    else:
        raise ValueError("unkown padding style: %s" % style) 
    return xx


def pad_batch_online(batch_lens, batch_data, style="zero"):
    max_len = int(max(batch_lens))
    padded_data = np.stack(list(batch_data.apply(
        lambda x: apply_padding(x, max_len, style=style))))
    return padded_data


SEED = 42
random.seed(SEED)

def set_random_seed(seed=42):
    SEED = seed
    random.seed(SEED)


def create_epoch_with_same_size_batching(length_with_index_dict,batch_size, shuffle=True):
    epoch = [] # list of batches
    foundlings = []  # rest samples for each length which do not fit into one batch
    
    for length in np.sort(list(length_with_index_dict.keys())): # iterate over each unique length in training data
        length_idxs = length_with_index_dict[length] # dictionary containing indices of samples with length
        rest = len(length_idxs) % batch_size
        if shuffle:
            random.shuffle(length_idxs) # shuffle indices
        epoch += [length_idxs[i * batch_size:(i * batch_size) + batch_size] for i in
                  range(int(len(length_idxs) / batch_size))] # cut into batches and append to epoch
        if rest > 0:
            foundlings += list(length_idxs[-rest:]) # remaining indices which do not fit into one batch are stored in foundling
    foundlings = np.asarray(foundlings)
    rest = len(foundlings) % batch_size
    epoch += [foundlings[i * batch_size:(i * batch_size) + batch_size] for i in
              range(int(len(foundlings) / batch_size))] # cut foudnlings into batches (because inserted sorted this ensures minimal padding)
    if rest > 0:
        epoch += [foundlings[-rest:]] # put rest into one batch (allow smaller batch)
    if shuffle:
        random.shuffle(epoch)
    return epoch






    
