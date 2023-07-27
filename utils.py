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


def tryconvert(value, default, *types):
    for t in types:
        try:
            return t(value)
        except (ValueError, TypeError):
            continue
    return default


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

