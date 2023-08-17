import numpy as np
import pandas as pd
import os
import re
import random
import skbio
import peptides 
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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

def get_clonset_info(rep, method, quant="proportion"):
    """
    chao1:  Non-parametric estimation of the number of classes in a population: Sest = Sobs + ((F2 / 2G + 1) - (FG / 2 (G + 1) 2))
            Sets = number classes
            Sobs = number classes observed in sample
            F = number singeltons (only one individual in class)
            G = number doubletons (exactly two individuals in class)

    gini index:  'inequality' among clonotypes. 0 for qual distribution and 1 for total unequal dstirbution only 1 clone in set
    
    simpson: Probability  that two random clones belong to the same clone type

    inv_simpson: 1 / simpson

    shannon:  Distribution of clones within a repertoire. Quotient between Shannon-Index and max Shannon-Index (all clones equal distributed) is called Evenness. 

    clonality: 1-evenness. 1 being a repertoire consisting of only one clone and 0 being a repertoire of maximal evennes (every clone in the repertoire was present at the same frequency).
    """


    n_aa_clones = len(rep["aaSeqCDR3"].unique())
    if quant == "count":
        counts = np.asarray(rep["cloneCount"])
    elif quant == "proportion":
        counts = np.asarray(rep["cloneFraction"])

    if method == "chao1":
        info = skbio.diversity.alpha.chao1(counts, bias_corrected=True)
    elif method == "gini":
        info = skbio.diversity.alpha.gini_index(counts, method='rectangles')
    elif method == "simpson":
        info = skbio.diversity.alpha.simpson(counts)
    elif method == "inv_simpson":
        info = skbio.diversity.alpha.enspie(counts)
    elif method == "shannon":
        info = skbio.diversity.alpha.shannon(counts, base=2)
    elif method == "clonality":
        hmax = np.log2(n_aa_clones)
        shannon = skbio.diversity.alpha.shannon(counts, base=2)
        eveness = shannon/hmax
        info = 1-eveness
    
    return info

def read_clones_txt(files, clones_txt_dict=None):
    """
    KF1: Helix/bend preference,
    KF2: Side-chain size,
    KF3: Extended structure preference,
    KF4: Hydrophobicity,
    KF5: Double-bend preference,
    KF6: Partial specific volume,
    KF7: Flat extended preference,
    KF8: Occurrence in alpha region,
    KF9: pK-C,
    KF10: Surrounding hydrophobicity
    """
    df_raw = []
    for file in tqdm(files):
        if not clones_txt_dict == None:
            file = os.path.join(clones_txt_dict, file)          
        df_file = pd.read_csv(file, sep="\t")
        df_file["clones.txt.name"] = os.path.basename(file)
        df_file["cloneFraction"] = df_file["cloneFraction"].apply(lambda x: float(x.replace(",",".").replace("+","-")) if isinstance(x, str) else float(x))
        df_file["clonality"] = get_clonset_info(df_file, "clonality")
        df_file["shannon"] = get_clonset_info(df_file, "shannon")
        df_file["inv_simpson"] = get_clonset_info(df_file, "inv_simpson")
        df_file["simpson"] = get_clonset_info(df_file, "simpson")
        df_file["gini"] = get_clonset_info(df_file, "gini")
        df_file["chao1"] = get_clonset_info(df_file, "chao1")

        df_file[['KF1', 'KF2', 'KF3', 'KF4', 'KF5',
                'KF6', 'KF7', 'KF8', 'KF9', 'KF10']] = df_file.aaSeqCDR3.apply(lambda x: list(peptides.Peptide(x).kidera_factors())).to_list()

        df_raw.append(df_file)
    df_raw = pd.concat(df_raw)
    return df_raw

def load_clone_files_data(project_path):
    clone_files = []
    for path, subdirs, files in os.walk(project_path):
        for name in files:
            file = os.path.join(path, name)
            if file.endswith("clones.txt"):
                clone_files.append(file)
    df = read_clones_txt(clone_files)
    return df

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

def get_top_n_clones(df, top_n_clones, file_id, clone_id):
    df_sorted = df.sort_values(by=[file_id, clone_id])
    if top_n_clones == "all":
        df_top_n = df_sorted
    else:
        df_top_n = df_sorted.groupby(file_id).head(top_n_clones)
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
        if isinstance(averaged_classification_report[group], (np.floating, float)):
            for report in classification_report_dicts[1:]:
                averaged_classification_report[group] += report[group]
        elif isinstance(averaged_classification_report[group], dict):
            for metric in averaged_classification_report[group].keys():
                for report in classification_report_dicts[1:]:
                    try:
                        averaged_classification_report[group][metric] += report[group][metric]
                    except KeyError:
                        print(group, " not found in validation fold...")

        else:
            raise ValueError("unkown format in classification report")
    
    for group in averaged_classification_report.keys():
        if isinstance(averaged_classification_report[group], (np.floating, float)):
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
        mcc = []
        
        summed_support = 0
        for k, v in averaged_classification_report.items():
            if "accuracy" in k:
                avg.append(v)
            elif "mcc" in k:
                mcc.append(v)
            else:
                row = [k] + list(v.values())
                if "avg" in k:  
                    rows.append(row)
                else:
                    rows.append(row)
                    summed_support += row[-1]
        avg.append(summed_support)
        mcc.append(summed_support)
            

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
        report += row_fmt_accuracy.format("mcc", "", "", *mcc, width=width, digits=digits)
        
        for row in rows[-2:]:
            report += row_fmt.format(*row, width=width, digits=digits)
    
    return report


def from_dosc_and_dob_to_age(dob, dosc):
    if pd.isnull(dob) or pd.isnull(dosc):
        return None
    else:
        if "," in str(dosc):
            dosc = str(dosc).split(",")[-1]
        elif "." in str(dosc):
            dosc = str(dosc).split(".")[-1]
        elif "/" in str(dosc):
            dosc = str(dosc).split("/")[-1]
        else:
            dosc = str(dosc)[-4:]
        
        if "," in str(dob):
            dob = str(dob).split(",")[-1]
        elif "." in str(dob):
            dob = str(dob).split(".")[-1]
        elif "/" in str(dob):
            dob = str(dob).split("/")[-1]
        else:
            dob = str(dob)[-4:]
        
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
    

def get_top_n_features_wide(df, fixed_feature_list, clone_feature_list, top_n_clones, file_id, clone_id, keep_remaining_columns=False):
    df_wide = pd.DataFrame()
    df = df.sort_values(by=[file_id, clone_id])

    clone_feature_list_wide = []

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
                        new_column = col +"_%d" % (j+1)
                        if j < len(top_n_clone_features):
                            df_wide.loc[i,new_column] = top_n_clone_features[j]
                        else:
                            df_wide.loc[i,new_column] = None
                        
                        clone_feature_list_wide.append(new_column)
                else:
                    if keep_remaining_columns:
                        df_wide[i, col] = group[col].iloc[0]
    
    return df_wide, clone_feature_list_wide



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



def create_feature_df(df, target_column, 
                      categorical_features, 
                      numerical_features, 
                      top_n_clones, 
                      file_id,
                      clone_id,
                      features_to_encode_wide=None,
                      wide_format =False, 
                      keep_remaining_columns=False,
                      ):

    if wide_format:
        assert not (features_to_encode_wide is None) and len(features_to_encode_wide) > 0, "Variable features_to_encode_wide undefined!\nYou have to specify the features which should be encoded in wide format..."
        
        fixed_feature_list = [target_column] + list(np.setdiff1d(categorical_features,features_to_encode_wide)) +  list(np.setdiff1d(numerical_features, features_to_encode_wide))
            
        X, feature_encoded_wide = get_top_n_features_wide(df, fixed_feature_list, features_to_encode_wide, top_n_clones, file_id, clone_id, keep_remaining_columns)

        for column in feature_encoded_wide:
            if X[column].dtype == float:
                if np.sum(X[column].isnull()):
                    X[column] = X[column].fillna(0)
            elif X[column].dtype == object:
                if type(X[column].loc[X[column].first_valid_index()]) == bool:
                    if np.sum(X[column].isnull()):
                        X[column] = X[column].fillna(False)
                elif type(X[column].loc[X[column].first_valid_index()]) == str:
                    if np.sum(X[column].isnull()):
                        X[column] = X[column].fillna("nan")
    else:
        feature_list = [file_id, clone_id, target_column] + categorical_features + numerical_features
        
        X = get_top_n_clones(df[feature_list], top_n_clones, file_id, clone_id)

    print("Created DataFrame with Features")
    for column in X.columns:
        print(column + " :", "%.2f %% NAN" % (X[column].isnull().sum()/len(X)*100), X[column].dtype)
    
    return X

def custom_combiner(feature, category):
    return str(feature) + "_" + str(category)

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def encode_target_for_classification(X, target_column):
    X_new = X.copy()
    target_preprocessor = LabelEncoder()
    Y_target = target_preprocessor.fit_transform(X[target_column])

    labels = []
    for i in range(len(X_new[target_column].unique())):
        labels.append(np.asarray(X[target_column])[Y_target==i][0])

    X_new.insert(loc=0, column = target_column + "_encoded", value=Y_target)
    return X_new, labels 


def encode_categorical_features_for_classification(X, 
                      categorical_features, 
                      top_n_clones,
                      features_to_encode_wide=None, 
                      wide_format=False 
                      ):
    X_new = X.copy() 
    categorical_preprocessor = OneHotEncoder(handle_unknown = 'ignore',feature_name_combiner=custom_combiner, sparse_output=False)

    if wide_format: 
        assert not (features_to_encode_wide is None) and len(features_to_encode_wide) > 0, "Variable features_to_encode_wide undefined!\nYou have to specify the features which are encoded in wide format..."
        wide_categorical_columns = list(np.intersect1d(features_to_encode_wide, categorical_features))      
        categorical_features =  list(np.setdiff1d(categorical_features,features_to_encode_wide))
        
        for feature in wide_categorical_columns:
            categorical_features += [feature + "_%d" % (i+1) for i in range(top_n_clones)]


    transformed_categories = categorical_preprocessor.fit_transform(X_new[categorical_features])
    transformed_categories = pd.DataFrame(data = transformed_categories, columns=categorical_preprocessor.get_feature_names_out())

    X_new = pd.concat([X_new.reset_index(drop=True), transformed_categories], axis=1) 

    return X_new, categorical_preprocessor.get_feature_names_out()

def scale_numerical_features(X, 
                      numerical_features, 
                      top_n_clones,
                      fit_transform,
                      features_to_encode_wide = None, 
                      wide_format=False,
                      numerical_preprocessor = None 
                      ):

    X_new = X.copy()
    if pd.isnull(numerical_preprocessor):
        numerical_preprocessor = StandardScaler()

    if wide_format:
        assert not (features_to_encode_wide is None) and len(features_to_encode_wide) > 0, "Variable features_to_encode_wide undefined!\nYou have to specify the features which are encoded in wide format..."
        wide_numerical_columns = list(np.intersect1d(features_to_encode_wide, numerical_features))      
        numerical_features =  list(np.setdiff1d(numerical_features, features_to_encode_wide))
        
        for feature in wide_numerical_columns:
            numerical_features += [feature + "_%d" % (i+1) for i in range(top_n_clones)]

    if fit_transform:
        X_new.loc[:, numerical_features] = numerical_preprocessor.fit_transform(X_new.loc[:, numerical_features])
    else:
        X_new.loc[:, numerical_features] = numerical_preprocessor.transform(X_new.loc[:, numerical_features])

    return X_new, numerical_features, numerical_preprocessor