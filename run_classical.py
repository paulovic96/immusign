import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score
import random, time, string
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import json
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm
import warnings
from utils import get_clonset_info
from networks import DeepSetClassifier
from utils import contaminated_hds
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

def create_vdj_index(class_files, family = False):
    
    # iterateo over all class files and store bestvgene, bestdgene, bestjgene to a unique list
    bestvgene = []
    bestdgene = []
    bestjgene = []
    for i, type in enumerate(class_files.keys()):
        for j, file in enumerate(class_files[type]):
        
            df = pd.read_csv("data/clones_mit_kidera/%s" % file, sep="\t")
                
            bestvgene.extend(df["bestVGene"].unique())
            bestdgene.extend(df["bestDGene"].unique())
            bestjgene.extend(df["bestJGene"].unique())

    # get unique values
    bestvgene = list(set(bestvgene))
    bestdgene = list(set(bestdgene))
    bestjgene = list(set(bestjgene))
    print("Number of unique V genes: %i" % len(bestvgene))
    print("Number of unique D genes: %i" % len(bestdgene))  
    print("Number of unique J genes: %i" % len(bestjgene))

    print(bestvgene)

    if family:
        # create short lists with only the first part of the gene name
        bestvgene_short = [str(gene).split("-")[0] for gene in bestvgene if gene != "nan"]
        bestdgene_short = [str(gene).split("-")[0] for gene in bestdgene if gene != "nan"]
        bestjgene_short = [str(gene).split("-")[0] for gene in bestjgene if gene != "nan"]
    else:
        bestvgene_short = [str(gene) for gene in bestvgene if gene != "nan"]
        bestdgene_short = [str(gene) for gene in bestdgene if gene != "nan"]
        bestjgene_short = [str(gene) for gene in bestjgene if gene != "nan"]

    print("Number of unique V genes: %i" % len(set(bestvgene_short)))
    print("Number of unique D genes: %i" % len(set(bestdgene_short)))
    print("Number of unique J genes: %i" % len(set(bestjgene_short)))

    uniqevgenefamilies = set(bestvgene_short)
    uniqedgenefamilies = set(bestdgene_short)
    uniqejgenefamilies = set(bestjgene_short)


    # create a mapping from gene to index, consider only the part before the "-" and map the same gene to the same index
    gene2index = {}
    # map nan to -1
    gene2index["nan"] = -1
    for i, gene in enumerate(bestvgene):
        # find the index of bestvgene_short[i] in uniqevgenes
        if gene != "nan":
            if family:
                gene2index[gene] = list(uniqevgenefamilies).index(str(bestvgene_short[i]).split("-")[0])
            else:
                gene2index[gene] = list(uniqevgenefamilies).index(str(bestvgene_short[i]))
    for i, gene in enumerate(bestdgene):
        if gene != "nan":
            if family:
                gene2index[gene] = list(uniqedgenefamilies).index(str(bestdgene_short[i]).split("-")[0])
            else:
                gene2index[gene] = list(uniqedgenefamilies).index(str(bestdgene_short[i]))
    for i, gene in enumerate(bestjgene):
        if gene != "nan":
            if family:
                gene2index[gene] = list(uniqejgenefamilies).index(str(bestjgene_short[i]).split("-")[0])
            else:
                gene2index[gene] = list(uniqejgenefamilies).index(str(bestjgene_short[i]))
    # save to json file
    if family:
        with open("data/gene2index_family.json", "w") as outfile:
            json.dump(gene2index, outfile)
    else:
        with open("data/gene2index.json", "w") as outfile:
            json.dump(gene2index, outfile)
    print(gene2index)

def _run_name(model_type):
    return time.strftime("output_%b_%d_%H%M%S_") + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5)) +  "_%s" %model_type

def _custom_combiner(feature, category):
    return str(feature) + "_" + str(category)

def read_feature(files, features , n_entries, flatten=True, return_filenames=False):
    """
     This method creates a feature vector for each repertoire by concatenating the given set of features 
     for the most n_entries frequent clones and concatenates them.
    """
    data = []
    cloneFraction = []
    filenames = []

    read_features = features.copy()
    for i, file in enumerate(files):
        df = pd.read_csv(file, sep="\t")
        df = df[df['cloneFraction'].apply(lambda x: isinstance(x, (int, float)))] 
        if (df.shape[0] == 0):
            continue
        cloneFraction.append(df['cloneFraction'].values[0])
        if "clonality" in read_features:
            clonality = get_clonset_info(df, "clonality")
            added_clonality = True
            read_features.remove("clonality")
        else:
            added_clonality = False
        if "shannon" in read_features:
            shannon = get_clonset_info(df, "shannon")
            added_shannon = True
            read_features.remove("shannon")
        else:
            added_shannon = False
        if "richness" in read_features:
            richness = get_clonset_info(df, "aminoacid_clones")
            added_richness = True
            read_features.remove("richness")
        else:
            added_richness = False
        if "hypermutatedFraction" in read_features:
            hypermutatedFraction = get_clonset_info(df, "hypermutation")
            added_hypermutatedFraction = True
            read_features.remove("hypermutatedFraction")
        else:
            added_hypermutatedFraction = False
        
        if "is_hypermutated" in read_features:
            df["is_hypermutated"] = df["vBestIdentityPercent"] < 0.98
    
        df = df.iloc[:n_entries] 
        d = df[read_features].values
        # PADD ING FOR RANDOM FOREST
        if (df[read_features].values.shape[0] < n_entries):
            pad = n_entries  - df[read_features].values.shape[0]
            d = [df[read_features].values, np.zeros((pad, len(read_features)))]
            d= np.concatenate(d)
        # flatten data such that it can be an input for the random forest
        if flatten:
            d = d.flatten()
            try: 
                if added_clonality:
                    d = np.append(d, clonality)
                    read_features += ["clonality"]
                if added_shannon:
                    d = np.append(d, shannon)
                    read_features += ["shannon"]
                if added_richness:
                    d = np.append(d, richness)
                    read_features += ["richness"]  
                if added_hypermutatedFraction:
                    d = np.append(d, hypermutatedFraction)
                    read_features += ["hypermutatedFraction"]  
                data.append(d)
            except NameError as e:
                data.append(d.flatten())
        else:
            try: 
                if added_clonality:
                    d = np.column_stack((d, np.repeat(clonality)))
                    read_features += ["clonality"]
                if added_shannon:
                    d = np.column_stack((d, np.repeat(shannon)))
                    read_features += ["shannon"]
                if added_richness:
                    d = np.column_stack((d, np.repeat(richness)))
                    read_features += ["richness"]
                if added_hypermutatedFraction:
                    d = np.column_stack((d, np.repeat(hypermutatedFraction)))
                    read_features += ["hypermutatedFraction"]
                data.append(d)
                
            except NameError: 
                data.append(d)
        filenames.append([file]* len(df))
    data = np.stack(data, axis=0)

    if return_filenames:
        return data, cloneFraction, filenames

    return data, cloneFraction

def scaler_range(X, feature_range=(-1, 1), min_x=None, max_x=None):
    X_scaled = X.copy()
    X_scaleable = X_scaled.select_dtypes(exclude=['category'])
    if min_x is None:
        min_x = np.nanmin(X_scaleable, axis=0)
        max_x = np.nanmax(X_scaleable, axis=0)

    X_std = (X_scaleable - min_x) / (max_x - min_x)
    X_scaleable = X_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    X_scaled[X_scaleable.columns] = X_scaleable
    return X_scaled, min_x, max_x

def create_features(class_files, feature_names, object_types, n_entries=5, onehot_encoding=False, ordinal_encoding=False, standardize=False, flatten=True, genefamily=True, return_filenames=False):
    # create feature names for each clone
    column_names = []
    column_to_features = {}
    column_to_type = {}
    for i in range(n_entries):
        for feature in feature_names:
            if feature == "clonality" or feature == "shannon" or feature == "richness" or feature == "hypermutatedFraction":
                continue
            else:
                column_names.append(feature + "_%i" %i)
                column_to_features[feature + "_%i" %i] = feature
                column_to_type[feature + "_%i" %i] = object_types[feature_names.index(feature)]
    
    if "clonality" in feature_names:
        column_names.append("clonality")
        column_to_features["clonality"] = "clonality"
        column_to_type["clonality"] = object_types[feature_names.index("clonality")]
    if "shannon" in feature_names:
        column_names.append("shannon")
        column_to_features["shannon"] = "shannon"
        column_to_type["shannon"] = object_types[feature_names.index("shannon")]
    if "richness" in feature_names:
        column_names.append("richness")
        column_to_features["richness"] = "richness"
        column_to_type["richness"] = object_types[feature_names.index("richness")]
    if "hypermutatedFraction" in feature_names:
        column_names.append("hypermutatedFraction")
        column_to_features["hypermutatedFraction"] = "hypermutatedFraction"
        column_to_type["hypermutatedFraction"] = object_types[feature_names.index("hypermutatedFraction")]

    X = []
    clone_fractions = []
    y = []
    filenames = []

    for i, type in enumerate(class_files.keys()):
        #print("Read feature for class %i" %i)
        if return_filenames:
            X_type, cf_type, filenames_type = read_feature(class_files[type], feature_names, n_entries, flatten=flatten, return_filenames=return_filenames)
            filenames.append(filenames_type)
        else:
            X_type, cf_type = read_feature(class_files[type], feature_names, n_entries, flatten=flatten)
        y_type = [int(type)] * len(X_type)
        X.append(X_type)
        y.extend(y_type)
        clone_fractions.extend(cf_type)
    X = np.concatenate(X)
    X = pd.DataFrame(X, columns=column_names)
    for i, feature in enumerate(column_names):
        X[feature] = X[feature].astype(column_to_type[feature])
        if column_to_type[feature] == 'object':
             X[feature] = X[feature].fillna('nan')
             X[feature] = X[feature].replace(0, 'nan')
        else:
            X[feature] = X[feature].fillna(0.0)
   
    for col in X.select_dtypes(include=['object']):
        X[col] = X[col].astype('category')
    y = np.array(y)
    clone_fractions = np.array(clone_fractions)

    if ordinal_encoding or genefamily or onehot_encoding:
        categorical_cols = X.columns[X.dtypes == 'category']
        if genefamily:
             # read gene2index mapping
            with open("data/gene2index.json", "r") as infile:
                gene2index = json.load(infile)
                for c in categorical_cols:
                     X[c] = X[c].apply(lambda x: gene2index[str(x)] if str(x) in gene2index else -1)
        elif ordinal_encoding:
            X[categorical_cols] = X[categorical_cols].astype(str)
            encoder = OrdinalEncoder()
            encoded_categorical_cols = encoder.fit_transform(X[categorical_cols])
            X[categorical_cols] = encoded_categorical_cols
        elif onehot_encoding:
            X[categorical_cols] = X[categorical_cols].astype(str)
            encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output=False, feature_name_combiner = _custom_combiner)
            encoded_categorical_cols = encoder.fit_transform(X[categorical_cols])
            encoded_categorical_cols = pd.DataFrame(data = encoded_categorical_cols, columns=encoder.get_feature_names_out())
            for col in encoded_categorical_cols:
                encoded_categorical_cols[col] = encoded_categorical_cols[col].astype('category')
            encoded_categorical_cols.set_index(X.index, inplace=True)
            X.drop(columns=categorical_cols,inplace=True)
            X = pd.concat([X,encoded_categorical_cols], axis=1)
            
    
    if standardize:
        X = scaler_range(X)[0]
    
    if return_filenames:
        return X, y, clone_fractions, filenames

    return X, y, clone_fractions

def read_features_deep(class_files, selected_features, scale=False, max_clones = 10000, genefamily=True, return_filenames=False):

    # read gene2index mapping
    if genefamily:
        with open("data/gene2index_family.json", "r") as infile:
            gene2index = json.load(infile)
    else:
        with open("data/gene2index.json", "r") as infile:
            gene2index = json.load(infile)

    X = []
    y = []
    filenames = []
    clone_fractions = []

    features_max = []
    features_min = []

    for i, type in enumerate(class_files.keys()):
        for j, file in enumerate(class_files[type]):
            print("File %s" % file)
            df = pd.read_csv(file, sep="\t")
            df = df[df['cloneFraction'].apply(lambda x: isinstance(x, (int, float)))]

            if (df.shape[0] == 0):
                continue
            clone_fractions.append(df['cloneFraction'].values[0])

            if "clonality" in selected_features:
                df["clonality"] = get_clonset_info(df, "clonality")
            if "shannon" in selected_features:
                df["shannon"] = get_clonset_info(df, "shannon")
            if "richness" in selected_features:
                df["richness"] = get_clonset_info(df, "aminoacid_clones")
            if "hypermutatedFraction" in selected_features:
                df["hypermutatedFraction"] = get_clonset_info(df, "hypermutation")
            if "is_hypermutated" in selected_features:
                df["is_hypermutated"] = df["vBestIdentityPercent"] < 0.98
    
            # encode v,d,j genes to indices
            df["bestVGene"] = df["bestVGene"].apply(lambda x: gene2index[str(x)] if str(x) in gene2index else -1)
            df["bestDGene"] = df["bestDGene"].apply(lambda x: gene2index[str(x)] if str(x) in gene2index else -1)
            df["bestJGene"] = df["bestJGene"].apply(lambda x: gene2index[str(x)] if str(x) in gene2index else -1)
            
            features_max.append(np.max(df.iloc[:max_clones][selected_features], axis=0))
            features_min.append(np.min(df.iloc[:max_clones][selected_features], axis=0))

            #features = df.iloc[:max_entries][selected_features].values.flatten()#.reshape(1, -1)
            
            X.append(df)
            filenames.append([file]* len(df))
            y.append(i)

    if scale:
        # get total min and max and apply z-scaling
        features_max = np.max(features_max, axis=0)
        features_min = np.min(features_min, axis=0)

        X_scaled = []
        for i, x in enumerate(X):
            tmp = x.iloc[:max_clones][selected_features]
            tmp = (tmp - features_min) / (features_max - features_min)
            X_scaled.append(tmp.values.flatten())

        X = X_scaled
    if return_filenames:
        return X, y, clone_fractions, filenames
    return X, y, clone_fractions

def get_model(model_name, settings):
        if model_name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=settings["n_estimators"], max_depth = settings["max_depth"], random_state=42)
        elif model_name == 'XGBoost':
            model = xgb.XGBClassifier(n_estimators=settings["n_estimators"], max_depth = settings["max_depth"], random_state=42)
        elif model_name == 'LightGBM':
            model = lightgbm.LGBMClassifier(n_estimators=settings["n_estimators"], max_depth = settings["max_depth"], random_state=42, bagging_fraction=1.0, boost_from_average=False, verbose=-1)
        elif model_name == "CatBoost":
            model = CatBoostClassifier(n_estimators=settings["n_estimators"], max_depth = settings["max_depth"], random_state=42, verbose=False)
        elif model_name == "TabPFN":
            raise NotImplementedError
        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=settings["max_iter"], penalty=settings['regularization'], C=settings['C'], solver=settings['solver'], l1_ratio=settings["l1_ratio"], n_jobs=8)
        elif model_name == "SVM":
            model = SVC(max_iter=settings["max_iter"], kernel = settings["kernel"], C=settings["C"])
        elif model_name == 'AttentionDeepSets':
            model = DeepSetClassifier(params=settings)
        else:
            raise NotImplementedError
        return model


def main(model_name,settings, selected_features, class_files, train_index, test_index, types, store_path=None):

    print("Start %s training..." %model_name)
    if store_path == None:
        store_dir  = os.path.join("outputs_%s" % model_name.replace(" ", ""), _run_name("classification") )
    else:
        store_dir = os.path.join(store_path, "outputs_%s" % model_name.replace(" ", ""), _run_name("classification") )

    os.makedirs(store_dir)

    with open(os.path.join(store_dir, "settings.json"), 'w') as outfile:
        json.dump(settings, outfile, indent=2)

    feature_names = ['cloneFraction', 'lengthOfCDR3', 'clonality', 'shannon', 'richness', 'hypermutatedFraction']  + ['bestVGene', 'bestDGene', 'bestJGene', "is_hypermutated"] + ['KF%i' %i for i in range(1, 11)] 
    object_types = ['float64', 'int64', 'float64', 'float64', 'float64', 'float64']  +  ['object', 'object', 'object', 'object']+['float64' for i in range(10)] 

    # create dict from feature name to object type
    feature_dict = {}
    for i in range(len(feature_names)):
        feature_dict[feature_names[i]] = object_types[i]
        
    selected_object_types = [feature_dict[feature] for feature in selected_features]

    if model_name == "AttentionDeepSets":
        X, y, clone_fractions = read_features_deep(class_files, selected_features, scale=True, max_clones = settings["n_clones"], genefamily=settings["genefamily"])
        X = pd.Series(X)
        y = np.array(y)
        clone_fractions = np.array(clone_fractions)
    else:
        X, y, clone_fractions = create_features(class_files, selected_features, selected_object_types, settings["n_clones"], onehot_encoding=settings["onehot_encoding"], ordinal_encoding=settings["ordinal_encoding"], standardize=settings["standardize"], genefamily=settings["genefamily"])
    
    if model_name == "CatBoost":
        categorical_cols = X.columns[X.dtypes == 'category']
        X[categorical_cols] = X[categorical_cols].astype('int64')
    

    X_test = X.iloc[test_index]
    y_test = y[test_index]
    clone_fractions_test = clone_fractions[test_index]

    X = X.iloc[train_index]
    y = y[train_index]
    clone_fractions = clone_fractions[train_index]
    
    k_fold = StratifiedKFold(n_splits=settings["n_splits"], shuffle=True, random_state=42)

    # init accuracy, recall, precision, roc_auc, mcc for validation set in one dict
    accuracies = []
    recalls = []
    precisions = []
    specificitys = []
    roc_aucs = []
    mccs = []
    masked_dlbcl_accuracies = []
    
    sampling_strategies = {
    'None': None,
    'adasyn': ADASYN(random_state=42),
    'random_over': RandomOverSampler(random_state=42),
    'smote': SMOTE(random_state=42),
    'random_under' : RandomUnderSampler(random_state=42)
    }
    
    
    sampler = sampling_strategies[settings["sampler"]]

    for k, (train, val) in enumerate(k_fold.split(X, y)):
      
        X_train, y_train = X.iloc[train], y[train]
        X_val, y_val = X.iloc[val], y[val]

        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            """for i, feature in enumerate(X_train.columns):
                feature_name = feature.split("_")[0]
                if feature_dict[feature_name] == 'object':
                    X_train[feature] = X_train[feature].fillna('nan')
                    X_train[feature] = X_train[feature].replace(0, 'nan')
                else:
                    X_train[feature] = X_train[feature].fillna(0.0)
            """
        
        model = get_model(model_name, settings)
        model.fit(X_train.values, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_val)

        # Measure the performance of the model using the metrics
        accuracies.append(accuracy_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0))
        precisions.append(precision_score(y_val, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0))
        roc_aucs.append(roc_auc_score(y_val, y_pred) if len(np.unique(y)) == 2 else np.nan)
        mccs.append(matthews_corrcoef(y_val, y_pred))

        # compute specificity
        if len(np.unique(y)) == 2:
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            s = tn / (tn+fp)
        else:
            conf_matrix = confusion_matrix(y_val, y_pred)
            num_classes = conf_matrix.shape[0]
            ss = []
            for i in range(num_classes):
                TP = conf_matrix[i, i]
                FN = np.sum(conf_matrix[i, :]) - TP
                FP = np.sum(conf_matrix[:, i]) - TP
                TN = np.sum(conf_matrix) - TP - FN - FP
                s = TN / (TN+FP)
                ss.append(s)
                
            support = np.sum(conf_matrix, axis=1)
            s = np.average(ss, weights=support)
        specificitys.append(s)

        low_cf = np.array([1 if ((clone_fractions[val][i] < 0.2) & (y_val[i] == 1)) else 0 for i in range(len(y_val))])
        low_mask = (low_cf== 1)
        if len(y_val[low_mask]) == 0:
            masked_dlbcl_accuracies.append([np.nan, np.nan, np.nan])
        else:
            masked_dlbcl_accuracies.append([accuracy_score(y_val[low_mask],y_pred[low_mask] ), 
                                precision_score(y_val[low_mask],y_pred[low_mask] , average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0), 
                                recall_score(y_val[low_mask],y_pred[low_mask], average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)])
        
        with open(os.path.join(store_dir,'valid_scores_%d.txt' % k), 'w') as f:
            digits = 2
            width = len("weighted avg")
            row_fmt_mcc = (
                        "{:>{width}s} "
                        + " {:>9.{digits}}" * 2
                        + " {:>9.{digits}f}"
                        + " {:>9.{digits}}\n"
                    )
            f.write("%s Validation Scores\n" % model_name)
            f.write(classification_report(y_val, y_pred, target_names=np.asarray(types), zero_division=0))  
            mcc = matthews_corrcoef(y_val, y_pred)
            f.write(row_fmt_mcc.format("mcc", "", "", mcc, "", width=width, digits=digits))  
            f.write("\n\n") 
            f.write("Low dlbcl Scores\n")    

            if model_name == "CatBoost":
                target_labels = np.unique(np.concatenate([y_val[low_mask], y_pred[low_mask].flatten()]).astype(int))                      
            else:
                target_labels = np.unique(np.concatenate([y_val[low_mask], y_pred[low_mask]]))               
            f.write(classification_report(y_val[low_mask], y_pred[low_mask], target_names=np.asarray(types)[target_labels], zero_division=0))
            mcc = matthews_corrcoef(y_val[low_mask], y_pred[low_mask])
            f.write(row_fmt_mcc.format("mcc", "", "", mcc, "", width=width, digits=digits)) 


    performance_df = pd.DataFrame({'accuracy': accuracies, 'recall': recalls, 'precision': precisions, 'roc_auc': roc_aucs, 'mcc' : mccs, 'specificity': specificitys})
  
    # store in storedir
    masked_scores = pd.DataFrame(columns=["low_dlbcl_accuracy", "low_dlbcl_precision", "low_dlbcl_recall"], data = masked_dlbcl_accuracies)
    performance_df = pd.concat([performance_df, masked_scores], axis=1)

    # add Model and Dataset column
    performance_df["Model"] = model_name
    performance_df["Dataset"] = 'Validation'
    

    # add performance on test set with Dataset 'Test'
    model = get_model(model_name, settings)
    if sampler is not None:
            X, y = sampler.fit_resample(X, y)
    
    model.fit(X.values, y)
    if model_name == 'AttentionDeepSets':
        model.save(os.path.join(store_dir, "model.pkl"))

    y_pred = model.predict(X_test)
    performance_df_test = pd.DataFrame({'accuracy': [accuracy_score(y_test, y_pred)], 'recall': [recall_score(y_test, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)], 'precision': [precision_score(y_test, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)], 'roc_auc': [roc_auc_score(y_test, y_pred) if len(np.unique(y)) == 2 else np.nan], 'mcc' : [matthews_corrcoef(y_test, y_pred)]})
    # add specificity
    if len(np.unique(y)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        s = tn / (tn+fp)
    else:
        conf_matrix = confusion_matrix(y_test, y_pred)
        num_classes = conf_matrix.shape[0]
        ss = []
        for i in range(num_classes):
            TP = conf_matrix[i, i]
            FN = np.sum(conf_matrix[i, :]) - TP
            FP = np.sum(conf_matrix[:, i]) - TP
            TN = np.sum(conf_matrix) - TP - FN - FP
            s = TN / (TN+FP)
            ss.append(s)
            
        support = np.sum(conf_matrix, axis=1)
        s = np.average(ss, weights=support)


    performance_df_test["specificity"] = s
    performance_df_test["Model"] = model_name
    performance_df_test["Dataset"] = 'Test'

    # measure performance on low dlbcl test set
    low_cf = np.array([1 if ((clone_fractions_test[i] < 0.2) & (y_test[i] == 1)) else 0 for i in range(len(y_test))])
    low_mask = (low_cf== 1)
    if len(y_test[low_mask]) == 0:
        masked_dlbcl_accuracies_test = [np.nan,  np.nan,np.nan]
    else:
        masked_dlbcl_accuracies_test = [accuracy_score(y_test[low_mask],y_pred[low_mask] ), 
                            precision_score(y_test[low_mask],y_pred[low_mask] , average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0), 
                            recall_score(y_test[low_mask],y_pred[low_mask], average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)]

    performance_df_test = pd.concat([performance_df_test, pd.DataFrame(columns=["low_dlbcl_accuracy", "low_dlbcl_precision", "low_dlbcl_recall"], data = [masked_dlbcl_accuracies_test])], axis=1)
    performance_df = pd.concat([performance_df, performance_df_test])

    performance_df.to_csv(os.path.join(store_dir, "performance.csv"), index=False)

    with open(os.path.join(store_dir,'test_scores.txt'), 'w') as f:
        digits = 2
        width = len("weighted avg")
        row_fmt_mcc = (
                    "{:>{width}s} "
                    + " {:>9.{digits}}" * 2
                    + " {:>9.{digits}f}"
                    + " {:>9.{digits}}\n"
                )
        f.write("%s Test Scores\n" % model_name)
        f.write(classification_report(y_test, y_pred, target_names=np.asarray(types), zero_division=0))  
        mcc = matthews_corrcoef(y_test, y_pred)
        f.write(row_fmt_mcc.format("mcc", "", "", mcc, "", width=width, digits=digits))  
        f.write("\n\n") 
        f.write("Low dlbcl Scores\n")    

        if model_name == "CatBoost":
            target_labels = np.unique(np.concatenate([y_test[low_mask], y_pred[low_mask].flatten()]).astype(int))                      
        else:
            target_labels = np.unique(np.concatenate([y_test[low_mask], y_pred[low_mask]]))               
        f.write(classification_report(y_test[low_mask], y_pred[low_mask], target_names=np.asarray(types)[target_labels], zero_division=0))
        mcc = matthews_corrcoef(y_test[low_mask], y_pred[low_mask])
        f.write(row_fmt_mcc.format("mcc", "", "", mcc, "", width=width, digits=digits)) 

    with open(os.path.join(store_dir, "selected_features.json"), 'w') as outfile:
        json.dump(selected_features, outfile, indent=2)
    
    try:
        if model_name in ["Logistic Regression" or "SVM"]:
            coefficients = np.mean(model.coef_, axis=0)
            feature_importance = coefficients
        else:
            feature_importance = model.feature_importances_
        with open(os.path.join(store_dir, "feature_importances.json"), 'w') as outfile:
                sorted_idx = np.argsort(np.asarray(abs(feature_importance)))[::-1]
                sorted_X = np.asarray(X.columns)[sorted_idx]
                sorted_features = np.asarray(feature_importance, dtype=np.float64)[sorted_idx]
                res = dict(zip(list(sorted_X), list(sorted_features)))
                json.dump(res, outfile)
    except:     
        print("Feature Importance not available for model: %s" % model_name)

def infer(model_name, model_path, settings, selected_features, class_files, test_index, experiment="results/deepsets"):

    
    with open(os.path.join(model_path, "settings.json"), 'w') as outfile:
        json.dump(settings, outfile, indent=2)

    X, y, clone_fractions = read_features_deep(class_files, selected_features, scale=True, max_clones = settings["n_clones"], genefamily=settings["genefamily"] if "genefamily" in settings else False)
    X = np.array(X)
    y = np.array(y)
    clone_fractions = np.array(clone_fractions)

    X_test = X[test_index]
    y_test = y[test_index]
    clone_fractions_test = clone_fractions[test_index]

    model = get_model(model_name, settings)  
    model.load(os.path.join(model_path, "model.pkl"))

    return X, y, X_test, y_test, model


def sample_setting(model_name):
    general_distributions = dict(
                        n_splits= [3],
                        n_clones=[1,2,3,4,5,10,20,50,100],
                        genefamily = [False],
                        standardize = [True, False], #[False, True],
                        ordinal_encoding = [False], #[False, True],
                        onehot_encoding = [True], #[False, True],
                        add_clonality = [True], #[True, False],
                        add_shannon = [True], #[True, False],
                        add_richness = [True], #[True, False],
                        add_hypermutation = [True],
                        sampler = ['random_over']
                        )
    
    tree_distributions = dict(
                        n_estimators = [200, 800], #[100, 200, 400, 800], #[100, 200, 400, 800],
                        max_depth=[8,16], #[3, 6, 8, 16],
    )

    regression_distributions = dict(
                        max_iter = [100000],
                        regularization = ['l2', 'l1'], #[None, 'l2', 'l1', 'elasticnet'],
                        C = [0.01, 0.1, 1, 10],
                        solver = ["lbfgs","saga"], #["lbfgs", "saga"],
                        l1_ratio= [0.5]
    )

    svm_distributions = dict(
                        max_iter = [100000],
                        kernel = ['rbf'],
                        C = [0.001, 0.01, 0.1, 1, 10, 100],
    )

    deepset_distributions = dict(
        var_input_dim=[17],
        batch_size=[64],
        phi_hidden_dim=[64],
        phi_output_dim=[128, 512, 1024],
        rho_hidden_dim=[64, 128],
        rho_output_dim=[2],
        attention_dim=[16,64, 128], # only for simple and scaled dot product
        #attention_dim = [None],
        attention_type=["simple"],#["simple", "scaled_dot_product"],cross_attention
        epochs=[3000],
        learning_rate=[0.001, 0.0001],
        dropout = [None, 0.1, 0.2]
    )


    tree_models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
    
    max_possible_combinations = 1
    tmp_setting = dict()

    for key in general_distributions:
        ind = int(np.random.randint(0, len(general_distributions[key])))
        tmp_setting[key] =general_distributions[key][ind]
        max_possible_combinations *= len(general_distributions[key])
    if model_name in tree_models:
        for key in tree_distributions:
            ind = int(np.random.randint(0, len(tree_distributions[key])))
            tmp_setting[key] =tree_distributions[key][ind]
            max_possible_combinations *= len(tree_distributions[key])
    elif model_name == "Logistic Regression":
        for key in regression_distributions:
            ind = int(np.random.randint(0, len(regression_distributions[key])))
            tmp_setting[key] =regression_distributions[key][ind]
            if not key == "solver":
                max_possible_combinations *= len(regression_distributions[key])
    elif model_name == "SVM":
        for key in svm_distributions:
            ind = int(np.random.randint(0, len(svm_distributions[key])))
            tmp_setting[key] =svm_distributions[key][ind]
            max_possible_combinations *= len(svm_distributions[key])
    
    elif model_name == "AttentionDeepSets":
        for key in deepset_distributions:
            ind = int(np.random.randint(0, len(deepset_distributions[key])))
            tmp_setting[key] =deepset_distributions[key][ind]
            max_possible_combinations *= len(deepset_distributions[key])
    
    if tmp_setting["ordinal_encoding"] == False:
        if tmp_setting["genefamily"] == False:
            if tmp_setting["onehot_encoding"] ==  False:
                print("Warning: onehot_encoding and ordinal_encoding set to False...")
                randomly_chosen = ["ordinal_encoding", "onehot_encoding"][np.random.randint(0, 2)]
                tmp_setting[randomly_chosen] = True
                print("Randomly set %s to true..." % randomly_chosen)
    
    if model_name == "Logistic Regression":
        if tmp_setting["regularization"] in ["elasticnet", "l1"]:
            tmp_setting["solver"] = "saga"         
        else:
            tmp_setting["solver"] = "lbfgs"         
    
    
    if (False in general_distributions["ordinal_encoding"]) and (False in general_distributions["onehot_encoding"]) and (False in general_distributions["genefamily"]):
        max_possible_combinations -= 1
    
    return tmp_setting, max_possible_combinations 


def hyperopt_classical(iterations, model_name, selected_features, class_files, train_index, test_index, types, store_path=None):
    already_trained_settings = []
    model_path = os.path.join(store_path ,"outputs_%s" % model_name.replace(" ", ""))
    for path, subdirs, files in os.walk(model_path):
        for name in files:
            file = os.path.join(path, name)
            if file.endswith("settings.json"):
                with open(file) as f:
                    settings = f.read()
                    settings = settings.replace("\n", "").strip()
                    settings = json.loads(settings)
                    already_trained_settings.append(settings.copy())

    for n in tqdm(range(iterations)):
        tmp_setting, max_possible_combinations = sample_setting(model_name)
        if tmp_setting in already_trained_settings:
            print("Already trained a model with same setting configuration...")
            print("Resample tmp_setting...")
            while True:
                tmp_setting, max_possible_combinations = sample_setting(model_name)
                if tmp_setting not in already_trained_settings:
                    break    
                if len(already_trained_settings) >= max_possible_combinations:
                    print("Exhausted parameter search...")
                    return
        
        
        temp_selected_features = selected_features.copy()
        if tmp_setting["add_clonality"]:
            temp_selected_features += ["clonality"]
        if tmp_setting["add_shannon"]:
            temp_selected_features += ["shannon"]
        if tmp_setting["add_richness"]:
            temp_selected_features += ["richness"]
        if tmp_setting["add_hypermutation"]:
            temp_selected_features += ["hypermutatedFraction"]

        main(model_name, tmp_setting, temp_selected_features, class_files, train_index, test_index, types, store_path)
        already_trained_settings.append(tmp_setting)

def load_metadata(types, target_locus, path_dir, clones_dir):
    df_meta = pd.read_csv(os.path.join(path_dir, "lymphoma-reps-file-infos.csv"))
    class_files = {}
    n_classes = len(types)
    number_of_repertoires = 0
    for i in range(n_classes):
        num_repertoires = 0
        files = {}
        class_files[i] = []
        for type in types[i]:
            if target_locus == "all":
                df_file = df_meta[df_meta["lymphoma_specification"] == type]
            else:
                df_file = df_meta[(df_meta["lymphoma_specification"] == type) & (df_meta["pcr_target_locus"].str.contains(target_locus))]
            if type == "hd":
                df_file = df_file[~df_file["clones.txt.name"].isin(contaminated_hds)]
            
            files[type] = df_file["clones.txt.name"].apply(lambda file : os.path.join(clones_dir, file)).values
            class_files[i].extend(files[type])
            num_repertoires += len(files[type])
        print("Number of Class %i repertoires: %i" %(i+1, num_repertoires))
        number_of_repertoires += num_repertoires
    return class_files, number_of_repertoires


def baseline(class_files, types, store_path = None):
        
    def evaluate_baseline(y, y_baseline, low_dlbcl_mask):
        df = pd.DataFrame()
        df['accuracy'] = [accuracy_score(y, y_baseline)]
        df['recall'] = [recall_score(y, y_baseline, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)]
        df['precision'] = [precision_score(y, y_baseline, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)]
        df['roc_auc'] = [roc_auc_score(y, y_baseline) if len(np.unique(y)) == 2 else np.nan]
        # add mathews correlation coefficient
        df['mcc'] = [matthews_corrcoef(y, y_baseline)]
        
        df['low_dlbcl_test_accuracy'] = [ accuracy_score(y[low_dlbcl_mask], y_baseline[low_dlbcl_mask])]
        df['low_dlbcl_test_precision'] =[ precision_score(y[low_dlbcl_mask], y_baseline[low_dlbcl_mask], average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)]
        df['low_dlbcl_test_recall'] = [ recall_score(y[low_dlbcl_mask], y_baseline[low_dlbcl_mask], average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)]
        
        return df
     
    feature_names = ['cloneFraction']
    object_types = ['float64'] 

    X, y, clone_fractions = create_features(class_files, feature_names, object_types, n_entries=1, ordinal_encoding=True)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index =  sss.split(X, y).__next__()
    
    y_baseline = np.array([1 if cf >= 0.2 else 0 for cf in clone_fractions])

    low_dlbcl_mask =np.array([1 if ((cf < 0.2) & (y[i] == 1)) else 0 for i, cf in enumerate(clone_fractions)])
    low_dlbcl_mask = low_dlbcl_mask == 1
    
    # evaluate baseline on train and test set   
    df_baseline_train = evaluate_baseline(y[train_index], y_baseline[train_index], low_dlbcl_mask[train_index])
    df_baseline_train["Model"] = 'Baseline'
    df_baseline_train["Dataset"] = 'Train'
    df_baseline_test = evaluate_baseline(y[test_index], y_baseline[test_index], low_dlbcl_mask[test_index])
    df_baseline_test["Model"] = 'Baseline'
    df_baseline_test["Dataset"] = 'Test'

    df_baseline = pd.concat([df_baseline_train, df_baseline_test])

    if store_path == None:
        store_dir = os.path.join("outputs_Baseline")
    else:
        store_dir = os.path.join(store_path, "outputs_Baseline")
    os.makedirs(store_dir, exist_ok=True)
    df_baseline.to_csv(os.path.join(store_dir, "performance.csv"), index=False)

    with open(os.path.join(store_dir,'test_scores.txt'), 'w') as f:
        digits = 2
        width = len("weighted avg")
        row_fmt_mcc = (
                    "{:>{width}s} "
                    + " {:>9.{digits}}" * 2
                    + " {:>9.{digits}f}"
                    + " {:>9.{digits}}\n"
                )
        
        f.write("Baseline Test Scores\n")
        f.write(classification_report(y[test_index], y_baseline[test_index], target_names=np.asarray(types), zero_division=0))
        mcc = matthews_corrcoef(y, y_baseline)
        f.write(row_fmt_mcc.format("mcc", "", "", mcc, "", width=width, digits=digits)) 
        f.write("\n\n") 
        
        f.write("Low dlbcl Scores\n")    
        target_labels = np.unique(np.concatenate([y[low_dlbcl_mask], y_baseline[low_dlbcl_mask]]))
        f.write(classification_report(y[low_dlbcl_mask], y_baseline[low_dlbcl_mask], target_names=np.asarray(types)[target_labels], zero_division=0))
        mcc = matthews_corrcoef(y[low_dlbcl_mask], y_baseline[low_dlbcl_mask])
        f.write(row_fmt_mcc.format("mcc", "","", mcc, "", width=width, digits=digits))
    
    return df_baseline, train_index, test_index



if __name__ == '__main__':
    store_paths = []
    path_dir = "data/"
    clones_dir = "data/clones_mit_kidera"
    store_path = "results/nlphl_dlbcl_hd_cll/"
    comparisons = [['nlphl'], ["dlbcl", "gcb_dlbcl", "abc_dlbcl"], ['hd'], ['cll']]#[['cll'], ["dlbcl", "gcb_dlbcl", "abc_dlbcl"], ['hd'], ['unspecified'], ['nlphl'], ['thrlbcl'], ['lymphadenitis']]
    comparison_labels = ['nlphl', 'dlbcl', 'hd', 'cll']#['cll', 'dlbcl', 'hd', 'unspecified','nlphl',  'thrlbcl', 'lymphadenitis']
 
    class_files, number_of_repertoires = load_metadata(comparisons, "IGH", path_dir, clones_dir)
    print(number_of_repertoires)

    df_baseline, train_index, test_index = baseline(class_files, comparison_labels, store_path=store_path)
    selected_features = ['cloneFraction', 'lengthOfCDR3']  + ['bestVGene', 'bestDGene', 'bestJGene'] + ['KF%i' %i for i in range(1, 11)]
    
    hyperopt_classical(72, "Random Forest", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)
    hyperopt_classical(144, "Logistic Regression", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)
    store_paths.append(store_path)

    store_path = "results/nlphl_dlbcl_hd/"
    comparisons = [['nlphl'], ["dlbcl", "gcb_dlbcl", "abc_dlbcl"], ['hd']]#[['cll'], ["dlbcl", "gcb_dlbcl", "abc_dlbcl"], ['hd'], ['unspecified'], ['nlphl'], ['thrlbcl'], ['lymphadenitis']]
    comparison_labels = ['nlphl', 'dlbcl', 'hd']#['cll', 'dlbcl', 'hd', 'unspecified','nlphl',  'thrlbcl', 'lymphadenitis']
    
    class_files, number_of_repertoires = load_metadata(comparisons, "IGH", path_dir, clones_dir)
    print(number_of_repertoires)

    df_baseline, train_index, test_index = baseline(class_files, comparison_labels, store_path=store_path)
    selected_features = ['cloneFraction', 'lengthOfCDR3']  + ['bestVGene', 'bestDGene', 'bestJGene'] + ['KF%i' %i for i in range(1, 11)]
    
    hyperopt_classical(72, "Random Forest", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)
    hyperopt_classical(144, "Logistic Regression", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)
    store_paths.append(store_path)

    store_path = "results/cll_dlbcl_hd/"
    comparisons = [['cll'], ["dlbcl", "gcb_dlbcl", "abc_dlbcl"], ['hd']]#[['cll'], ["dlbcl", "gcb_dlbcl", "abc_dlbcl"], ['hd'], ['unspecified'], ['nlphl'], ['thrlbcl'], ['lymphadenitis']]
    comparison_labels = ['cll', 'dlbcl', 'hd']#['cll', 'dlbcl', 'hd', 'unspecified','nlphl',  'thrlbcl', 'lymphadenitis']
 
    class_files, number_of_repertoires = load_metadata(comparisons, "IGH", path_dir, clones_dir)
    print(number_of_repertoires)

    df_baseline, train_index, test_index = baseline(class_files, comparison_labels, store_path=store_path)
    selected_features = ['cloneFraction', 'lengthOfCDR3']  + ['bestVGene', 'bestDGene', 'bestJGene'] + ['KF%i' %i for i in range(1, 11)]
    
    hyperopt_classical(72, "Random Forest", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)
    hyperopt_classical(144, "Logistic Regression", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)
    store_paths.append(store_path)

    score_to_choose_best = "f1"
    best_score_test = -np.inf
    best_score_valid = -np.inf
    best_model_test = ""
    best_model_valid = ""
    scores_txt_test = ""
    scores_txt_valid = ""
    for store_path in store_paths:
        print("\n\n\n\n", store_path)
        for path, subdirs, files in os.walk(store_path):
            for name in files:
                    file = os.path.join(path, name)
                    if file.endswith("performance.csv"):
                        if "Baseline" in file:
                            baseline_results = pd.read_csv(file)
                        else:
                            model_results = pd.read_csv(file)
                            if score_to_choose_best == "f1":
                                prec_test = model_results[model_results["Dataset"] == "Test"]["precision"].iloc[0]
                                rec_test = model_results[model_results["Dataset"] == "Test"]["recall"].iloc[0]
                                score_test = 2 * prec_test * rec_test / (prec_test + rec_test)
                                prec_valid = model_results[model_results["Dataset"] == "Validation"]["precision"]
                                rec_valid = model_results[model_results["Dataset"] == "Validation"]["recall"]
                                score_valid = np.mean(2 * prec_valid * rec_valid / (prec_valid + rec_valid))
                            else:
                                score_test = model_results[model_results["Dataset"] == "Test"][score_to_choose_best].iloc[0]
                                score_valid = np.mean(model_results[model_results["Dataset"] == "Validation"][score_to_choose_best])
                            if score_test > best_score_test:
                                best_score_test = score_test
                                best_model_test = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)) 
                                with open(os.path.join(path,"test_scores.txt")) as f:
                                    scores_txt_test = f.read()
                            if score_valid > best_score_valid:
                                best_score_valid = score_valid
                                best_model_valid = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)) 
                                with open(os.path.join(path,"test_scores.txt")) as f:
                                    scores_txt_valid = f.read()
        print("\n\n")
        print("Best model Test: ", best_model_test)
        print(scores_txt_test)
        print("\n\n")
        print("Best model Validation: ", best_model_valid)
        print(scores_txt_valid)