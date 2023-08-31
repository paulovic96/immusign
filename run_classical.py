import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random, time, string
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import json
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

import skbio


contaminated_hds = ['105-D28-Ig-gDNA-PB-Nuray-A250_S180.clones.txt',
 '108-D0-Ig-gDNA-PB-Nuray-A250_S185.clones.txt',
 '109-D0-Ig-gDNA-PB-Nuray-A250_S187.clones.txt',
 'Barbara-hs-IGH-Zeller-1-HD-PB-gDNA_S140.clones.txt',
 'Barbara-hs-IGH-Zeller-110-HD-PB-gDNA_S148.clones.txt',
 'Christoph-hs-IGH-HD078-PB-gDNA-HLA-DRB1-AIH2-liver-gDNA_S255.clones.txt',
 'ChristophS-hs-IGH-HD078-31-01-2017-gDNA_S126.clones.txt',
 'HD-078-IGH-Dona_S52.clones.txt',
 'HD-Mix2-250ng-10hoch6-FR1-Ig-Anna-m-binder-A250_S95.clones.txt',
 'HD-Mix2-250ng-200000-FR1-Ig-Anna-m-binder-A250_S97.clones.txt']

def get_clonset_info(rep, method, quant="proportion"):
    """
    chao1:  Non-parametric estimation of the number of classes in a population: Sest = Sobs + ((F2 / 2G + 1) - (FG / 2 (G + 1) 2))
            Sets = number classes
            Sobs = number classes observed in sample
            F = number singeltons (only one individual in class)
            G = number doubletons (exactly two individuals in class)

    gini index:  'inequality' among clonotypes. 0 for equal distribution and 1 for total unequal dstribution only 1 clone in set
    
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
        if np.isnan(info) or np.isinf(info):
            info = 1

    
    return info

def create_vdj_index(class_files, family = False):
    
    # iterateo over all class files and store bestvgene, bestdgene, bestjgene to a unique list
    bestvgene = []
    bestdgene = []
    bestjgene = []
    for i, type in enumerate(class_files.keys()):
        for j, file in enumerate(class_files[type]):
        
            df = pd.read_csv("immusign/data/clones_mit_kidera/%s" % file, sep="\t")
                
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
        with open("immusign/data/gene2index_family.json", "w") as outfile:
            json.dump(gene2index, outfile)
    else:
        with open("immusign/data/gene2index.json", "w") as outfile:
            json.dump(gene2index, outfile)
    print(gene2index)

def _run_name(model_type):
    return time.strftime("output_%b_%d_%H%M%S_") + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5)) +  "_%s" %model_type

def _custom_combiner(feature, category):
    return str(feature) + "_" + str(category)

def read_feature(files, features , n_entries, flatten=True):
    """
     This method creates a feature vector for each repertoire by concatenating the given set of features 
     for the most n_entries frequent clones and concatenates them.
    """
    data = []
    cloneFraction = []

    for i, file in enumerate(files):
        df = pd.read_csv("immusign/data/clones_mit_kidera/%s" %file, sep="\t")
        df = df[df['cloneFraction'].apply(lambda x: isinstance(x, (int, float)))] 
        if (df.shape[0] == 0):
            continue
        cloneFraction.append(df['cloneFraction'].values[0])
        if "clonality" in features:
            clonality = get_clonset_info(df, "clonality")
            features.remove("clonality")
    
        df = df.iloc[:n_entries]
        d = df[features].values
        # PADD ING FOR RANDOM FOREST
        if (df[features].values.shape[0] < n_entries):
            pad = n_entries  - df[features].values.shape[0]
            d = [df[features].values, np.zeros((pad, len(features)))]
            d= np.concatenate(d)
            
        # flatten data such that it can be an input for the random forest
        if flatten:
            try: 
                data.append(np.append(d.flatten(),clonality))
                features += ["clonality"]
            except NameError: 
                data.append(d.flatten())
        else:
            try: 
                d = np.column_stack((d, np.repeat(clonality)))
                data.append(d)
                features += ["clonality"]
            except NameError: 
                data.append(d)
            
    data = np.stack(data, axis=0)
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

def create_features(class_files, feature_names, object_types, n_entries=5, onehot_encoding=False, ordinal_encoding=False, standardize=False, flatten=True, genefamily=True):
    # create feature names for each clone
    column_names = []
    column_to_features = {}
    column_to_type = {}
    for i in range(n_entries):
        for feature in feature_names:
            if feature == "clonality":
                continue
            else:
                column_names.append(feature + "_%i" %i)
                column_to_features[feature + "_%i" %i] = feature
                column_to_type[feature + "_%i" %i] = object_types[feature_names.index(feature)]
    if "clonality" in feature_names:
        column_names.append("clonality")
        column_to_features["clonality"] = "clonality"
        column_to_type["clonality"] = object_types[feature_names.index("clonality")]

    X = []
    clone_fractions = []
    y = []
    for i, type in enumerate(class_files.keys()):
        #print("Read feature for class %i" %i)
        X_type, cf_type = read_feature(class_files[type], feature_names, n_entries, flatten=flatten)
        y_type = [int(type)] * len(X_type)
        X.append(X_type)
        y.extend(y_type)
        clone_fractions.extend(cf_type)
    X = np.concatenate(X)
    X = pd.DataFrame(X, columns=column_names)
    for i, feature in enumerate(column_names):
        X[feature] = X[feature].astype(column_to_type[feature])
    X = X.fillna('nan')
    for col in X.select_dtypes(include=['object']):
        X[col] = X[col].astype('category')
    y = np.array(y)
    clone_fractions = np.array(clone_fractions)

    if ordinal_encoding or genefamily or onehot_encoding:
        categorical_cols = X.columns[X.dtypes == 'category']
        if genefamily:
             # read gene2index mapping
            with open("immusign/data/gene2index.json", "r") as infile:
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
   

    return X, y, clone_fractions

def main(model_name,settings, selected_features, class_files, train_index, test_index, types, store_path=None):

    print("Start %s training..." %model_name)
    if store_path == None:
        store_dir  = os.path.join("immusign/outputs_%s" % model_name.replace(" ", ""), _run_name("classification") )
    else:
        store_dir = os.path.join(store_path, "outputs_%s" % model_name.replace(" ", ""), _run_name("classification") )

    os.makedirs(store_dir)

    with open(os.path.join(store_dir, "settings.json"), 'w') as outfile:
        json.dump(settings, outfile, indent=2)

    feature_names = ['cloneFraction', 'lengthOfCDR3', 'clonality']  + ['bestVGene', 'bestDGene', 'bestJGene'] + ['KF%i' %i for i in range(1, 11)] 
    object_types = ['float64', 'int64', 'float64']  +  ['object', 'object', 'object']+['float64' for i in range(10)] 

    # create dict from feature name to object type
    feature_dict = {}
    for i in range(len(feature_names)):
        feature_dict[feature_names[i]] = object_types[i]
        
    selected_object_types = [feature_dict[feature] for feature in selected_features]

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
    
    k_fold = StratifiedKFold(n_splits=settings["n_splits"], shuffle=True, random_state=15)

    # init accuracy, recall, precision, roc_auc, mcc for validation set in one dict
    accuracies = []
    recalls = []
    precisions = []
    roc_aucs = []
    mccs = []
    masked_dlbcl_accuracies = []
    
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
            model = LogisticRegression(max_iter=settings["max_iter"])
        elif model_name == "SVM":
            model = SVC(max_iter=settings["max_iter"], kernel = settings["kernel"])
        return model

    for k, (train, val) in enumerate(k_fold.split(X, y)):
      
        X_train, y_train = X.iloc[train], y[train]
        X_val, y_val = X.iloc[val], y[val]

        model = get_model(model_name, settings)
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_val)

        # Measure the performance of the model using the metrics
        accuracies.append(accuracy_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0))
        precisions.append(precision_score(y_val, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0))
        roc_aucs.append(roc_auc_score(y_val, y_pred) if len(np.unique(y)) == 2 else np.nan)
        mccs.append(matthews_corrcoef(y_val, y_pred))

        low_cf = np.array([1 if ((clone_fractions[val][i] < 0.2) & (y_val[i] == 1)) else 0 for i in range(len(y_val))])
        low_mask = (low_cf== 1)
        if len(y_val[low_mask]) == 0:
            masked_dlbcl_accuracies.append([np.nan, np.nan, np.nan])
        else:
            masked_dlbcl_accuracies.append([accuracy_score(y_val[low_mask],y_pred[low_mask] ), 
                                precision_score(y_val[low_mask],y_pred[low_mask] , average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0), 
                                recall_score(y_val[low_mask],y_pred[low_mask], average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)])


    performance_df = pd.DataFrame({'accuracy': accuracies, 'recall': recalls, 'precision': precisions, 'roc_auc': roc_aucs, 'mcc' : mccs})
  
    # store in storedir
    masked_scores = pd.DataFrame(columns=["low_dlbcl_accuracy", "low_dlbcl_precision", "low_dlbcl_recall"], data = masked_dlbcl_accuracies)
    performance_df = pd.concat([performance_df, masked_scores], axis=1)

    # add Model and Dataset column
    performance_df["Model"] = model_name
    performance_df["Dataset"] = 'Validation'
    

    # add performance on test set with Dataset 'Test'
    model = get_model(model_name, settings)
    model.fit(X, y)

    y_pred = model.predict(X_test)
    performance_df_test = pd.DataFrame({'accuracy': [accuracy_score(y_test, y_pred)], 'recall': [recall_score(y_test, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)], 'precision': [precision_score(y_test, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)], 'roc_auc': [roc_auc_score(y_test, y_pred) if len(np.unique(y)) == 2 else np.nan], 'mcc' : [matthews_corrcoef(y_test, y_pred)]})
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
                        
    


def hyperopt_classical(iterations, model_name, selected_features, class_files, train_index, test_index, types, store_path=None):

    distributions = dict(
                        n_splits= [3],
                        standardize = [False, True],
                        ordinal_encoding = [False, True],
                        onehot_encoding = [False, True],
                        max_depth=[3, 6, 8, 16],
                        n_clones = [1, 3, 5, 10, 20, 50],
                        genefamily = [False],
                        max_iter = [100, 500, 1000, 5000],
                        kernel = ['rbf', 'poly', 'sigmoid'],
                        n_estimators = [100, 200, 400, 800]
                        )
    already_trained_settings = []
    model_path = os.path.join(store_path ,"outputs_%s" % model_name)
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
        tmp_setting = dict()
        for key in distributions:
            ind = int(np.random.randint(0, len(distributions[key])))
            tmp_setting[key] =distributions[key][ind]
        if tmp_setting["ordinal_encoding"] == False:
            if tmp_setting["genefamily"] == False:
                if tmp_setting["onehot_encoding"] ==  False:
                    print("Warning: onehot_encoding and ordinal_encoding set to False...")
                    randomly_chosen = ["ordinal_encoding", "onehot_encoding"][np.random.randint(0, 2)]
                    tmp_setting[randomly_chosen] = True
                    print("Randomly set %s to true..." % randomly_chosen)
        
        if tmp_setting in already_trained_settings:
            print("Already trained a model with same setting configuration...")
            print("Resample tmp_setting...")
            while True:
                tmp_setting = dict()
                for key in distributions:
                    ind = int(np.random.randint(0, len(distributions[key])))
                    tmp_setting[key] =distributions[key][ind]
                if tmp_setting["ordinal_encoding"] == False:
                    if tmp_setting["genefamily"] == False:
                        if tmp_setting["onehot_encoding"] ==  False:
                            print("Warning: onehot_encoding and ordinal_encoding set to False...")
                            randomly_chosen = ["ordinal_encoding", "onehot_encoding"][np.random.randint(0, 2)]
                            tmp_setting[randomly_chosen] = True
                            print("Randomly set %s to true..." % randomly_chosen)
                if tmp_setting not in already_trained_settings:
                    break    
        
        main(model_name, tmp_setting, selected_features, class_files, train_index, test_index, types, store_path)

def load_metadata(types, target_locus, path_dir):

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
            files[type] = df_file["clones.txt.name"].values
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
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
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
        store_dir = os.path.join("immusign/results_twoclass/outputs_Baseline")
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
    path_dir = "immusign/data/"
    store_path = "immusign/results_cll_dlbcl_hd/"
    comparisons = [['cll'], ["dlbcl", "gcb_dlbcl", "abc_dlbcl"], ['hd']]
    comparison_labels = ['cll', 'dlbcl', 'hd']

    #'unspecified', 'dlbcl', 'nlphl', 'abc_dlbcl', 'thrlbcl', 'lymphadenitis', hd
    
    class_files, number_of_repertoires = load_metadata(comparisons, "IGH", path_dir)
    print(number_of_repertoires)

    selected_features = ['cloneFraction', 'lengthOfCDR3']  + ['bestVGene', 'bestDGene', 'bestJGene'] + ['KF%i' %i for i in range(1, 11)]
 
    df_baseline, train_index, test_index = baseline(class_files, comparison_labels, store_path=store_path)

    hyperopt_classical(20, "Logistic Regression", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)
    hyperopt_classical(20, "SVM", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)
    hyperopt_classical(20, "Random Forest", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)
    hyperopt_classical(20, "LightGBM", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)
    #hyperopt_classical(20, "CatBoost", selected_features, class_files, train_index, test_index, comparison_labels, store_path=store_path)

    score_to_choose_best = "mcc"
    best_score_test = -np.inf
    best_score_valid = -np.inf
    best_model_test = ""
    best_model_valid = ""
    scores_txt_test = ""
    scores_txt_valid = ""
    for path, subdirs, files in os.walk(store_path):
        for name in files:
                file = os.path.join(path, name)
                if file.endswith("performance.csv"):
                    if "Baseline" in file:
                        baseline_results = pd.read_csv(file)
                    else:
                        model_results = pd.read_csv(file)
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