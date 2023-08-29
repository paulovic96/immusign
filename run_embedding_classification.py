import numpy as np
import pandas as pd
from model_zoo import *
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef, classification_report
import matplotlib.pyplot as plt
import os
import json
import utils
import time
import random
import string

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

def _run_name(model_type):
    return time.strftime("output_%b_%d_%H%M%S_") + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5)) +  "_%s" %model_type

def load_data(comparisons, settings):
    df = pd.read_pickle("immusign/immusign_not_normalized_with_out_of_frame_merged_raw_data_cleaned_olga_pgen_embeddings_igh.pkl")
    df = df[~df["clones.txt.name"].isin(contaminated_hds)]
    df = df[df.lymphoma_specification.isin(sum(comparisons,[]))]

    embedded_df = pd.DataFrame(columns=["clones.txt.name", "label", "embedding"])
    cloneFraction = []
    for name, group in df.groupby("clones.txt.name"):
        if settings["embedding_method"] == "sum":
            rep_embedding = sum(group.cloneCount * group.esm_embedding)
        elif settings["embedding_method"] == "mean":
            rep_embedding = np.mean(group.cloneCount * group.esm_embedding)
        
        for i, comp in enumerate(comparisons):
            if group.lymphoma_specification.iloc[0] in comp:
                label = i
        embedded_df.loc[len(embedded_df)] = [name, label, rep_embedding]
        cloneFraction.append(group['cloneFraction'].values[0])
    
    X = np.concatenate(list(embedded_df["embedding"].apply(lambda x: x.reshape((1,-1)))))
    y = list(embedded_df["label"])
    
    if settings["standardize"]:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        X = (X-means)/stds
    return X, np.asarray(y), np.asarray(cloneFraction)



def initialize_model(model_name, settings):
    if model_name == "ResNet": 
        model = ResnetModel(input_channel = settings["input_channel"], output_channel=settings["output_channel"], 
                            hidden_units=settings["hidden_units"], hidden_layers=settings["hidden_layers"], 
                            global_average_pooling=False)
        
    elif model_name == "Perceptron":
        model = NonLinearModel(input_channel=settings["input_channel"], output_channel=settings["output_channel"],
                               hidden_units = settings["hidden_units"],
                               hidden_layers= settings["hidden_layers"],
                               global_average_pooling=False)
    device = settings["device"]
    model = model.to(device)
    return model



def train_model(model_name, comparisons, settings, train_index, test_index, types,store_path=None):
    
    print("Start %s training..." % model_name)
    if store_path == None:
        store_dir  = os.path.join("immusign/outputs_%s" % model_name.replace(" ", ""), _run_name("classification") )
    else:
        store_dir = os.path.join(store_path, "outputs_%s" % model_name.replace(" ", ""), _run_name("classification") )

    os.makedirs(store_dir)

    with open(os.path.join(store_dir, "settings.json"), 'w') as outfile:
        json.dump(settings, outfile, indent=2)

    X, y, clone_fractions  = load_data(comparisons, settings)

    X_test = X[test_index]
    y_test = y[test_index]
    clone_fractions_test = clone_fractions[test_index]
    
    X = X[train_index]
    y = y[train_index]
    
    k_fold = StratifiedKFold(n_splits=settings["n_splits"], shuffle=True, random_state=15)
    folds = k_fold.split(X, y)
    
    # init accuracy, recall, precision, roc_auc, mcc for validation set in one dict
    accuracies = []
    recalls = []
    precisions = []
    roc_aucs = []
    mccs = []
    masked_dlbcl_accuracies = []

    for k, (train, val) in enumerate(folds):
      
        X_train, y_train = X[train], y[train]
        X_val, y_val = X[val], y[val]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(settings["device"])
        y_train_tensor = torch.tensor(utils.one_hot_from_label(y_train, n_classes=len(comparisons)), dtype=torch.float32).to(settings["device"])

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(settings["device"])
        y_val_tensor = torch.tensor(utils.one_hot_from_label(y_val, n_classes=len(comparisons)), dtype=torch.float32).to(settings["device"])
        
        dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
        dataset_val = TensorDataset(X_val_tensor, y_val_tensor)

        dataloader_train = DataLoader(dataset_train, batch_size=settings["batch_size"], shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)
        
        n_epochs = settings["n_epochs"]
        
        model = initialize_model(model_name, settings)
        
        optimizer = optim.Adam(model.parameters(),lr=settings["lr"])
        criterion = torch.nn.CrossEntropyLoss()

        epoch_loss = []
        for i in tqdm(range(n_epochs), desc ='Training  on Fold %d...' % (k+1)):
            average_epoch_loss = []
            running_loss = 0.0

            for batch_x, batch_y in dataloader_train:                
                Y_hat = model(batch_x)
                optimizer.zero_grad() # set gradients to zero before performing backprop
                
                loss = criterion(Y_hat, batch_y)
                average_epoch_loss += [loss.item()]
                
                running_loss += loss.item()
                loss.backward()  # compute dloss/dx and accumulated into x.grad
                optimizer.step()  # compute x += -learning_rate * x.grad
            epoch_loss += [np.mean(average_epoch_loss)]

        val_loss = []
        y_pred = []      
        for batch_x, batch_y in dataloader_val:
            with torch.no_grad():
                Y_hat = model(batch_x)
                loss = criterion(Y_hat, batch_y)
                Y_hat = torch.nn.functional.softmax(Y_hat, dim=1)
            
            y_pred += list(np.argmax(Y_hat.detach().cpu().numpy(), axis=1))
            val_loss += [loss.item()]

        y_pred = np.asarray(y_pred)
        # Measure the performance of the model using the metrics
        accuracies.append(accuracy_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0))
        precisions.append(precision_score(y_val, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0 ))
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
    X_tensor = torch.tensor(X, dtype=torch.float32).to(settings["device"])
    y_tensor = torch.tensor(utils.one_hot_from_label(y, n_classes=len(comparisons)), dtype=torch.float32).to(settings["device"])

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(settings["device"])
    y_test_tensor = torch.tensor(utils.one_hot_from_label(y_test, n_classes=len(comparisons)), dtype=torch.float32).to(settings["device"])

    dataset = TensorDataset(X_tensor, y_tensor)
    dataset_test = TensorDataset(X_test_tensor, y_test_tensor)
    dataloader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    
    model = initialize_model(model_name, settings)
    optimizer = optim.Adam(model.parameters(), lr=settings["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    epoch_loss = []
    for i in tqdm(range(n_epochs), desc ='Final Training...'):
        average_epoch_loss = []
        running_loss = 0.0

        for batch_x, batch_y in dataloader:                
            Y_hat = model(batch_x)
            optimizer.zero_grad() # set gradients to zero before performing backprop    
            loss = criterion(Y_hat, batch_y)
            average_epoch_loss += [loss.item()]
            
            running_loss += loss.item()
            loss.backward()  # compute dloss/dx and accumulated into x.grad
            optimizer.step()  # compute x += -learning_rate * x.grad
        epoch_loss += [np.mean(average_epoch_loss)]

    test_loss = []
    y_pred = []      
    for batch_x, batch_y in dataloader_test:
        with torch.no_grad():
            Y_hat = model(batch_x)
            loss = criterion(Y_hat, batch_y)
            Y_hat = torch.nn.functional.softmax(Y_hat, dim=1)
        
        y_pred += list(np.argmax(Y_hat.detach().cpu().numpy(), axis=1))
        test_loss += [loss.item()]
    y_pred = np.asarray(y_pred)
    performance_df_test = pd.DataFrame({'accuracy': [accuracy_score(y_test, y_pred)], 'recall': [recall_score(y_test, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted',zero_division=0.0)], 'precision': [precision_score(y_test, y_pred, average='binary' if len(np.unique(y)) == 2 else 'weighted', zero_division=0.0)], 'roc_auc': [roc_auc_score(y_test, y_pred) if len(np.unique(y)) == 2 else np.nan], 'mcc' : [matthews_corrcoef(y_test, y_pred)]})
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
                            recall_score(y_test[low_mask],y_pred[low_mask], average='binary' if len(np.unique(y)) == 2 else 'weighted',zero_division=0.0 )]

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

def hyperopt_classical(iterations, model_name, comparisons, train_index, test_index, types, store_path=None):
    distributions = dict(
                        device = ["mps"],
                        n_splits= [3],
                        embedding_method = ["sum", "mean"],
                        input_channel = [320],
                        output_channel = [len(types)],
                        hidden_units = [320, 640, 960, 1280],
                        hidden_layers = [1, 3, 5, 7],
                        standardize = [False, True],
                        lr = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                        n_epochs = [100, 200, 300, 400, 500],
                        batch_size = [8, 16, 32]
                        )

    for n in tqdm(range(iterations)):
        tmp_setting = dict()
        for key in distributions:
            ind = int(np.random.randint(0, len(distributions[key])))
            tmp_setting[key] =distributions[key][ind]
        train_model(model_name, comparisons, tmp_setting, train_index, test_index, types, store_path)

    
if __name__ == '__main__':
    store_path = "immusign/results_nlphl_dlbcl_hd/"
    comparisons = [['nlphl'], ["dlbcl", "gcb_dlbcl", "abc_dlbcl"], ['hd']]
    comparison_labels = ['nlphl', 'dlbcl', 'hd']

    X, y, clone_fraction = load_data(comparisons, dict(embedding_method = "sum", standardize=True))
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_index, test_index =  sss.split(X, y).__next__()

    hyperopt_classical(20, "Perceptron", comparisons, train_index, test_index, comparison_labels, store_path=store_path)
    hyperopt_classical(20, "ResNet", comparisons, train_index, test_index, comparison_labels, store_path=store_path)
    
    #store_path = "immusign/results_nlphl_dlbcl_hd/outputs_DNNs/"
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
    
    
    