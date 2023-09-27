import numpy as np
import pandas as pd
import os
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

def get_clonset_info(rep, method, quant="cloneFraction"):
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
    if quant == "cloneCount":
        counts = np.asarray(rep["cloneCount"])
    elif quant == "cloneFraction":
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
        if hmax == 0 or shannon==0:
            eveness = 0 
        else:
            eveness = shannon/hmax
        info = 1-eveness
        if np.isnan(info) or np.isinf(info):
            info = 1
    elif method == "nucleotid_clones":
        info = len(rep["nSeqCDR3"].unique())
    elif method == "out_of_frames":
        info = rep["aaSeqCDR3"].apply(lambda x: "_" in x or "*" in x)
    elif method == "reads":
        info = sum(rep["cloneCount"])
    elif method == "aminoacid_clones":
        info = n_aa_clones

    elif method == "hypermutation":
        hyper = rep[rep.vBestIdentityPercent < 0.98]
        info = np.sum(hyper.cloneFraction)
    
    return info

def read_clones_txt(files, clones_txt_dict=None, normalize_read_count = None):
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

        if normalize_read_count == None:
            #df_file["cloneFraction"] = df_file["cloneFraction"].apply(lambda x: float(x.replace(",",".").replace("+","-")) if isinstance(x, str) else float(x)) 
            df_file["cloneFraction"] = df_file["cloneFraction"].apply(lambda x:float(x))
        else:
            norm_factor = sum(df_file["cloneCount"])/normalize_read_count
            df_file["cloneCount"] = df_file["cloneCount"]/norm_factor
            df_file = df_file[~(df_file["cloneCount"]<2)]
            df_file["cloneFraction"] = (df_file["cloneCount"]/sum(df_file["cloneCount"]))
            df_file["cloneCount"] = df_file["cloneCount"].apply(np.ceil)

        df_file["clonality"] = get_clonset_info(df_file, "clonality")
        df_file["shannon"] = get_clonset_info(df_file, "shannon")
        df_file["inv_simpson"] = get_clonset_info(df_file, "inv_simpson")
        df_file["simpson"] = get_clonset_info(df_file, "simpson")
        df_file["gini"] = get_clonset_info(df_file, "gini")
        df_file["chao1"] = get_clonset_info(df_file, "chao1")
        df_file["#nucleotide_clonotypes"] = get_clonset_info(df_file, "nucleotid_clones")
        df_file["#aminoacid_clonotypes"] = get_clonset_info(df_file, "aminoacid_clones")
        df_file["total_reads"] = get_clonset_info(df_file, "reads")
        df_file["out_of_frames"] = get_clonset_info(df_file, "out_of_frames")

        df_file[['KF1', 'KF2', 'KF3', 'KF4', 'KF5',
                'KF6', 'KF7', 'KF8', 'KF9', 'KF10']] = df_file.aaSeqCDR3.apply(lambda x: list(peptides.Peptide(x).kidera_factors())).to_list()
        
        df_file["low_reads"] = df_file["total_reads"] < 30000
        df_file["hypermutated"] = df_file["vBestIdentityPercent"] < 0.98
        df_file["hypermutatedFraction"] = get_clonset_info(df_file, "hypermutation")

        df_raw.append(df_file)
    df_raw = pd.concat(df_raw)
    return df_raw

def load_clone_files_data(project_path, normalize_read_count=None):
    clone_files = []
    for path, subdirs, files in os.walk(project_path):
        for name in files:
            file = os.path.join(path, name)
            if file.endswith("clones.txt"):
                clone_files.append(file)
    df = read_clones_txt(clone_files, normalize_read_count=normalize_read_count)
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


def load_model_results(results_path, condition, models, metric, weighted_metric = False):
    results_df = pd.DataFrame()
    settings_df = pd.DataFrame()

    for model in models:
        model_path = os.path.join(results_path, condition,"outputs_" + model)
        model_settings = pd.DataFrame()
        model_results = pd.DataFrame()
        runs = []
        for path, subdirs, files in os.walk(model_path):
            output_identifier = os.path.basename(path)
            if weighted_metric:
                results_ = pd.DataFrame(columns = ["run", "Dataset", "F1", "recall", "precision", 
                                                   "low_dlbcl_F1", "low_dlbcl_recall", "low_dlbcl_precision"]
                                       )
            for name in files:
                file = os.path.join(path, name)
                if file.endswith("settings.json"):
                    with open(file) as f:
                        settings = f.read()
                        settings = settings.replace("\n", "").strip()
                        settings = json.loads(settings)

                        if len(model_settings) == 0:
                            model_settings = pd.DataFrame(settings, index=[0])
                        else:
                            model_settings = pd.concat([model_settings,  pd.DataFrame(settings, index=[0])], ignore_index=True)
                    runs.append(output_identifier)
                if weighted_metric:
                    if file.endswith("test_scores.txt"):
                        row = [output_identifier, "Test"]
                        with open(file) as f: 
                            low_dlbcl = False
                            for line in f.readlines():
                                if "low dlbcl" in line:
                                    low_dlbcl = True
                                if "weighted avg" in line:
                                    if low_dlbcl:
                                        row.append(float(list(filter(None, line.split(" ")))[-2]))
                                        row.append(float(list(filter(None, line.split(" ")))[-3]))
                                        row.append(float(list(filter(None, line.split(" ")))[-4]))
                                    else:
                                        row.append(float(list(filter(None, line.split(" ")))[-2]))
                                        row.append(float(list(filter(None, line.split(" ")))[-3]))
                                        row.append(float(list(filter(None, line.split(" ")))[-4]))
                        
                        results_.loc[len(results_)] = row
                    elif "valid_scores" in file:
                        row = [output_identifier, "Validation"]
                        with open(file) as f:
                            low_dlbcl = False
                            for line in f.readlines():
                                if "low dlbcl" in line:
                                    low_dlbcl = True
                                if "weighted avg" in line:
                                    if low_dlbcl:
                                        row.append(float(list(filter(None, line.split(" ")))[-2]))
                                        row.append(float(list(filter(None, line.split(" ")))[-3]))
                                        row.append(float(list(filter(None, line.split(" ")))[-4]))
                                    else:
                                        row.append(float(list(filter(None, line.split(" ")))[-2]))
                                        row.append(float(list(filter(None, line.split(" ")))[-3]))
                                        row.append(float(list(filter(None, line.split(" ")))[-4]))
                        
                        results_.loc[len(results_)] = row
                        

                else:
                    if file.endswith("performance.csv"):
                        results_ = pd.read_csv(file)
                        results_["run"] = output_identifier
                        results_["F1"] = 2 * (results_.precision * results_.recall) / (results_.precision + results_.recall)
                        results_["low_dlbcl_F1"] = 2 * (results_.low_dlbcl_precision * results_.low_dlbcl_recall) / (results_.low_dlbcl_precision + results_.low_dlbcl_recall)
                        if len(model_results) == 0:
                            model_results = results_
                        else:
                            model_results = pd.concat([model_results, results_], ignore_index = True)
            if weighted_metric: 
                model_results = pd.concat([model_results, results_], ignore_index = True)
                
        model_settings["run"] = runs
        model_settings["model"] = model
        if len(settings_df) == 0:
            settings_df = model_settings
            results_df = model_results
        else:
            results_df = pd.concat([results_df, model_results])
            settings_df = pd.concat([settings_df, model_settings])
    
    df = pd.DataFrame(settings_df.values)
    df.columns = settings_df.columns
    
    mean_validation_metric = []
    test_metric = []
    mean_validation_metric_low_dlbcl = []
    test_metric_low_dlbcl = []
    
    for i,row in settings_df.iterrows():
        run = row.run
        sub_df = results_df[results_df.run == run]
        if len(sub_df) == 0:
            mean_validation_metric += [np.nan]
            test_metric += [np.nan]
            mean_validation_metric_low_dlbcl += [np.nan]
            test_metric_low_dlbcl += [np.nan] 
        else:
            mean_validation_metric += [np.mean(sub_df[sub_df.Dataset == "Validation"][metric])]
            test_metric += [sub_df[sub_df.Dataset == "Test"][metric].iloc[0]]
            mean_validation_metric_low_dlbcl += [np.mean(sub_df[sub_df.Dataset == "Validation"]["low_dlbcl_" + metric])]
            test_metric_low_dlbcl += [sub_df[sub_df.Dataset == "Test"]["low_dlbcl_" + metric].iloc[0]]
    
    
    df["mean_validation_%s" % metric] = mean_validation_metric
    df["test_%s" % metric] = test_metric
    df["mean_validation_low_dlbcl_%s" % metric] = mean_validation_metric_low_dlbcl
    df["test_low_dlbcl_%s" % metric] = test_metric_low_dlbcl
    
    
    return df, results_df, settings_df