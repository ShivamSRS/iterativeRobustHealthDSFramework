from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot
from collections import Counter
import random
from numpy import where
random_state = np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import statistics
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.metrics import accuracy_score as acc
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE 
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
random_state = np.random.RandomState(42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,  make_scorer, precision_score, recall_score, \
average_precision_score, accuracy_score, average_precision_score
from sklearn.metrics import roc_curve, auc  , precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import random
import seaborn as sns
import joblib
from sklearn import metrics
from scipy.stats import ks_2samp
import numpy as np
from datetime import datetime
import ast
from sklearn.feature_selection import f_regression, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn import feature_selection
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import time
from os import listdir
from os.path import isfile, join
import math
import ast
from dataselectutils import get_dataset,get_test_dataset
from dataselectutils import get_dataset,statistical_filter,mutual_info, RFE_features,permutation_importance_features
from arguments import time_window,data_files,test_folder,train_folder,project_folder,data_folder,label_col,pt_col
from arguments import ventDataFolder,ventDataFiles_median
from configs import Unbalanced,Downsample_25,feature_selection_method,feature_import_path,algorithm,use_features,prefered_columns,use_prefered_cols
from configs import data_setting
from VentWaveData import VentData

data_files = data_files

project_folder = project_folder
data_folder = data_folder
train_folder = train_folder
test_folder = test_folder
label_col,pt_col = label_col,pt_col

import warnings
warnings.filterwarnings("ignore")


calibration_df  =pd.read_csv("/data0/May12/ehrdata2/results/12h/all_features/May12_12h_vent_summary/Unbalanced_25_DS/calibration_all_feats_test_set.csv")#("/data0/ehrdata2/results/CFS_400_50_True/MEDIAN_ONLY_OS_48h_Ventilator_Waveform_summary_EHR_both_CFS_400_50_True/Unbalanced_25_DS/calibration_CFS_400_50_True_test_set.csv")
results_path = "/data0/May12/ehrdata2/results/12h/all_features/May12_12h_vent_summary/Unbalanced_25_DS/"
print(set(calibration_df["buckets"]))
indbuckets = calibration_df.groupby("buckets")
calibration_graph_counts= {}
calibration_graph_ards_count = {}
calibration_graph_ards_prevalence_observed ={}
for name, group in indbuckets:
    print(name)
    print(group["counts"].sum(),group["FR_count"].sum())
    calibration_graph_counts[name] = group["counts"].sum()
    calibration_graph_ards_count[name] = group["FR_count"].sum()

    calibration_graph_ards_prevalence_observed[name] = round(group["FR_count"].sum()/group["counts"].sum(),2)*100
print(calibration_graph_counts,"\n\n",calibration_graph_ards_count,"\n\n",calibration_graph_ards_prevalence_observed)

import matplotlib.pyplot as plt

# Extracting the data
buckets = list(calibration_graph_counts.keys())
count_values = list(calibration_graph_counts.values())
prop_values = list(calibration_graph_ards_prevalence_observed.values())

# Create the figure and axis objects
fig, ax1 = plt.subplots(figsize=(10, 6)) 

# Plot the count as a bar graph
ax1.bar(buckets, count_values, color='lightblue', alpha=0.7)
ax1.set_xlabel('Probability deciles')
ax1.set_ylabel('Total ARDS predictions in each decile', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for the proportion
ax2 = ax1.twinx()
ax2.plot(buckets, prop_values, color='green', marker='o', linestyle='-', linewidth=2)
ax2.set_ylabel('Proportion of Actual ARDS patients among those predicted', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Title and display
plt.title('Count and Correctness Proportion by probability deciles OS2 30h _Vent +  EHR_CFS_400_50_True_W/o_BG')
# Rotate the x-axis labels and adjust spacing
plt.xticks(rotation=90, ha='right')

# Adjust spacing around the plot
plt.subplots_adjust(bottom=0.2)
plt.show()

plt.savefig(results_path+'calibration_double_graph.png', dpi=300)

import numpy as np
patient_probabilities= np.load("/data0/May12/ehrdata2/results/12h/all_features/May12_12h_vent_summary/Unbalanced_25_DS/patient_probabilities.npz",allow_pickle=True)#("/data0/ehrdata2/results/CFS_400_50_True/MEDIAN_ONLY_OS_48h_Ventilator_Waveform_summary_EHR_both_CFS_400_50_True/Unbalanced_25_DS/patient_probabilities.npz", allow_pickle=True)

import ast

import ast
import pandas as pd
import numpy as np
import os

# Load the patientsInTestSet.xlsx file
patientsInTestSet = pd.read_excel(os.path.join(project_folder, "testpatients", "patientsInTestSet.xlsx"))
patientsInTestSet['Test_patient_IDs'] = patientsInTestSet['Test_patient_IDs'].apply(ast.literal_eval)

all_rows = []  # List to collect all rows

for split_name in patient_probabilities.files:
    test_df = pd.read_csv(os.path.join("/data0/May12/ehrdata2/test/30h", split_name))
    patients_in_a_test_file = patient_probabilities[split_name].item()
    
    # Get the list of patient IDs used in holdout.py for this split
    lisst_ = patientsInTestSet[patientsInTestSet['Splits'] == split_name]['Test_patient_IDs'].values
    if len(lisst_) > 0:
        lisst_ = lisst_[0]
        # Filter test_df to include only these patients
        test_df = test_df[test_df[pt_col].isin(lisst_)].reset_index(drop=True)
    else:
        continue  # Skip if no patients found
    
    # Map predicted probabilities to ground truth labels
    for pt_id, probas in patients_in_a_test_file.items():
        if pt_id in test_df[pt_col].values:
            ground_truth_label = int(test_df[test_df[pt_col] == pt_id][label_col].values[0])
            new_values = {
                "pt_id": pt_id,
                "splitname": split_name,
                "ground_truth": ground_truth_label,
                "predicted_probability_class1": probas[1]
            }
            all_rows.append(new_values)
        else:
            continue
            print(f"Patient ID {pt_id} not found in test_df for split {split_name}")

# Create the DataFrame once, outside the loop
all_predictions_dataframe = pd.DataFrame(all_rows)

print("all predictions df",all_predictions_dataframe)

# Function to calculate metrics
def calculate_metrics(tp, tn, fp, fn):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    precision = ppv  # Precision is same as PPV
    recall = sensitivity  # Recall is same as Sensitivity
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return sensitivity, specificity, ppv, npv, precision, recall, accuracy


# Initialize a list to store results for each threshold
results = []

# Iterate over thresholds from 0 to 1 with a step of 0.1
for threshold in np.arange(0, 1.1, 0.1):
    tp = tn = fp = fn = 0
    # threshold=0.5
    # Iterate through each patient
    for index, row in all_predictions_dataframe.iterrows():
        predicted_prob = row['predicted_probability_class1']  # Probability for class 1
        predicted_class = 1 if predicted_prob >= threshold else 0
        true_class = row['ground_truth']
        # print(threshold,predicted_prob,predicted_class,true_class)
        
        if predicted_class == 1 and true_class == 1:
            tp += 1
        elif predicted_class == 1 and true_class == 0:
            fp += 1
        elif predicted_class == 0 and true_class == 0:
            tn += 1
        elif predicted_class == 0 and true_class == 1:
            fn += 1
    # print(tp, tn, fp, fn)
    # Calculate metrics for the current threshold
    sensitivity, specificity, ppv, npv, precision, recall, accuracy = calculate_metrics(tp, tn, fp, fn)
    
    # Append results for this threshold
    results.append({
        'Threshold': threshold,
        'Accuracy': accuracy,
        # 'Sensitivity': sensitivity,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        # 'PPV': ppv,
        'NPV': npv,
        
        
    })
    # break

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
print(results_df)
# exit()
results_df.to_excel(results_path+"new_calibration_confusion_matrix.xlsx")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Example function to calculate ROC points for each threshold
def calculate_roc_points(ground_truth, predicted_probs):
    fpr_list = []
    tpr_list = []
    thresholds = np.arange(0, 1.01, 0.01)  # Vary thresholds from 0 to 1 with step 0.01

    # Calculate FPR and TPR at each threshold
    for threshold in thresholds:
        predicted_labels = (predicted_probs >= threshold).astype(int)
        tp = np.sum((predicted_labels == 1) & (ground_truth == 1))
        fp = np.sum((predicted_labels == 1) & (ground_truth == 0))
        tn = np.sum((predicted_labels == 0) & (ground_truth == 0))
        fn = np.sum((predicted_labels == 0) & (ground_truth == 1))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds

# Sample function to simulate multiple predictions per patient
def pool_patient_predictions(predictions_df):
    # Group by patient and calculate mean probabilities
    pooled_df = predictions_df.groupby('pt_id').agg({
        'ground_truth': 'first',
        'predicted_probability_class1': ['mean', 'std']  # Mean and stddev of predictions
    }).reset_index()

    pooled_df.columns = ['pt_id', 'ground_truth', 'mean_predicted_probability', 'stddev_predicted_probability']
    return pooled_df

# Generate mock data (replace with your actual data)
# Assume we have a DataFrame 'all_predictions_dataframe' with multiple predictions per patient


# Pool the patient predictions (mean and stddev)
pooled_predictions_df = pool_patient_predictions(all_predictions_dataframe)

# Prepare lists to store the ROC points for each patient
fpr_list = []
tpr_list = []
for i, row in pooled_predictions_df.iterrows():
    ground_truth = row['ground_truth']
    predicted_probs = np.random.normal(row['mean_predicted_probability'], row['stddev_predicted_probability'], 100)  # Simulating multiple predictions

    # Calculate ROC points for this patient's pooled probabilities
    fpr, tpr, thresholds = calculate_roc_points(np.full_like(predicted_probs, ground_truth), predicted_probs)
    fpr_list.append(fpr)
    tpr_list.append(tpr)

# Convert lists to arrays for easier handling
fpr_array = np.array(fpr_list)
tpr_array = np.array(tpr_list)

# Calculate mean and stddev for the ROC points across patients
mean_fpr = np.mean(fpr_array, axis=0)
stddev_fpr = np.std(fpr_array, axis=0)
mean_tpr = np.mean(tpr_array, axis=0)
stddev_tpr = np.std(tpr_array, axis=0)

# Plot the ROC curve with mean and standard deviation as shaded area
plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, label=f"AUC: {np.trapz(mean_tpr, mean_fpr):.3f}")
plt.fill_between(mean_fpr, mean_tpr - stddev_tpr, mean_tpr + stddev_tpr, color='b', alpha=0.2)

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlabel('1 - Specificity (FPR)')
plt.ylabel('Sensitivity (TPR)')
plt.title('ROC of Validation Set')
plt.legend(loc="lower right")
# plt.show()/
plt.savefig(results_path+"roc_curve_with_confidence_interval.png", dpi=300)

# Clear the figure to avoid overwriting
plt.clf()
plt.close()

# .cla()

calibration_df_both  =pd.read_csv("/data0/May12/ehrdata2/results/30h/CFS_400_50_True/May12_30h_EHR_CFS_400_50_True_VWD_both_summary/Unbalanced_25_DS/calibration_CFS_400_50_True_test_set.csv")
results_path_both = "/data0/May12/ehrdata2/results/30h/CFS_400_50_True/May12_30h_EHR_CFS_400_50_True_VWD_both_summary/Unbalanced_25_DS/"
patient_probabilities_both= np.load("/data0/May12/ehrdata2/results/30h/CFS_400_50_True/May12_30h_EHR_CFS_400_50_True_VWD_both_summary/Unbalanced_25_DS/patient_probabilities.npz",allow_pickle=True)

calibration_df_vent = pd.read_csv("/data0/May12/ehrdata2/results/12h/all_features/May12_12h_vent_summary/Unbalanced_25_DS/calibration_all_feats_test_set.csv")
results_path_vent = "/data0/May12/ehrdata2/results/12h/all_features/May12_12h_vent_summary/Unbalanced_25_DS/"
patient_probabilities_vent= np.load("/data0/May12/ehrdata2/results/12h/all_features/May12_12h_vent_summary/Unbalanced_25_DS/patient_probabilities.npz",allow_pickle=True)

calibration_df_ehr = pd.read_csv("/data0/May12/ehrdata2/results/30h/CFS_400_50_True/May12_30h_EHR_CFS_400_50_True/Unbalanced_25_DS/calibration_CFS_400_50_True_test_set.csv")
results_path_ehr = "/data0/May12/ehrdata2/results/30h/CFS_400_50_True/May12_30h_EHR_CFS_400_50_True/Unbalanced_25_DS"
patient_probabilities_ehr= np.load("/data0/May12/ehrdata2/results/30h/CFS_400_50_True/May12_30h_EHR_CFS_400_50_True/Unbalanced_25_DS/patient_probabilities.npz",allow_pickle=True)

calibration_df_u = [calibration_df_both,calibration_df_vent,calibration_df_ehr]
results_path_u  = [results_path_both,results_path_vent,results_path_ehr]
patient_probabilities_u = [patient_probabilities_both,patient_probabilities_vent,patient_probabilities_ehr]

import numpy as np

import matplotlib.pyplot as plt
# Create the figure and axis objects
fig, ax1 = plt.subplots(figsize=(10, 6)) 

# Plot the count as a bar graph
# ax1.bar(buckets, count_values, color='lightblue', alpha=0.7)
ax1.set_xlabel('Probability deciles')
ax1.set_ylabel('Proportion of Correct ARDS patients among those predicted', color='black')

#("/data0/ehrdata2/results/CFS_400_50_True/MEDIAN_ONLY_OS_48h_Ventilator_Waveform_summary_EHR_both_CFS_400_50_True/Unbalanced_25_DS/patient_probabilities.npz", allow_pickle=True)
colors = ["green","red","black"]
# ["VWD +6hr", "EHR -24 to +6hr", "VWD + EHR -24 to +6 hr "]
labels = ["VWD + EHR -24 to +6 hr ","VWD +6 hr","EHR -24 to +6 hr"]
for idx,(calibration_df,results_path,patient_probabilities) in enumerate(zip(calibration_df_u,results_path_u,patient_probabilities_u)):

    print(set(calibration_df["buckets"]))
    indbuckets = calibration_df.groupby("buckets")
    calibration_graph_counts= {}
    calibration_graph_ards_count = {}
    calibration_graph_ards_prevalence_observed ={}
    for name, group in indbuckets:
        print(name)
        print(group["counts"].sum(),group["FR_count"].sum())
        calibration_graph_counts[name] = group["counts"].sum()
        calibration_graph_ards_count[name] = group["FR_count"].sum()

        calibration_graph_ards_prevalence_observed[name] = round(group["FR_count"].sum()/group["counts"].sum(),2)*100
    print(calibration_graph_counts,"\n\n",calibration_graph_ards_count,"\n\n",calibration_graph_ards_prevalence_observed)

    # Extracting the data
    buckets = list(calibration_graph_counts.keys())
    count_values = list(calibration_graph_counts.values())
    prop_values = list(calibration_graph_ards_prevalence_observed.values())


    ax1.plot(buckets, prop_values, color=colors[idx], marker='o', linestyle='-', linewidth=2,label=labels[idx])



# ax1.tick_params(axis='y', labelcolor='green')

plt.legend()
# Title and display
plt.title('Proportion of Correct ARDS Predictions by probability deciles')
# Rotate the x-axis labels and adjust spacing
plt.xticks(rotation=90, ha='right')

# Adjust spacing around the plot
plt.subplots_adjust(bottom=0.2)
plt.show()

plt.savefig("/data0/May12/ehrdata2/results/30h/"+'calibration_30h_calibration_graph.png', dpi=300)