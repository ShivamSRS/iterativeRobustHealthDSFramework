
import numpy as np
import pandas as pd
import os
import random
import warnings
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")

# For reproducibility
np.random.seed(42)
random.seed(42)

# ------------------------------------------------------------------------------
# You may already have these or similar variables in your environment:
# ------------------------------------------------------------------------------
from arguments import test_folder,pt_col ,label_col # <- Adjust if needed
# pt_col = "patient_id"                       # <- Adjust if your ID column is different
# label_col = "ARDS_label"                    # <- Adjust if your label column is different

# ------------------------------------------------------------------------------
# 1. Helper function to collect fold-level ROC data and AUC from an npz file
# ------------------------------------------------------------------------------
def get_fold_roc_data(probabilities_file):
    """
    Loads the patient_probabilities.npz from a model, iterates over each fold/split,
    matches predictions with ground truth in the corresponding test CSV, then
    computes the fold-level ROC (FPR, TPR) and AUC. Returns a list of (fpr, tpr, auc)
    for each fold.
    """
    patient_probabilities = np.load(probabilities_file, allow_pickle=True)
    
    fold_roc_data = []
    
    # Each key in the .npz is typically the name of the test CSV file for that fold
    for split_name in patient_probabilities.files:
        # This is a dict of {pt_id_str: [prob_0, prob_1]} for all patients in that fold
        patients_pred_dict = patient_probabilities[split_name].item()
        
        # Read the test CSV for this fold:
        test_csv_path = os.path.join(test_folder, split_name)
        if not os.path.exists(test_csv_path):
            print(f"Warning: Test file not found for fold={split_name}, skipping.")
            continue
        
        test_df = pd.read_csv(test_csv_path)
        
        y_true_fold = []
        y_prob_fold = []
        
        # Match each patient’s predicted probability to ground truth in test_df
        for pt_id_str, probas in patients_pred_dict.items():
            pt_id = int(pt_id_str)
            row = test_df[test_df[pt_col] == pt_id]
            print(pt_id,row)
            if row.empty:
                continue
            gt = row[label_col].values[0]
            if pd.isna(gt):
                continue
            
            # Ground truth
            y_true_fold.append(int(gt))
            # Probability of the positive class
            y_prob_fold.append(probas[1])  
        
        if len(y_true_fold) == 0:
            print(f"No matching patients or labels for fold={split_name}, skipping.")
            continue
        
        # Compute ROC for this fold
        fpr_fold, tpr_fold, _ = roc_curve(y_true_fold, y_prob_fold)
        auc_fold = auc(fpr_fold, tpr_fold)
        
        fold_roc_data.append((fpr_fold, tpr_fold, auc_fold))
    
    return fold_roc_data

# ------------------------------------------------------------------------------
# 2. Aggregate the fold-level ROC data onto a common FPR grid and compute 95% CI
# ------------------------------------------------------------------------------
def aggregate_roc_across_folds(fold_roc_data):
    """
    Given a list of (fpr, tpr, auc) for each fold, interpolate TPR on a common FPR grid,
    compute mean TPR and standard error at each point. Also compute mean AUC and 95% CI.
    Returns:
        mean_fpr, mean_tpr, lower_tpr, upper_tpr, mean_auc, lower_auc, upper_auc
    """
    # Number of folds
    n_folds = len(fold_roc_data)
    if n_folds == 0:
        print("No fold ROC data found!")
        return None
    
    # We will use a common FPR grid from 0 to 1 in 100 steps
    mean_fpr = np.linspace(0, 1, 100)
    
    # Store each fold's tpr (interpolated) and the AUC
    tprs_interpolated = []
    aucs = []
    
    for (fpr_fold, tpr_fold, auc_fold) in fold_roc_data:
        aucs.append(auc_fold)
        # Interpolate TPR onto mean_fpr
        tpr_interp = np.interp(mean_fpr, fpr_fold, tpr_fold)
        tprs_interpolated.append(tpr_interp)
    
    # Convert to array for easier math
    tprs_interpolated = np.array(tprs_interpolated)  # shape = (n_folds, len(mean_fpr))
    
    # Compute mean TPR and standard error across folds
    mean_tpr = tprs_interpolated.mean(axis=0)
    std_tpr  = tprs_interpolated.std(axis=0, ddof=1)  # sample std
    # Standard Error of the Mean (SEM)
    sem_tpr = std_tpr / np.sqrt(n_folds)
    
    # 95% CI for TPR: mean_tpr ± 1.96 * sem
    lower_tpr = mean_tpr - 1.96 * sem_tpr
    upper_tpr = mean_tpr + 1.96 * sem_tpr
    
    # Clip at 0,1 just in case
    lower_tpr = np.clip(lower_tpr, 0, 1)
    upper_tpr = np.clip(upper_tpr, 0, 1)
    
    # Now compute mean AUC and 95% CI for AUC
    aucs = np.array(aucs)
    mean_auc = aucs.mean()
    std_auc  = aucs.std(ddof=1)
    sem_auc  = std_auc / np.sqrt(n_folds)
    
    ci_95 = 1.96 * sem_auc
    lower_auc = mean_auc - ci_95
    upper_auc = mean_auc + ci_95
    
    return mean_fpr, mean_tpr, lower_tpr, upper_tpr, mean_auc, lower_auc, upper_auc

# ------------------------------------------------------------------------------
# 3. Get the ROC data for each of the three models and plot
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # ------------------------------------------------------------------
    # Adjust these three file paths to your actual .npz predictions:
    # ------------------------------------------------------------------
    vwd_only_file    = "/data0/May12/ehrdata2/results/12h/all_features/May12_12h_vent_summary/Unbalanced_25_DS/patient_probabilities.npz"
    ehr_only_file    = "/data0/May12/ehrdata2/results/30h/CFS_400_50_True/May12_30h_EHR_CFS_400_50_True/Unbalanced_25_DS/patient_probabilities.npz"
    vwd_ehr_file     = "/data0/May12/ehrdata2/results/30h/CFS_400_50_True/May12_30h_EHR_CFS_400_50_True_VWD_both_summary/Unbalanced_25_DS/patient_probabilities.npz"
    
    # Labels for the legend
    model_labels = ["VWD +6hr", "EHR -24 to +6hr", "VWD + EHR -24 to +6 hr "]
    model_files  = [vwd_only_file, ehr_only_file, vwd_ehr_file]
    colors       = ["blue", "green", "red"]
    
    plt.figure(figsize=(8, 8))
    
    for (file_path, label, color) in zip(model_files, model_labels, colors):
        fold_roc_data = get_fold_roc_data(file_path)
        if len(fold_roc_data) == 0:
            print(f"No folds found for: {label}. Skipping plot.")
            continue
        
        agg_res = aggregate_roc_across_folds(fold_roc_data)
        if agg_res is None:
            continue
        
        mean_fpr, mean_tpr, lower_tpr, upper_tpr, mean_auc, lower_auc, upper_auc = agg_res
        
        print(label,":",mean_auc)
        # Plot the mean ROC curve
        plt.plot(
            mean_fpr, 
            mean_tpr, 
            color=color, 
            label=f"{label} (AUC = {mean_auc:.3f} [{lower_auc:.3f}, {upper_auc:.3f}])", 
            lw=2
        )
        # Plot the confidence band
        plt.fill_between(
            mean_fpr, 
            lower_tpr, 
            upper_tpr, 
            color=color, 
            alpha=0.2
        )
    
    # Diagonal line for reference
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    
    plt.title("Combined AUC-ROC with 95% CI\n(VWD Only, EHR Only, VWD+EHR)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save the figure if you like:
    plt.savefig("three_models_roc_with_ci.png", dpi=300)
    
    # plt.show()


exit()
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import warnings

warnings.filterwarnings("ignore")

# (The following imports and definitions are kept from your original code.)
from dataselectutils import get_dataset,get_test_dataset
from arguments import time_window, data_files, test_folder, train_folder, project_folder, data_folder, label_col, pt_col
from configs import Unbalanced, Downsample_25, feature_selection_method, feature_import_path, algorithm, use_features, prefered_columns, use_prefered_cols, data_setting
from VentWaveData import VentData

# For reproducibility
random_state = np.random.RandomState(42)

# (Optional) Calibration graph code remains unchanged
calibration_df = pd.read_csv("/data0/NEW_OS/ehrdata2/results/30h/CFS_400_50_True/OS2_30h_EHR_CFS_400_50_True/Unbalanced_25_DS/calibration_CFS_400_50_True_test_set.csv")
results_path = "/data0/NEW_OS/ehrdata2/results/30h/CFS_400_50_True/OS2_30h_EHR_CFS_400_50_True/Unbalanced_25_DS/"
print(set(calibration_df["buckets"]))
indbuckets = calibration_df.groupby("buckets")
calibration_graph_counts= {}
calibration_graph_ards_count = {}
calibration_graph_ards_prevalence_observed = {}

for name, group in indbuckets:
    print(name)
    print(group["counts"].sum(), group["FR_count"].sum())
    calibration_graph_counts[name] = group["counts"].sum()
    calibration_graph_ards_count[name] = group["FR_count"].sum()
    calibration_graph_ards_prevalence_observed[name] = round(group["FR_count"].sum() / group["counts"].sum(), 2) * 100

print(calibration_graph_counts, "\n\n", calibration_graph_ards_count, "\n\n", calibration_graph_ards_prevalence_observed)

# Plot calibration graph (unchanged)
buckets = list(calibration_graph_counts.keys())
count_values = list(calibration_graph_counts.values())
prop_values = list(calibration_graph_ards_prevalence_observed.values())

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(buckets, count_values, color='lightblue', alpha=0.7)
ax1.set_xlabel('Probability deciles')
ax1.set_ylabel('Total ARDS predictions in each decile', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(buckets, prop_values, color='green', marker='o', linestyle='-', linewidth=2)
ax2.set_ylabel('Proportion of Actual ARDS patients among those predicted', color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('Count and Correctness Proportion by probability deciles OS2 30h _Vent +  EHR_CFS_400_50_True_W/o_BG')
plt.xticks(rotation=90, ha='right')
plt.subplots_adjust(bottom=0.2)
plt.show()
# plt.savefig(results_path + 'calibration_double_graph.png')


# ---------------------------------------------------------------------
# NEW CODE: No patient-level aggregation; every ground truth/prediction is a sample.
# ---------------------------------------------------------------------

# Load the patient probabilities saved from different splits
patient_probabilities = np.load(
    "/data0/NEW_OS/ehrdata2/results/30h/CFS_400_50_True/OS2_30h_MEDIANOnly_VentVWD_and_EHR_CFS_400_50_True/Unbalanced_25_DS/patient_probabilities.npz", 
    allow_pickle=True
)

# Initialize lists to hold all ground truth labels and predicted probabilities
all_ground_truths = []
all_pred_probs = []

# Loop over each split file (each key in the npz file)
for split_name in patient_probabilities.files:
    # Each file contains a dictionary mapping pt_id to the model's predicted probability vector.
    patients_in_a_test_file = patient_probabilities[split_name].item()
    
    # Read the corresponding test file; assume that the test file contains columns for patient id and the ground truth label.
    test_df = pd.read_csv(os.path.join(test_folder, split_name))
    
    # Loop over each patient in this split.
    for pt_id_str, probas in patients_in_a_test_file.items():
        # Convert the pt_id to integer (if needed)
        pt_id = int(pt_id_str)
        
        # Retrieve the ground truth for this patient from the test_df.
        # Assumes that the column name for patient id is stored in the variable `pt_col`
        # and the label column name is in `label_col`
        row = test_df[test_df[pt_col] == pt_id]
        if row.empty:
            # If no matching row is found, skip this patient.
            continue
        gt = row[label_col].values[0]
        
        # Skip if ground truth is missing
        if pd.isna(gt):
            continue
        
        # Append the ground truth value (ensure it is an integer 0 or 1)
        all_ground_truths.append(int(gt))
        
        # Instead of using ground truth as an index, we now simply choose the predicted probability for the positive class.
        # (If you want to reproduce your earlier behavior of using probas[int(gt)], use that instead.)
        pred_prob = probas[1]
        all_pred_probs.append(pred_prob)

# Check that we have some samples
print("Number of samples:", len(all_ground_truths))

# Compute the ROC curve and AUC score using the flattened lists
fpr, tpr, thresholds = roc_curve(all_ground_truths, all_pred_probs)
roc_auc = auc(fpr, tpr)
print("AUC =", roc_auc)

# Plot the ROC Curve
plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random classifier
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("AUC-ROC Curve (No Patient-level Aggregation)")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("finalaucroc2.png")
plt.show()

exit()