from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
import imblearn

import random
random_state = np.random.RandomState(42)
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

from VentWaveData import VentData


import warnings
warnings.filterwarnings("ignore")



from dataselectutils import get_dataset,statistical_filter,mutual_info, RFE_features,permutation_importance_features
from arguments import time_window,data_files,test_folder,train_folder,project_folder,data_folder,label_col,pt_col,ventDataFiles_median,ventDataFolder


repeat_flag = 'Y'

data_files = data_files

project_folder = project_folder
data_folder = data_folder
train_folder = train_folder
test_folder = test_folder
label_col,pt_col = label_col,pt_col

hyperparameter_catalog = {
    
    'RF': {
        'bootstrap': [True],
        'max_depth': [2, 5, 10], # maximum depth of the tree
        'max_features': ['log2','sqrt'], # maximum number of features to use at each split
        'min_samples_leaf': [5,10], # minimum number of samples to split a node
        'min_samples_split': range(2,10,2),
        'n_estimators': [100,200, 500], # number of trees
        'criterion' : ['gini','entropy']  # criterion for evaluating a split
    },
    # 'XGB': {
    #     'learning_rate': [0.2, 0.4, 0.6, 0.8],
    #     'n_estimators': [100,200, 500],
    #     'max_depth': [2, 5, 10],
    #     'max_features': ['auto','sqrt'],
    #     'min_samples_leaf': [5,10],
    #     'min_samples_split': range(2,10,2)
        
    # },
    # 'SVM': {
    #     'C': [0.5, 1, 1.5],
    #     'kernel': ['poly', 'sigmoid'],
    #     'degree': [3,4],
    #     'gamma': ['scale', 'auto']
        
    # },
    # 'LR': {
    #     'penalty': ['l1', 'l2', 'elasticnet'],
    #     'solver': ['lbfgs', 'liblinear'] 
        
    # }
}

from scipy.stats import ks_2samp



rp_list = [['n','n'], ['y', 'n'], ['n', 'y']]


# data_folder = 'bootstraps_sv2/'

filtered_col_list = []
fold_perf = []
only_comm = 'n'
comm_feats_90 = ['median_dia_area',
 'median_pulse_pres',
 'median_sys_dec_area',
 'median_t_dia',
 'median_t_sys',
 'std_avg_dia',
 'std_dias_pres',
 'std_dic_pres',
 'std_pp_area']
comm_feats_50 = ['std_dias_pres',
 'std_sys_dec_area_nor',
 'std_sys_pres',
 'std_sys_area_nor',
 'std_avg_dia',
 'std_avg_sys',
 'std_sys_rise_area_nor',
 'std_avg_sys_rise',
 'median_sys_dec_area',
 'std_dic_pres',
 'std_pp_area',
 'std_pp_area_nor']





from configs import num_splits,data_setting
num_files = num_splits
train_or_test = 'train/'



#### Select Algorithm Selection Method ####
# set selection_method to: 
# 1 for random forest, 2 for XGBoost,
# 3 Logistic Regression, 4 for Support Vector Machine

algorithm_catalog = {
    'RF': 1,
    'XGB': 2,
    'LR': 3,
    'SVM': 4
}
from configs import algorithm,use_features


import argparse



algorithm = algorithm
algorithm_no = algorithm_catalog[algorithm]
hyperparameter_grid = hyperparameter_catalog[algorithm]



file_list = [f for f in listdir(train_folder) if isfile(join(train_folder, f))]
# print(file_list,train_folder)
# exit()
filtered_col_list = []
fold_perf = []
import argparse
parser = argparse.ArgumentParser()

# parser.add_argument("--use_features",default='N')
# parser.add_argument("--featselection",default='SFS')
# args = parser.parse_args()

from configs import feature_selection_method,feature_import_path,use_prefered_cols,prefered_columns,expt_name

import_feature_list = use_features # 'Y' to use saved features from feature selection code
if import_feature_list == 'Y':
    #both should be user input
    feature_selection_method = feature_selection_method
    feature_import_path = feature_import_path #'pickled_features/{}/{}_top{}_features.pkl'.format(feature_selection_method,feature_selection_method,suffix_str)

resultdfcols = ["params","mean_fit_time","rank_test_roc_auc","mean_test_roc_auc","split0_test_roc_auc","split1_test_roc_auc","split2_test_roc_auc","split3_test_roc_auc","split4_test_roc_auc","rank_test_accuracy","mean_test_accuracy","rank_test_prc_auc","mean_test_prc_auc","rank_test_precision","mean_test_precision","rank_test_recall","mean_test_recall","rank_test_specificity","mean_test_specificity","rank_test_brier_score","mean_test_brier_score"]
classification_metrics = ["mean_test_roc_auc","mean_test_accuracy","mean_test_prc_auc","mean_test_precision","mean_test_recall","mean_test_specificity","mean_test_brier_score"]
# trainsplit_metrics = [m[:5] + 'train'+m[10:] for m in classification_metrics]
# "mean_train_roc_auc","split1_train_roc_auc","split2_train_roc_auc","split3_train_roc_auc","split4_train_roc_auc",
classification_metrics_ci = [m[5:] + ' 95% CI' for m in classification_metrics]

forcols = pd.read_excel(project_folder+"/fullbestmodels_cv.xlsx")
fullmodels_cv = pd.DataFrame(columns = forcols.columns)
bestmodels_cv = pd.DataFrame(columns =["Filename"]+resultdfcols+classification_metrics_ci)

# if args.featselection!='SFS':
#     feature_selection_method = 'RFE'
# else:
#     feature_selection_method = "statistical_feature_selection"
print(feature_selection_method)
for file_num in range(num_files):
    print('')
    # print(file_num)
    # if file_num==4:
    #     break

    start_time = time.time()
    print("ssss",file_list,file_num)
    
    if import_feature_list == 'N':
        All_file_pickle_folder = project_folder +  'algorithm_selection/' + expt_name + "/" + \
                                 algorithm + '/all_features' + '/model_'+file_list[file_num][:-4] + '/'
    else:
        All_file_pickle_folder = project_folder +  '/algorithm_selection/' + expt_name + "/" + \
                                 algorithm + '/' + feature_selection_method + \
                                 '/model_' + file_list[file_num][:-4] + '/'
    

    # print()
    # exit()
    data_file = file_list[file_num]
    print('Processing file ' + data_file)
    if not os.path.isdir(All_file_pickle_folder):
        print(All_file_pickle_folder)
        os.makedirs(All_file_pickle_folder)
    
    
    from sklearn.metrics import brier_score_loss
    param_grid = hyperparameter_grid
    hyperparameters = {'classification_model__' + key: param_grid[key] for key in param_grid}
    def roc_auc_brier_score(y_true, y_pred):
        return roc_auc_score(y_true, y_pred)/brier_score_loss(y_true,y_pred)

    import numpy as np
    from sklearn.metrics import roc_auc_score

    def make_patient_level_roc_auc_scorer(patient_ids):
        """
        Creates a custom scorer that computes ROC AUC on the patient level based on majority vote.
        
        Args:
        patient_ids (array-like): Array of patient IDs corresponding to each row in the input data.
        
        Returns:
        scorer (callable): Custom scorer compatible with scikit-learn's make_scorer function.
        """
        def patient_level_roc_auc(y_true, y_pred):
            # Convert predictions to binary labels based on a threshold (e.g., 0.5 for probabilities)
            print(y_pred,"inside roc_auc y true y pred",y_pred.shape,patient_ids)
            
            y_pred_labels = (y_pred >= 0.5).astype(int)
            
            # Aggregate predictions by patient ID, finding the majority label for each patient
            patient_predictions = {}
            for patient_id, prediction in zip(patient_ids, y_pred_labels):
                print(patient_id, prediction)
                if patient_id not in patient_predictions:
                    patient_predictions[patient_id] = []
                patient_predictions[patient_id].append(prediction)
            
            patient_majority_label = {patient_id: np.argmax(np.bincount(labels))
                                    for patient_id, labels in patient_predictions.items()}
            
            # Aggregate true labels by patient ID, assuming the true label is consistent for all rows of a patient
            patient_true_label = {}
            # exit()
            for patient_id, true_label in zip(patient_ids, y_true):
                patient_true_label[patient_id] = true_label
            
            # Prepare lists of aggregated true labels and predicted labels for ROC AUC calculation
            y_true_agg = [patient_true_label[pid] for pid in patient_predictions.keys()]
            y_pred_agg = [patient_majority_label[pid] for pid in patient_predictions.keys()]
            
            # Calculate and return the ROC AUC score
            return roc_auc_score(y_true_agg, y_pred_agg)
        
        return patient_level_roc_auc

    
    print("This is the modality setting",data_setting)
        
    if data_setting=='ehr':
        
        X,y,df_dataset, cv,train_patient_ids,test_patient_ids = get_dataset(os.path.join(project_folder,train_or_test,time_window,data_file),file_num,label_col,pt_col)
        
        print(test_patient_ids)
        count_ones = y.value_counts()
        # count_zeros = y.count(0)

        print("Number of 1s:", count_ones)
        # exit()
        print(patient_ids,set(patient_ids))
        scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
               'specificity': make_scorer(recall_score,pos_label=0),\
               'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True),'brier_score':make_scorer(brier_score_loss,needs_proba=True)}
    elif data_setting=='vent':
        obj = VentData(ventDataFolder)
        
        X,y,df_dataset, cv,train_pigs =obj.get_train_test_file(data_file,int(data_file[-data_file[::-1].find("_"):data_file.find(".")]),ventDataFiles_median,time_window)
        patient_ids = df_dataset[pt_col].values
        count_ones = y.value_counts()
        # count_zeros = y.count(0)
        print(df_dataset[pt_col].value_counts(),len(df_dataset))
        print("Number of 1s:", count_ones)
        # exit()

        # print("patient ids",patient_ids,sep="#####$$$$$#####")
        #'patient_level_roc_auc_scorer' : make_scorer(make_patient_level_roc_auc_scorer(patient_ids), needs_proba=True),
        scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
               'specificity': make_scorer(recall_score,pos_label=0),\
               'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True),'brier_score':make_scorer(brier_score_loss,needs_proba=True)}
    elif data_setting=='oversample_both':
        obj = VentData(ventDataFolder)
        X,y,df_dataset, cv =obj.get_oversample_vent_ehr_train_test_file(data_file,int(data_file[-data_file[::-1].find("_"):data_file.find(".")]),ventDataFiles_median,time_window)
        scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
               'specificity': make_scorer(recall_score,pos_label=0),\
               'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True),'brier_score':make_scorer(brier_score_loss,needs_proba=True)}
        print(cv)
        # exit()   
    elif data_setting=='vent_summary':
        
        obj = VentData(ventDataFolder)
        
        X,y,df_dataset, cv,train_pigs =obj.get_train_test_file_summary(data_file,int(data_file[-data_file[::-1].find("_"):data_file.find(".")]),ventDataFiles_median,time_window,median_only=True)
        patient_ids = df_dataset[pt_col].values
        count_ones = y.value_counts()
        # count_zeros = y.count(0)
        print(X.columns)
        print("Number of 1s:", count_ones)
        print(len(y),len(X))
        print(X.head())
        # X.to_csv("48h_summary_vent_features_{}.csv".format(int(data_file[-data_file[::-1].find("_"):data_file.find(".")])))
        # exit()

        # print("patient ids",patient_ids,sep="#####$$$$$#####")
        #'patient_level_roc_auc_scorer' : make_scorer(make_patient_level_roc_auc_scorer(patient_ids), needs_proba=True),
        scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
               'specificity': make_scorer(recall_score,pos_label=0),\
               'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True),'brier_score':make_scorer(brier_score_loss,needs_proba=True)}

    elif data_setting=='both_summary':
        obj = VentData(ventDataFolder)
        
        X,y,df_dataset, cv,train_patient_ids,test_patient_ids  = get_dataset(os.path.join(project_folder,train_or_test,time_window,data_file),file_num,label_col,pt_col,give_pt=True)
        # print(X.columns)
        
        # print(int(data_file[-data_file[::-1].find("_"):data_file.find(".")]))
        ventX,venty,ventdf_dataset, cv,train_pigs  =obj.get_train_test_file_summary(data_file,int(data_file[-data_file[::-1].find("_"):data_file.find(".")]),ventDataFiles_median,time_window,give_pt=True,median_only=True)
        # print(ehrX.columns,ventX.columns)
        # print(data_file,int(data_file[-data_file[::-1].find("_"):data_file.find(".")]),os.path.join(project_folder,train_or_test,time_window,data_file))
        print(int(data_file[-data_file[::-1].find("_"):data_file.find(".")]))
        print("\n\n\n\n #######cv",cv)
        # exit() 
        patient_ids = ventdf_dataset[pt_col].values
        print("patient ids",patient_ids,sep="#####$$$$$#####")
        # Counting 1s and 0s
        count_ones = venty.value_counts()
        # count_zeros = y.count(0)

        print("Number of 1s:", count_ones)
        count_ones = y.value_counts()
        # count_zeros = y.count(0)

        print("Number of EHR 1s:", count_ones)
        # print("Number of 0s:", count_zeros)
        # exit()
        scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
               'specificity': make_scorer(recall_score,pos_label=0),\
               'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True),'brier_score':make_scorer(brier_score_loss,needs_proba=True)}
        # print(X)
        y=venty

        print(X.columns,len(X),len(y))
    
    elif data_setting=='oversample_both_summary':
        obj = VentData(ventDataFolder)
        print("now enteting ehr only")
        X,y,df_dataset, cv,train_patient_ids,test_patient_ids  = get_dataset(os.path.join(project_folder,train_or_test,time_window,data_file),file_num,label_col,pt_col,give_pt=True)
        
        df_dataset.sort_values(by=pt_col, inplace=True)
        df_dataset.reset_index(drop=True, inplace=True)
        X,y = df_dataset,df_dataset[label_col] 
        if 'Unnamed: 0' in X.columns.tolist():
            X = X.drop(['Unnamed: 0'], axis =1)
        if 'Unnamed: 0.1' in X.columns.tolist():
            X = X.drop(['Unnamed: 0.1'], axis =1)
        if label_col in X.columns.tolist():
            X = X.drop([label_col], axis =1)

        print("finish ehr")

        ventX,venty,ventdf_dataset, cv,train_pigs  =obj.get_train_test_file_summary(data_file,int(data_file[-data_file[::-1].find("_"):data_file.find(".")]),ventDataFiles_median,time_window,give_pt=True,median_only=True)
        
        print("oversam")
        
        ventdf_dataset.sort_values(by=pt_col, inplace=True)
        ventdf_dataset.reset_index(drop=True, inplace=True)
        columns_to_include = [col for col in ventdf_dataset.columns if col not in [label_col]]
        ventX,venty = ventdf_dataset.loc[:,columns_to_include],ventdf_dataset[label_col] 

        print(y.value_counts(),venty.value_counts(),(y-venty).value_counts())
        df_dataset.to_excel("ehrCHEDCK.xlsx")
        ventdf_dataset.to_excel("ventCHECK.xlsx")

        # exit()
        scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
               'specificity': make_scorer(recall_score,pos_label=0),\
               'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True),'brier_score':make_scorer(brier_score_loss,needs_proba=True)}
        print(cv)

    else:
        obj = VentData(ventDataFolder)
        
        X,y,df_dataset, cv,train_patient_ids,test_patient_ids  = get_dataset(os.path.join(project_folder,train_or_test,time_window,data_file),file_num,label_col,pt_col,give_pt=True)
        print(int(data_file[-data_file[::-1].find("_"):data_file.find(".")]))
        ventX,venty,ventdf_dataset, cv,train_pigs  =obj.get_train_test_file(data_file,int(data_file[-data_file[::-1].find("_"):data_file.find(".")]),ventDataFiles_median,time_window,give_pt=True)
        # print(ehrX.columns,ventX.columns)
        print("cv",cv,len(X))
        # exit()
        patient_ids = ventdf_dataset[pt_col].values
        print("patient ids",patient_ids,sep="#####$$$$$#####")
        # Counting 1s and 0s
        count_ones = venty.value_counts()
        # count_zeros = y.count(0)

        print("Number of 1s:", count_ones)
        count_ones = y.value_counts()
        # count_zeros = y.count(0)

        print("Number of EHR 1s:", count_ones)
        print("Number o:", len(ventX))
        # print("Number of 0s:", count_zeros)
        # exit()
        scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
               'specificity': make_scorer(recall_score,pos_label=0),\
               'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True),'brier_score':make_scorer(brier_score_loss,needs_proba=True)}
        # print(X)
        y=venty

        print(X.columns,len(X),len(y))
    # print(y.shape)
    # exit()
    print()
    

    print()
    if data_setting=='ehr' or data_setting=='both' or data_setting=='both_summary' or data_setting=="oversample_both_summary":
        print("inside inmport featu")
        if import_feature_list == 'Y':

            if os.path.exists(feature_import_path):
                feature_dict = joblib.load(feature_import_path)
                try:
                    selected_features = feature_dict[data_file[:-4]]
                    if selected_features==[]:
                        print("no feature was selected, passing whole data instead")
                        selected_features = X.columns
                except:
                    print("probably key error passing all features instead")
                    selected_features = X.columns
            
                
            else:
                print("cant load the feature selection path")
            
            if use_prefered_cols:
                selected_features = prefered_columns
            print("selected features are ",len(selected_features),selected_features)
            
            if data_setting=="both" or data_setting=='both_summary' or data_setting=="oversample_both_summary":
                
                if pt_col not in selected_features:
                    selected_features.append(pt_col)
                print(X.columns,selected_features)
                # exit()
                X = X[selected_features]#[list(X.columns[:51]) + list(selected_features)]
                # print(X.columns,ventX.columns)
                # print(X,"VentX",ventX,sep="\n\n")
                X = pd.merge(ventX, X, on=pt_col, how='left')
                # print("after",X,sep="\n\n")
                # exit()
                print("after merging ehr and vent data",X.shape,y,venty,y.value_counts(),venty.value_counts(),(y-venty).value_counts())
                # exit()
                
                print(y.shape,"sjsjmdaokp",X.shape)
                # print(X.columns,y,(venty-y).value_counts())
                # exit()

                if data_setting=="oversample_both_summary":
                    
                    
                    train_patients = train_patient_ids
                    train_dataset = pd.concat([X,y.rename(label_col)],axis=1)
                    
                    print("These are columns",train_dataset.columns,train_dataset.shape)
                    # exit()
                    x_new=[]
                    y_new=[]
                    for fold_idx,fold in enumerate(train_patients):

                        CV_train_pts_list = [f for f_idx,f in enumerate(train_patients) if f_idx!=fold_idx]
                        # print("CV",CV_train_pts_list,len(CV_train_pts_list),fold,sep="\n####")
                        # exit()
                        # break
                        train_pigs = []
                        for l in range(len(CV_train_pts_list[0])):
                            # print(CV_train_pts_list[l][0])
                            # exit()
                            train_pigs+=CV_train_pts_list[0][l]

                        # print("train pigs",sorted(train_pigs))
                        
                        # print("\n\n\n ",train_dataset[pt_col].astype(int).isin(sorted(fold[0])))
                        # exit()
                        train_pt_data =train_dataset[train_dataset[pt_col].isin(train_pigs)]
                        test_pt_data =  train_dataset[train_dataset[pt_col].isin(fold[0])]
                        ########################
                        #for holdout set make sure u concatenate the patient IDs after this
                        ######################

                        # print(train_pt_data.shape,test_pt_data.shape)
                        sm = SMOTE(sampling_strategy={0:3*len(train_pt_data[train_pt_data[label_col]==1]),1:len(train_pt_data[train_pt_data[label_col]==1])})
                        # nan_mask = np.any(train_pt_data[feature_cols].isna(), axis=1)
                        nan_mask = np.any(train_pt_data.isna(), axis=1)
                        # print(train_pt_data.isna().sum())
                        selcted_cols = train_dataset.columns.difference(pd.Index([pt_col,label_col]))
                        # print(train_pt_data)
                        print(selcted_cols)
                        # exit()
                        fold_x_res, fold_y_res = sm.fit_resample(train_pt_data[~nan_mask][selcted_cols], train_pt_data[~nan_mask][label_col])
                        # set non feature cols to nan because theres no actual reference to real
                        # world values with synthetic data
                        # fold_x_res[non_feature_cols] = np.nan
                        print(len(fold_x_res),len(train_pt_data[train_pt_data[label_col]==1]))
                        # exit()
                        fold_y_res = fold_y_res.rename(label_col)
                        print(fold_x_res.isna().sum(),"\n\n",fold_y_res.isna().sum(),type(fold_y_res),fold_y_res)
                        
                        # exit()
                        
                        
                        y_new.extend([fold_y_res, test_pt_data[label_col]])
                        
                        
                        x_new.extend([fold_x_res, test_pt_data])
                        print(fold_idx)
                    print(len(y_new),len(x_new))
                    # exit()
                    print("INSIDNE ED")
                    # exit()
                    # print(x_new,y_new)
                    # exit()
                    idxs=[]
                    cur_idx = 0
                    for i in range(0, len(x_new), 2):
                        x_tr_idx = pd.Index(range(cur_idx, cur_idx+len(x_new[i])))
                        cur_idx += len(x_new[i])
                        x_tst_idx = pd.Index(range(cur_idx, cur_idx+len(x_new[i+1])))
                        cur_idx += len(x_new[i+1])
                        idxs.append((x_tr_idx, x_tst_idx))
                    combined_dfx_list = []
                    combined_dfy_list = []
                    cv =idxs
                    
                    for ctr,(fold_x,fold_y) in enumerate(zip(x_new,y_new)):
                        
                        print(ctr)
                        combined_dfx_list.append(fold_x)
                        
                        if ctr%2==1:
                            combined_dfy_list.append(fold_y.rename(label_col))
                        else:
                            combined_dfy_list.append(fold_y)
                    
                        

                    X, y = pd.concat(combined_dfx_list, ignore_index=True), pd.concat(combined_dfy_list, ignore_index=True)
                    # print(y,X)
                    print(X.isna().sum())
                    # exit()
                    print(X.shape,y.shape)

                    print(X.columns)
                    # exit()
                
            else:
                X = X[selected_features]
            
        # exit()
        column_list =[]
        column_list.append(X.columns.tolist())
        # print(column_list)
        filtered_col_list.append(X.columns.tolist())

    # exit()
    inner_cv = cv
    
    # print("K folds",inner_cv)

    # for x1,y1 in inner_cv:
    #     print(x1,len(x1),"x",y1,len(y1),"y",sep='\n###\n')
    #     print(x1[17],y1[7],df_dataset.loc[x1[17],:],sep="^^^^^^^^^")
    # print("Thisb is innercv",inner_cv,len(inner_cv),len(inner_cv[1][0]),len(inner_cv[1][1]))
    # exit()
    
    if algorithm_no == 1:
        classification_model=RandomForestClassifier(random_state=1)
    elif algorithm_no == 2:
        classification_model=GradientBoostingClassifier(random_state=1)
    elif algorithm_no == 3:
        classification_model=LogisticRegression()
    elif algorithm_no == 4:
        classification_model=SVC(probability=True, random_state=1)
    
    
    noimb_pipeline = Pipeline([('classification_model', classification_model)])
    
    
    clf = GridSearchCV(noimb_pipeline, param_grid= hyperparameters, verbose =0,cv=inner_cv, scoring= scoring, refit = 'roc_auc', n_jobs=-1,error_score="raise",return_train_score=True)
    import time
    # startfit = time.time()
    # print("This is ",X,y,sep="\n\n")
    print(list(X.columns),len(X),len(y))

    if data_setting=='both' or data_setting=='both_summary':
        X = X.drop(pt_col,axis=1)
    if pt_col in X.columns:
        X = X.drop(pt_col,axis=1)
    if label_col in X.columns:
        X = X.drop(label_col,axis=1)
    print(list(X.columns),len(X),len(y),"\n\n\n")
    
    
    # X =X[['mean_flow_from_pef_median', 'inst_RR_median', 'minF_to_zero_median', 'pef_+0.16_to_zero_median', 'iTime_median', 'eTime_median', 'I:E ratio_median', 'dyn_compliance_median', 'tve:tvi ratio_median', 'stat_compliance_median', 'resist_median', 'sf_median', 'lab_pf_ratio_res_median', 'lab_pf_ratio_res_min', 'sf97']]
    from configs import CFS_400_50_True, CFS_400_50_Alt
    if data_setting=='both_summary' or data_setting=='oversample_both_summary':
        vent_features_median = [ i +"_median" for i in obj.vent_features ]
        if feature_selection_method=="CFS_400_50_True":
            CFS_400_50_True.remove(pt_col)
            
            sorted_features =  vent_features_median + CFS_400_50_True 
            print(sorted_features)
        elif feature_selection_method == "CFS_400_50_Alt":
            CFS_400_50_Alt.remove(pt_col)
            sorted_features = vent_features_median + CFS_400_50_Alt
            print(sorted_features)
    X =X[sorted_features]
    print(list(X.columns))
    # X =X[]
    # exit()
    print(X,X.isna().sum())

    clf.fit(X, y)
    # exit()
    
    # endfit = time.time()
    # print(endfit-startfit,"fitting took this much time")
    # print(clf.best_estimator_)
    # print("BEst results",clf.best_score_,"index",clf.best_index_,"best params",clf.best_params_)
    fold_perf.append(clf.cv_results_)
    
     
    results_df = pd.DataFrame(clf.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_roc_auc"])#["rank_test_roc_auc"]) #change this when doing EHR
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("kernel")

   
    
    temp = results_df[results_df["params"]==clf.best_params_]

    tmp1 = results_df[results_df["params"]==clf.best_params_][resultdfcols].values.tolist()
    
    
    # print(tmp1)
    splitlevel = {j:["split" + str(i) + "_"+j[5:] for i in range(5) ] for j in classification_metrics}
    # print(splitlevel) 
    # calc1 = time.time()

    for metric in classification_metrics:
        print(metric)
        metric_ci = round(1.96*np.std(temp[splitlevel[metric]].values.tolist())/np.sqrt(len(temp[splitlevel[metric]].values.tolist())),3)
        # print(temp[metric].values.tolist(),temp[splitlevel[metric]].values.tolist(),np.std(temp[splitlevel[metric]].values.tolist()),np.sqrt(len(temp[splitlevel[metric]].values.tolist()[0])),metric_ci)    

        tmp1[0].append(metric_ci)
    # endcal =time.time()
    # print("\n\n\n\n",endcal-calc1,"calculation time ")
    # print(tmp1)
    # exit() 
    tmp_bootstrap_summary = pd.DataFrame([[data_file.split("/")[-1]]+tmp1[0]],columns=["Filename"]+resultdfcols+classification_metrics_ci)
    tmp_bootstrap_summary.to_excel("temp.xlsx")
    # exit()
    bestmodels_cv=pd.concat([bestmodels_cv,tmp_bootstrap_summary],ignore_index= True)
    
    # print("best model",temp[temp["params"]==clf.best_params_])
    # print(results_df.columns)
    # results_df.to_excel("prelimresults.xlsx")
    # exit()
    model_to_choose =clf.best_estimator_ 
    
    model_file = "classification_model_"+file_list[file_num][:-4]+".pkl"
    
    
    temppath = project_folder + 'algorithm_selection/'+ expt_name + '/' + algorithm + '/' + feature_selection_method 
    bestmodels_cv.to_excel(temppath+"/bestmodels_cv.xlsx")
    
    fullmodels_cv=pd.concat([fullmodels_cv,temp],ignore_index=True)
    fullmodels_cv.to_excel(temppath+"/fullbestmodels_cv.xlsx")
    # exit()
    joblib.dump(model_to_choose, All_file_pickle_folder+model_file)
    print('')
    processing_time = (round(time.time() - start_time, 2))
    if processing_time > 60:
        print('Processing time for file ' + data_file + ': ' + str(round(processing_time/60, 2)) + ' minutes')
    else:
        print('Processing time for file ' + data_file + ': ' + str(processing_time/60) + ' seconds')
    
    print('')
    print('Processed file number ' + str(file_num + 1))
    print(str(round(100*(file_num+1)/num_files, 2)) + '% complete')
    print('')
    print('-------------------------------------------------------')
performance_file = "gridsearch_classification_performance_"+file_list[file_num][:-4]+".pkl"
joblib.dump(fold_perf, All_file_pickle_folder+performance_file)




