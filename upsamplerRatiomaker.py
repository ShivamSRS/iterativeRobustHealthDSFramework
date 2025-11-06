from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
import random
from collections import Counter
from imblearn.over_sampling import SMOTE 
from configs import data_setting,feature_import_path,use_prefered_cols,prefered_columns
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
from sklearn.linear_model import LogisticRegression
# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.metrics import accuracy_score as acc
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
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
import random

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
from random import sample
from configs import num_splits, Unbalanced, Downsample_25,use_features
from argparse import ArgumentParser
from arguments import data_files,test_folder,train_folder,project_folder,data_folder,label_col,pt_col,time_window
import warnings
from dataselectutils import get_test_dataset,get_dataset
from VentWaveData import VentData
from arguments import ventDataFolder, ventDataFiles_median

warnings.filterwarnings("ignore")

data_files = data_files

project_folder = project_folder
data_folder = data_folder
train_folder =train_folder
test_folder =test_folder
repeat_flag = 'Y'
number_of_splits = num_splits
label_col,pt_col = label_col,pt_col

# print(train_folder,test_folder)
train_file_list = [f for f in listdir(train_folder) if isfile(join(train_folder, f)) and (f[-3:]=='csv' or f[-3]=='xls' or f[-4:]=='xlsx')]
print(train_folder,test_folder,train_file_list,train_file_list[58])
# exit()
rows = []

##USe this code block wehn u alrrady have the downsampled splits and want to apply to other time windows
from arguments import fold_information_flag, fold_information_file
if fold_information_flag ==True:
    if Downsample_25 is False:
        
        folds_info = pd.read_csv(fold_information_file)

        for file_num in range(len(train_file_list)):

            print(file_num,train_file_list[file_num])
            

            train_or_test="train"
            if data_setting=='ehr':
                train_or_test="train"
                print("file is ",file_num,train_file_list[file_num])
                X,y,df_dataset, cv,train_patient_ids,test_patient_ids = get_dataset(os.path.join(project_folder,train_or_test,time_window,train_file_list[file_num]),file_num,label_col,pt_col)
                
                print(test_patient_ids)
                count_ones = y.value_counts()
                # count_zeros = y.count(0)

                print("Number of 1s:", count_ones)
                # exit()
                train_df = df_dataset

            elif data_setting=='vent_summary':
                obj = VentData(ventDataFolder)
                train_or_test="train"
                X,y,df_dataset, cv,train_pigs =obj.get_train_test_file_summary(os.path.join(project_folder,train_or_test,time_window,train_file_list[file_num]),file_num,ventDataFiles_median,time_window,median_only=True)
                patient_ids = df_dataset[pt_col].values
                count_ones = y.value_counts()
                # count_zeros = y.count(0)
                train_df = df_dataset
                print(X.columns)
                print("Number of 1s:", count_ones)
                print(len(y),len(X))
                print(X.head())

            elif data_setting=='both_summary':
                obj = VentData(ventDataFolder)
                print("now enteting ehr only")
                X,y,df_dataset, cv,train_patient_ids,test_patient_ids  = get_dataset(os.path.join(project_folder,train_or_test,time_window,train_file_list[file_num]),file_num,label_col,pt_col,give_pt=True)
                
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

                ventX,venty,ventdf_dataset, cv,train_pigs  =obj.get_train_test_file_summary(os.path.join(project_folder,train_or_test,time_window,train_file_list[file_num]),int(file_num),ventDataFiles_median,time_window,give_pt=True,median_only=True)
                
                print("oversam")
                
                ventdf_dataset.sort_values(by=pt_col, inplace=True)
                ventdf_dataset.reset_index(drop=True, inplace=True)
                columns_to_include = [col for col in ventdf_dataset.columns if col not in [label_col]]
                ventX,venty = ventdf_dataset.loc[:,columns_to_include],ventdf_dataset[label_col] 

                print("TALLYING BOTH VENT AND EHR",y.value_counts(),venty.value_counts(),(y-venty).value_counts())
                df_dataset.to_excel("ehrCHEDCK.xlsx")
                ventdf_dataset.to_excel("ventCHECK.xlsx")

                # exit()
                scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
                    'specificity': make_scorer(recall_score,pos_label=0),\
                    'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True)}
                # print(cv)

                if data_setting=='ehr' or data_setting=='both' or data_setting=='both_summary' or data_setting=="oversample_both_summary":
                    print("inside inmport featu")
                    if use_features == 'Y':

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
                        else:
                            selected_features = X.columns
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
                            y = pd.DataFrame(y.values,columns=[label_col])
                            print(y,X)
                            # exit()
                            
                            train_df = pd.concat([X,y],axis=1)
                            print(train_df)
            # elif data_setting=='both_summary':
            #     obj = VentData(ventDataFolder)
            #     train_or_test="train"
            #     X,y,df_dataset, cv,train_pigs =obj.get_train_test_file_summary(os.path.join(project_folder,train_or_test,time_window,train_file_list[file_num]),file_num,ventDataFiles_median,time_window,median_only=True)
            #     patient_ids = df_dataset[pt_col].values
            #     count_ones = y.value_counts()
            #     # count_zeros = y.count(0)
            #     train_df = df_dataset
            #     print(X.columns)
            #     print("Number of 1s:", count_ones)
            #     print(len(y),len(X))
            #     print(X.head())
                
            else:
                train_df = pd.read_csv(train_folder+train_file_list[file_num])
                train_df = train_df.drop('Unnamed: 0',axis=1)
            if 'Unnamed: 0' in train_df.columns:
                train_df = train_df.drop('Unnamed: 0',axis=1)
            print(train_df.columns)
            print(train_df[label_col].value_counts())
            # train_df = pd.read_csv(os.path.join(train_folder,train_file_list[file_num]))

            # print(train_df['lab_chloride_res_mean'].isna().sum())
            
            # if time_window == '12h':
            #     train_df['lab_chloride_res_mean'] = train_df['lab_chloride_res_mean'].fillna(100)
            # print(train_df['lab_chloride_res_mean'].isna().sum())
            # exit()
            # transform the dataset
            newIDs = train_df.loc[:,pt_col]
            train_df = train_df.drop(pt_col,axis=1)
            for i in range(84*2):
                newIDs.at[len(train_df)+i] =  241+i
            #     print(len(newIDs),len(train_df))
            print(len(newIDs))
            
            print(train_df.loc[:,train_df.columns!=label_col], train_df[label_col])
            
            counter = Counter(train_df[label_col])
            print(counter)
            # exit()
            print(train_df[train_df.isna()].columns)
            oversample = SMOTE(sampling_strategy={0:84*3,1:84})
            X, y = oversample.fit_resample(train_df.loc[:,train_df.columns!=label_col], train_df[label_col])
            # summarize the new class distribution

            
            train_df = pd.concat([newIDs, y, X],axis = 1)
            counter = Counter(train_df[label_col])

            print("after oversampling",counter)
            # exit()
            print(X.shape,y.shape,train_df.shape)
            from arguments import fold_information_file_ForUpsampling

            folds_info = pd.read_csv(fold_information_file_ForUpsampling)

            folds_for_current_split = folds_info[folds_info['filename']==train_file_list[file_num]]
            print(folds_for_current_split)

            fold_1_pigs = ast.literal_eval(folds_for_current_split['fold_1'].tolist()[0])
            fold_2_pigs = ast.literal_eval(folds_for_current_split['fold_2'].tolist()[0])
            fold_3_pigs = ast.literal_eval(folds_for_current_split['fold_3'].tolist()[0])
            fold_4_pigs = ast.literal_eval(folds_for_current_split['fold_4'].tolist()[0])
            fold_5_pigs = ast.literal_eval(folds_for_current_split['fold_5'].tolist()[0])


            all_fold_pigs = [np.array(fold_1_pigs),np.array(fold_2_pigs),np.array(fold_3_pigs),np.array(fold_4_pigs),np.array(fold_5_pigs)]
            print(all_fold_pigs)


            train_fold_1 = train_df[train_df[pt_col].isin(fold_1_pigs)] #all_data.take(list(indices_to_keep))
            print(len(train_fold_1),len(fold_1_pigs))
            train_fold_2 = train_df[train_df[pt_col].isin(fold_2_pigs)] #all_data.take(list(indices_to_keep))
            print(len(train_fold_2),len(fold_2_pigs))

            train_fold_3 = train_df[train_df[pt_col].isin(fold_3_pigs)] #all_data.take(list(indices_to_keep))
            print(len(train_fold_3),len(fold_3_pigs))
            train_fold_4 = train_df[train_df[pt_col].isin(fold_4_pigs)] #all_data.take(list(indices_to_keep))
            print(len(train_fold_4),len(fold_4_pigs))
            train_fold_5 = train_df[train_df[pt_col].isin(fold_5_pigs)] #all_data.take(list(indices_to_keep))
            print(len(train_fold_5),len(fold_5_pigs))

            train = pd.concat([train_fold_1,train_fold_2,train_fold_3,train_fold_4,train_fold_5],ignore_index=True,axis = 0)
            pew = list(train.deidentified_study_id)
            print(pew,len(pew))
            print(train)
            print(os.path.join(train_folder,train_file_list[file_num]))

            print(train_df[label_col].value_counts(),os.path.join(project_folder,"upsampled_data",data_setting,time_window,train_file_list[file_num]))
            train_df = train_df.sort_values(by=pt_col,ignore_index=True)
            train_df = train_df.reset_index(drop=True)
            print(train_df,train_df.columns,os.path.join(project_folder,"upsampled_data",data_setting,time_window,train_file_list[file_num]))
            # exit()
            if time_window=='':
                train_df.to_csv(os.path.join(project_folder,"upsampled_data",data_setting,"48h",train_file_list[file_num]),index=False)
            else:
                train_df.to_csv(os.path.join(project_folder,"upsampled_data",data_setting,time_window,train_file_list[file_num]),index=False)
                    
            # exit()




# exit()

for file_num in range(len(train_file_list)):
    print('Creating folds for file ' + str(file_num + 1))
    print('')
    
    train_or_test="train"
    if data_setting=='ehr':
        train_or_test="train"
        print("file is ",file_num,train_file_list[file_num])
        X,y,df_dataset, cv,train_patient_ids,test_patient_ids = get_dataset(os.path.join(project_folder,train_or_test,time_window,train_file_list[file_num]),file_num,label_col,pt_col)
        
        print(test_patient_ids)
        count_ones = y.value_counts()
        # count_zeros = y.count(0)

        print("Number of 1s:", count_ones)
        # exit()
        train_df = df_dataset

    elif data_setting=='vent_summary':
        obj = VentData(ventDataFolder)
        train_or_test="train"
        X,y,df_dataset, cv,train_pigs =obj.get_train_test_file_summary(os.path.join(project_folder,train_or_test,time_window,train_file_list[file_num]),file_num,ventDataFiles_median,time_window,median_only=True)
        patient_ids = df_dataset[pt_col].values
        count_ones = y.value_counts()
        # count_zeros = y.count(0)
        train_df = df_dataset
        print(X.columns)
        print("Number of 1s:", count_ones)
        print(len(y),len(X))
        print(X.head())

    elif data_setting=='both_summary':
        obj = VentData(ventDataFolder)
        print("now enteting ehr only")
        X,y,df_dataset, cv,train_patient_ids,test_patient_ids  = get_dataset(os.path.join(project_folder,train_or_test,time_window,train_file_list[file_num]),file_num,label_col,pt_col,give_pt=True)
        
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

        ventX,venty,ventdf_dataset, cv,train_pigs  =obj.get_train_test_file_summary(os.path.join(project_folder,train_or_test,time_window,train_file_list[file_num]),int(file_num),ventDataFiles_median,time_window,give_pt=True,median_only=True)
        
        print("oversam")
        
        ventdf_dataset.sort_values(by=pt_col, inplace=True)
        ventdf_dataset.reset_index(drop=True, inplace=True)
        columns_to_include = [col for col in ventdf_dataset.columns if col not in [label_col]]
        ventX,venty = ventdf_dataset.loc[:,columns_to_include],ventdf_dataset[label_col] 

        print("TALLYING BOTH VENT AND EHR",y.value_counts(),venty.value_counts(),(y-venty).value_counts())
        df_dataset.to_excel("ehrCHEDCK.xlsx")
        ventdf_dataset.to_excel("ventCHECK.xlsx")

        # exit()
        scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
               'specificity': make_scorer(recall_score,pos_label=0),\
               'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True)}
        # print(cv)

        if data_setting=='ehr' or data_setting=='both' or data_setting=='both_summary' or data_setting=="oversample_both_summary":
            print("inside inmport featu")
            if use_features == 'Y':

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
                else:
                    selected_features = X.columns
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
                    y = pd.DataFrame(y.values,columns=[label_col])
                    print(y,X)
                    # exit()
                    
                    train_df = pd.concat([X,y],axis=1)
                    print(train_df)
                    # exit()
    # elif data_setting=='both_summary':
    #     obj = VentData(ventDataFolder)
    #     train_or_test="train"
    #     X,y,df_dataset, cv,train_pigs =obj.get_train_test_file_summary(os.path.join(project_folder,train_or_test,time_window,train_file_list[file_num]),file_num,ventDataFiles_median,time_window,median_only=True)
    #     patient_ids = df_dataset[pt_col].values
    #     count_ones = y.value_counts()
    #     # count_zeros = y.count(0)
    #     train_df = df_dataset
    #     print(X.columns)
    #     print("Number of 1s:", count_ones)
    #     print(len(y),len(X))
    #     print(X.head())
        
    else:
        train_df = pd.read_csv(train_folder+train_file_list[file_num])
        train_df = train_df.drop('Unnamed: 0',axis=1)
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop('Unnamed: 0',axis=1)
    print(train_df.columns)
    print(train_df[label_col].value_counts())
    # exit()
    
    # ds = train_dftolist()
    # pig = train_df.Pigs.tolist()
    # X,y,_ = get_test_dataset(os.path.join(train_folder,train_file_list[file_num]),label_col,pt_col)
    # print(X.columns)
    
    

    
    # counter = Counter(y)
    # print(unique_pts_list_0,unique_pts_list_1)
    # exit()
    # exit()
    # continue
    
    
    if Unbalanced is True:
        if Downsample_25 is False:
            if time_window == '12h':
                train_df['lab_chloride_res_mean'] = train_df['lab_chloride_res_mean'].fillna(100)
            # transform the dataset
            newIDs = train_df.loc[:,pt_col]
            train_df = train_df.drop(pt_col,axis=1)
            for i in range(84*2):
                newIDs.at[len(train_df)+i] =  241+i
            #     print(len(newIDs),len(train_df))
            # print(len(newIDs))
            
            # print(train_df.loc[:,train_df.columns!=label_col], train_df[label_col])
            # exit()
            counter = Counter(train_df[label_col])
            print(counter)
            oversample = SMOTE(sampling_strategy={0:84*3,1:84})
            X, y = oversample.fit_resample(train_df.loc[:,train_df.columns!=label_col], train_df[label_col])
            # summarize the new class distribution

            
            train_df = pd.concat([newIDs, y, X],axis = 1)
            counter = Counter(train_df[label_col])
            print(counter)
            print(X.shape,y.shape,train_df.shape)
    print(list(train_df.deidentified_study_id),len(list(train_df.deidentified_study_id)))
    acceptable = False
    
    unique_pts_list_0 = list(set(train_df[train_df[label_col]==0][pt_col]))
    unique_pts_list_1 = list(set(train_df[train_df[label_col]==1][pt_col]))
    # while acceptable != True:
        
    indices_0 = [i for i in range(len(unique_pts_list_0))]
    # print(indices_0)
    pt_indices_1 = sample(indices_0,int(0.2*len(unique_pts_list_0)))
    indices_0 = [i for i in indices_0 if i not in pt_indices_1]
    pt_indices_2 = sample(indices_0,int(0.2*len(unique_pts_list_0)))
    indices_0 = [i for i in indices_0 if i not in pt_indices_2]
    pt_indices_3 = sample(indices_0,int(0.2*len(unique_pts_list_0)))
    indices_0 = [i for i in indices_0 if i not in pt_indices_3]
    pt_indices_4 = sample(indices_0,int(0.2*len(unique_pts_list_0)))
    indices_0 = [i for i in indices_0 if i not in pt_indices_4]
    pt_indices_5 = indices_0


    indices_1 = [i for i in range(len(unique_pts_list_1))]
    # print(indices_1)
    # exit()
    ards_pt_indices_1 = sample(indices_1,int(0.2*len(unique_pts_list_1)))
    indices_1 = [i for i in indices_1 if i not in ards_pt_indices_1]
    ards_pt_indices_2 = sample(indices_1,int(0.2*len(unique_pts_list_1)))
    indices_1 = [i for i in indices_1 if i not in ards_pt_indices_2]
    ards_pt_indices_3 = sample(indices_1,int(0.2*len(unique_pts_list_1)))
    indices_1 = [i for i in indices_1 if i not in ards_pt_indices_3]
    ards_pt_indices_4 = sample(indices_1,int(0.2*len(unique_pts_list_1)))
    indices_1 = [i for i in indices_1 if i not in ards_pt_indices_4]
    ards_pt_indices_5 = indices_1
    
    # print(pt_indices_1,pt_indices_2,pt_indices_3,pt_indices_4,pt_indices_5,sep="\n##")
    # print("\n\n")
    # print(train_df.loc[train_df[train_df[pt_col]==83].index,label_col],set(train_df[pt_col]))
    # print(ards_pt_indices_1,ards_pt_indices_2,ards_pt_indices_3,ards_pt_indices_4,ards_pt_indices_5,sep="\n##")
    # print(y[pt_indices_1])
    # cnt1,cnt2,cnt3,cnt4,cn5 = Counter(y[pt_indices_1]),Counter(y[pt_indices_2]),Counter(y[pt_indices_3]),Counter(y[pt_indices_4]),Counter(y[pt_indices_5])
    # print(cnt1,cnt2,cnt3,cnt4,cn5)
    # exit()]
    import math
    
    if len(pt_indices_5)/len(ards_pt_indices_5)<0.9*3 :
        print("set 5",len(pt_indices_5)/len(ards_pt_indices_5))
        if 3*len(ards_pt_indices_5)<len(pt_indices_5):
            remove_n = len(pt_indices_5) - 3*len(ards_pt_indices_5)
            drop_indices = np.random.choice(pt_indices_5, remove_n, replace=False)
            
            drop_pts = [unique_pts_list_0[k] for k in drop_indices]
            train_df=train_df.drop(train_df[train_df[pt_col].isin(drop_pts)].index)
            
            pt_indices_5 = list(set(pt_indices_5) - set(drop_indices))

            # print("less",len(ards_pt_indices_5),len(pt_indices_5))
        else:
            remove_n =   len(ards_pt_indices_5)-int(len(pt_indices_5)/3)
            drop_indices = np.random.choice(ards_pt_indices_5, remove_n, replace=False)
            print("more b4",len(ards_pt_indices_5),len(pt_indices_5))
            drop_pts = [unique_pts_list_1[k] for k in drop_indices]
            train_df=train_df.drop(train_df[train_df[pt_col].isin(drop_pts)].index)
            
            ards_pt_indices_5 = list(set(ards_pt_indices_5) - set(drop_indices))
             
            print("more",len(ards_pt_indices_5),len(pt_indices_5))
    
    if len(pt_indices_4)/len(ards_pt_indices_4)<0.9*3:
        if 3*len(ards_pt_indices_4)<len(pt_indices_4):
            remove_n = len(pt_indices_4) - 3*len(ards_pt_indices_4)
            drop_indices = np.random.choice(pt_indices_4, remove_n, replace=False)
            
            drop_pts = [unique_pts_list_0[k] for k in drop_indices]
            train_df=train_df.drop(train_df[train_df[pt_col].isin(drop_pts)].index)
            
            pt_indices_4 = list(set(pt_indices_4) - set(drop_indices))

            # print("less",len(ards_pt_indices_5),len(pt_indices_5))
        else:
            remove_n =   len(ards_pt_indices_4)-int(math.ceil(len(pt_indices_4)/3))
            drop_indices = np.random.choice(ards_pt_indices_4, remove_n, replace=False)
            
            drop_pts = [unique_pts_list_1[k] for k in drop_indices]
            train_df=train_df.drop(train_df[train_df[pt_col].isin(drop_pts)].index)
            
            ards_pt_indices_4 = list(set(ards_pt_indices_4) - set(drop_indices))
    
    if len(pt_indices_3)/len(ards_pt_indices_3)<0.9*3:
        if 3*len(ards_pt_indices_5)<len(pt_indices_3):
            remove_n = len(pt_indices_3) - 3*len(ards_pt_indices_3)
            drop_indices = np.random.choice(pt_indices_3, remove_n, replace=False)
            
            drop_pts = [unique_pts_list_0[k] for k in drop_indices]
            train_df=train_df.drop(train_df[train_df[pt_col].isin(drop_pts)].index)
            
            pt_indices_3 = list(set(pt_indices_3) - set(drop_indices))

            # print("less",len(ards_pt_indices_5),len(pt_indices_5))
        else:
            remove_n =   len(ards_pt_indices_3)-int(math.ceil(len(pt_indices_3)/3))
            drop_indices = np.random.choice(ards_pt_indices_3, remove_n, replace=False)
            
            drop_pts = [unique_pts_list_1[k] for k in drop_indices]
            train_df=train_df.drop(train_df[train_df[pt_col].isin(drop_pts)].index)
            
            ards_pt_indices_3 = list(set(ards_pt_indices_3) - set(drop_indices))


    if len(pt_indices_2)/len(ards_pt_indices_2)<0.9*3:
        if 3*len(ards_pt_indices_2)<len(pt_indices_2):
            remove_n = len(pt_indices_2) - 3*len(ards_pt_indices_2)
            drop_indices = np.random.choice(pt_indices_2, remove_n, replace=False)
            
            drop_pts = [unique_pts_list_0[k] for k in drop_indices]
            train_df=train_df.drop(train_df[train_df[pt_col].isin(drop_pts)].index)
            
            pt_indices_2 = list(set(pt_indices_2) - set(drop_indices))

            # print("less",len(ards_pt_indices_5),len(pt_indices_5))
        else:
            remove_n =   len(ards_pt_indices_2)-int(math.ceil(len(pt_indices_2)/3))
            drop_indices = np.random.choice(ards_pt_indices_2, remove_n, replace=False)
            
            drop_pts = [unique_pts_list_1[k] for k in drop_indices]
            train_df=train_df.drop(train_df[train_df[pt_col].isin(drop_pts)].index)
            
            ards_pt_indices_2 = list(set(ards_pt_indices_2) - set(drop_indices))

    if len(pt_indices_1)/len(ards_pt_indices_1)<0.9*3:
        if 3*len(ards_pt_indices_1)<len(pt_indices_1):
            remove_n = len(pt_indices_1) - 3*len(ards_pt_indices_1)
            drop_indices = np.random.choice(pt_indices_1, remove_n, replace=False)
            
            drop_pts = [unique_pts_list_0[k] for k in drop_indices]
            train_df=train_df.drop(train_df[train_df[pt_col].isin(drop_pts)].index)
            
            pt_indices_1 = list(set(pt_indices_1) - set(drop_indices))

            # print("less",len(ards_pt_indices_5),len(pt_indices_5))
        else:
            remove_n =   len(ards_pt_indices_1)-int(math.ceil(len(pt_indices_1)/3))
            drop_indices = np.random.choice(ards_pt_indices_1, remove_n, replace=False)
            
            drop_pts = [unique_pts_list_1[k] for k in drop_indices]
            train_df=train_df.drop(train_df[train_df[pt_col].isin(drop_pts)].index)
            
            ards_pt_indices_1 = list(set(ards_pt_indices_1) - set(drop_indices))


    split1 = [unique_pts_list_0[i] for i in pt_indices_1]
    split2 = [unique_pts_list_0[i] for i in pt_indices_2]
    split3 = [unique_pts_list_0[i] for i in pt_indices_3]
    split4 = [unique_pts_list_0[i] for i in pt_indices_4]
    split5 = [unique_pts_list_0[i] for i in pt_indices_5]

    print(len(train_df))
    split1 += [unique_pts_list_1[i] for i in ards_pt_indices_1]
    split2 +=[unique_pts_list_1[i] for i in ards_pt_indices_2]
    split3 += [unique_pts_list_1[i] for i in ards_pt_indices_3]
    split4 += [unique_pts_list_1[i] for i in ards_pt_indices_4]
    split5 += [unique_pts_list_1[i] for i in ards_pt_indices_5]

    # print(split1)           
    # print(train_df)
    # print([train_df[train_df[pt_col]==k][label_col] for k in split1])

    split1Count = Counter([int(train_df[train_df[pt_col]==k][label_col])  for k in split1])
    print(split1Count[0],split1Count[1],3*0.9,3*1.1)
    
    split2Count = Counter([int(train_df[train_df[pt_col]==k][label_col])  for k in split2])
    print(split2Count[0],split2Count[1])

    split3Count = Counter([int(train_df[train_df[pt_col]==k][label_col])  for k in split3])
    print(split3Count[0],split3Count[1])

    split4Count = Counter([int(train_df[train_df[pt_col]==k][label_col])  for k in split4])
    print(split4Count[0],split4Count[1])

    split5Count = Counter([int(train_df[train_df[pt_col]==k][label_col])  for k in split5])
    print(split5Count[0],split5Count[1])

    
    rows.append([train_file_list[file_num], file_num+1, [split1[i]for i in range(len(split1))],
                                            [split2[i] for i in range(len(split2))],
                                            [split3[i] for i in range(len(split3))],
                                            [split4[i] for i in range(len(split4))],
                                            [split5[i] for i in range(len(split5))]])
    print(rows,len(train_df))
    # exit()
    train_df = train_df.sort_values(by=pt_col,ignore_index=True)
    print(train_df)
    # exit()
    if time_window=='':
        train_df.to_csv(os.path.join(project_folder,"upsampled_data",data_setting,"48h",train_file_list[file_num]),index=False)
    else:
        train_df.to_csv(os.path.join(project_folder,"upsampled_data",data_setting,time_window,train_file_list[file_num]),index=False)
    # exit()
    print('Finished creating folds for file ' + str(file_num + 1))
    print(str(round(100*(file_num+1)/len(train_file_list), 2)) + '% completed')
    print('')
    print('------------------------------------------------------------------------')
    print('')
    
cols = ['filename', 'split', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
fold_df = pd.DataFrame(rows, columns = cols)
if Downsample_25 is True:
    fold_df.to_csv(project_folder+"Downsample 25 fold_information_"+data_setting+".csv", index=False)
else:
    if time_window=='':
        fold_df.to_csv(project_folder+"/Upsample 25 fold_information_"+data_setting+"_48h"+".csv", index=False)
    else:
        fold_df.to_csv(project_folder+"/Upsample 25 fold_information_"+data_setting+"_"+time_window+".csv", index=False)
    
