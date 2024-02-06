from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
import random
from collections import Counter
from imblearn.over_sampling import SMOTE 

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
from configs import num_splits, Unbalanced, Downsample_25
from argparse import ArgumentParser
from arguments import data_files,test_folder,train_folder,project_folder,data_folder,label_col,pt_col
import warnings
from dataselectutils import get_test_dataset
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
train_file_list = [f for f in listdir(train_folder) if isfile(join(train_folder, f))]
# print(train_folder,test_folder,train_file_list)
# exit()
rows = []

##USe this code block wehn u alrrady have the downsampled splits and want to apply to other time windows
from arguments import fold_information_flag, fold_information_file
if fold_information_flag ==True:
    folds_info = pd.read_csv(fold_information_file)
    main_df = pd.read_csv(os.path.join(data_folder,data_files[0]))
    df_list = [pd.read_csv(os.path.join(data_folder,i)) for i in data_files]
    all_data = main_df
    print(main_df)
    print(all_data.columns)
    total_pts = []
    for df_list_member in df_list:
        total_pts += df_list_member[pt_col].tolist()


    all_pts = list(set(total_pts))

    for file_num in range(number_of_splits):

        print("fold infromation",folds_info)
        # exit()
        folds_for_current_split = folds_info[folds_info['split']==file_num+1]
        train_filename = folds_for_current_split['filename'].tolist()[0]
        print(train_filename)
        fold_1_pigs = ast.literal_eval(folds_for_current_split['fold_1'].tolist()[0])
        fold_2_pigs = ast.literal_eval(folds_for_current_split['fold_2'].tolist()[0])
        fold_3_pigs = ast.literal_eval(folds_for_current_split['fold_3'].tolist()[0])
        fold_4_pigs = ast.literal_eval(folds_for_current_split['fold_4'].tolist()[0])
        fold_5_pigs = ast.literal_eval(folds_for_current_split['fold_5'].tolist()[0])


        all_fold_pigs = [np.array(fold_1_pigs),np.array(fold_2_pigs),np.array(fold_3_pigs),np.array(fold_4_pigs),np.array(fold_5_pigs)]
        
        test_patients = []

        for i in all_pts:
            if i not in fold_1_pigs:
                if i not in fold_2_pigs:
                    if i not in fold_3_pigs:
                        if i not in fold_4_pigs:
                            if i not in fold_5_pigs:
                                test_patients.append(i)
        # print(test_patients)
        train_fold_1 = all_data[all_data[pt_col].isin(fold_1_pigs)] #all_data.take(list(indices_to_keep))
        # print(len(train_fold_1))
        train_fold_2 = all_data[all_data[pt_col].isin(fold_2_pigs)] #all_data.take(list(indices_to_keep))
        # print(len(train_fold_2))

        train_fold_3 = all_data[all_data[pt_col].isin(fold_3_pigs)] #all_data.take(list(indices_to_keep))
        # print(len(train_fold_3))
        train_fold_4 = all_data[all_data[pt_col].isin(fold_4_pigs)] #all_data.take(list(indices_to_keep))
        # print(len(train_fold_4))
        train_fold_5 = all_data[all_data[pt_col].isin(fold_5_pigs)] #all_data.take(list(indices_to_keep))
        # print(len(train_fold_5),train_fold_5)

        train = pd.concat([train_fold_1,train_fold_2,train_fold_3,train_fold_4,train_fold_5],ignore_index=True,axis = 0)
        test = all_data[all_data[pt_col].isin(test_patients)] 
        print(train,os.path.join(train_folder,train_filename))
        print("ending")
        # exit()
        print(train_folder+train_filename)
        train.to_csv(os.path.join(train_folder,train_filename),index=False)
        # test.to_csv(test_folder+train_filename.replace('train','test'),index=False)


    exit()

for file_num in range(len(train_file_list)):
    print('Creating folds for file ' + str(file_num + 1))
    print('')
    
    train_df = pd.read_csv(train_folder+train_file_list[file_num])
    train_df = train_df.drop('Unnamed: 0',axis=1)
    
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
        if Downsample_25 is True:
            
            counter = Counter(train_df[label_col])
            print(counter)
            # print(len(y),len((y[y==1]).index),len((y[y==1]).index)/2,(y[y==1]).index)
            remove_n = 28*2 
            # print(remove_n,X.loc[y[y==1].index,:].index,y[y==1].index)
            drop_indices = np.random.choice(train_df[train_df[label_col]==1].index, remove_n, replace=False)
            # X = X.drop(drop_indices).reset_index(drop=True)
            # y = y.drop(drop_indices).reset_index(drop=True)
            print(" patients to be dropped are",list(train_df.loc[drop_indices,pt_col]))
            # print(train_df[train_df[pt_col]==206][label_col])
            train_df = train_df.drop(drop_indices).reset_index(drop=True)
            counter = Counter(train_df[label_col])
            print(train_df.shape)
            print(counter)
            
            # exit()
        else:
            
            # transform the dataset
            newIDs = train_df.loc[:,pt_col]
            train_df = train_df.drop(pt_col,axis=1)
            for i in range(84*2):
                newIDs.at[len(train_df)+i] =  241+i
            
            
            # print(train_df.loc[:,train_df.columns!=label_col], train_df[label_col])
            # exit()
            counter = Counter(train_df[label_col])
            print(counter)
            oversample = SMOTE(sampling_strategy={0:84*3,1:84})
            X, y = oversample.fit_resample(train_df.loc[:,train_df.columns!=label_col], train_df[label_col])
            # summarize the new class distribution

            
            train_df = pd.concat([newIDs,X, y],axis = 1)
            counter = Counter(train_df[label_col])
            print(counter)
            print(X.shape,y.shape,train_df.shape)
    print(list(train_df.deidentified_study_id))
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
    
    if len(pt_indices_5)/len(ards_pt_indices_5)<0.9*3:
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
    train_df.to_csv(os.path.join(train_folder,train_file_list[file_num]),index=False)
    # exit()
    print('Finished creating folds for file ' + str(file_num + 1))
    print(str(round(100*(file_num+1)/len(train_file_list), 2)) + '% completed')
    print('')
    print('------------------------------------------------------------------------')
    print('')
    
cols = ['filename', 'split', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
fold_df = pd.DataFrame(rows, columns = cols)
fold_df.to_csv(project_folder+'Downsample 25 fold_information.csv', index=False)

