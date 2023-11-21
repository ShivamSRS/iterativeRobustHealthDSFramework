from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
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
from configs import num_splits
from argparse import ArgumentParser
from arguments import data_files,test_folder,train_folder,project_folder,data_folder
import warnings
warnings.filterwarnings("ignore")

data_files = data_files

project_folder = project_folder
data_folder = data_folder
train_folder =train_folder
test_folder =test_folder
repeat_flag = 'Y'
number_of_splits = num_splits
split_ratio = 0.7


parser = ArgumentParser()

parser.add_argument("--race",default=True)



args = parser.parse_args()


if not os.path.isdir(train_folder):
    os.makedirs(train_folder)
if not os.path.isdir(test_folder):
    os.makedirs(test_folder)

main_df = pd.read_csv(data_folder+data_files[0])



df_list = [pd.read_csv(data_folder+i) for i in data_files]



all_data = main_df
racial_features = ['AA', 'Asian', 'Declined', 'Hispanic', 'Middle Eastern', 'Mixed (Asian, White)', 'Mixed (Asian, White, AA)', 'Other', 'White']


if args.race is False:
    all_data = all_data[all_data.columns[~all_data.columns.isin(racial_features)]]
print(main_df)
print(all_data.columns)
# exit()
# remove_n = abs(len(all_data[all_data['ards_flag']==0])-len(all_data[all_data['ards_flag']==1]))
# drop_indices = np.random.choice(all_data[all_data['ards_flag']==0].index, remove_n, replace=False)
# all_data = all_data.drop(drop_indices)
# all_data=all_data.reset_index(drop=True)

label_col = 'ards_flag'
pt_col = 'deidentified_study_id'
total_pts = []
for df_list_member in df_list:
    total_pts += df_list_member[pt_col].tolist()


all_pts = list(set(total_pts))
if len(data_files)>1:
    main_df = df_list[0]
    for dfs in df_list:
        main_df = main_df.merge(dfs, on='pt_col', how='inner')
print(all_pts)
def one_hot_encode(df,feat):
    # Get one hot encoding of columns B
    one_hot = pd.get_dummies(df[feat],prefix=feat)
    # Drop column B as it is now encoded
    df = df.drop(feat,axis = 1)
    # Join the encoded df
    df = df.join(one_hot)
    return df

from arguments import fold_information_flag, fold_information_file
if fold_information_flag ==True:
    folds_info = pd.read_csv(fold_information_file)

    for file_num in range(number_of_splits):

        # print("fold infromation",folds_info)
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
        print(train_folder+train_filename)
        train.to_csv(train_folder+train_filename,index=False)
        test.to_csv(test_folder+train_filename.replace('train','test'),index=False)


    exit()

for split in range(number_of_splits):
    
    print('Creating split ' + str(split+1))
    print('')

    
    
    while repeat_flag == 'Y':
        
        #for binary classificaiton only 
        # all_pts_0_indices = list(main_df[main_df[label_col]==0].index)
        # all_pts_1_indices = list(main_df[main_df[label_col]==1].index)
        # print(all_pts_0,all_pts_1,)
        # exit()
        pt_indices = np.random.choice(len(all_pts), replace = False, 
                                         size = int(split_ratio*len(all_data)))
        test = pd.DataFrame(columns=all_data.columns)
        #197

        full = set(range(0,len(all_pts)))
        excluded_pts = full - set(pt_indices)

        train_pts = [all_pts[i] for i in pt_indices]
        test_pts = [all_pts[i] for i in excluded_pts]

        for i in all_data['ards_flag'].unique():
            val_1  = all_data[all_data['ards_flag']==i].take(np.random.permutation(len(all_data[all_data['ards_flag']==i]))[:int((1-split_ratio)*len(all_data[all_data['ards_flag']==i]))])
            test = pd.concat([test,val_1])
        
        # train_pts = [all_data[i] for i in pt_indices]
        # test_pts = [all_data[i] for i in excluded_indices
        # ]    
        full = set(range(0,len(all_data)))
        indices_to_drop = set(test.index)
        # print(indices_to_drop,len(indices_to_drop))
        # exit()
        indices_to_keep = full - indices_to_drop
        # # full[indices_to_drop] = False
        train = all_data.take(list(indices_to_keep))

        print(len(train),len(test))

        train= train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # train = all_data.loc[all_data.index.isin(pt_indices)]
        # train = train.reset_index(drop=True)
        # train = train.drop('Unnamed: 0',axis=1)

        # test = all_data.loc[all_data.index.isin(excluded_indices)]
        # test = test.reset_index(drop=True)
        # test = test.drop('Unnamed: 0',axis=1)
        # print(train,test,sep="\n####\n")
        # exit()
        print(len(test[test['ards_flag']==1]),len(test[test['ards_flag']==0]),len(pt_indices))
        # print(((train[train['ards_flag']==1].shape[0]) - (train[train['ards_flag']==0].shape[0]))/0.20*train.shape[0])
        # exit()
        if abs((train[train['ards_flag']==1].shape[0]) - \
               (train[train['ards_flag']==0].shape[0])) <= 0.20*train.shape[0]:
            
            repeat_flag = 'N'
        else:
            print('Repeating Split to meet balance criterion')
            print('')
        
    
    repeat_flag = 'Y'

    # train = one_hot_encode(train,'tobacco_user')
    # train = one_hot_encode(train,'smoking_usage')
    # test = one_hot_encode(test,'tobacco_user')
    # test = one_hot_encode(test,'smoking_usage')

    train_filename = 'EHR_train_'+str(split+1)+'.csv'
    test_filename = 'EHR_test_'+str(split+1)+'.csv'
    # train.to_csv(train_folder+train_filename,index=False)
    # test.to_csv(test_folder+test_filename,index=False)

    print('Saved split ' + str(split+1))
    print(str(round(100*(split+1)/number_of_splits, 2)) + '% completed')
    print('')
    print('------------------------------------------------------------------------')
    print('')
    
print('Split formation complete')    
split_summary_df = pd.DataFrame()
train_file_list = [f for f in listdir(train_folder) if isfile(join(train_folder, f))]
test_file_list = [f for f in listdir(test_folder) if isfile(join(test_folder, f))]


###############################               SUMMARIZING               ###############################

all_rows = []
for file_num in range(len(train_file_list)):
    train_data_file = train_file_list[file_num]
    test_data_file = test_file_list[file_num]
    train_data = pd.read_csv(train_folder+train_data_file)
    test_data = pd.read_csv(test_folder+test_data_file)
        
    train_pigs = 0
    test_pigs = 0
    
    train_pigs+=len(train_data)
    
    test_pigs+=len(test_data)
    vals = []
    
    # vals.append(len(list(set(train_data))))
    # vals.append(len(list(set(test_data))))
                
    vals.append(['Total', 
                 train_data[train_data['ards_flag']==1].shape[0],
                 train_data[train_data['ards_flag']==0].shape[0],
                 test_data[test_data['ards_flag']==1].shape[0],
                 test_data[test_data['ards_flag']==0].shape[0],
                 train_pigs,
                 test_pigs])
    print(vals)
    final_df = pd.DataFrame(vals, columns = ['sr no','train 1s', 'train 0s', 'test 1s', 'test 0s',
                                  'train pts', 'test pts'])
    final_df['split'] = [file_num+1]*final_df.shape[0]
    final_df['train_file'] = train_data_file
    final_df['test_file'] = test_data_file
    split_summary_df = pd.concat([split_summary_df, final_df])
# split_summary_df.to_csv(project_folder + 'split_formation_summary.csv', index = False)
print('Split formation summarization complete')
# exit()
###############################               FOLD FORMATION               ###############################

rows = []
for file_num in range(len(train_file_list)):
    print('Creating folds for file ' + str(file_num + 1))
    print('')
    
    train_df = pd.read_csv(train_folder+train_file_list[file_num])
    # ds = train_dftolist()
    # pig = train_df.Pigs.tolist()
    unique_pts_list = list(set(train_df[pt_col]))
    
    bolus_dict = {}
    # for p in unique_pts_list:
    #     count = 0
    #     for i in range(len(pig)):
    #         if ds[i]+':'+pig[i] == p:
    #             count+=1
    #     bolus_dict[p] = count
    
    acceptable = False
    # while acceptable != True:
        
    indices = [i for i in range(len(unique_pts_list))]
    pt_indices_1 = sample(indices,int(0.2*len(unique_pts_list)))
    indices = [i for i in indices if i not in pt_indices_1]
    pt_indices_2 = sample(indices,int(0.2*len(unique_pts_list)))
    indices = [i for i in indices if i not in pt_indices_2]
    pt_indices_3 = sample(indices,int(0.2*len(unique_pts_list)))
    indices = [i for i in indices if i not in pt_indices_3]
    pt_indices_4 = sample(indices,int(0.2*len(unique_pts_list)))
    indices = [i for i in indices if i not in pt_indices_4]
    pt_indices_5 = indices
    
    
    split1 = [unique_pts_list[i] for i in pt_indices_1]
    split2 = [unique_pts_list[i] for i in pt_indices_2]
    split3 = [unique_pts_list[i] for i in pt_indices_3]
    split4 = [unique_pts_list[i] for i in pt_indices_4]
    split5 = [unique_pts_list[i] for i in pt_indices_5]
    
        
        
        
        
            
    rows.append([train_file_list[file_num], file_num+1, [split1[i]for i in range(len(split1))],
                                            [split2[i] for i in range(len(split2))],
                                            [split3[i] for i in range(len(split3))],
                                            [split4[i] for i in range(len(split4))],
                                            [split5[i] for i in range(len(split5))]])
    
    print('Finished creating folds for file ' + str(file_num + 1))
    print(str(round(100*(file_num+1)/len(train_file_list), 2)) + '% completed')
    print('')
    print('------------------------------------------------------------------------')
    print('')
    
cols = ['filename', 'split', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
fold_df = pd.DataFrame(rows, columns = cols)
# fold_df.to_csv(project_folder+'fold_information.csv', index=False)