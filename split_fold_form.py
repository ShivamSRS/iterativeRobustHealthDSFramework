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
from random import sample
from configs import num_splits
from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore")

data_files = ['preeclampsia_final']

project_folder = '/data/srs/zipcode/'
data_folder = project_folder + 'datafile/'
train_folder = project_folder + 'train/'
test_folder = project_folder + 'test/'
repeat_flag = 'Y'
number_of_splits = num_splits
split_ratio = 0.7

split_summary_df = pd.DataFrame()
train_file_list = [f for f in listdir(train_folder) if isfile(join(train_folder, f))]
test_file_list = [f for f in listdir(test_folder) if isfile(join(test_folder, f))]

parser = ArgumentParser()

parser.add_argument("--race",default=True)
parser.add_argument("--original",default=False)
parser.add_argument("--adi_only",default=False)


args = parser.parse_args()


if not os.path.isdir(train_folder):
    os.makedirs(train_folder)
if not os.path.isdir(test_folder):
    os.makedirs(test_folder)

main_df = pd.read_csv(data_folder+data_files[0]+'.csv')



# df_list = [main_df]



all_data = main_df
racial_features = ['AA', 'Asian', 'Declined', 'Hispanic', 'Middle Eastern', 'Mixed (Asian, White)', 'Mixed (Asian, White, AA)', 'Other', 'White']

if args.original and (args.adi_only==False):
    all_data = all_data.loc[:,all_data.columns[:51]]

if args.adi_only:
    all_data = all_data.loc[:,all_data.columns[:52]]
if args.race is False:
    all_data = all_data[all_data.columns[~all_data.columns.isin(racial_features)]]
all_data["Delta BMI"] = all_data['BMI close to delivery'] - all_data['BMI prepregnancy']

print(all_data.columns)
# exit()
remove_n = abs(len(all_data[all_data['Readmitted (Y/N)']==0])-len(all_data[all_data['Readmitted (Y/N)']==1]))
drop_indices = np.random.choice(all_data[all_data['Readmitted (Y/N)']==0].index, remove_n, replace=False)
all_data = all_data.drop(drop_indices)
all_data=all_data.reset_index(drop=True)


for split in range(number_of_splits):
    
    print('Creating split ' + str(split+1))
    print('')
    
    while repeat_flag == 'Y':
        
        
        pt_indices = np.random.choice(len(all_data), replace = False, 
                                         size = int(split_ratio*len(all_data)))
        test = pd.DataFrame(columns=all_data.columns)
        #197
        
        
        for i in all_data['Readmitted (Y/N)'].unique():
            val_1  = all_data[all_data['Readmitted (Y/N)']==i].take(np.random.permutation(len(all_data[all_data['Readmitted (Y/N)']==i]))[:int((1-split_ratio)*len(all_data[all_data['Readmitted (Y/N)']==i]))])
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
        print(len(test[test['Readmitted (Y/N)']==1]),len(test[test['Readmitted (Y/N)']==0]),len(pt_indices))
        # print(((train[train['Readmitted (Y/N)']==1].shape[0]) - (train[train['Readmitted (Y/N)']==0].shape[0]))/0.20*train.shape[0])
        # exit()
        if abs((train[train['Readmitted (Y/N)']==1].shape[0]) - \
               (train[train['Readmitted (Y/N)']==0].shape[0])) <= 0.20*train.shape[0]:
            
            repeat_flag = 'N'
        else:
            print('Repeating Split to meet balance criterion')
            print('')
        
    
    repeat_flag = 'Y'

    train_filename = 'PZ_train_'+str(split+1)+'.csv'
    test_filename = 'PZ_test_'+str(split+1)+'.csv'
    train.to_csv(train_folder+train_filename,index=False)
    test.to_csv(test_folder+test_filename,index=False)

    print('Saved split ' + str(split+1))
    print(str(round(100*(split+1)/number_of_splits, 2)) + '% completed')
    print('')
    print('------------------------------------------------------------------------')
    print('')
    
print('Split formation complete')    


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
    print
    # vals.append(len(list(set(train_data))))
    # vals.append(len(list(set(test_data))))
                
    vals.append(['Total', 
                 train_data[train_data['Readmitted (Y/N)']==1].shape[0],
                 train_data[train_data['Readmitted (Y/N)']==0].shape[0],
                 test_data[test_data['Readmitted (Y/N)']==1].shape[0],
                 test_data[test_data['Readmitted (Y/N)']==0].shape[0],
                 train_pigs,
                 test_pigs])
    print(vals)
    final_df = pd.DataFrame(vals, columns = ['sr no','train 1s', 'train 0s', 'test 1s', 'test 0s',
                                  'train pigs', 'test pigs'])
    final_df['split'] = [file_num+1]*final_df.shape[0]
    final_df['train_file'] = train_data_file
    final_df['test_file'] = test_data_file
    split_summary_df = pd.concat([split_summary_df, final_df])
split_summary_df.to_csv(project_folder + 'split_formation_summary.csv', index = False)
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
    unique_pts_list = list(set(train_df['Mutated MRN']))
    
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
fold_df.to_csv(project_folder+'fold_information.csv', index=False)