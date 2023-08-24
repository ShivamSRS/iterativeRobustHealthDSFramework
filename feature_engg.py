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

from dataselectutils import get_dataset,statistical_filter,mutual_info, RFE_features,permutation_importance_features,Lasso,chi_square


import warnings
warnings.filterwarnings("ignore")
data_files = ['preeclampsia_final']

project_folder = '/data/srs/zipcode/'
data_folder = project_folder + 'datafile/'
train_folder = project_folder + 'train/'
test_folder = project_folder + 'test/'

repeat_flag = 'Y'
hyperparameter_catalog = {

    'RF': {
        'bootstrap': [True],
        'max_depth': [2, 5, 10], # maximum depth of the tree
        'max_features': ['auto','sqrt'], # maximum number of features to use at each split
        'min_samples_leaf': [5,10], # minimum number of samples to split a node
        'min_samples_split': range(2,10,2),
        'n_estimators': [100,200, 500], # number of trees
        'criterion' : ['gini','entropy']  # criterion for evaluating a split
    }
}

rp_list = [['n','n'], ['y', 'n'], ['n', 'y']]


data_folder = 'bootstraps_sv2/'

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








#### Select Feature Selection Method ####
# set selection_method to:
# 1 for statistical feature selection, 2 for mutual information,
# 3 for permutation importance, 4 for recursive feature elimination

feature_selection_method_catalog = {
    'statistical_feature_selection': 1,
    'mutual_information': 2,
    'permutation_importance': 3,
    'RFE': 4,
    'Lasso' : 5,
    'Chi2' : 6
}
from configs import feature_selection_method

method = feature_selection_method
selection_method = feature_selection_method_catalog[method]
from configs import num_splits
num_files = num_splits
data_folder = ''
train_or_test = 'train/'


file_list = [f for f in listdir(data_folder+train_or_test) if isfile(join(data_folder+train_or_test, f))]




feature_list_dict = {}
for file_num in range(num_files):

    start_time = time.time()
    print("method is",method)

    file_name = file_list[file_num]
    X,y,df_dataset, cv = get_dataset(data_folder+train_or_test+file_name,file_num)

    if selection_method == 1:
        k = 'NA'
        Corr_threshold = 0.9 #'Not applicable'

        for correlation_th in [Corr_threshold]:

        #    print(correlation_th)
        #     X,y,df_dataset = get_dataset(os.path.join(input_folder,data_file))

            if correlation_th != 'Not applicable':

                correlation_th = np.round(correlation_th,1)
                feats = statistical_filter(df_dataset,X,y, correlation_th)

                feature_list_dict[file_name[:-4]] = feats

        print('Processed file ' + str(file_num + 1))
        print(str(round(100*(file_num+1)/num_files, 2)) + '% complete')




    elif selection_method == 2:
        k = 10
        print(len(X.columns),len(df_dataset.columns))
        feats = mutual_info(X,y,df_dataset, k)
        feature_list_dict[file_name[:-4]] = feats
        print("top features according to mutual info are",feats)
        print('Processed file ' + str(file_num + 1))
        print(str(round(100*(file_num+1)/num_files, 2)) + '% complete')




    elif selection_method == 3:
        k = 10
        feats = permutation_importance_features(data_folder+train_or_test+file_name, file_num, k)
        feature_list_dict[file_name[:-4]] = feats

        print('Processed file ' + str(file_num + 1))
        print(str(round(100*(file_num+1)/num_files, 2)) + '% complete')




    elif selection_method == 4:
        k = 10
        feats = RFE_features(data_folder+train_or_test+file_name, file_num, k)
        feature_list_dict[file_name[:-4]] = feats
        print("featurs selected by rfe are",feats)
        print('Processed file ' + str(file_num + 1))
        print(str(round(100*(file_num+1)/num_files, 2)) + '% complete')
    
    elif selection_method == 5:
        k = 10
        feats = Lasso(df_dataset,X,y)
        feature_list_dict[file_name[:-4]] = feats
        print("featurs selected by rfe are",feats)
        print('Processed file ' + str(file_num + 1))
        print(str(round(100*(file_num+1)/num_files, 2)) + '% complete')
    
    elif selection_method == 6:
        k = 10
        feats = chi_square(df_dataset,X,y,True)
        feature_list_dict[file_name[:-4]] = feats
        print("featurs selected by chi2 are",feats)
        print('Processed file ' + str(file_num + 1))
        print(str(round(100*(file_num+1)/num_files, 2)) + '% complete')

    print('')
    print('Processing time for file ' + str(file_num + 1) + ': ' + str((round(time.time() - start_time, 2))) + ' seconds')
    print('-------------------------------------------------------')

save_to = "pickled_features/"+method
if not os.path.isdir(save_to):
    os.makedirs(save_to)
if k!= 'NA':
    joblib.dump(feature_list_dict, save_to+'/'+method+'_top_'+str(k)+'_features.pkl')
else:
    joblib.dump(feature_list_dict, save_to+'/'+method+'_top_features.pkl')