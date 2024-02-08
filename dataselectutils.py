from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import linear_model, datasets
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import chi2,SelectFromModel
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

from arguments import data_files,test_folder,train_folder,project_folder,data_folder,fold_information_file
from configs import remove_diagnostic_features, diagnostic_features

import warnings
warnings.filterwarnings("ignore")
data_files = data_files

project_folder = project_folder
data_folder = data_folder
train_folder =train_folder
test_folder =test_folder

repeat_flag = 'Y'
hyperparameter_catalog_RFE = {

    'RF': {
        'bootstrap': [True],
        'max_depth': [10], # maximum depth of the tree
        'max_features': ['log2'], # maximum number of features to use at each split
        # 'min_samples_leaf': [5,10], # minimum number of samples to split a node
        'min_samples_split': [8],#range(8,10,2),
        'n_estimators': [100], # number of trees
        'criterion' : ['gini','entropy']  # criterion for evaluating a split
    }
}

hyperparameter_catalog = {

    'RF': {
        'bootstrap': [True],
        'max_depth': [2, 5, 10], # maximum depth of the tree
        'max_features': ['log2','sqrt'], # maximum number of features to use at each split
        'min_samples_leaf': [5,10], # minimum number of samples to split a node
        'min_samples_split': range(2,10,2),
        'n_estimators': [100,200, 500], # number of trees
        'criterion' : ['gini','entropy']  # criterion for evaluating a split
    }
}
from scipy.stats import ks_2samp
def statistical_filter(df,X,y, corr_th,label_col,pt_col):
    #FOR PERFORMING STATISTICAL FEATURE SELECTION
    
    df_ks= pd.DataFrame(columns =('Features', 'KS score', 'p value'))
    def ks(feat1,feat2):
        # print(feat1,feat2)
        d,p_val = ks_2samp(feat1,feat2)
        return d,p_val
    
    cohort = df.groupby([label_col])
    cols = X.columns
    for feat in cols:
        d,p_val = ks(cohort.get_group(0)[feat],cohort.get_group(1)[feat])
        data_append = {'Features': [feat], 'KS score': [d], 'p value': [p_val]}
        df_ks = pd.concat([df_ks,pd.DataFrame(data=data_append)], ignore_index= True)
    df_ks_sorted = df_ks.sort_values('KS score', ascending=False)
    df_ks_sorted.reset_index(drop = True, inplace= True)
    cols_ML = df_ks_sorted[df_ks_sorted['p value']<=0.1]['Features'].values
    X_KS_filtered= X[cols_ML]


    correlated_features = pd.DataFrame(columns = ('Feature A', 'Feature B', 'Correlation values (-1 to +1)'))
    data_col =set()
    # data_file = 'Expert_abp_vent_large_data.csv'
    # X, y, df = get_dataset(data_file)
    correlation_matrix = X_KS_filtered.corr()
    for i in range(len(correlation_matrix.columns)):

        for j in range(i):
            #print(str(i) + '-'+ str(j))
            if abs(correlation_matrix.iloc[i, j]) > corr_th:
                colnameA = correlation_matrix.columns[i]
                colnameB = correlation_matrix.columns[j]
                indexA = np.where(cols_ML == colnameA)
                indexB = np.where(cols_ML == colnameB)
                if indexA < indexB:
                    
                    data_col.add(colnameB) ## to remove the feature with 
                else: 
                
                    data_col.add(colnameA)
                #data_col.add(colnameB)
                data = {'Feature A':[colnameA], 'Feature B': [colnameB], 'Correlation values (-1 to +1)':[correlation_matrix.iloc[i, j]] }
                correlated_features = pd.concat([correlated_features,pd.DataFrame(data=data)], ignore_index = True)

    colm_drop =list(data_col)
    df_corrFiltered = X_KS_filtered.drop(colm_drop, axis =1)
#     print('Number of reduced features: {}'.format(df_corrFiltered.shape[1]))
#    print(df_corrFiltered.columns)
    return df_corrFiltered.columns.tolist()#,correlated_features,df_ks_sorted,X_KS_filtered

def chi_square(df,X,y,with_corr=False,corr_th=0.9):
    chi_scores = chi2(X,y)
    d = {'Features': X.columns, 'pvalues': chi_scores[1]}
    df_ks = pd.DataFrame(d,index =[i for i in range(len(X.columns))])
    df_ks_sorted = df_ks.sort_values('pvalues', ascending=False)
    df_ks_sorted.reset_index(drop = True, inplace= True)
    cols_ML = df_ks_sorted[df_ks_sorted['p value']<=0.1]['Features'].values
    X_KS_filtered= X[cols_ML]
    if with_corr is False:
        return X_KS_filtered
    correlated_features = pd.DataFrame(columns = ('Feature A', 'Feature B', 'Correlation values (-1 to +1)'))
    data_col =set()
    # data_file = 'Expert_abp_vent_large_data.csv'
    # X, y, df = get_dataset(data_file)
    correlation_matrix = X_KS_filtered.corr()
    for i in range(len(correlation_matrix.columns)):

        for j in range(i):
            #print(str(i) + '-'+ str(j))
            if abs(correlation_matrix.iloc[i, j]) > corr_th:
                colnameA = correlation_matrix.columns[i]
                colnameB = correlation_matrix.columns[j]
                indexA = np.where(cols_ML == colnameA)
                indexB = np.where(cols_ML == colnameB)
                if indexA < indexB:
                    
                    data_col.add(colnameB) ## to remove the feature with 
                else: 
                
                    data_col.add(colnameA)
                #data_col.add(colnameB)
                data = {'Feature A':[colnameA], 'Feature B': [colnameB], 'Correlation values (-1 to +1)':[correlation_matrix.iloc[i, j]] }
                correlated_features = pd.concat([correlated_features,pd.DataFrame(data=data)], ignore_index = True)

    colm_drop =list(data_col)
    df_corrFiltered = X_KS_filtered.drop(colm_drop, axis =1)
#     print('Number of reduced features: {}'.format(df_corrFiltered.shape[1]))
#    print(df_corrFiltered.columns)
    return df_corrFiltered.columns.tolist()#,correlated_features,df_ks_sorted,X_KS_filtered

def Lasso(df,X,y):
    lsvc = LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=10).fit(X, y)
    # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    print(X.columns[model.get_support()],"number of features selected are : ",len(X.columns[model.get_support()]),sep="\n")
    return list(X.columns[model.get_support()])

def RandomForestFeatSelection(df,X,y):
    rf = RandomForestClassifier().fit(X, y)
    model = SelectFromModel(rf,threshold=0.25, prefit=True)
    feat_dict = {i : j for i,j in zip(X.columns,model.estimator.feature_importances_) }
    sortedFeats = sorted(feat_dict.keys(), key=lambda k: feat_dict[k], reverse=True)
    # print(feat_dict,lsn(feat_dict))
    top25th_value = int(len(feat_dict)*0.25)
    # print(top25th_value,sortedFeats[:top25th_value], max(zip(feat_dict.values(), feat_dict.keys()))[1],feat_dict[sortedFeats[:top25th_value][0]])
    # exit()
    # print(sortedFeats[:top25th_value])
    # print(X.columns[model.get_support()])
    return sortedFeats[:top25th_value]#list(X.columns[model.get_support()])

def mutual_info(X,y,data, k):
    # Top k features according to mutual information score
    # file_num is number of split, k is the number of features to select

    input_type_parameter = 'ABP'

    if input_type_parameter=='ABP':
        ignore_fields = ['Pig', 'batch', 'ppv', 'binary_class', 'dataset', 'Pigs', 'id', 'Unnamed: 0',
                         'median_beats_mean_cvp', 'std_beats_mean_cvp']
    else:
        ignore_fields = ['Pig', 'batch', 'ppv', 'binary_class', 'dataset', 'Pigs', 'id', 'Unnamed: 0']


    
    relevant_cols = [col for col in X.columns.tolist() if col not in ignore_fields]
    print(len(relevant_cols))

    mi_dict = {}
    rel_data = X[relevant_cols]
    
    mi = mutual_info_classif(X, y)
    #mi
    print(mi,len(mi))
    # round(mutual_info_regression(x, y)[0],2)
    for idx,col in enumerate(relevant_cols):
        print(idx,col)
        if col not in ['label', 'index']:
            mi_dict[col] = round(mi[idx],2)
    import operator
    sorted_d = dict( sorted(mi_dict.items(), key=operator.itemgetter(1),reverse=True))
    #sorted_d


    #int(len(sorted_d))
    #mi_top25pct_cols = int(len(sorted_d))
    #top10features = list(sorted_d.keys())[:10]
    #top20features = list(sorted_d.keys())[:20]
    #top30features = list(sorted_d.keys())[:30]
    #allfeatures = list(sorted_d.keys())
    top_k_features = list(sorted_d.keys())[:k]
    
    #feature_set_list = [top10features,top20features,top30features,allfeatures]
    return top_k_features



def permutation_importance_features(data_folder, file_num, k, hyperparams='RF'):


    all_folds_perf = []
    retained_features = []

    All_file_pickle_folder = data_folder+'models_fixed_kf_pi/model_'+str(file_num+1) #data_folder+'models/model_'+str(file_num+1) # change folder name as required

    input_folder ='' # change folder name as required


    #'pl'+str(file_num+1)+'_train_clean.csv' ## Put the right cohort file
    Corr_threshold = 'Not applicable' ## select the correct cross-correlation threshold in the featurization step that optimized the performance


    output_file_string = data_folder.split('.csv')[0]
    print(output_file_string)
    if not os.path.isdir(All_file_pickle_folder):
        #print(1)
        os.makedirs(All_file_pickle_folder)


    param_grid = hyperparameter_catalog[hyperparams]
    hyperparameters = {'rf_model__' + key: param_grid[key] for key in param_grid}

    #scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True)}

    scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
                      'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True)}

    # rf_model=RandomForestClassifier(random_state= 42)

    # def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
    # def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
    # def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
    # def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

    fpr_total =[]
    tpr_total =[]
    precision_total =[]
    recall_total =[]

    finalResult = pd.DataFrame()
    print(data_folder)
    # scoring_outercv= {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
    #                  'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True),\
    #                   'tp': make_scorer(tp), 'tn': make_scorer(tn),'fp': make_scorer(fp), \
    #                   'fn': make_scorer(fn) , 'specificity': make_scorer(specificity)}
    X,y,df_dataset, cv = get_dataset(data_folder,file_num)
    #print(X.columns)


    #filtered_col_list.append(X.columns.tolist())



    column_mean =['accuracy (%)', 'roc_auc','precision', 'sensitivity','specificity','prc_auc']
    column_CI = list(r + ' 95% CI' for r in column_mean)

    #exp_pipe = Pipeline([ ('scaler', StandardScaler()),('logreg', LogisticRegression(max_iter= 5000))])
    column_list =[]

    # correlation_list = np.round([np.arange(0.5, 1.05, 0.1)],1)
    # correlation_list = list(correlation_list[0]) + ['Not applicable']

    df_corr= pd.DataFrame()
    for correlation_th in [Corr_threshold]:
        print(correlation_th)
    #     X,y,df_dataset = get_dataset(os.path.join(input_folder,data_file))

        if correlation_th != 'Not applicable':
            correlation_th = np.round(correlation_th,1)
            X = statistical_filter(df_dataset,X,y, correlation_th)
        column_list.append(X.columns)

        filtered_col_list.append(X.columns.tolist())

    #     result = []
        No_features = X.shape[1]


        for i in range(1,2):
            #inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
            inner_cv = cv

            startTime = datetime.now()
            rf_model=RandomForestClassifier(random_state=i)

            noimb_pipeline = Pipeline([('rf_model', rf_model)])


            clf = GridSearchCV(noimb_pipeline, param_grid= hyperparameters, verbose =0,cv=inner_cv, scoring= scoring, refit = 'roc_auc', n_jobs=-1)

            clf.fit(X, y)
            print(clf.best_estimator_)
            fold_perf.append(clf.cv_results_)
            model_to_choose =clf.best_estimator_

            #print(str(i+1) + ' completed')


    result_test = permutation_importance(
        model_to_choose[0], X, y, n_repeats=20, random_state=42, n_jobs=2
    )

    sorted_importances_idx_test = result_test.importances_mean.argsort()
    importances_test = pd.DataFrame(
        result_test.importances[sorted_importances_idx_test].T,
        columns=X.columns[sorted_importances_idx_test],
    )

    t = importances_test.columns.tolist()
    t.reverse()
   #t1 = t[:10]
   #t2 = t[:20]
   #t3 = t[:30]
   #t4 = t
   #f_list = [t1,t2,t3,t4]
   #retained_features.append(f_list)
    top_k_features = t[:k]
    return top_k_features



def RFE_features(data_folder, file_num, k,label_col,pt_col, hyperparams = 'RF'):

    filtered_col_list = []
    fold_perf = []


    input_folder =''
    only_comm = 'n'

    All_file_pickle_folder = data_folder+'models_fixed_kf_RFE/model_'+str(file_num+1)


    output_file_string = data_folder.split('.csv')[0]
    if not os.path.isdir(All_file_pickle_folder):
        #print(1)
        os.makedirs(All_file_pickle_folder)


    param_grid = hyperparameter_catalog_RFE[hyperparams]
    hyperparameters = {'estimator__' + key: param_grid[key] for key in param_grid}

    #scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True)}

    scoring = {'roc_auc':make_scorer(roc_auc_score, needs_proba= True), 'precision': 'precision', 'recall': 'recall',\
                      'accuracy': 'accuracy','prc_auc': make_scorer(average_precision_score,needs_proba=True)}

    fpr_total =[]
    tpr_total =[]
    precision_total =[]
    recall_total =[]

    finalResult = pd.DataFrame()

    X,y,df_dataset, cv = get_dataset(data_folder,file_num,label_col,pt_col,)

    print(X.columns.tolist())

    column_mean =['accuracy (%)', 'roc_auc','precision', 'sensitivity','specificity','prc_auc']
    column_CI = list(r + ' 95% CI' for r in column_mean)
    column_list =[]


    for i in range(1,2):

        inner_cv = cv

        startTime = datetime.now()
        rf_model=RandomForestClassifier(random_state=i)

        #est = SVR(kernel="linear")
        selector = feature_selection.RFE(RandomForestClassifier(random_state=i), n_features_to_select=10)
        #clf = GridSearchCV(selector, param_grid={'estimator__C': [1, 10, 100]})
        #selector = feature_selection.RFE(rf_model, n_features_to_select=10)

        clf = GridSearchCV(selector, param_grid= hyperparameters, verbose =0,cv=inner_cv, scoring= scoring, refit = 'roc_auc', n_jobs=-1)

        clf.fit(X, y)
        print(clf.best_estimator_)
        rfe_features=list(X.columns[clf.best_estimator_.support_])
        return rfe_features


def get_dataset(data_file,file_num,label_col,pt_col):
    #Processing input file for ingestion into training functions


    
    df = pd.read_csv(data_file, index_col= None)

    print(data_file,"FOld info file is",os.path.join(project_folder,fold_information_file))

    fold_df = pd.read_csv(os.path.join(project_folder,fold_information_file)) #reading file specifying which pigs belong to which splits
    # print(file_num+1,fold_df[fold_df['filename']==data_file.split("/")[-1]],sep="\n\n")
    folds_for_current_split = fold_df[fold_df['filename']==data_file.split("/")[-1]]
    print("########")
    print(fold_df[fold_df['filename']==data_file.split("/")[-1]]['filename'],file_num,data_file.split("/")[-1],folds_for_current_split)
    print("########")
    fold_1_pigs = ast.literal_eval(folds_for_current_split['fold_1'].tolist()[0])
    fold_2_pigs = ast.literal_eval(folds_for_current_split['fold_2'].tolist()[0])
    fold_3_pigs = ast.literal_eval(folds_for_current_split['fold_3'].tolist()[0])
    fold_4_pigs = ast.literal_eval(folds_for_current_split['fold_4'].tolist()[0])
    fold_5_pigs = ast.literal_eval(folds_for_current_split['fold_5'].tolist()[0])

    all_fold_pigs = [fold_1_pigs,fold_2_pigs,fold_3_pigs,fold_4_pigs,fold_5_pigs]

    k_folds = []
    for fold in all_fold_pigs:

        train_pig_list = [f for f in all_fold_pigs if f!=fold]
        train_pigs = []
        for l in range(len(train_pig_list)):
            train_pigs+=train_pig_list[l]

        val_idxs = []
        train_idxs = []
        for p in range(len(fold)):
            idx = df.index[(df[pt_col] == fold[p])].tolist()
            val_idxs+=idx
        #print(val_idxs)
        for p in range(len(train_pigs)):
            idx = df.index[(df[pt_col] == train_pigs[p])].tolist()
            train_idxs+=idx
        #print(train_idxs)
        k_folds.append((np.array(train_idxs), np.array(val_idxs)))





    # binary_label = [1 if l >=15 else 0 for l in df['label'].values]
    # df['label']= binary_label
    if 'ppv' in data_file.lower():
        X = df[df.columns[0:53]]
    else:
        X = df[df.columns[:]]

    if 'std_beats_mean_cvp' in X.columns.tolist():
        X = X.drop(['std_beats_mean_cvp'], axis =1)
    if 'median_beats_mean_cvp' in X.columns.tolist():
        X = X.drop(['median_beats_mean_cvp'], axis =1)
    if 'Unnamed: 0' in X.columns.tolist():
        X = X.drop(['Unnamed: 0'], axis =1)
    if 'Unnamed: 0.1' in X.columns.tolist():
        X = X.drop(['Unnamed: 0.1'], axis =1)
    if pt_col in X.columns.tolist():
        X = X.drop([pt_col], axis =1)
    if label_col in X.columns.tolist():
        X = X.drop([label_col], axis =1)
    if 'ZIPCODE' in X.columns.tolist():
        X = X.drop(['ZIPCODE'], axis =1)

    y = df[label_col]
    if remove_diagnostic_features==True:
        X=X.drop(diagnostic_features,axis =1)


    if only_comm == 'y':
        #X = df[comm_feats_90]
        X = df[comm_feats_50]

    return X, y, df, k_folds

def get_test_dataset(data_file,label_col,pt_col):
    #Processing input file for ingestion into training functions


    
    df = pd.read_csv(data_file, index_col= None)



    if 'ppv' in data_file.lower():
        X = df[df.columns[0:53]]
    else:
        X = df[df.columns[:]]

    
    if 'Unnamed: 0.1' in X.columns.tolist():
        X = X.drop(['Unnamed: 0.1'], axis =1)
    if pt_col in X.columns.tolist():
        X = X.drop([pt_col], axis =1)
    if label_col in X.columns.tolist():
        X = X.drop([label_col], axis =1)
    if 'ZIPCODE' in X.columns.tolist():
        X = X.drop(['ZIPCODE'], axis =1)

    y = df[label_col]

    if only_comm == 'y':
        #X = df[comm_feats_90]
        X = df[comm_feats_50]

    return X, y, df


rp_list = [['n','n'], ['y', 'n'], ['n', 'y']]


# data_folder = 'bootstraps_sv2/'

filtered_col_list = []
fold_perf = []
only_comm = 'n'







