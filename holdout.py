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

from dataselectutils import get_dataset,get_test_dataset
from dataselectutils import get_dataset,statistical_filter,mutual_info, RFE_features,permutation_importance_features
from arguments import time_window,data_files,test_folder,train_folder,project_folder,data_folder,label_col,pt_col

from configs import Unbalanced,Downsample_25,feature_selection_method,feature_import_path,algorithm,use_features,prefered_columns,use_prefered_cols


data_files = data_files

project_folder = project_folder
data_folder = data_folder
train_folder = train_folder
test_folder = test_folder
label_col,pt_col = label_col,pt_col

import warnings
warnings.filterwarnings("ignore")

validation = 'test'
from configs import num_splits, expt_name
num_files = num_splits
data_folder = ''
train_or_test = validation+'/'
results_path = 'results/'+time_window+"/"+ feature_selection_method+"/"
if Unbalanced is True:
    if Downsample_25 is True:
        results_path += "/Unbalanced_25_DS/"
    else:
        results_path += "/Unbalanced_25_US/"
if not os.path.isdir(results_path):
    print(results_path)
    if not os.path.isdir('results/'+time_window+"/"):
        os.makedirs('results/'+time_window+"/")
    if not os.path.isdir('results/'+time_window+"/"+feature_selection_method+"/"):
        os.makedirs('results/'+time_window+"/"+feature_selection_method+"/")
    
    if Unbalanced is True:
        print("will make result subfolders now")
        if not os.path.isdir(results_path):
            print("created the subfolder unbalanced 25 ds")
            os.makedirs(results_path)
        
# exit()
file_list = [f for f in listdir(test_folder) if isfile(join(test_folder, f))]

filtered_col_list = []
fold_perf = []

holdout_results = []
probas_fr_list = []
final_bootstrap_summary = []
results_df = pd.DataFrame()
calibration_df = pd.DataFrame()

algorithm = algorithm
import_feature_list = use_features # 'Y' to use saved features from feature selection code
if import_feature_list == 'Y':
    #both should be user input
    feature_selection_method = feature_selection_method
    feature_import_path = feature_import_path#'pickled_features/statistical_feature_selection/statistical_feature_selection_top_features.pkl'
else:
    input_type_parameter = 'ABP'
    include_ppv = False
    
    if input_type_parameter=='ABP':
        ignore_fields = ['Pig', 'batch', 'binary_class', 'dataset', 'Pigs', 'id', 'Unnamed: 0', 
                         'Unnamed: 1', 'median_beats_mean_cvp', 'std_beats_mean_cvp']
    else:
        ignore_fields = ['Pig', 'batch', 'binary_class', 'dataset', 'Pigs', 'id', 'Unnamed: 0', 
                         'Unnamed: 1']
    
    if include_ppv == False:
        ignore_fields.append('ppv')

# if import_feature_list == 'N':
#     feats = filtered_col_list[0]

for file_num in range(num_files):
    
    print(test_folder,file_list[file_num],feature_selection_method)
    data_file = file_list[file_num]
    X,y,test_df = get_test_dataset(os.path.join(test_folder,data_file),label_col,pt_col)
    if 'Unnamed: 0' in test_df.columns.tolist():
        test_df = test_df.drop(['Unnamed: 0'], axis =1)
    if 'Unnamed: 0.1' in test_df.columns.tolist():
        test_df = test_df.drop(['Unnamed: 0.1'], axis =1)
    

    if import_feature_list == 'Y':
        feature_dict = joblib.load(feature_import_path)
        # print(feature_dict)
        try:
            selected_features = feature_dict[data_file[:-4].replace('test','train')]
        except:
            print("probably key error passing all features instead")
            selected_features = X.columns
    else:
        selected_features = [col for col in X.columns.tolist() if col not in ignore_fields]
    pickle_folder = project_folder + 'algorithm_selection/' + expt_name + '/' + \
                        algorithm + '/' + feature_selection_method + \
                        '/model_' + data_file[:-4].replace('test','train') + '/'

    target =15
    if not os.path.isdir(pickle_folder):
        print(pickle_folder)
        os.makedirs(pickle_folder)
    print()
    try:
        saved_model = joblib.load(pickle_folder + 'classification_model_'+ data_file[:-4].replace('test','train')+'.pkl')
    except:
        print(pickle_folder)
        print("##############")
        print("ERROR FOR",data_file)
        print("###############")
        continue
    if selected_features==[]:
            print("no feature was selected, passing whole data instead")
            selected_features = X.columns
    if use_prefered_cols:
        selected_features = prefered_columns
    print(selected_features)
    if pt_col in X.columns.tolist():
        X = X.drop([pt_col], axis =1)
    if label_col in X.columns.tolist():
        X = X.drop([label_col], axis =1)
    X = X[selected_features]#[list(X.columns[:51]) + list(selected_features)]#[selected_features]

    print("######",X.shape,type(X),test_df.shape,selected_features)
    # exit()
    if Unbalanced is True:
        if Downsample_25 is True:
            print(len(y),len((y[y==1]).index),len((y[y==1]).index)/2,(y[y==1]).index)
            remove_n = 24 #int(len((y[y==1]).index)/2)
            print(remove_n,X.loc[y[y==1].index,:].index,y[y==1].index)
            drop_indices = np.random.choice(X.loc[y[y==1].index,:].index, remove_n, replace=False)
            X = X.drop(drop_indices).reset_index(drop=True)
            y = y.drop(drop_indices).reset_index(drop=True)
            test_df = test_df.drop(drop_indices).reset_index(drop=True)
            print(X.index, y.index)
            print(X.shape,y.shape,test_df.shape)
        else:
            counter = Counter(y)
            print(counter)
            print(X.shape,y.shape,test_df.shape)
            # transform the dataset
            oversample = SMOTE(sampling_strategy={0:108,1:36})
            X, y = oversample.fit_resample(X, y)
            # summarize the new class distribution
            counter = Counter(y)
            print(counter)
            print(X.shape,y.shape,test_df.shape)
            # scatter plot of examples by class label
            # for label, _ in counter.items():
            #     row_ix = where(y == label)[0]
            #     pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
            # pyplot.legend()
            # pyplot.show()
    
    # exit()
    
    Y_pred = saved_model.predict(X)
    print(Y_pred)
    probas_=saved_model.predict_proba(X)
    
    if algorithm=='RF' or algorithm=='XGB':
        feat_importances = pd.Series(saved_model['classification_model'].feature_importances_, index=X.columns)
        
        print("important features are ",feat_importances.nlargest(5))
    y_test = y#test_df[label_col].values
    # x_test = test_df.drop(label_col, axis=1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test, probas_[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), probas_.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    from itertools import cycle

    result_images_path = project_folder+'result_images/'+time_window+"/" + expt_name+"/"+algorithm+"/"+feature_selection_method
    if Unbalanced is True:
        if Downsample_25 is True:
            result_images_path += "/Unbalanced_25_DS/"
        else:
            result_images_path += "/Unbalanced_25_US/"

    if os.path.exists(result_images_path) is False:
        if os.path.exists(project_folder+"result_images/"+time_window+"/" + expt_name+"/") is False:
            os.mkdir(project_folder+"result_images/"+time_window+"/" + expt_name+"/")
        if os.path.exists(project_folder+"result_images/"+time_window+"/" + expt_name+"/"+algorithm) is False:
            os.mkdir(project_folder+"result_images/"+time_window+"/" + expt_name+"/"+algorithm)
        
        if os.path.exists(project_folder+"result_images/"+time_window+"/" + expt_name+"/"+algorithm+"/"+feature_selection_method) is False:
            os.mkdir(project_folder+"result_images/"+time_window+"/" + expt_name+"/"+algorithm+"/"+feature_selection_method)
        if Unbalanced is True:
            print("will make result images subfolders now")
            if os.path.isdir(result_images_path) is False:
                os.makedirs(result_images_path)
                print("made")
    
    plt.plot(
        fpr[1],
        tpr[1],
        color="aqua",
        lw=2,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(" Receiver operating characteristic")
    plt.legend(loc="lower right")

    # if not os.path.isdir(project_folder+"result_images/"+ expt_name+"/"):
    #     print(project_folder+"result_images/"+ expt_name+"/")
    #     os.makedirs(project_folder+"result_images/"+ expt_name+"/")

    plt.savefig(result_images_path+data_file[:-4]+'_roc'+'.png')
    plt.clf()
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, probas_[:, 1])
    # print("FPR",fpr,"TPR",tpr)
    # exit()

    from sklearn.calibration import calibration_curve
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    x_cali, y_cali=calibration_curve(y_test, probas_[:, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x_cali,y_cali)
    ref = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    ref.set_transform(transform)
    ax.add_line(ref)
    fig.suptitle('Calibration – Neptune.ai')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Fraction of positive')
    plt.legend()
    plt.savefig(result_images_path+data_file[:-4]+'_calibration.png')
    plt.clf()
    
    
    
    
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1]) 
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, probas_[:, 1])

    
    
    
    fp_iloc_list = []
    fn_iloc_list = []
    for i in range(len(y_test)):
        
        if (Y_pred[i] == 1) & (y_test[i] == 0):
            fp_iloc_list.append(i)
        if (Y_pred[i] == 0) & (y_test[i] == 1):
            fn_iloc_list.append(i)
    fp_x = X.iloc[fp_iloc_list]
    fn_x = X.iloc[fn_iloc_list]
    # boundary_fp_pct = round(((fp_x[(fp_x[label_col]==1) & (fp_x[label_col]==0)].shape[0]/x_test.shape[0])*100),2)
    # boundary_fn_pct = round(((fn_x[(fn_x['label']>10) & (fn_x[label_col]<20)].shape[0]/x_test.shape[0])*100),2)
    # boundary_all_errors_pct = round(((fp_x[(fp_x['label']>10) & (fp_x['label']<20)].shape[0]
    #                           + fn_x[(fn_x['label']>10) & (fn_x['label']<20)].shape[0])/x_test.shape[0])*100,2)
    
    
    tn = confusion_matrix(y_test, Y_pred)[0, 0]
    fp= confusion_matrix(y_test, Y_pred)[0, 1]
    fn= confusion_matrix(y_test, Y_pred)[1, 0]
    tp= confusion_matrix(y_test, Y_pred)[1, 1]
    print('TP: ' + str(tp))
    print('TN: ' + str(tn))
    print('FP: ' + str(fp))
    print('FN: ' + str(fn))
    from configs import prevalence
    prevalence = prevalence
    sensitivity =tp / (tp + fn)
    specificity = tn/(tn + fp)
    PPV = ( sensitivity * prevalence) / ( (sensitivity * prevalence) + ((1 - specificity) * (1 - prevalence)) )
    NPV = (specificity * (1 - prevalence)) / ((specificity * (1 - prevalence)) + ((1 - sensitivity) * prevalence))
    precision = tp/(tp+fp)
    recall=tp/(tp+fn)

    data = [[ data_file[:-4], target, 100*(accuracy_score(y_test, Y_pred)),  roc_auc_score(y_test, probas_[:, 1]),\
              precision, recall, sensitivity,specificity  , auc(lr_recall, lr_precision),PPV,NPV ]]
    
    holdout_results.append(data[0])
    print(data[0],results_df)
    df_temp = pd.DataFrame([data[0]], index=[0],\
                              columns=(['file','FR_threshold', 'Accuracy', 'AUROC', 
                                        'Precision', 'Recall','Sensitivity', 'Specificity', 'AUPRC','PPV','NPV']))
    results_df = pd.concat([results_df, df_temp])

    
    
    probas_fr = [probas_[i][1] for i in range(len(probas_))]
    probas_fr_list.append(probas_fr)
    fr_list = []
    probas_fr_buckets = {}
    for i in range(10):
        c = 0
        fr = 0
        for j in range(len(probas_fr)):
            if (probas_fr[j]*100>=i*10) and (probas_fr[j]*100<(i*10)+10):
                c+=1
                if y_test[j] == 1:
                    fr+=1
        probas_fr_buckets[str(i*10)+'-'+str((i*10)+10)+'%']=c
        fr_list.append(fr)
    
    
    temp_calibration_df = pd.DataFrame({'file': [data_file[:-4]]*10,
                                        'buckets': list(probas_fr_buckets.keys()),
                                        'counts': list(probas_fr_buckets.values()),
                                        'FR_count': fr_list,
                                        'FR_proportion': [round(fr_list[i]/list(probas_fr_buckets.values())[i],2)*100
                                                          if list(probas_fr_buckets.values())[i] !=0
                                                          else 0
                                                          for i in range(len(fr_list))]})
    calibration_df = pd.concat([calibration_df, temp_calibration_df])
    
    print('Finished processing file ' + str(file_num+1) + ', ' + str(round(100*file_num/num_files,2)) + '% complete')

if import_feature_list == 'N':
    results_df.to_csv(results_path + 'split_level_validation_results_all_feats_' + \
                      validation + '_set.csv', index=False)
    calibration_df.to_csv(results_path + 'calibration_all_feats_' + \
                          validation + '_set.csv', index=False)
    
    
else:
    results_df.to_csv(results_path + 'split_level_validation_results_' + feature_selection_method + '_' + \
                      validation + '_set.csv', index=False)
    calibration_df.to_csv(results_path + 'calibration_' + feature_selection_method + '_' + \
                          validation + '_set.csv', index=False)
    
classification_metrics = ['Accuracy', 'AUROC', 'Precision', 'Recall','Sensitivity', 'Specificity', 'AUPRC','PPV','NPV']
classification_metrics_ci = [m + ' 95% CI' for m in classification_metrics]

for metric in classification_metrics:
    avg_metric = round(sum(results_df[metric].tolist())/len(results_df[metric].tolist()), 2)
    final_bootstrap_summary.append(avg_metric)
for metric in classification_metrics:    
    metric_ci = round(1.96*np.std(results_df[metric].tolist())/np.sqrt(len(results_df[metric].tolist())),3)
    final_bootstrap_summary.append(metric_ci)
    
summary_df = pd.DataFrame([final_bootstrap_summary], 
                          columns = classification_metrics + classification_metrics_ci)
if import_feature_list == 'N':
    summary_df.to_excel(results_path + 'summarized_validation_results_all_feats_' + \
                      algorithm + '_set.xlsx', index=False)    
else:
    summary_df.to_excel(results_path + 'summarized_validation_results_' + feature_selection_method + '_' + \
                      algorithm + '_set.xlsx', index=False)
print('')
print('Storing results complete')