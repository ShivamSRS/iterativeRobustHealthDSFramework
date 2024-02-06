import pandas as pd
from configs import feature_selection_method
from arguments import train_folder,test_folder,pt_col,label_col
import joblib
import os


foldinfo = pd.read_csv("/data1/srsrai/ehrdata_DS/Downsample 25 fold_information.csv")
datafile = pd.read_csv("/data1/srsrai/ehrdata_DS/datafile/ehr48h_summary_imputed.csv")
print(foldinfo.columns,len(foldinfo))
new = pd.DataFrame([[-1 ]* len(foldinfo.columns)]*len(foldinfo),index = [i for i in range(100)],columns=foldinfo.columns)

print(new)

longlistARDSprevalence = []

for idx,i in enumerate(foldinfo.iterrows()):
    # print(i[1]['fold_1'])
    print("IIÎ",i[1]['filename'])
    # print(i[1]['fold_1'])
    new.loc[idx,'filename']=i[1]['filename']
    new.loc[idx,'split']=i[1]['split']
    # print(new)
    i[1]['fold_1']=list(map(str.strip, i[1]['fold_1'].strip('][').replace('"', '').split(',')))
    i[1]['fold_2']=list(map(str.strip, i[1]['fold_2'].strip('][').replace('"', '').split(',')))
    i[1]['fold_3']=list(map(str.strip, i[1]['fold_3'].strip('][').replace('"', '').split(',')))
    i[1]['fold_4']=list(map(str.strip, i[1]['fold_4'].strip('][').replace('"', '').split(',')))
    i[1]['fold_5']=list(map(str.strip, i[1]['fold_5'].strip('][').replace('"', '').split(',')))
    # print(i[1]['fold_1'])
    # continue
    prevalence_count = 0
    for patient in i[1]['fold_1']:
        # print(datafile.columns)
        
        patient = int(patient)
        # print(patient,len(i[1]['fold_1']))

        flag = int(datafile.loc[datafile['deidentified_study_id']==patient,'ards_flag'])
        # print(patient,flag,prevalence_count)
        prevalence_count+=flag
    # print(prevalence_count,len(i[1]['fold_1']),15/33)    
    # exit()
    
    new.loc[idx,'fold_1']=prevalence_count*100/len(i[1]['fold_1'])
    longlistARDSprevalence.append(prevalence_count*100/len(i[1]['fold_1']))
    prevalence_count = 0
    for patient in i[1]['fold_2']:
        # print(datafile.columns)
        patient = int(patient)
        # print(patient,len(i[1]['fold_2']))

        flag = int(datafile.loc[datafile['deidentified_study_id']==patient,'ards_flag'])
        prevalence_count+=flag
    new.loc[idx,'fold_2']=prevalence_count*100/len(i[1]['fold_2'])

    longlistARDSprevalence.append(prevalence_count*100/len(i[1]['fold_2']))
    prevalence_count = 0
    for patient in i[1]['fold_3']:
        # print(datafile.columns)
        patient = int(patient)
        # print(patient,len(i[1]['fold_3']))

        flag = int(datafile.loc[datafile['deidentified_study_id']==patient,'ards_flag'])
        prevalence_count+=flag
    new.loc[idx,'fold_3']=prevalence_count*100/len(i[1]['fold_3'])
    longlistARDSprevalence.append(prevalence_count*100/len(i[1]['fold_3']))
    prevalence_count = 0
    
    for patient in i[1]['fold_4']:
        # print(datafile.columns)
        patient = int(patient)
        # print(patient,len(i[1]['fold_4']))

        flag = int(datafile.loc[datafile['deidentified_study_id']==patient,'ards_flag'])
        prevalence_count+=flag
    new.loc[idx,'fold_4']=prevalence_count*100/len(i[1]['fold_4'])
    longlistARDSprevalence.append(prevalence_count*100/len(i[1]['fold_4']))
    prevalence_count = 0
    for patient in i[1]['fold_5']:
        # print(datafile.columns)
        patient = int(patient)
        # print(patient,len(i[1]['fold_5']))

        flag = int(datafile.loc[datafile['deidentified_study_id']==patient,'ards_flag'])
        prevalence_count+=flag
    
    new.loc[idx,'fold_5']=prevalence_count*100/(len(i[1]['fold_5']))
    longlistARDSprevalence.append(prevalence_count*100/len(i[1]['fold_5']))
    # break
print(new)
new.to_excel("Fold_wise_ARDSPrevalenceDistribution.xlsx")
exit()

# trainfile = train_folder + "EHR_train_1.csv"
# testfile = test_folder + "EHR_test_1.csv"
new = pd.read_excel("Fold_wise_ARDSPrevalenceDistribution.xlsx")
for idx,i in enumerate(new.iterrows()):
    # print(idx,i)
    longlistARDSprevalence.append(i[1]['fold_1'])
    longlistARDSprevalence.append(i[1]['fold_2'])
    longlistARDSprevalence.append(i[1]['fold_3'])
    longlistARDSprevalence.append(i[1]['fold_4'])
    longlistARDSprevalence.append(i[1]['fold_5'])


CFS_400_50_True = pd.read_excel("/data1/srsrai/ehrdata/algorithm_selection/CFS_400_50_True/RF/CFS_400_50_True/bestmodels_cv.xlsx")
CFS_400_50_True_ARDS_prevalence = []
for idx,i in enumerate(CFS_400_50_True.iterrows()):
    # print(idx,i)
    CFS_400_50_True_ARDS_prevalence.append(i[1]['split0_test_roc_auc'])
    CFS_400_50_True_ARDS_prevalence.append(i[1]['split1_test_roc_auc'])
    CFS_400_50_True_ARDS_prevalence.append(i[1]['split2_test_roc_auc'])
    CFS_400_50_True_ARDS_prevalence.append(i[1]['split3_test_roc_auc'])
    CFS_400_50_True_ARDS_prevalence.append(i[1]['split4_test_roc_auc'])


CFS_400_50_Alt = pd.read_excel("/data1/srsrai/ehrdata/algorithm_selection/CFS_400_50_Alt/RF/CFS_400_50_Alt/bestmodels_cv.xlsx")   

CFS_400_50_Alt_ARDS_prevalence = []
for idx,i in enumerate(CFS_400_50_Alt.iterrows()):
    # print(idx,i)
    CFS_400_50_Alt_ARDS_prevalence.append(i[1]['split0_test_roc_auc'])
    CFS_400_50_Alt_ARDS_prevalence.append(i[1]['split1_test_roc_auc'])
    CFS_400_50_Alt_ARDS_prevalence.append(i[1]['split2_test_roc_auc'])
    CFS_400_50_Alt_ARDS_prevalence.append(i[1]['split3_test_roc_auc'])
    CFS_400_50_Alt_ARDS_prevalence.append(i[1]['split4_test_roc_auc'])

CFS_300_50_True = pd.read_excel("/data1/srsrai/ehrdata/algorithm_selection/CFS_300_50_True/RF/CFS_300_50_True/bestmodels_cv.xlsx")   

CFS_300_50_True_ARDS_prevalence = []
for idx,i in enumerate(CFS_300_50_True.iterrows()):
    # print(idx,i)
    CFS_300_50_True_ARDS_prevalence.append(i[1]['split0_test_roc_auc'])
    CFS_300_50_True_ARDS_prevalence.append(i[1]['split1_test_roc_auc'])
    CFS_300_50_True_ARDS_prevalence.append(i[1]['split2_test_roc_auc'])
    CFS_300_50_True_ARDS_prevalence.append(i[1]['split3_test_roc_auc'])
    CFS_300_50_True_ARDS_prevalence.append(i[1]['split4_test_roc_auc'])


CFS_300_50_Alt = pd.read_excel("/data1/srsrai/ehrdata/algorithm_selection/CFS_300_50_Alt/RF/CFS_300_50_Alt/bestmodels_cv.xlsx")   

CFS_300_50_Alt_ARDS_prevalence = []
for idx,i in enumerate(CFS_300_50_Alt.iterrows()):
    # print(idx,i)
    CFS_300_50_Alt_ARDS_prevalence.append(i[1]['split0_test_roc_auc'])
    CFS_300_50_Alt_ARDS_prevalence.append(i[1]['split1_test_roc_auc'])
    CFS_300_50_Alt_ARDS_prevalence.append(i[1]['split2_test_roc_auc'])
    CFS_300_50_Alt_ARDS_prevalence.append(i[1]['split3_test_roc_auc'])
    CFS_300_50_Alt_ARDS_prevalence.append(i[1]['split4_test_roc_auc'])




import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()
colors = ['b', 'c', 'y', 'm', 'r']
sc1= plt.scatter(longlistARDSprevalence,CFS_400_50_True_ARDS_prevalence,marker='x', color=colors[0],label="CFS_400_50_True")
sc2 = plt.scatter(longlistARDSprevalence,CFS_400_50_Alt_ARDS_prevalence,marker='x', color=colors[1],label="CFS_400_50_Alt")
# sc3 =  plt.scatter(longlistARDSprevalence,CFS_300_50_True_ARDS_prevalence,marker='x', color=colors[2],label="CFS_300_50_True")
sc4 =  plt.scatter(longlistARDSprevalence,CFS_300_50_Alt_ARDS_prevalence,marker='x', color=colors[3],label="CFS_300_50_Alt")


plt.legend((sc1, sc2,sc4),
           ('CFS_400_50_True', 'CFS_400_50_Alt','CFS_300_50_Alt'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)


plt.title("title")
plt.xlabel("ARDS PREVALENCE RATIO")
plt.ylabel("ROC ACCURACY")
# plt.plot(, linestyle = 'dotted')
plt.savefig('kfoldanalysis'+'.png')