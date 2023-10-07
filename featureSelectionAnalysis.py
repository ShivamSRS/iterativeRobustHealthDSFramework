import pandas as pd
from configs import feature_selection_method
from arguments import train_folder,test_folder,pt_col,label_col
import joblib
import os
feature_folder = "pickled_features/"+feature_selection_method+"/"
print("Feature selectin is ",feature_selection_method)
if feature_selection_method =="statistical_feature_selection":
    feature_file = feature_folder+feature_selection_method+"_top_features.pkl"
else:
    feature_file = feature_folder+feature_selection_method+"_top_10_features.pkl"

featureSelectedDict = joblib.load(feature_file)

trainfile = train_folder + "EHR_train_1.csv"
testfile = test_folder + "EHR_test_1.csv"

forCols = pd.read_csv(trainfile)
forAnal= pd.read_csv(testfile)

print("in train file patients",forCols[pt_col].unique(),"dist of classes",forCols[label_col].value_counts())

print("in test file patients",forAnal[pt_col].unique(),"dist of classes",forAnal[label_col].value_counts())


# print(forCols.columns)
import numpy as np
zeroArr = np.zeros((1,186))
featSelectCount = pd.DataFrame(zeroArr,index=[0],columns = forCols.columns)
zeroArr2 = np.zeros((240,2))
# print(featSelectCount)

for i in featureSelectedDict.keys():
    for j in featureSelectedDict[i]:
        featSelectCount.loc[0,j] = featSelectCount.loc[0,j] + 1
    # print(featSelectCount[0,])
featSelectCount.to_excel(feature_folder+feature_selection_method+"_featAnalysis.xlsx")
# print(featSelectCount)

def visualize_ptDistributions(train_folder,test_folder):
    ptDistributionCount = pd.DataFrame(zeroArr2,index=[i for i in range(240)],columns=["pt_id","no of times in training set"])
    for files in os.listdir(train_folder):
        if files[-3:]=='csv':
            train_file = pd.read_csv(os.path.join(train_folder,files))
            ptDistributionCount.loc[train_file[pt_col],"no of times in training set"]+=1
    ptDistributionCount.to_excel("ptDistributionCount.xlsx")


# visualize_ptDistributions(train_folder,test_folder)
