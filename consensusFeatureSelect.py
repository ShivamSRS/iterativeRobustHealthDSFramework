import pandas as pd
from configs import feature_selection_method
from arguments import train_folder,test_folder,pt_col,label_col
import joblib
import os
feature_selection_method = "RandomForestFeatSelection"

feature_folder = "pickled_features/"+feature_selection_method+"/"
print("Feature selectin is ",feature_selection_method)
if feature_selection_method =="RFE":
    feature_analysis_file = feature_folder+feature_selection_method+"_featAnalysis.xlsx"
else:
    feature_analysis_file = feature_folder+feature_selection_method+"_featAnalysis.xlsx"

featureSelectedDict = pd.read_excel(feature_analysis_file)

trainfile = train_folder + "EHR_train_1.csv"
testfile = test_folder + "EHR_test_1.csv"
threshold = 50
forCols = pd.read_csv(trainfile)
forAnal= pd.read_csv(testfile)

print("in train file patients",forCols[pt_col].unique(),"dist of classes",forCols[label_col].value_counts())

print("in test file patients",forAnal[pt_col].unique(),"dist of classes",forAnal[label_col].value_counts())


# print(forCols.columns)
import numpy as np
zeroArr = np.zeros((1,2))
featSelectCount = pd.DataFrame(zeroArr,index=[0],columns = ["Feature name","Count"])
zeroArr2 = np.zeros((240,2))
# print(featSelectCount)
featureSelectedDict = pd.read_excel(feature_analysis_file)



for i in featureSelectedDict.keys():
    # print(i)
    for j in featureSelectedDict[i]:
        # print(i,j)
        if j>threshold:
            print(i,j)
            temp = pd.DataFrame(np.array([i,j]).reshape(1,2),index=[0],columns = ["Feature name","Count"])
        # featSelectCount.loc[0,j] = featSelectCount.loc[0,j] + 1
            featSelectCount = pd.concat([featSelectCount,temp],ignore_index=True)
    # print(featSelectCount[0,:])
print(featSelectCount)
featSelectCount.to_excel(feature_folder+feature_selection_method+"_"+str(threshold)+"_featConsensus.xlsx")
# print(featSelectCount)




# visualize_ptDistributions(train_folder,test_folder)
