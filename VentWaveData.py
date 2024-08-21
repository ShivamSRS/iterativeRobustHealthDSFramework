import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from io import open
from scipy import stats
import os
import glob
from ventmap.breath_meta import get_file_breath_meta,get_file_experimental_breath_meta
import pickle
import joblib
import ast
import time


import pandas as pd
import os
from datetime import datetime, timedelta

from ventmap.raw_utils import extract_raw

from arguments import project_folder,ventDataFolder,data_folder,data_files,pt_col,label_col, ventDataFiles_median,fold_information_file
from configs import no_of_pts
from dataselectutils import get_dataset

class VentData:
    def __init__(self,fold_info_file):
        
        
        self.vent_features = ['mean_flow_from_pef','inst_RR','minF_to_zero','pef_+0.16_to_zero','iTime','eTime','I:E ratio','dyn_compliance','tve:tvi ratio','stat_compliance','resist']
        # for pt in range(1,no_of_pts+1):
        #     print("processing patient ",pt)
        #     start=time.time()
        #     if os.path.exists(os.path.join(ventDataFile,str(pt))) is False:
        #         os.mkdir(os.path.join(ventDataFile,str(pt)))

        #     if os.path.exists(os.path.join(ventDataFile,str(pt),"patientid_{}_vwd_summary.xlsx".format(pt,pt))):
        #         continue
        #         try:
        #             print("opening patient excel file",pt)
        #             data = pd.read_excel(os.path.join(ventDataFile,str(pt),"patientid_{}_vwd_summary.xlsx".format(pt,pt)))
        #             continue
        #         except:
        #             print("failed opening",pt,"excel file, trying csv file")
        #             try:
        #                 data = pd.read_csv(os.path.join(ventDataFile,str(pt),"patientid_{}_vwd_summary.csv".format(pt,pt)))
        #                 print("opened csv file instead")
        #                 continue
        #             except: 
        #                 print("failed opening",pt,"csv file, making csv file")   
        #                 self.make_breath_vector_for_a_single_split(ventDataFolder,pt,tocsv=True)
        #                 data = pd.read_csv(os.path.join(ventDataFile,str(pt),"patientid_{}_vwd_summary.csv".format(pt,pt)))

        #         ehr = pd.read_csv(data_folder+data_files[0])
                
        #         data.rename({'Patient Id': 'deidentified_study_id'}, axis=1, inplace=True)
        #         print("flag is",ehr[(ehr[pt_col] == pt)][label_col])
        #         if label_col in data.columns:
        #             data=data.drop(label_col,axis=1)
        #         # data.insert(1,label_col,ehr[(ehr[pt_col] == pt)][label_col])
        #         data[label_col] = pd.Series([int(ehr[(ehr[pt_col] == pt)][label_col].astype(int).values) for x in range(len(data.index))])
        #         if 'Unnamed: 0' in data.columns:
        #             data=data.drop('Unnamed: 0',axis=1)
        #         if 'Unnamed: 0.1' in data.columns:
        #             data=data.drop('Unnamed: 0.1',axis=1)
                
        #         print("saving median files")
        #         self.save_median_breath_vectors(pt,data)
        #         data.to_csv(os.path.join(ventDataFile,str(pt),"patientid_{}_vwd_summary.csv".format(pt,pt)),index=False)
                
        #             # exit()
        #             # continue
        #         # self.make_breath_vector_for_a_single_split(ventDataFolder,pt)#continue
        #     # else:
        #     #     self.make_breath_vector_for_a_single_split(ventDataFolder,pt)
        #     end = time.time()
        #     print("Processing patient",pt,"took",time.strftime("%Hh%Mm%Ss", time.gmtime(end-start)),"time")
        #     print("completely processed patient",pt)
            
        
        self.fold_infos = pd.read_csv(os.path.join(ventDataFolder,fold_information_file))
        print("INSIDE VENT WAVE DATA fold info",fold_information_file,os.path.join(ventDataFolder,fold_information_file))

        fold_names = ["fold_1","fold_2","fold_3","fold_4","fold_5"]
        self.fold_pts = {'EHR_train_{}.csv'.format(i):[] for i in range(1,len(self.fold_infos)+1)}
        
    
    
    def analyse_breath_missing_report(self,time_window):
        missingReport = pd.DataFrame(columns=['deidentified_study_id','mean_flow_from_pef','inst_RR','minF_to_zero','pef_+0.16_to_zero','iTime','eTime','I:E ratio','dyn_compliance','tve:tvi ratio','stat_compliance','resist'])
        for i in range(1,240+1):
            filepath = os.path.join(ventDataFiles_median,str(i),"patientid_{}_vwd_summary.csv".format(i))
            data = pd.read_csv(filepath)
            data = data[['mean_flow_from_pef','inst_RR','minF_to_zero','pef_+0.16_to_zero','iTime','eTime','I:E ratio','dyn_compliance','tve:tvi ratio','stat_compliance','resist']]
            print("File number",i)
            # print(data.isna().sum()/len(data))
            missingROW = pd.DataFrame(100*(data.isna().sum()/len(data)))
            studyIDDF = pd.DataFrame([i],columns=['deidentified_study_id'])
            missingROWFull = pd.concat([studyIDDF.T,missingROW])
            # print(studyIDDF,missi ngROWFull,"skjnfrhvn")
            missingReport = pd.concat([missingReport,missingROWFull.T],axis=0)
        print(missingReport,missingReport.columns)
        missingReport.to_excel("missing_report.xlsx") 
    
    def make_breath_vector_for_a_single_split(self,ventDataFolder,num_pt,tocsv=False):
        if num_pt//10==0:
            num_pt_formatted ='00'+str(num_pt)
        elif num_pt//100==0 and num_pt//10!=0:
            num_pt_formatted ='0'+str(num_pt)
        elif num_pt//1000==0 and num_pt//100!=0:
            num_pt_formatted =str(num_pt)
        print(num_pt_formatted)
        filepath = glob.glob(ventDataFolder+'/{}/*{}*consolidate*.csv'.format(num_pt_formatted,num_pt_formatted))#"/data/vwd-deidentified-data/{}/138-consolidate-vwd-2016-06-07-22-04-37.623.csv".format(num_pt)
        print(filepath)

        try:
            breath_meta = get_file_experimental_breath_meta(filepath[0], to_data_frame=True,new_format=True)
        except:
            breath_meta = get_file_experimental_breath_meta(filepath[0], ignore_missing_bes=False,to_data_frame=True,new_format=True)
        
            
        if tocsv==True:
            breath_meta.to_csv(os.path.join(ventDataFile,str(num_pt),"patientid_{}_vwd_summary.csv".format(num_pt,num_pt)))
        else:
            breath_meta.to_excel(os.path.join(ventDataFile,str(num_pt),"patientid_{}_vwd_summary.xlsx".format(num_pt,num_pt)))

    def save_median_breath_vectors(self,num_pt,data):
        newDf = pd.DataFrame(columns = data.columns)

        for i in range(0,len(data),100):
            # print(data[i:i+100],data[i:i+100].median())
            # toAdd = pd.DataFrame([data[i:i+100]].astype(str).astype(float).median(),index=[0],columns = data.columns)
            newDf  = pd.concat([newDf,data[i:i+100].median().to_frame().T],axis=0)
        print(newDf,"this is df")
        if os.path.exists(os.path.join(project_folder,'ventDataFiles_median',str(num_pt))) is False:
            os.mkdir(os.path.join(project_folder,'ventDataFiles_median',str(num_pt)))
        newDf.to_csv(os.path.join(project_folder,'ventDataFiles_median',str(num_pt),"patientid_{}_vwd_summary.csv".format(num_pt,num_pt)),index=False)
        # exit()
        
    def process_dataset(self):
        pt=5
        # = get_dataset(data_file,file_num,label_col,pt_col)
        df = pd.read_excel(os.path.join(ventDataFile,str(pt),"patientid_{}_vwd_summary.xlsx".format(pt,pt)))
        idx = df.index[(df[pt_col] == 5)].tolist() 
        print("indeinces are",idx)
    
    
    def read_pickled_breath(self,median=True):
        folder = '/data/vwd_processed/138_processed_1.pkl'
        file = open(folder,'rb')
        peek = pickle.load(file)
        print(peek)
        peek = pd.DataFrame(peek)
        print(peek)
        print(peek.columns)
        print(peek.loc[0,'exp_feature'])
        print("MEDIAN IS",peek.median())
        # peek.to_excel("138processed.xlsx")
        return peek.median()

    def get_patients_from_folds_for_file(self,data_file,file_num,train_flag=True):
        fold_df = self.fold_infos #reading file specifying which pigs belong to which splits
        print("THIS IS VVVENT WAVE PATIENT FOLD INFO FILE, FILE NUM IS ",file_num)
        
        if train_flag==True:
            folds_for_current_split =  fold_df[fold_df['filename']==data_file.split("/")[-1]]
        else:
            folds_for_current_split =  fold_df[fold_df['filename']==data_file.split("/")[-1].replace('test','train')]

        # print(train_flag)
        print("foldsss",folds_for_current_split,folds_for_current_split['fold_1'],"foldss")
        
        # exit()
        fold_1_pigs = ast.literal_eval(folds_for_current_split['fold_1'].tolist()[0])
        fold_2_pigs = ast.literal_eval(folds_for_current_split['fold_2'].tolist()[0])
        fold_3_pigs = ast.literal_eval(folds_for_current_split['fold_3'].tolist()[0])
        fold_4_pigs = ast.literal_eval(folds_for_current_split['fold_4'].tolist()[0])
        fold_5_pigs = ast.literal_eval(folds_for_current_split['fold_5'].tolist()[0])
        

        all_pts = [i for i in range(1,241)]

        

        all_fold_pigs = [np.array(fold_1_pigs),np.array(fold_2_pigs),np.array(fold_3_pigs),np.array(fold_4_pigs),np.array(fold_5_pigs)]
        
        test_patients = []

        for i in all_pts:
            if i not in fold_1_pigs:
                if i not in fold_2_pigs:
                    if i not in fold_3_pigs:
                        if i not in fold_4_pigs:
                            if i not in fold_5_pigs:
                                test_patients.append(i)
        
        
        train_patients = all_fold_pigs
        print("Test patients",len(test_patients),test_patients,sep ="\n\n")

        print("Train patients",len(all_fold_pigs),len(all_fold_pigs[0]),len(all_fold_pigs[1]),len(all_fold_pigs[2]),len(all_fold_pigs[3]),len(all_fold_pigs[4]),all_fold_pigs,sep ="\n\n")

        train_patients = all_fold_pigs
        
        # print(k_folds)

        return train_patients,test_patients

    def get_train_test_file(self,filename,file_num,folder,time_window ='',train = True,give_pt=False):
        """
        pass time window as 6h or 24h as a string
        """
        train_patients,test_patients = self.get_patients_from_folds_for_file(filename,file_num,train)
        # print(train_patients,"train",test_patients,"test")
        if train == True:
            train_dataset = pd.DataFrame()
            train_filepaths = []
            for fold in train_patients:
                for idx,patientID in enumerate(fold):
                    train_filepaths.append(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    tempFile = pd.read_csv(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    if time_window =='' or time_window=='48h':
                        # print("time iwndo is 48h")
                        tempFile = tempFile[tempFile['BE']<86400]
                        # print(len(tempFile))
                    elif time_window =='12h' or time_window=='30h':
                        # print("time iwndo is ",time_window)
                        tempFile = tempFile[tempFile['BE']<21600]
                        # print(len(tempFile))
                    else:
                        # print("why is this coming here?")
                        tempFile = tempFile[tempFile['BE']<int(time_window[:-1])*3600]
                        # print(len(tempFile))
                    # print(tempFile.columns)
                    # exit()
                    tempFile = tempFile[[pt_col,label_col]+self.vent_features]
                    train_dataset = pd.concat([train_dataset,tempFile],axis=0)
                    
            train_dataset=train_dataset.dropna().reset_index(drop=True)
            k_folds=[]            
            for fold_idx,fold in enumerate(train_patients):


                CV_train_pts_list = [f for f_idx,f in enumerate(train_patients) if f_idx!=fold_idx]
                print("CV",CV_train_pts_list,len(CV_train_pts_list),fold,sep="\n####")
                # exit()
                # break
                train_pigs = []
                for l in range(len(CV_train_pts_list)):
                    train_pigs+=list(CV_train_pts_list[l])

                # print("train pigs",train_pigs)

                val_idxs = []
                train_idxs = []
                val_idxs = train_dataset[train_dataset[pt_col].isin(fold)].index.tolist()
                train_idxs = train_dataset[train_dataset[pt_col].isin(train_pigs)].index.tolist()

                k_folds.append((np.array(train_idxs), np.array(val_idxs)))
            
            # print("K folds",k_folds)

            if give_pt==False:
                columns_to_include = [col for col in train_dataset.columns if col not in [pt_col,label_col]]
            else:
                columns_to_include = [col for col in train_dataset.columns if col not in [label_col]]
            # for x,y in k_folds:
                # print(x,len(x),"x",y,len(y),"y",sep='\n###\n')
                # print(x[2000],y[2000],train_dataset.loc[x[2000],:],sep="^^^^^^^^^")
            X,y = train_dataset.loc[:,columns_to_include],train_dataset[label_col] 
            # print("x",X)
            # exit()
            return X,y,train_dataset,k_folds,train_pigs



        else:
            test_dataset = pd.DataFrame()
            test_filepaths = []
            for patientID in test_patients:
            
                test_filepaths.append(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                tempFile = pd.read_csv(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                if time_window =='' or time_window=='48h':
                    tempFile = tempFile[tempFile['BE']<86400]
                elif time_window =='12h' or time_window=='30h':
                    tempFile = tempFile[tempFile['BE']<21600]
                else:
                    tempFile = tempFile[tempFile['BE']<int(time_window[:-1])*3600]
                # print(tempFile.columns)
                
                tempFile = tempFile[[pt_col,label_col]+self.vent_features]
                test_dataset = pd.concat([test_dataset,tempFile],axis=0)
                
            test_dataset=test_dataset.dropna().reset_index(drop=True)
            if give_pt==False:
                columns_to_include = [col for col in test_dataset.columns if col not in [pt_col,label_col]]
            else:
                columns_to_include = [col for col in test_dataset.columns if col not in [label_col]]
            X,y = test_dataset.loc[:,columns_to_include],test_dataset[label_col] 
            print("x",X)
            return X,y,test_dataset,test_patients
    
    def get_train_test_file_summary(self,filename,file_num,folder,time_window='',train = True,give_pt=False,median_only=False):
        """
        pass time window as 6h or 24h as a string
        """
        train_patients,test_patients = self.get_patients_from_folds_for_file(filename,file_num,train)
        # print(train_patients,"train",test_patients,"test")
        if train == True:
            train_dataset = pd.DataFrame()
            train_filepaths = []
            for fold in train_patients:
                # print(len(fold),len(train_patients))
                
                for idx,patientID in enumerate(fold):
                    # print(idx,patientID)
                    train_filepaths.append(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    tempFile = pd.read_csv(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    
                    if time_window =='' or time_window=='48h':
                        # print("time iwndo is 48h")
                        tempFile = tempFile[tempFile['BE']<86400]
                        # print(len(tempFile))
                    elif time_window =='12h' or time_window=='30h':
                        # print("time iwndo is ",time_window)
                        tempFile = tempFile[tempFile['BE']<21600]
                        # print(len(tempFile))
                    else:
                        # print("why is this coming here?")
                        tempFile = tempFile[tempFile['BE']<int(time_window[:-1])*3600]
                        # print(len(tempFile))
                    # print(tempFile.columns)
                    # exit()
                    
                    tempFile = tempFile[[pt_col,label_col,'rel_time_at_BS']+self.vent_features]
                    new_values=[]

                    df =tempFile
                    if median_only==False:
                        statistics = pd.DataFrame({
                            'mean': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).mean(),
                            'median': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).median(),
                            'min': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).min(),
                            'max': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).max(),
                            'stddev': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).std(),
                        })
                    else:
                        statistics = pd.DataFrame({
                            'median': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).median(),
                        })

                    
                    # Prepare a DataFrame to collect all statistics
                    all_stats = {pt_col:tempFile[pt_col][0],label_col:tempFile[label_col][0]}

                    if median_only==False:
                        # Function to calculate slopes for each column
                        def calculate_slope(y, x=df['rel_time_at_BS']):
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                            return slope

                        # Append slope to statistics
                        # print(df.drop(['rel_time_at_BS',pt_col,label_col], axis=1))
                        statistics['slope'] = df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).apply(calculate_slope)
                        # statistics = statistics.T
                        # print(statistics['slope'])
                        # exit()

                    
                    # Flatten statistics into a single row with concatenated names
                    for stat, series in statistics.items():

                        for col_name in series.index:
                            
                            all_stats[f'{col_name}_{stat}'] = series[col_name]

                    # Convert the dictionary to DataFrame
                    summary_df = pd.DataFrame([all_stats])
                   
                    
                    train_dataset = pd.concat([train_dataset,summary_df],axis=0,ignore_index=True)
            # print(train_dataset[pt_col])
            # exit()
            # train_dataset=train_dataset.dropna().reset_index(drop=True)

            k_folds = []
            # print(train_dataset.columns[train_dataset.isna().any()].tolist(),train_dataset.isna().sum())
            # print(len(train_dataset))
            for col in train_dataset.columns:
                if 'stddev' in col or 'slope' in col:
                    train_dataset[col].fillna(0, inplace=True)
            # exit()

            for fold_idx,fold in enumerate(train_patients):


                CV_train_pts_list = [f for f_idx,f in enumerate(train_patients) if f_idx!=fold_idx]
                
                train_pigs = []
                for l in range(len(CV_train_pts_list)):
                    train_pigs+=list(CV_train_pts_list[l])

                # print("train pigs",train_pigs)

                val_idxs = []
                train_idxs = []
                val_idxs = train_dataset[train_dataset[pt_col].isin(fold)].index.tolist()
                train_idxs = train_dataset[train_dataset[pt_col].isin(train_pigs)].index.tolist()
                k_folds.append((np.array(train_idxs), np.array(val_idxs)))

            # print("K folds",k_folds)

            if give_pt==False:
                columns_to_include = [col for col in train_dataset.columns if col not in [pt_col,label_col]]
            else:
                columns_to_include = [col for col in train_dataset.columns if col not in [label_col]]
            # train_dataset.sort_values(by=pt_col, inplace=True)
            X,y = train_dataset.loc[:,columns_to_include],train_dataset[label_col] 
            
            print(X,y,train_dataset,train_pigs)
            print("leaving vent wave")
            # exit()
            return X,y,train_dataset,k_folds,train_pigs



        else:
            test_dataset = pd.DataFrame()
            test_filepaths = []
            print("no of patients",len(test_patients))
            # exit()
            for idx,patientID  in enumerate(test_patients):
                # print("jnfvjfnjvnfsjvnfjvnf",test_patients,fold)
                # exit()
                # for idx,patientID in enumerate(fold):
                test_filepaths.append(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                tempFile = pd.read_csv(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                if time_window =='' or time_window=='48h':
                    tempFile = tempFile[tempFile['BE']<86400]
                elif time_window =='12h' or time_window=='30h':
                    tempFile = tempFile[tempFile['BE']<21600]
                else:
                    tempFile = tempFile[tempFile['BE']<int(time_window[:-1])*3600]
                # print(tempFile.columns)
                
                tempFile = tempFile[[pt_col,label_col,'rel_time_at_BS']+self.vent_features]

                df =tempFile

                if median_only==False:
                    statistics = pd.DataFrame({
                        'mean': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).mean(),
                        'median': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).median(),
                        'min': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).min(),
                        'max': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).max(),
                        'stddev': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).std(),
                    })
                else:
                    statistics = pd.DataFrame({
                        'median': df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).median(),
                    })

                # Prepare a DataFrame to collect all statistics
                all_stats = {pt_col:tempFile[pt_col][0],label_col:tempFile[label_col][0]}

                if median_only==False:
                    # Function to calculate slopes for each column
                    def calculate_slope(y, x=df['rel_time_at_BS']):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        return slope

                    # Append slope to statistics
                    print(df.drop(['rel_time_at_BS',pt_col,label_col], axis=1))
                    statistics['slope'] = df.drop(['rel_time_at_BS',pt_col,label_col], axis=1).apply(calculate_slope)
                    # statistics = statistics.T
                    # print(statistics['slope'])

                
                # Flatten statistics into a single row with concatenated names
                for stat, series in statistics.items():

                    for col_name in series.index:
                        
                        all_stats[f'{col_name}_{stat}'] = series[col_name]

                # Convert the dictionary to DataFrame
                summary_df = pd.DataFrame([all_stats])
                # print(summary_df)                    
                
                test_dataset = pd.concat([test_dataset,summary_df],axis=0)
                    
            # exit()
            # train_dataset=train_dataset.dropna().reset_index(drop=True)

            k_folds = []
            # print(train_dataset.columns[train_dataset.isna().any()].tolist(),train_dataset.isna().sum())
            # print(len(train_dataset))
            for col in test_dataset.columns:
                if 'stddev' in col or 'slope' in col:
                    test_dataset[col].fillna(0, inplace=True)
            if give_pt==False:
                columns_to_include = [col for col in test_dataset.columns if col not in [pt_col,label_col]]
            else:
                columns_to_include = [col for col in test_dataset.columns if col not in [label_col]]
            X,y = test_dataset.loc[:,columns_to_include],test_dataset[label_col] 
            print("x",X)
            return X,y,test_dataset, test_patients



    def get_oversample_vent_train_test_file(self,filename,file_num,folder,time_window='',train = True,give_pt=False):
        """
        pass time window as 6h or 24h as a string
        """
        train_patients,test_patients = self.get_patients_from_folds_for_file(filename,file_num)
        # print(train_patients,"train",test_patients,"test")
        if train == True:
            train_dataset = pd.DataFrame()
            train_filepaths = []
            for fold in train_patients:
                for idx,patientID in enumerate(fold):
                    train_filepaths.append(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    tempFile = pd.read_csv(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    if time_window =='' or time_window=='48h':
                        # print("time iwndo is 48h")
                        tempFile = tempFile[tempFile['BE']<86400]
                        # print(len(tempFile))
                    elif time_window =='12h' or time_window=='30h':
                        # print("time iwndo is ",time_window)
                        tempFile = tempFile[tempFile['BE']<21600]
                        # print(len(tempFile))
                    else:
                        # print("why is this coming here?")
                        tempFile = tempFile[tempFile['BE']<int(time_window[:-1])*3600]
                        # print(len(tempFile))
                    # print(tempFile.columns)
                    # exit()
                    tempFile = tempFile[[pt_col,label_col]+self.vent_features]
                    train_dataset = pd.concat([train_dataset,tempFile],axis=0)
                    
            train_dataset=train_dataset.dropna().reset_index(drop=True)
            print(train_patients)
            # exit()
            from imblearn.over_sampling import SMOTE
            
            k_folds = []
            x_new,y_new=[],[]
            idxs = []
            print(train_dataset[label_col].value_counts())
            # exit()
            for fold_idx,fold in enumerate(train_patients):


                CV_train_pts_list = [f for f_idx,f in enumerate(train_patients) if f_idx!=fold_idx]
                print("CV",CV_train_pts_list,len(CV_train_pts_list),fold,sep="\n####")
                # exit()
                # break
                train_pigs = []
                for l in range(len(CV_train_pts_list)):
                    train_pigs+=list(CV_train_pts_list[l])

                # print("train pigs",train_pigs)

                train_pt_data =train_dataset[train_dataset[pt_col].isin(train_pigs)]
                test_pt_data =  train_dataset[train_dataset[pt_col].isin(fold)]
                nan_mask = np.any(train_pt_data.isna(), axis=1)

                sm = SMOTE(sampling_strategy={0:3*len(train_pt_data[train_pt_data[label_col]==1]),1:len(train_pt_data[train_pt_data[label_col]==1])})
                # nan_mask = np.any(train_pt_data[feature_cols].isna(), axis=1)
                fold_x_res, fold_y_res = sm.fit_resample(train_pt_data[~nan_mask][self.vent_features], train_pt_data[~nan_mask][label_col])
                # set non feature cols to nan because theres no actual reference to real
                # world values with synthetic data
                # fold_x_res[non_feature_cols] = np.nan
                x_new.extend([fold_x_res, test_pt_data])
                y_new.extend([fold_y_res, test_pt_data[label_col]])

            
            cur_idx = 0
            for i in range(0, len(x_new), 2):
                x_tr_idx = pd.Index(range(cur_idx, cur_idx+len(x_new[i])))
                cur_idx += len(x_new[i])
                x_tst_idx = pd.Index(range(cur_idx, cur_idx+len(x_new[i+1])))
                cur_idx += len(x_new[i+1])
                idxs.append((x_tr_idx, x_tst_idx))
            train_dataset, y = pd.concat(x_new, ignore_index=True), pd.concat(y_new, ignore_index=True)
            print(y.value_counts(),train_dataset[pt_col].unique(),len(train_datasetx[pt_col].unique()))
            print(train_dataset.columns)
            
            
            # print("K folds",k_folds)

            if give_pt==False:
                columns_to_include = [col for col in train_dataset.columns if col not in [pt_col,label_col]]
            else:
                columns_to_include = [col for col in train_dataset.columns if col not in [label_col]]
            # for x,y in k_folds:
                # print(x,len(x),"x",y,len(y),"y",sep='\n###\n')
                # print(x[2000],y[2000],train_dataset.loc[x[2000],:],sep="^^^^^^^^^")
            X,y = train_dataset.loc[:,columns_to_include],y 
            # print("x",X)
            return X,y,train_dataset,idxs



        else:
            test_dataset = pd.DataFrame()
            test_filepaths = []
            for fold in test_patients:
                for idx,patientID in enumerate(fold):
                    test_filepaths.append(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    tempFile = pd.read_csv(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    if time_window =='' or time_window=='48h':
                        tempFile = tempFile[tempFile['BE']<86400]
                    elif time_window =='12h' or time_window=='30h':
                        tempFile = tempFile[tempFile['BE']<21600]
                    else:
                        tempFile = tempFile[tempFile['BE']<int(time_window[:-1])*3600]
                    # print(tempFile.columns)
                    
                    tempFile = tempFile[[pt_col,label_col]+self.vent_features]
                    test_dataset = pd.concat([test_dataset,tempFile],axis=0)
                    
            test_dataset=test_dataset.dropna().reset_index(drop=True)
            columns_to_include = [col for col in train_dataset.columns if col not in [pt_col,label_col]]
            X,y = test_dataset.loc[:,columns_to_include],test_dataset[label_col] 
            print("x",X)
            return X,y,test_dataset

    def get_oversample_vent_ehr_train_test_file(self,filename,file_num,folder,time_window='',train = True,give_pt=False):
        """
        pass time window as 6h or 24h as a string
        """
        from dataselectutils import get_dataset
        X,y,df_dataset, cv,train_patient_ids,test_patient_ids  = get_dataset(os.path.join(project_folder,"train",time_window, filename),file_num,label_col,pt_col,give_pt=True)

        train_patients,test_patients = self.get_patients_from_folds_for_file(filename,file_num)
        print(train_patients,"train","\n\n",train_patient_ids,"meow")
        # exit()
        if train == True:
            train_dataset = pd.DataFrame()
            train_filepaths = []
            for fold in train_patients:
                for idx,patientID in enumerate(fold):
                    train_filepaths.append(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    tempFile = pd.read_csv(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    
                    if time_window =='' or time_window=='48h':
                        # print("time iwndo is 48h")
                        tempFile = tempFile[tempFile['BE']<86400]
                        # print(len(tempFile))
                    elif time_window =='12h' or time_window=='30h':
                        # print("time iwndo is ",time_window)
                        tempFile = tempFile[tempFile['BE']<21600]
                        # print(len(tempFile))
                    else:
                        # print("why is this coming here?")
                        tempFile = tempFile[tempFile['BE']<int(time_window[:-1])*3600]
                        # print(len(tempFile))
                    # print(tempFile.columns)
                    # print(tempFile)
                    # exit()
                    tempFile = tempFile[[pt_col,label_col]+self.vent_features]
                    train_dataset = pd.concat([train_dataset,tempFile],axis=0)
                    
            train_dataset=train_dataset.dropna().reset_index(drop=True)
            from configs import prefered_columns
            selected_features = prefered_columns
            
            X = X[[pt_col]+selected_features]#[list(X.columns[:51]) + list(selected_features)]
            train_dataset = pd.merge(train_dataset, X, on=pt_col, how='left')
            train_dataset.to_excel("temp.xlsx")
            print(train_dataset.isna().sum())
            # exit()
            print("after merging ehr and vent data",X.shape)
            # y = venty
            print(train_dataset.columns, train_dataset[label_col].value_counts())
            # exit()
            print(y.shape,"sjsjmdaokp",X.shape)


            # print(train_patients)
            # exit()
            from imblearn.over_sampling import SMOTE
            
            k_folds = []
            x_new,y_new=[],[]
            idxs = []
            print(train_dataset[label_col].value_counts())
            # exit()
            for fold_idx,fold in enumerate(train_patients):


                CV_train_pts_list = [f for f_idx,f in enumerate(train_patients) if f_idx!=fold_idx]
                print("CV",CV_train_pts_list,len(CV_train_pts_list),fold,sep="\n####")
                # exit()
                # break
                train_pigs = []
                for l in range(len(CV_train_pts_list)):
                    train_pigs+=list(CV_train_pts_list[l])

                print("train pigs",train_pigs)
                
                
                train_pt_data =train_dataset[train_dataset[pt_col].isin(train_pigs)]
                test_pt_data =  train_dataset[train_dataset[pt_col].isin(fold)]
                ########################
                #for holdout set make sure u concatenate the patient IDs after this
                ######################

                
                sm = SMOTE(sampling_strategy={0:3*len(train_pt_data[train_pt_data[label_col]==1]),1:len(train_pt_data[train_pt_data[label_col]==1])})
                # nan_mask = np.any(train_pt_data[feature_cols].isna(), axis=1)
                nan_mask = np.any(train_pt_data.isna(), axis=1)
                selcted_cols = train_dataset.columns.difference(pd.Index([pt_col,label_col]))
                fold_x_res, fold_y_res = sm.fit_resample(train_pt_data[~nan_mask][selcted_cols], train_pt_data[~nan_mask][label_col])
                # set non feature cols to nan because theres no actual reference to real
                # world values with synthetic data
                # fold_x_res[non_feature_cols] = np.nan
                print(fold_x_res.isna().sum())
                
                # exit()
                
                
                y_new.extend([fold_y_res, test_pt_data[label_col]])
                
                
                x_new.extend([fold_x_res, test_pt_data])

            
            cur_idx = 0
            for i in range(0, len(x_new), 2):
                x_tr_idx = pd.Index(range(cur_idx, cur_idx+len(x_new[i])))
                cur_idx += len(x_new[i])
                x_tst_idx = pd.Index(range(cur_idx, cur_idx+len(x_new[i+1])))
                cur_idx += len(x_new[i+1])
                idxs.append((x_tr_idx, x_tst_idx))
            combined_dfx_list = []
            combined_dfy_list = []
            
            for fold_x,fold_y in zip(x_new,y_new):
                
                
                combined_dfx_list.append(fold_x)
                

                combined_dfy_list.append(fold_y)
                

            train_dataset, y = pd.concat(combined_dfx_list, ignore_index=True), pd.concat(combined_dfy_list, ignore_index=True)
            print("sep",train_dataset.isna().sum())
            # exit()
            print(y.value_counts(),train_dataset[pt_col].unique(),len(train_dataset[pt_col].unique()))
            print(train_dataset.columns)
            
            
            # print("K folds",k_folds)

            if give_pt==False:
                columns_to_include = [col for col in train_dataset.columns if col not in [pt_col,label_col]]
            else:
                columns_to_include = [col for col in train_dataset.columns if col not in [label_col]]
            # for x,y in k_folds:
                # print(x,len(x),"x",y,len(y),"y",sep='\n###\n')
                # print(x[2000],y[2000],train_dataset.loc[x[2000],:],sep="^^^^^^^^^")
            X,y = train_dataset.loc[:,columns_to_include] ,y
            # print("x",X)
            train_dataset.to_excel("temp.xlsx")
            return X,y,train_dataset,idxs



        else:
            test_dataset = pd.DataFrame()
            test_filepaths = []
            for fold in test_patients:
                for idx,patientID in enumerate(fold):
                    test_filepaths.append(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    tempFile = pd.read_csv(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    if time_window =='' or time_window=='48h':
                        tempFile = tempFile[tempFile['BE']<86400]
                    elif time_window =='12h' or time_window=='30h':
                        tempFile = tempFile[tempFile['BE']<21600]
                    else:
                        tempFile = tempFile[tempFile['BE']<int(time_window[:-1])*3600]
                    # print(tempFile.columns)
                    
                    tempFile = tempFile[[pt_col,label_col]+self.vent_features]
                    test_dataset = pd.concat([test_dataset,tempFile],axis=0)
                    
            test_dataset=test_dataset.dropna().reset_index(drop=True)
            columns_to_include = [col for col in train_dataset.columns if col not in [pt_col,label_col]]
            X,y = test_dataset.loc[:,columns_to_include],test_dataset[label_col] 
            print("x",X)
            return X,y,test_dataset

#     def get_oversample_vent_ehr_train_test_file(self,filename,file_num,folder,time_window='',train = True,give_pt=False):
#         """
#         pass time window as 6h or 24h as a string
#         """
#         from dataselectutils import get_dataset
#         X,y,df_dataset, cv,train_patient_ids,test_patient_ids  = get_dataset(os.path.join(project_folder,"train",time_window, filename),file_num,label_col,pt_col,give_pt=True)

#         train_patients,test_patients = self.get_patients_from_folds_for_file(filename,file_num)
#         print(train_patients,"train","\n\n",train_patient_ids,"meow")
#         # exit()
#         if train == True:
#             train_dataset = pd.DataFrame()
#             train_filepaths = []
#             for fold in train_patients:
#                 for idx,patientID in enumerate(fold):
#                     train_filepaths.append(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
#                     tempFile = pd.read_csv(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
#                     if time_window =='' or time_window=='48h':
#                         # print("time iwndo is 48h")
#                         tempFile = tempFile[tempFile['BE']<86400]
#                         # print(len(tempFile))
#                     elif time_window =='12h' or time_window=='30h':
#                         # print("time iwndo is ",time_window)
#                         tempFile = tempFile[tempFile['BE']<21600]
#                         # print(len(tempFile))
#                     else:
#                         # print("why is this coming here?")
#                         tempFile = tempFile[tempFile['BE']<int(time_window[:-1])*3600]
#                         # print(len(tempFile))
#                     # print(tempFile.columns)
#                     # exit()
#                     tempFile = tempFile[[pt_col,label_col]+self.vent_features]
#                     train_dataset = pd.concat([train_dataset,tempFile],axis=0)
                    
#             train_dataset=train_dataset.dropna().reset_index(drop=True)
#             from configs import prefered_columns
#             selected_features = prefered_columns
            
#             X = X[[pt_col]+selected_features]#[list(X.columns[:51]) + list(selected_features)]
#             train_dataset = pd.merge(train_dataset, X, on=pt_col, how='left')
#             train_dataset.to_excel("temp.xlsx")
#             print(train_dataset.isna().sum())
#             # exit()
#             print("after merging ehr and vent data",X.shape)
#             # y = venty
#             print(train_dataset.columns, train_dataset[label_col].value_counts())
#             # exit()
            


#             # print(train_patients)
#             # exit()
#             from imblearn.over_sampling import SMOTE
            
#             k_folds = []
#             x_new,y_new=[],[]
#             idxs = []
#             print(train_dataset[label_col].value_counts())
#             # exit()

#             sm = SMOTE(sampling_strategy={0:3*len(train_dataset[train_dataset[label_col]==1]),1:len(train_dataset[train_dataset[label_col]==1])})
#             # nan_mask = np.any(train_pt_data[feature_cols].isna(), axis=1)
#             nan_mask = np.any(train_dataset.isna(), axis=1)
#             train_x_res, train_y_res = sm.fit_resample(train_dataset[~nan_mask][:,[col for col in train_dataset.columns if col != label_col]], train_dataset[~nan_mask][label_col])

#             print(train_x_res.sum().isna(),train_y_res.value_counts())
#             exit()
#             for fold_idx,fold in enumerate(train_patients):


#                 CV_train_pts_list = [f for f_idx,f in enumerate(train_patients) if f_idx!=fold_idx]
#                 print("CV",CV_train_pts_list,len(CV_train_pts_list),fold,sep="\n####")
#                 # exit()
#                 # break
#                 train_pigs = []
#                 for l in range(len(CV_train_pts_list)):
#                     train_pigs+=list(CV_train_pts_list[l])

#                 print("train pigs",train_pigs)
                
                
#                 train_pt_data =train_dataset[train_dataset[pt_col].isin(train_pigs)]
#                 test_pt_data =  train_dataset[train_dataset[pt_col].isin(fold)]
                
#                 sm = SMOTE(sampling_strategy={0:3*len(train_pt_data[train_pt_data[label_col]==1]),1:len(train_pt_data[train_pt_data[label_col]==1])})
#                 # nan_mask = np.any(train_pt_data[feature_cols].isna(), axis=1)
#                 nan_mask = np.any(train_pt_data.isna(), axis=1)
#                 fold_x_res, fold_y_res = sm.fit_resample(train_pt_data[~nan_mask][self.vent_features], train_pt_data[~nan_mask][label_col])
#                 # set non feature cols to nan because theres no actual reference to real
#                 # world values with synthetic data
#                 # fold_x_res[non_feature_cols] = np.nan
                
#                 x_new.extend([fold_x_res, test_pt_data])
#                 y_new.extend([fold_y_res, test_pt_data[label_col]])

#             for fold_idx,fold in enumerate(train_patients):


#                 CV_train_pts_list = [f for f_idx,f in enumerate(train_patients) if f_idx!=fold_idx]
#                 print("CV",CV_train_pts_list,len(CV_train_pts_list),fold,sep="\n####")
#                 # exit()
#                 # break
#                 train_pigs = []
#                 for l in range(len(CV_train_pts_list)):
#                     train_pigs+=list(CV_train_pts_list[l])

#                 # print("train pigs",train_pigs)

#                 val_idxs = []
#                 train_idxs = []
#                 val_idxs = train_dataset[train_dataset[pt_col].isin(fold)].index.tolist()
#                 train_idxs = train_dataset[train_dataset[pt_col].isin(train_pigs)].index.tolist()

#                 k_folds.append((np.array(train_idxs), np.array(val_idxs)))

            
            

# # obj = VentData(ventDataFolder)
# # # obj.analyse_breath_missing_report()
# # obj.get_train_test_file(1,ventDataFiles_median)
