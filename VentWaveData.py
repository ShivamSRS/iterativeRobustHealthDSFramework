import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from io import open
import os
import glob
from ventmap.breath_meta import get_file_breath_meta,get_file_experimental_breath_meta
import pickle
import joblib
import ast
import time


# filepath = "/data/vwd-deidentified-data/011/011-consolidate-vwd-2009-03-03-22-33-02.957.csv"
# breath_meta = get_file_breath_meta(filepath, to_data_frame=True,new_format=True)

from ventmap.raw_utils import extract_raw

from arguments import project_folder,ventDataFolder,ventDataFile,data_folder,data_files,pt_col,label_col, ventDataFiles_median
from configs import no_of_pts
from dataselectutils import get_dataset

class VentData:
    def __init__(self,fold_info_file):
        
        # print(self.fold_pts)

        
        # self.read_pickled_breath()
        # self.get_train_test_file(1)
        # exit()
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
            

        self.fold_infos = pd.read_csv(ventDataFolder+"/fold_information.csv")
        fold_names = ["fold_1","fold_2","fold_3","fold_4","fold_5"]
        self.fold_pts = {'EHR_train_{}.csv'.format(i):[] for i in range(1,len(self.fold_infos)+1)}
        # print(self.fold_infos,self.fold_pts)
        # for num_pt in range(1,len(self.fold_infos)+1):
        #     # print("nm pts",num_pt)
        #     for fold in fold_names:
        #         fold_1 = list(self.fold_infos[fold][num_pt-1].split(", "))
        #         fold_1[0] = fold_1[0][1:]
        #         fold_1[-1] = fold_1[-1][:-1]
        #         fold_1 = [ int(i) for i in fold_1]
        #         # print(fold_1)
        #         self.fold_pts['EHR_train_{}.csv'.format(num_pt)].append(fold_1)
        # print(self.fold_pts,len(self.fold_pts['EHR_train_1.csv']))
    
    
    def analyse_breath_missing_report(self,time_window='48h'):
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

    def get_patients_from_folds_for_file(self,file_num):
        fold_df = self.fold_infos #reading file specifying which pigs belong to which splits

        folds_for_current_split = fold_df[fold_df['split']==file_num]
        # print("foldsss",folds_for_current_split,folds_for_current_split['fold_1'],"foldss")
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

    def get_train_test_file(self,file_num,folder,time_window='',train = True):
        """
        pass time window as 6h or 24h as a string
        """
        train_patients,test_patients = self.get_patients_from_folds_for_file(file_num)
        # print(train_patients,"train",test_patients,"test")
        if train == True:
            train_dataset = pd.DataFrame()
            train_filepaths = []
            for fold in train_patients:
                for idx,patientID in enumerate(fold):
                    train_filepaths.append(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    tempFile = pd.read_csv(os.path.join(ventDataFiles_median,str(patientID),"patientid_{}_vwd_summary.csv".format(patientID)))
                    if time_window =='' or time_window=='48h':
                        tempFile = tempFile[tempFile['BE']<86400]
                    elif time_window =='12h' or time_window=='30h':
                        tempFile = tempFile[tempFile['BE']<21600]
                    else:
                        tempFile = tempFile[tempFile['BE']<int(time_window[:-1])*3600]
                    # print(tempFile.columns)
                    
                    tempFile = tempFile[[pt_col,label_col]+self.vent_features]
                    train_dataset = pd.concat([train_dataset,tempFile],axis=0)
                    # master+=1
                    # print(master,ptid,"inside train datasets",patients)
            # print(train_filepaths,train_dataset)
            # print(train_dataset.isna().sum())
            train_dataset=train_dataset.dropna().reset_index(drop=True)

            k_folds = []

            for fold_idx,fold in enumerate(train_patients):


                CV_train_pts_list = [f for f_idx,f in enumerate(train_patients) if f_idx!=fold_idx]
                # print("CV",CV_train_pts_list,len(CV_train_pts_list),fold,sep="\n####")
                # break
                train_pigs = []
                for l in range(len(CV_train_pts_list)):
                    train_pigs+=list(CV_train_pts_list[l])

                # print("train pigs",train_pigs)

                val_idxs = []
                train_idxs = []
                val_idxs = train_dataset[train_dataset[pt_col].isin(fold)].index.tolist()
                train_idxs = train_dataset[train_dataset[pt_col].isin(train_pigs)].index.tolist()
                # val_idxs = train_dataset.index.tolist()
                # print("val dataframe,",val_slice)
                # print("train indices",train_dataset.loc[train_idxs[0],:],"rrrr patirnt is",train_pigs[0])
                # for p in range(len(fold)):
                    
                #     idx = train_dataset.index[(train_dataset[pt_col] == fold[p])].tolist()
                #     print("p",fold[p])
                #     print("selected val indices are",idx,len(idx))
                #     print("Dataset for that val indice is",train_dataset.loc[idx,:])
                #     exit()
                #     val_idxs+=idx
                #print(val_idxs)
                # exit()
                # for p in range(len(train_pigs)):
                #     idx = train_dataset.index[(train_dataset[pt_col] == train_pigs[p])].tolist()
                #     print("selected indices are",idx)
                #     print("Dataset for that indice is",train_dataset.loc[idx,:])
                #     exit()
                #     train_idxs+=idx
                # print("Length of trainnig and validation indices",len(train_idxs),len(val_idxs))
                k_folds.append((np.array(train_idxs), np.array(val_idxs)))

            # print("K folds",k_folds)


            columns_to_include = [col for col in train_dataset.columns if col not in [pt_col,label_col]]
            # for x,y in k_folds:
                # print(x,len(x),"x",y,len(y),"y",sep='\n###\n')
                # print(x[2000],y[2000],train_dataset.loc[x[2000],:],sep="^^^^^^^^^")
            X,y = train_dataset.loc[:,columns_to_include],train_dataset[label_col] 
            # print("x",X)
            return X,y,train_dataset,k_folds



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
                    # master+=1
                    # print(master,ptid,"inside train datasets",patients)
            # print(test_filepaths,test_dataset)
            # print(train_dataset.isna().sum())
            test_dataset=test_dataset.dropna().reset_index(drop=True)
            columns_to_include = [col for col in train_dataset.columns if col not in [pt_col,label_col]]
            # for x,y in k_folds:
                # print(x,len(x),"x",y,len(y),"y",sep='\n###\n')
                # print(x[2000],y[2000],train_dataset.loc[x[2000],:],sep="^^^^^^^^^")
            X,y = test_dataset.loc[:,columns_to_include],test_dataset[label_col] 
            print("x",X)
            return X,y,test_dataset

        

        



obj = VentData(ventDataFolder)
# obj.analyse_breath_missing_report()
obj.get_train_test_file(1,ventDataFiles_median)
