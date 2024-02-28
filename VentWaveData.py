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

from arguments import ventDataFolder,ventDataFile
from configs import no_of_pts

class VentData:
    def __init__(self,fold_info_file):
        
        # print(self.fold_pts)

        
        # self.read_pickled_breath()
        # self.get_train_test_file(1)
        # exit()

        for pt in range(1,no_of_pts+1):
            print("processing patient ",pt)
            start=time.time()
            if os.path.exists(os.path.join(ventDataFile,str(pt))) is False:
                os.mkdir(os.path.join(ventDataFile,str(pt)))

            if os.path.exists(os.path.join(ventDataFile,str(pt),"patientid_{}_vwd_summary.xlsx".format(pt,pt))):
                statinfo = os.stat(os.path.join(ventDataFile,str(pt),"patientid_{}_vwd_summary.xlsx".format(pt,pt)))
                if statinfo.st_size>10560:
                    continue
                self.make_breath_vector_for_a_single_split(ventDataFolder,pt)#continue
            else:
                self.make_breath_vector_for_a_single_split(ventDataFolder,pt)
            end = time.time()
            print("Processing patient",pt,"took",time.strftime("%Hh%Mm%Ss", time.gmtime(end-start)),"time")
            print("completely processed patient",pt) 

        self.fold_infos = pd.read_csv(ventDataFolder+"/fold_information.csv")
        fold_names = ["fold_1","fold_2","fold_3","fold_4","fold_5"]
        self.fold_pts = {'EHR_train_{}.csv'.format(i):[] for i in range(1,len(self.fold_infos)+1)}
        print(self.fold_infos,self.fold_pts)
        for num_pt in range(1,len(self.fold_infos)+1):
            # print("nm pts",num_pt)
            for fold in fold_names:
                fold_1 = list(self.fold_infos[fold][num_pt-1].split(", "))
                fold_1[0] = fold_1[0][1:]
                fold_1[-1] = fold_1[-1][:-1]
                fold_1 = [ int(i) for i in fold_1]
                # print(fold_1)
                self.fold_pts['EHR_train_{}.csv'.format(num_pt)].append(fold_1)
        print(self.fold_pts,len(self.fold_pts['EHR_train_1.csv']))
    
    def make_breath_vector_for_a_single_split(self,ventDataFolder,num_pt):
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
        
            
        
        breath_meta.to_excel(os.path.join(ventDataFile,str(num_pt),"patientid_{}_vwd_summary.xlsx".format(num_pt,num_pt)))

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

        folds_for_current_split = fold_df[fold_df['split']==file_num+1]
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
        
        print("Test",test_patients,sep ="\n\n")

        print("Train",all_fold_pigs,sep ="\n\n")


        
        print(k_folds)

        return train_patients,test_patients

    def get_train_test_file(self,file_num,folder):

        train_patients,test_patients = self.get_patients_from_folds_for_file(file_num)
        
        inner_cv = train_patients

        train_dataset = pd.DataFrame()

        return train_dataset,inner_cv,test_patients



obj = VentData(ventDataFolder)