data_files = ['ehr12h_summary_imputed.csv']
project_folder = '/data0/ehrdata/'#'/data0/ehrdata2/'
time_window = '12h'
data_folder = project_folder + 'datafile/' + time_window
train_folder = project_folder + 'train/' + time_window
test_folder = project_folder + 'test/' +time_window
fold_information_flag = True
fold_information_file = 'fold_information.csv'#'Downsample 25 fold_information.csv'#'fold_information.csv'
#'Downsample 25 fold_information.csv'
# pickle_folder
label_col = 'ards_flag'
pt_col = 'deidentified_study_id'

ventDataFolder = project_folder#'/data/vwd-deidentified-data'
# ventDataFile = project_folder + '/ventDataFiles'
ventDataFiles_median = project_folder + 'ventDataFiles_median'

#for ehr data with aBG-  12h both and 48h cfs 40050true both have a design flaw if reading wrong fold info file

# Index(['I:E ratio_median', 'dyn_compliance_median', 'eTime_median',
#        'iTime_median', 'inst_RR_median', 'lab_pf_ratio_res_median',
#        'lab_pf_ratio_res_min', 'mean_flow_from_pef_median',
#        'minF_to_zero_median', 'pef_+0.16_to_zero_median', 'resist_median',
#        'sf97', 'sf_median', 'stat_compliance_median', 'tve:tvi ratio_median',
#        'deidentified_study_id', 'ards_flag'],
#       dtype='object')
# (base) bash-4.4$ 
# ['mean_flow_from_pef_median', 'inst_RR_median', 'minF_to_zero_median', 'pef_+0.16_to_zero_median', 'iTime_median', 'eTime_median', 'I:E ratio_median', 'dyn_compliance_median', 'tve:tvi ratio_median', 'stat_compliance_median', 'resist_median', 'sf_median', 'lab_pf_ratio_res_median', 'lab_pf_ratio_res_min', 'sf97']



