data_files = ['ehr30h_summary_imputed.csv']
project_folder = '/data0/ehrdata/'#'/data0/ehrdata2/'
time_window = '30h'
data_folder = project_folder + 'datafile/' + time_window
train_folder = project_folder + 'train/' + time_window
test_folder = project_folder + 'test/' +time_window
fold_information_flag = True
fold_information_file ='fold_information.csv'
#'Downsample 25 fold_information.csv'
# pickle_folder
label_col = 'ards_flag'
pt_col = 'deidentified_study_id'

ventDataFolder = project_folder#'/data/vwd-deidentified-data'
ventDataFile = project_folder + '/ventDataFiles'
ventDataFiles_median = project_folder + 'ventDataFiles_median'

#for ehr data with aBG-  12h both and 48h cfs 40050true both have a design flaw if reading wrong fold info file