data_files = ['ehr48h_summary_imputed.csv']
project_folder = '/data/srs/ehrdata_US/'
time_window = ''
data_folder = project_folder + 'datafile/' + time_window
train_folder = project_folder + 'train/' + time_window
test_folder = project_folder + 'test/' +time_window
fold_information_flag = False
fold_information_file = 'Downsample 25 fold_information.csv'
# pickle_folder
label_col = 'ards_flag'
pt_col = 'deidentified_study_id'

ventDataFolder = '/data/vwd-deidentified-data'
ventDataFile = project_folder + '/ventDataFiles'