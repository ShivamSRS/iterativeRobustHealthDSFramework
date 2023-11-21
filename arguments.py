data_files = ['ehr30h_summary_imputed.csv']
project_folder = '/data1/srsrai/ehrdata/'
time_window = '30h/'
data_folder = project_folder + 'datafile/' + time_window
train_folder = project_folder + 'train/' + time_window
test_folder = project_folder + 'test/' +time_window
fold_information_flag = True
fold_information_file = 'fold_information.csv'

label_col = 'ards_flag'
pt_col = 'deidentified_study_id'

ventDataFolder = '/data/vwd-deidentified-data'
ventDataFile = project_folder + '/ventDataFiles'