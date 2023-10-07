feature_selection_method = 'Lasso'
use_features='Y'
if feature_selection_method=='statistical_feature_selection':
    suffix_str=''
else:
    suffix_str='_10'
    
feature_import_path = 'pickled_features/{}/{}_top{}_features.pkl'.format(feature_selection_method,feature_selection_method,suffix_str)
algorithm = 'RF'
num_splits = 100
prefered_columns =[]#['Delta BMI', 'ACS_PCT_NO_WORK_NO_SCHL_16_19_ZC', 'Yes Induction', 'POS_DIST_TRAUMA_ZP', 'Y_ECG', 'ACS_PCT_OTH_LANG_ZC']


mutual_info_cols = ['POS_DIST_OBSTETRICS_ZP' ,'ACS_PCT_MEDICAID_ANY_BELOW64_ZC','ACS_PCT_ENGLISH_ZC','ACS_TOT_OWN_CHILD_BELOW17_ZC','ACS_TOT_POP_US_ABOVE1_ZC']
#mutual_info_cols2=['superimposed with SF w/ SF', 'chronic hypertension', 'POS_DIST_OBSTETRICS_ZP', 'N_HDP Diag', 'ACS_TOT_OWN_CHILD_BELOW17_ZC', 'Multiple', 'Y_ECG', 'ACS_PCT_GRADUATE_DGR_ZC', 'No Episode', 'N_ECG']
use_prefered_cols = False


"""'statistical_feature_selection': 1,
    'mutual_information': 2,
    'permutation_importance': 3,
    'RFE': 4,
    'Lasso' : 5,

    'RandomForestFeatSelection':7
}"""
