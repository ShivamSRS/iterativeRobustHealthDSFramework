#modify the feature method a t three places
expt_name = "12h"
feature_selection_method = "CFS_300_50_Alt"
use_features='Y'
if feature_selection_method=='statistical_feature_selection':
    suffix_str=''
else:
    suffix_str='_10'

no_of_pts = 240  
prevalence =0.25
Unbalanced = True
Downsample_25 = True
  
feature_import_path = 'pickled_features/{}/{}_top{}_features.pkl'.format(feature_selection_method,feature_selection_method,suffix_str)
algorithm = 'RF'
num_splits = 100

CFS_300_50_True = ["sf_median","lab_pf_ratio_res_median","lab_pf_ratio_res_min","sf97","osi_max","sf_min","respiratory_rate_mean","lab_pf_ratio_res_max","spo2_min","total_output_ml_slope","bp_cuff_diastolic_max","bp_map_cuff_max"]
CFS_300_50_Alt = CFS_300_50_True + ["sf_max","fio2_max","spo2_mean","lab_pf_ratio_res_stddev","respiratory_rate_max","fio2_min","bp_cuff_systolic_max","bp_map_cuff_min","bp_cuff_systolic_min","fio2_mean","osi_median","total_output_ml_stddev","bp_cuff_diastolic_min","lab_vbg_pco2_res","lab_neutrophil_abs_auto_res_max","lab_pf_ratio_res_slope","osi97","lab_pf_ratio_res_mean","sf_mean","spo2_mean","osi_stddev","osi_median","fio2_median","osi_min","bp_a_line_systolic","spo2_median","bp_a_line_diastolic","bp_map_a_line","lab_chloride_res_median","temperature_celsius_stddev"]

CFS_400_50_True = ["sf_median","lab_pf_ratio_res_median","lab_pf_ratio_res_min","sf97"]
CFS_400_50_Alt = CFS_400_50_True + ["osi_max","sf_min","respiratory_rate_mean","lab_pf_ratio_res_max","spo2_min","total_output_ml_slope","bp_cuff_diastolic_max","bp_map_cuff_max","fio2_mean","spo2_mean","lab_pf_ratio_res_mean","osi97"]

CFS_300_75_True = ["sf_min","lab_pf_ratio_res_min","lab_pf_ratio_res_max","sf_median","lab_pf_ratio_res_max"]
CFS_300_75_Alt = CFS_300_75_True + ["sf_max","fio2_max","spo2_mean","fio2_min","sf97","total_output_ml_slope","bp_map_cuff_max","bp_cuff_systolic_max","respiratory_rate_mean","osi_max","lab_pf_ratio_res_median","osi97","fio2_median"]

CFS_400_75_True = ["lab_pf_ratio_res_min"]
CFS_400_75_Alt = CFS_400_75_True + ["sf_min","spo2_mean","lab_pf_ratio_res_min","lab_pf_ratio_res_max","sf97","sf_median","spo2_min","lab_pf_ratio_res_median"]

prefered_columns =CFS_300_50_Alt#[]#['Delta BMI', 'ACS_PCT_NO_WORK_NO_SCHL_16_19_ZC', 'Yes Induction', 'POS_DIST_TRAUMA_ZP', 'Y_ECG', 'ACS_PCT_OTH_LANG_ZC']

use_prefered_cols = True



remove_diagnostic_features = False
diagnostic_features = ['lab_vbg_pco2_res','lab_abg_ph_res','lab_abg_po2_res', 'lab_abg_pco2_res_mean',	'lab_abg_pco2_res_median',	'lab_abg_pco2_res_min',	'lab_abg_pco2_res_max',	'lab_abg_pco2_res_stddev',	'lab_abg_pco2_res_slope',	'lab_pf_ratio_res_mean',	'lab_pf_ratio_res_median',	'lab_pf_ratio_res_min',	'lab_pf_ratio_res_max',	'lab_pf_ratio_res_stddev',	'lab_pf_ratio_res_slope']

mutual_info_cols = ['POS_DIST_OBSTETRICS_ZP' ,'ACS_PCT_MEDICAID_ANY_BELOW64_ZC','ACS_PCT_ENGLISH_ZC','ACS_TOT_OWN_CHILD_BELOW17_ZC','ACS_TOT_POP_US_ABOVE1_ZC']
#mutual_info_cols2=['superimposed with SF w/ SF', 'chronic hypertension', 'POS_DIST_OBSTETRICS_ZP', 'N_HDP Diag', 'ACS_TOT_OWN_CHILD_BELOW17_ZC', 'Multiple', 'Y_ECG', 'ACS_PCT_GRADUATE_DGR_ZC', 'No Episode', 'N_ECG']



"""'statistical_feature_selection': 1,
    'mutual_information': 2,
    'permutation_importance': 3,
    'RFE': 4,
    'Lasso' : 5,

    'RandomForestFeatSelection':7
}"""
