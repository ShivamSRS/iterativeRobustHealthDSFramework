#modify the feature method a t three places
expt_name = "CFS_400_50_True"
feature_selection_method = "CFS_400_50_True"
use_features='Y'
if feature_selection_method=='statistical_feature_selection':
    suffix_str=''
else:
    suffix_str='_10'

no_of_pts = 240  
prevalence =0.25
Unbalanced = True
Downsample_25 = False
  
  
feature_import_path ='pickled_features/{}/{}_top{}_features.pkl'.format(feature_selection_method,feature_selection_method,suffix_str)
algorithm = 'RF'
num_splits = 100
CFS_300_50_True = ["sf_median","sf_min","osi_max","sf97","respiratory_rate_mean","spo2_min","total_output_ml_slope","sf_max","bp_map_cuff_max","bp_cuff_diastolic_max","bp_cuff_systolic_max"]
CFS_300_50_Alt = CFS_300_50_True + ["sf_mean","osi_median","osi_stddev","fio2_mean","spo2_mean","osi_mean","osi97","fio2_median","osi_min","spo2_median","bp_map_a_line","sf_slope","lab_chloride_res_median","temperature_celsius_stddev","fio2_slope","fio2_max","respiratory_rate_max","fio2_min","respiratory_rate_min","bp_map_cuff_min","bp_cuff_systolic_min","total_output_ml_stddev","bp_cuff_diastolic_min","bp_cuff_systolic_mean","lab_neutrophil_abs_auto_res_max"]

CFS_400_50_True = ["sf_median","sf_min","respiratory_rate_mean","sf97"]
CFS_400_50_Alt = CFS_400_50_True + ["spo2_mean","osi97","fio2_mean","osi_median","osi_mean","sf_mean","osi_max","fio2_mean","spo2_min","bp_map_cuff_max","sf_max","bp_cuff_diastolic_max","bp_cuff_systolic_max","total_output_ml_slope"]


prefered_columns =CFS_300_50_Alt#[]#['Delta BMI', 'ACS_PCT_NO_WORK_NO_SCHL_16_19_ZC', 'Yes Induction', 'POS_DIST_TRAUMA_ZP', 'Y_ECG', 'ACS_PCT_OTH_LANG_ZC']



remove_diagnostic_features = True
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
