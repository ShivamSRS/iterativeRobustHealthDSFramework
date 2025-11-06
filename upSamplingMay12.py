import os
import ast
import random
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from dataselectutils import get_dataset
from VentWaveData import VentData

# === CONFIGURATION ===
project_folder = '/data0/May12/ehrdata2'
fold_csv       = os.path.join(project_folder, 'fold_information.csv')
pt_col         = 'deidentified_study_id'
label_col      = 'ards_flag'

# fixed pool of 168 fake IDs, same for every setting
FAKE_ID_START  = 241
FAKE_POOL_SIZE = 168
FAKE_ID_POOL   = list(range(FAKE_ID_START, FAKE_ID_START + FAKE_POOL_SIZE))

# the eight (modality, time-window) combos
modalities = {
    'both_summary': ['48h','12h','30h'],
    'ehr':          ['12h','30h','48h'],
    'vent_summary': ['12h','48h'],
    
}

# === STEP 1: load original fold definitions ===
folds_df = pd.read_csv(fold_csv)

# === STEP 2: compute synthetic-negative counts per split (ref = EHR/12h) ===
ref_folder  = os.path.join(project_folder, 'train', '12h')
n_synth_map = {}
for row in folds_df.itertuples(index=False):
    fn   = row.filename
    df_r = pd.read_csv(os.path.join(ref_folder, fn))
    df_r = df_r.loc[:, ~df_r.columns.str.startswith('Unnamed')]
    n_pos = int((df_r[label_col] == 1).sum())
    n_neg = int((df_r[label_col] == 0).sum())
    target_neg = 3 * n_pos           # 25% positives → 3×pos negatives
    n_synth    = max(0, target_neg - n_neg)
    n_synth_map[fn] = n_synth

# === HELPER: reproducible random subset of FAKE_ID_POOL per split ===
def get_fake_ids_for_split(filename):
    split_idx = int(filename.split('_')[-1].split('.')[0])
    n_synth   = n_synth_map[filename]
    if n_synth > FAKE_POOL_SIZE:
        raise RuntimeError(
            f"Split {split_idx} needs {n_synth} fake IDs, but pool size is {FAKE_POOL_SIZE}"
        )
    rng = random.Random(42 + split_idx)
    return rng.sample(FAKE_ID_POOL, n_synth)

# === STEP 3: build & write unified upsampled fold info ===
upsampled_rows = []
for row in folds_df.itertuples(index=False):
    new = row._asdict()
    fakes = get_fake_ids_for_split(new['filename'])
    for fcol in ['fold_1','fold_2','fold_3','fold_4','fold_5']:
        orig = ast.literal_eval(new[fcol])
        new[fcol] = orig + fakes
    upsampled_rows.append(new)
pd.DataFrame(upsampled_rows).to_csv(
    os.path.join(project_folder, 'fold_information_upsampled.csv'),
    index=False
)

# === HELPER: oversample NEG to hit 25% positives with SMOTE ===
def oversample_neg_to_25(df, n_synth):
    # report NaNs
    cols_with_na = df.columns[df.isna().any()].tolist()
    if cols_with_na:
        print("Warning: columns with NA:", cols_with_na)
        rows_with_na = df.index[df.isna().any(axis=1)].tolist()
        for idx in rows_with_na:
            missing = df.columns[df.loc[idx].isna()].tolist()
            print(f"  Row {idx} missing in {missing}")

    # split features & labels
    X = df.drop([pt_col, label_col], axis=1)
    y = df[label_col]

    X_imp   = X

    curr_neg = int((y == 0).sum())
    sm = SMOTE(sampling_strategy={0: curr_neg + n_synth}, random_state=42)
    X_res, y_res = sm.fit_resample(X_imp, y)
    return pd.DataFrame(X_res, columns=X.columns), y_res

# === STEP 4: regenerate train CSVs for all settings ===
for mod, windows in modalities.items():
    for tw in windows:
        train_dir = (os.path.join(project_folder, 'train', tw)
                     if tw != '48h' else os.path.join(project_folder, 'train'))
        out_dir   = os.path.join(project_folder, 'upsampled', mod, tw)
        os.makedirs(out_dir, exist_ok=True)

        for row in folds_df.itertuples(index=False):
            fn = row.filename

            # load raw train
            if mod == 'ehr':
                df_train = pd.read_csv(os.path.join(train_dir, fn))

            elif mod == 'vent_summary':
                obj = VentData(project_folder)
                _, _, df_train, _, _ = obj.get_train_test_file_summary(
                    os.path.join(train_dir, fn),
                    int(fn.split('_')[-1].split('.')[0]),
                    project_folder + 'ventDataFiles_median',
                    tw,
                    median_only=True
                )

            else:  # both_summary
                # --- load EHR split with label ---
                X_ehr, y_ehr, df_ehr, _, _, _ = get_dataset(
                    os.path.join(project_folder, 'train', tw, fn) if tw != '48h' else os.path.join(project_folder, 'train',fn),
                    int(fn.split('_')[-1].split('.')[0]),
                    label_col, pt_col,
                    give_pt=True
                )
                # --- load Vent split with label but drop it ---
                obj = VentData(project_folder)
                _, _, df_vent, _, _ = obj.get_train_test_file_summary(
                    os.path.join(train_dir, fn),
                    int(fn.split('_')[-1].split('.')[0]),
                    project_folder + 'ventDataFiles_median',
                    tw,
                    give_pt=True, median_only=True
                )
                if label_col in df_vent.columns:
                    df_vent = df_vent.drop(columns=[label_col])
                # merge, preserving df_ehr[label_col] only
                df_train = pd.merge(df_vent, df_ehr, on=pt_col)

            # clean & standardize
            df_train = df_train.loc[:, ~df_train.columns.str.startswith('Unnamed')]
            df_train[pt_col] = df_train[pt_col].astype(int)
            if 'lab_chloride_res_mean' in df_train.columns:
                df_train['lab_chloride_res_mean'] = df_train['lab_chloride_res_mean'].fillna(100)

            # oversample & rebuild
            n_synth      = n_synth_map[fn]
            X_res, y_res = oversample_neg_to_25(df_train, n_synth)
            fake_ids     = get_fake_ids_for_split(fn)
            all_ids      = np.concatenate([df_train[pt_col].to_numpy(), fake_ids]).astype(int)

            df_out               = pd.DataFrame(X_res, columns=X_res.columns)
            df_out[pt_col]       = all_ids
            df_out[label_col]    = y_res.values

            # reorder so pt_col,label_col first
            cols = [pt_col, label_col] + [c for c in df_out.columns if c not in [pt_col, label_col]]
            df_out = df_out[cols]

            df_out.to_csv(os.path.join(out_dir, fn), index=False)

# === STEP 5: analyze alignment & prevalence, emit Excel ===
alignment_records = []
prevalence_records = []

for row in folds_df.itertuples(index=False):
    fn = row.filename
    split_idx = int(fn.split('_')[-1].split('.')[0])

    setting_ids = {}
    for mod, windows in modalities.items():
        for tw in windows:
            setting = f"{mod}_{tw}"
            df = pd.read_csv(os.path.join(project_folder, 'upsampled', mod, tw, fn))
            df[pt_col] = df[pt_col].astype(int)
            ids = set(df[pt_col])
            setting_ids[setting] = ids
            prevalence_records.append({
                'split': split_idx,
                'setting': setting,
                'prevalence_percent': df[label_col].mean() * 100
            })

    common = set.intersection(*setting_ids.values())
    for setting, ids in setting_ids.items():
        non_overlap = sorted(ids - common)
        alignment_records.append({
            'split': split_idx,
            'setting': setting,
            'non_overlapping_ids': non_overlap,
            'count_non_overlapping': len(non_overlap)
        })

df_align = pd.DataFrame(alignment_records)
df_prev  = pd.DataFrame(prevalence_records)

analysis_path = os.path.join(project_folder, 'alignment_prevalence_analysis.xlsx')
with pd.ExcelWriter(analysis_path) as writer:
    df_align.to_excel(writer, sheet_name='non_overlap', index=False)
    df_prev.to_excel(writer, sheet_name='prevalence', index=False)

print(f"Analysis saved to {analysis_path}")
