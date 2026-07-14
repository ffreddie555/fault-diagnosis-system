import pandas as pd
import numpy as np
import pickle
import joblib
import os

print("=" * 60)
print("诊断特征提取问题")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
code_folder = os.path.join(project_folder, '02_Code')
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')

feature_cols = joblib.load(os.path.join(code_folder, 'feature_columns.pkl'))
print(f"模型需要的特征数量: {len(feature_cols)}")

sample_path = os.path.join(raw_data_folder, 'hv_double_line_90kv_preprocessed_data', '1_sample_hv_double_line_90kv.pkl')
with open(sample_path, 'rb') as f:
    df = pickle.load(f)

def extract_features_from_df(df):
    features = {}
    all_vol_cols = [col for col in df.columns if 'vol' in col.lower()]
    all_cur_cols = [col for col in df.columns if 'cur' in col.lower()]
    v_cols = []
    for phase in ['L1', 'L2', 'L3']:
        for col in all_vol_cols:
            if phase in col:
                v_cols.append(col)
                break
    v_cols = v_cols[:3]
    c_cols = []
    for phase in ['L1', 'L2', 'L3']:
        for col in all_cur_cols:
            if phase in col:
                c_cols.append(col)
                break
    c_cols = c_cols[:3]
    for i, col in enumerate(v_cols):
        vals = df[col].values
        features[f'voltage_L{i+1}_rms'] = float(np.sqrt(np.mean(vals**2)))
        features[f'voltage_L{i+1}_peak'] = float(np.max(np.abs(vals)))
        features[f'voltage_L{i+1}_std'] = float(np.std(vals))
        features[f'voltage_L{i+1}_max'] = float(np.max(vals))
        features[f'voltage_L{i+1}_min'] = float(np.min(vals))
        features[f'voltage_L{i+1}_peak_to_peak'] = float(np.max(vals) - np.min(vals))
    for i, col in enumerate(c_cols):
        vals = df[col].values
        features[f'current_L{i+1}_rms'] = float(np.sqrt(np.mean(vals**2)))
        features[f'current_L{i+1}_peak'] = float(np.max(np.abs(vals)))
        features[f'current_L{i+1}_std'] = float(np.std(vals))
        features[f'current_L{i+1}_max'] = float(np.max(vals))
        features[f'current_L{i+1}_min'] = float(np.min(vals))
        features[f'current_L{i+1}_peak_to_peak'] = float(np.max(vals) - np.min(vals))
    v_rms = [features.get(f'voltage_L{i+1}_rms', 0) for i in range(3)]
    c_rms = [features.get(f'current_L{i+1}_rms', 0) for i in range(3)]
    features['voltage_imbalance'] = float(np.std(v_rms))
    features['current_imbalance'] = float(np.std(c_rms))
    for i in range(3):
        features[f'v_i_ratio_L{i+1}'] = float(v_rms[i] / (c_rms[i] + 1e-6))
        features[f'crest_factor_L{i+1}'] = float(features.get(f'voltage_L{i+1}_peak', 0) / (v_rms[i] + 1e-6))
    for col in feature_cols:
        if col not in features:
            features[col] = 0.0
    return features

features = extract_features_from_df(df)
print(f"实际提取的特征数量: {len(features)}")

missing = []
for col in feature_cols:
    if col not in features:
        missing.append(col)
if missing:
    print(f"缺失的特征: {missing[:5]}")
else:
    print("所有特征都已提取")

extra = []
for col in features:
    if col not in feature_cols:
        extra.append(col)
if extra:
    print(f"多余的特征: {extra[:5]}")
else:
    print("没有多余特征")

print("诊断完成")