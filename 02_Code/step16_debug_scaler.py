import joblib
import pandas as pd
import numpy as np
import pickle
import os

print("=" * 60)
print("完整诊断 scaler 和特征匹配问题")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
code_folder = os.path.join(project_folder, '02_Code')
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')

# 1. 加载所有文件
feature_cols = joblib.load(os.path.join(code_folder, 'feature_columns.pkl'))
scaler = joblib.load(os.path.join(code_folder, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(code_folder, 'label_encoder.pkl'))

print(f"\n1. feature_cols 数量: {len(feature_cols)}")
print(f"2. scaler.mean_ 数量: {scaler.mean_.shape[0]}")
print(f"3. scaler.scale_ 数量: {scaler.scale_.shape[0]}")
print(f"4. scaler.var_ 数量: {scaler.var_.shape[0] if hasattr(scaler, 'var_') else 'N/A'}")

# 2. 加载一个样本
sample_path = os.path.join(raw_data_folder, 'hv_double_line_90kv_preprocessed_data', '1_sample_hv_double_line_90kv.pkl')
with open(sample_path, 'rb') as f:
    df = pickle.load(f)

print(f"\n5. 样本数据列数: {len(df.columns)}")

# 3. 定义特征提取函数
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
        zero_crossings = np.where(np.diff(np.sign(vals)))[0]
        if len(zero_crossings) > 1:
            time_span = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
            features[f'voltage_L{i+1}_freq_est'] = float(len(zero_crossings) / (2 * time_span)) if time_span > 0 else 0.0
        else:
            features[f'voltage_L{i+1}_freq_est'] = 0.0
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
    features['total_voltage_rms'] = float(np.sqrt(np.mean([v**2 for v in v_rms])))
    features['total_current_rms'] = float(np.sqrt(np.mean([c**2 for c in c_rms])))
    features['voltage_unbalance'] = float(np.std(v_rms))
    features['current_unbalance'] = float(np.std(c_rms))
    for col in feature_cols:
        if col not in features:
            features[col] = 0.0
    return features

# 4. 提取特征
features = extract_features_from_df(df)
print(f"\n6. 提取的特征数量: {len(features)}")

# 5. 按 feature_cols 顺序排列
input_df = pd.DataFrame([features])
input_df = input_df[feature_cols]
print(f"7. input_df 形状: {input_df.shape}")

# 6. 尝试标准化
print("\n8. 尝试 scaler.transform()...")
try:
    result = scaler.transform(input_df.values)
    print(f"✅ 标准化成功！结果形状: {result.shape}")
except Exception as e:
    print(f"❌ 标准化失败: {e}")
    print(f"   input_df 列数: {input_df.shape[1]}")
    print(f"   scaler 期望列数: {scaler.mean_.shape[0]}")

print("\n" + "=" * 60)
print("诊断完成")