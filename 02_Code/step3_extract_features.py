import pickle
import pandas as pd
import numpy as np
import os

data_folder = 'D:\\Fault_Diagnosis_Project\\01_Raw_Data\\hv_double_line_90kv_preprocessed_data'
test_file = '1_sample_hv_double_line_90kv.pkl'

def extract_features_from_df(df):
    """
    从DataFrame里提取特征
    输入: df (包含 time_s, 电压列, 电流列)
    输出: 一个字典，包含所有特征值
    """
    features = {}
    
    voltage_cols = [col for col in df.columns if 'vol' in col.lower() and 'L1' in col]
    current_cols = [col for col in df.columns if 'cur' in col.lower() and 'L1' in col]
    
    if not voltage_cols:
        voltage_cols = [col for col in df.columns if 'vol' in col.lower()][:3]
    if not current_cols:
        current_cols = [col for col in df.columns if 'cur' in col.lower()][:3]
    
    print(f"找到电压列: {voltage_cols}")
    print(f"找到电流列: {current_cols}")
    
    for i, col in enumerate(voltage_cols[:3]):
        data = df[col].values
        features[f'voltage_L{i+1}_rms'] = np.sqrt(np.mean(data**2))
        features[f'voltage_L{i+1}_peak'] = np.max(np.abs(data))
        features[f'voltage_L{i+1}_std'] = np.std(data)
        features[f'voltage_L{i+1}_max'] = np.max(data)
        features[f'voltage_L{i+1}_min'] = np.min(data)
        features[f'voltage_L{i+1}_peak_to_peak'] = np.max(data) - np.min(data)
        zero_crossings = np.where(np.diff(np.sign(data)))[0]
        if len(zero_crossings) > 1 and len(df['time_s']) > 1:
            time_span = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
            features[f'voltage_L{i+1}_freq_est'] = len(zero_crossings) / (2 * time_span) if time_span > 0 else 0
        else:
            features[f'voltage_L{i+1}_freq_est'] = 0
    
    for i, col in enumerate(current_cols[:3]):
        data = df[col].values
        features[f'current_L{i+1}_rms'] = np.sqrt(np.mean(data**2))
        features[f'current_L{i+1}_peak'] = np.max(np.abs(data))
        features[f'current_L{i+1}_std'] = np.std(data)
        features[f'current_L{i+1}_max'] = np.max(data)
        features[f'current_L{i+1}_min'] = np.min(data)
        features[f'current_L{i+1}_peak_to_peak'] = np.max(data) - np.min(data)
    
    features['total_voltage_rms'] = np.sqrt(np.mean([features[f'voltage_L{i+1}_rms']**2 for i in range(min(3, len(voltage_cols)))]))
    features['total_current_rms'] = np.sqrt(np.mean([features[f'current_L{i+1}_rms']**2 for i in range(min(3, len(current_cols)))]))
    features['voltage_unbalance'] = np.std([features[f'voltage_L{i+1}_rms'] for i in range(min(3, len(voltage_cols)))])
    features['current_unbalance'] = np.std([features[f'current_L{i+1}_rms'] for i in range(min(3, len(current_cols)))])
    
    features['sample_id'] = test_file
    
    return features


print("=" * 50)
print(f"处理测试文件: {test_file}")
print("=" * 50)

pkl_path = os.path.join(data_folder, test_file)
with open(pkl_path, 'rb') as f:
    df = pickle.load(f)

features = extract_features_from_df(df)

print("\n提取到的特征:")
for key, value in features.items():
    print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

print(f"\n总共提取了 {len(features)} 个特征（含sample_id）")
print("✅ 特征提取测试成功！")
print("\n" + "=" * 50)
print("开始批量处理所有样本...")
print("=" * 50)

all_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
print(f"找到 {len(all_files)} 个 .pkl 文件")

files_to_process = all_files[:1000]  
print(f"本次处理前 {len(files_to_process)} 个文件")

all_features = []

for i, file_name in enumerate(files_to_process):
    if i % 50 == 0:
        print(f"正在处理第 {i+1}/{len(files_to_process)} 个文件: {file_name}")
    
    try:
        pkl_path = os.path.join(data_folder, file_name)
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)
        
        features = extract_features_from_df(df)
        features['sample_id'] = file_name  
        all_features.append(features)
    except Exception as e:
        print(f"  处理 {file_name} 时出错: {e}")
        continue

feature_df = pd.DataFrame(all_features)
print(f"\n✅ 成功处理了 {len(feature_df)} 个样本")

# 保存到 CSV
output_path = os.path.join('..', '01_Raw_Data', 'extracted_features.csv')
feature_df.to_csv(output_path, index=False)
print(f"✅ 特征表格已保存到: {output_path}")
print(f"表格形状: {feature_df.shape} (行数, 列数)")
print("\n前5行预览:")
print(feature_df.head())