import pickle
import os
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

print("=" * 60)
print("Step 25: RTE 真实数据评估")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
code_folder = os.path.join(project_folder, '02_Code')
rte_folder = os.path.join(project_folder, '01_Raw_Data', 'rte_preprocessed')

# 加载模型
model = joblib.load(os.path.join(code_folder, 'fault_classifier_model.pkl'))
scaler = joblib.load(os.path.join(code_folder, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(code_folder, 'label_encoder.pkl'))
feature_cols = joblib.load(os.path.join(code_folder, 'feature_columns.pkl'))

# 定义特征提取函数（与 app.py 一致）
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

# 测试前 100 个 RTE 样本
print("测试前 100 个 RTE 样本...")
predictions = []
confidences = []

for i in range(100):
    file_path = os.path.join(rte_folder, f'rte_sample_{i}.pkl')
    if not os.path.exists(file_path):
        break
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    
    features = extract_features_from_df(df)
    input_df = pd.DataFrame([features])
    input_df = input_df[feature_cols]
    input_scaled = scaler.transform(input_df.values)
    
    pred_encoded = model.predict(input_scaled)[0]
    pred = label_encoder.inverse_transform([pred_encoded])[0]
    proba = model.predict_proba(input_scaled)[0]
    
    predictions.append(pred)
    confidences.append(np.max(proba))

print(f"预测结果: {predictions[:10]}...")
print(f"置信度: {np.mean(confidences)*100:.2f}% (平均)")
print(f"最高置信度: {np.max(confidences)*100:.2f}%")
print(f"最低置信度: {np.min(confidences)*100:.2f}%")
print(f"\n类别分布: {pd.Series(predictions).value_counts().to_dict()}")
print("\n✅ RTE 数据评估完成")