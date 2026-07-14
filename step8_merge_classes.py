import pandas as pd
import numpy as np
import os
import joblib
import warnings
import pickle
import gc
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 配置 ==========
project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
data_folder = os.path.join(raw_data_folder, 'hv_double_line_90kv_preprocessed_data')
label_path = os.path.join(raw_data_folder, 'hv_double_line_90kv_labels.csv')
code_folder = os.path.join(project_folder, '02_Code')

print("=" * 60)
print("加载全部原始波形数据 (9,022个样本) 并提取 时域 + 频域 特征")
print("=" * 60)

# ========== 1. 加载标签 ==========
labels_df = pd.read_csv(label_path)
labels_df['sample_id_num'] = labels_df['sample_id'].astype(int)
print(f"标签文件加载成功，共 {len(labels_df)} 条记录")

# ========== 2. 获取所有文件列表 ==========
all_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
print(f"找到 {len(all_files)} 个 .pkl 文件")

# 只处理有标签的样本（防止标签不匹配）
valid_ids = set(labels_df['sample_id_num'].values)
filtered_files = []
for f in all_files:
    try:
        sid = int(f.split('_')[0])
        if sid in valid_ids:
            filtered_files.append(f)
    except:
        continue

print(f"匹配到标签的样本: {len(filtered_files)} 个")

# ========== 3. 定义特征提取函数 ==========
def extract_features_from_df(df):
    """从单个样本提取时域 + 频域特征"""
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
    
    # ---- 时域特征 ----
    for i, col in enumerate(v_cols):
        vals = df[col].values
        features[f'voltage_L{i+1}_rms'] = np.sqrt(np.mean(vals**2))
        features[f'voltage_L{i+1}_peak'] = np.max(np.abs(vals))
        features[f'voltage_L{i+1}_std'] = np.std(vals)
        features[f'voltage_L{i+1}_max'] = np.max(vals)
        features[f'voltage_L{i+1}_min'] = np.min(vals)
        features[f'voltage_L{i+1}_peak_to_peak'] = np.max(vals) - np.min(vals)
    
    for i, col in enumerate(c_cols):
        vals = df[col].values
        features[f'current_L{i+1}_rms'] = np.sqrt(np.mean(vals**2))
        features[f'current_L{i+1}_peak'] = np.max(np.abs(vals))
        features[f'current_L{i+1}_std'] = np.std(vals)
        features[f'current_L{i+1}_max'] = np.max(vals)
        features[f'current_L{i+1}_min'] = np.min(vals)
        features[f'current_L{i+1}_peak_to_peak'] = np.max(vals) - np.min(vals)
    
    # ---- 频域特征（FFT）----
    for i, col in enumerate(v_cols[:3]):
        vals = df[col].values
        fft_vals = np.abs(np.fft.fft(vals))
        for k in range(1, 6):
            if k < len(fft_vals):
                features[f'voltage_L{i+1}_fft_{k}'] = fft_vals[k]
            else:
                features[f'voltage_L{i+1}_fft_{k}'] = 0
    
    for i, col in enumerate(c_cols[:3]):
        vals = df[col].values
        fft_vals = np.abs(np.fft.fft(vals))
        for k in range(1, 6):
            if k < len(fft_vals):
                features[f'current_L{i+1}_fft_{k}'] = fft_vals[k]
            else:
                features[f'current_L{i+1}_fft_{k}'] = 0
    
    # ---- 组合特征 ----
    v_rms = [features.get(f'voltage_L{i+1}_rms', 0) for i in range(3)]
    c_rms = [features.get(f'current_L{i+1}_rms', 0) for i in range(3)]
    features['voltage_imbalance'] = np.std(v_rms)
    features['current_imbalance'] = np.std(c_rms)
    
    for i in range(3):
        features[f'v_i_ratio_L{i+1}'] = v_rms[i] / (c_rms[i] + 1e-6)
        features[f'crest_factor_L{i+1}'] = features.get(f'voltage_L{i+1}_peak', 0) / (v_rms[i] + 1e-6)
    
    return features

# ========== 4. 逐个处理所有样本（内存安全） ==========
print("\n开始逐个处理样本（每次只加载一个文件，内存安全）...")

all_features = []
sample_ids = []
failed_count = 0

for i, file_name in enumerate(filtered_files):
    if i % 500 == 0:
        print(f"  处理进度: {i+1}/{len(filtered_files)}")
    
    try:
        # 提取样本ID
        sample_id = int(file_name.split('_')[0])
        
        # 读取单个 .pkl 文件
        with open(os.path.join(data_folder, file_name), 'rb') as f:
            df = pickle.load(f)
        
        # 提取特征
        features = extract_features_from_df(df)
        features['sample_id'] = sample_id
        all_features.append(features)
        sample_ids.append(sample_id)
        
        # 释放内存（重要！）
        del df
        gc.collect()
        
    except Exception as e:
        failed_count += 1
        if failed_count <= 5:
            print(f"  处理 {file_name} 时出错: {e}")
        continue

print(f"\n✅ 成功处理 {len(all_features)} 个样本")
print(f"❌ 失败 {failed_count} 个样本")

# ========== 5. 转为 DataFrame ==========
features_df = pd.DataFrame(all_features)
print(f"特征表形状: {features_df.shape}")
print(f"特征数量: {len(features_df.columns) - 1}")  # 减去 sample_id

# ========== 6. 合并标签 ==========
merged_df = features_df.merge(labels_df[['sample_id_num', 'fault_target']], 
                               left_on='sample_id', right_on='sample_id_num', 
                               how='inner')
print(f"合并后样本数: {len(merged_df)}")

# 合并类别
merged_df['fault_target_merged'] = merged_df['fault_target'].replace({
    'Line_2_3_a': 'Line_2_3',
    'Line_2_3_b': 'Line_2_3'
})

print("\n合并后的类别分布:")
print(merged_df['fault_target_merged'].value_counts())

# ========== 7. 准备训练数据 ==========
exclude_cols = ['sample_id', 'sample_id_num', 'fault_target', 'fault_target_merged']
feature_cols = [col for col in merged_df.columns if col not in exclude_cols]

X = merged_df[feature_cols]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(merged_df['fault_target_merged'])
class_names = label_encoder.classes_

print(f"\n最终特征数量: {len(feature_cols)}")

# ========== 8. 划分并训练 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n训练集: {len(X_train)} 个样本")
print(f"测试集: {len(X_test)} 个样本")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\n训练模型中...")
model.fit(X_train, y_train)
print("✅ 模型训练完成！")

# ========== 9. 评估 ==========
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "=" * 60)
print("随机森林 + 时域+频域特征 (全部数据 9,022样本) 评估")
print("=" * 60)
print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"加权精确率 (Precision): {precision:.4f}")
print(f"加权召回率 (Recall): {recall:.4f}")
print(f"加权F1分数 (F1-Score): {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(cm)

print("\n" + "=" * 60)
print("详细分类报告")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=class_names))

# ========== 10. 特征重要性 ==========
feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)
print("\nTop 10 特征:")
print(feature_importance.head(10))

# ========== 11. 保存模型 ==========
model_path = os.path.join(code_folder, 'fault_classifier_model.pkl')
joblib.dump(model, model_path)
print(f"\n✅ 模型已保存到: {model_path}")

feature_cols_path = os.path.join(code_folder, 'feature_columns.pkl')
joblib.dump(feature_cols, feature_cols_path)
print(f"✅ 特征列名已保存到: {feature_cols_path}")

label_encoder_path = os.path.join(code_folder, 'label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_path)
print(f"✅ 标签编码器已保存到: {label_encoder_path}")

# ========== 12. 混淆矩阵图 ==========
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Random Forest + FFT (All 9,022 Samples)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(code_folder, 'confusion_matrix_all_data.png'), dpi=300)
print("✅ 混淆矩阵图已保存为: confusion_matrix_all_data.png")
plt.show()

print("\n" + "=" * 60)
print("🎉 全部完成！")
print(f"   - 处理样本数: {len(all_features)}")
print(f"   - 特征数: {len(feature_cols)}")
print(f"   - 准确率: {accuracy*100:.2f}%")
print("=" * 60)