import pandas as pd
import numpy as np
import os
import joblib
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("XGBoost + SMOTE + GridSearchCV 全量训练 (9022 样本)")
print("=" * 70)

project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
code_folder = os.path.join(project_folder, '02_Code')
data_folder = os.path.join(raw_data_folder, 'hv_double_line_90kv_preprocessed_data')

# ========== 1. 加载标注文件 ==========
label_path = os.path.join(raw_data_folder, 'hv_double_line_90kv_labels.csv')
df_labels = pd.read_csv(label_path)
print(f"\n1. 标注文件: {df_labels.shape[0]} 条记录")

# ========== 2. 提取结构特征 ==========
struct_cols = [
    'sample_id',
    'fault_target',
    'sc_type',
    'fault_resistance',
    'phase_select',
    'line_1_2_a_rline', 'line_1_2_a_xline', 'line_1_2_a_cline',
    'line_2_3_a_rline', 'line_2_3_a_xline', 'line_2_3_a_cline',
    'line_2_3_b_on', 'line_1_2_b_on', 'ext_grid_3_on',
    'load_3_plini', 'load_3_qlini'
]
df_struct = df_labels[struct_cols].copy()
df_struct['sc_type_encoded'] = df_struct['sc_type'].astype('category').cat.codes
df_struct['phase_select_encoded'] = df_struct['phase_select'].astype('category').cat.codes
df_struct.drop(['sc_type', 'phase_select'], axis=1, inplace=True)
df_struct['sample_id_num'] = df_struct['sample_id'].astype(int)

# ========== 3. 定义波形特征提取函数 ==========
def extract_waveform_features(df):
    features = {}
    all_vol_cols = [c for c in df.columns if 'vol' in c.lower()]
    all_cur_cols = [c for c in df.columns if 'cur' in c.lower()]
    
    v_cols, c_cols = [], []
    for phase in ['L1', 'L2', 'L3']:
        for col in all_vol_cols:
            if phase in col:
                v_cols.append(col); break
        for col in all_cur_cols:
            if phase in col:
                c_cols.append(col); break
    v_cols, c_cols = v_cols[:3], c_cols[:3]
    
    for i, col in enumerate(v_cols):
        vals = df[col].values
        features[f'voltage_L{i+1}_rms'] = float(np.sqrt(np.mean(vals**2)))
        features[f'voltage_L{i+1}_peak'] = float(np.max(np.abs(vals)))
        features[f'voltage_L{i+1}_std'] = float(np.std(vals))
        features[f'voltage_L{i+1}_max'] = float(np.max(vals))
        features[f'voltage_L{i+1}_min'] = float(np.min(vals))
        features[f'voltage_L{i+1}_peak_to_peak'] = float(np.max(vals) - np.min(vals))
        zc = np.where(np.diff(np.sign(vals)))[0]
        if len(zc) > 1:
            time_span = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
            features[f'voltage_L{i+1}_freq_est'] = float(len(zc) / (2 * time_span)) if time_span > 0 else 0.0
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
    return features

# ========== 4. 遍历所有 .pkl 文件 ==========
all_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
print(f"\n2. 找到 {len(all_files)} 个波形文件，开始提取特征...")

all_waveform_features = []
for i, fname in enumerate(all_files):
    if i % 1000 == 0:
        print(f"   进度: {i}/{len(all_files)}")
    try:
        sid = int(fname.split('_')[0])
        with open(os.path.join(data_folder, fname), 'rb') as f:
            df = pickle.load(f)
        feats = extract_waveform_features(df)
        feats['sample_id'] = sid
        all_waveform_features.append(feats)
        del df
        gc.collect()
    except Exception as e:
        continue

df_waveform = pd.DataFrame(all_waveform_features)
df_waveform['sample_id_num'] = df_waveform['sample_id'].astype(int)
print(f"\n3. 波形特征提取完成: {df_waveform.shape[0]} 个样本, {df_waveform.shape[1]} 个特征")

# ========== 5. 合并结构特征 ==========
merged = df_waveform.merge(df_struct, on='sample_id_num', how='inner')
print(f"\n4. 合并后总样本: {merged.shape[0]}, 总特征: {merged.shape[1]}")

# ========== 6. 准备训练数据 ==========
merged['fault_target_3class'] = merged['fault_target'].replace({
    'Line_2_3_a': 'Line_2_3',
    'Line_2_3_b': 'Line_2_3'
})

exclude_cols = [
    'sample_id', 'sample_id_x', 'sample_id_y', 'sample_id_num',
    'fault_target', 'fault_target_3class', 'fault_resistance'
]
feature_cols = [col for col in merged.columns if col not in exclude_cols]

for col in feature_cols:
    if merged[col].dtype == 'object':
        merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)

X = merged[feature_cols].values.astype(np.float64)
y = merged['fault_target_3class'].values
X = np.nan_to_num(X)

print(f"\n5. 最终训练数据: {X.shape[0]} 样本, {X.shape[1]} 特征")

# ========== 7. 标签编码 ==========
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ========== 8. 划分训练集和测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

# ========== 9. 标准化 ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 10. SMOTE 数据增强 ==========
print("\n应用 SMOTE 进行数据平衡...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"SMOTE 前训练集: {X_train_scaled.shape[0]} 样本")
print(f"SMOTE 后训练集: {X_train_resampled.shape[0]} 样本")
print("SMOTE 后各类别数量:")
for cls in np.unique(y_train_resampled):
    print(f"  类别 {cls}: {np.sum(y_train_resampled == cls)}")

# ========== 11. GridSearchCV 自动调参 ==========
print("\n" + "=" * 70)
print("GridSearchCV 自动搜索最优参数 (预计 3-5 分钟)")
print("=" * 70)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

xgb_base = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1
)

grid_search = GridSearchCV(
    xgb_base,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

print(f"\n最优参数: {grid_search.best_params_}")
print(f"交叉验证准确率: {grid_search.best_score_:.4f}")

# ========== 12. 用最优参数训练 ==========
best_model = grid_search.best_estimator_

# ========== 13. 评估 ==========
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "=" * 70)
print(f"✅ 最终模型准确率: {accuracy*100:.2f}%")
print("=" * 70)
print(f"加权精确率: {precision:.4f}")
print(f"加权召回率: {recall:.4f}")
print(f"加权F1分数: {f1:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ========== 14. 混淆矩阵 ==========
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(cm)

# ========== 15. 混淆矩阵图 ==========
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - XGBoost + SMOTE + GridSearchCV')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(code_folder, 'confusion_matrix_optimized.png'), dpi=300)
print("\n混淆矩阵图已保存: confusion_matrix_optimized.png")
plt.show()

# ========== 16. 保存模型 ==========
joblib.dump(best_model, os.path.join(code_folder, 'fault_classifier_model.pkl'))
joblib.dump(scaler, os.path.join(code_folder, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(code_folder, 'label_encoder.pkl'))
joblib.dump(feature_cols, os.path.join(code_folder, 'feature_columns.pkl'))

print(f"\n✅ 模型已保存，最终特征数: {len(feature_cols)}")
print("=" * 70)