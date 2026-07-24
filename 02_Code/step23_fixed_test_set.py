import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Step 23: 固定测试集 + 模型训练 (方案2)")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
code_folder = os.path.join(project_folder, '02_Code')

# ========== 1. 加载全量数据 ==========
data_path = os.path.join(raw_data_folder, 'full_dataset.csv')
df = pd.read_csv(data_path)
print(f"加载数据: {df.shape[0]} 个样本, {df.shape[1]} 列")

# ========== 2. 合并类别 ==========
df['fault_target_3class'] = df['fault_target'].replace({
    'Line_2_3_a': 'Line_2_3',
    'Line_2_3_b': 'Line_2_3'
})

print("\n类别分布:")
print(df['fault_target_3class'].value_counts())

# ========== 3. 准备特征和标签 ==========
exclude_cols = ['sample_id', 'sample_id_num', 'fault_target', 'fault_target_3class', 'fault_resistance']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].values
y = df['fault_target_3class'].values

print(f"\n特征数量: {len(feature_cols)}")

# ========== 4. 标签编码 ==========
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ========== 5. 固定划分 (80/20, random_state=42) ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n训练集: {len(X_train)} 个样本")
print(f"测试集: {len(X_test)} 个样本")

# ========== 6. 标准化 ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 7. 训练 XGBoost ==========
print("\n训练 XGBoost 模型...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# ========== 8. 评估 ==========
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f"准确率: {accuracy*100:.2f}%")
print("=" * 60)
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# ========== 9. 保存 ==========
# 保存固定测试集
test_set = {
    'X_test': X_test,
    'y_test': y_test,
    'feature_cols': feature_cols
}
joblib.dump(test_set, os.path.join(code_folder, 'fixed_test_set.pkl'))
print(f"\n✅ 固定测试集已保存: {len(X_test)} 个样本")

# 保存模型文件
joblib.dump(model, os.path.join(code_folder, 'fault_classifier_model.pkl'))
joblib.dump(scaler, os.path.join(code_folder, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(code_folder, 'label_encoder.pkl'))
joblib.dump(feature_cols, os.path.join(code_folder, 'feature_columns.pkl'))
print("✅ 模型文件已更新")