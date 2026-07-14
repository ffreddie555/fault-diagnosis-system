import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("Step 18: 添加结构化特征 + 重新训练")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
code_folder = os.path.join(project_folder, '02_Code')

# 1. 加载现有的特征表
data_path = os.path.join(raw_data_folder, 'full_dataset.csv')
df_features = pd.read_csv(data_path)
print(f"1. 加载波形特征表: {df_features.shape}")

# 2. 加载标注信息
label_path = os.path.join(raw_data_folder, 'hv_double_line_90kv_labels.csv')
df_labels = pd.read_csv(label_path)
print(f"2. 加载标注参数表: {df_labels.shape}")

# 3. 选择要加入的结构特征
struct_cols = [
    'sample_id',
    'sc_type',
    'fault_resistance',
    'phase_select',
    'line_1_2_a_rline', 'line_1_2_a_xline', 'line_1_2_a_cline',
    'line_2_3_a_rline', 'line_2_3_a_xline', 'line_2_3_a_cline',
    'line_2_3_b_on', 'line_1_2_b_on', 'ext_grid_3_on',
    'load_3_plini', 'load_3_qlini'
]

df_struct = df_labels[struct_cols].copy()
print(f"3. 提取结构特征: {df_struct.shape}")

# 4. 处理分类变量 (sc_type, phase_select) 转为数值
df_struct['sc_type_encoded'] = df_struct['sc_type'].astype('category').cat.codes
df_struct['phase_select_encoded'] = df_struct['phase_select'].astype('category').cat.codes
df_struct.drop(['sc_type', 'phase_select'], axis=1, inplace=True)

print(f"4. 编码分类变量后: {df_struct.shape}")

# 5. 合并特征
df_features['sample_id_num'] = df_features['sample_id'].str.extract(r'(\d+)').astype(int)
df_struct['sample_id_num'] = df_struct['sample_id'].astype(int)

merged_df = df_features.merge(df_struct, on='sample_id_num', how='inner')
print(f"5. 合并后总特征表: {merged_df.shape}")

# 6. 准备训练数据 (3分类)
merged_df['fault_target_3class'] = merged_df['fault_target'].replace({
    'Line_2_3_a': 'Line_2_3',
    'Line_2_3_b': 'Line_2_3'
})

# ===== 排除所有非数值列 =====
exclude_cols = [
    'sample_id',           # 文本
    'sample_id_x',         # 合并后产生的文本列
    'sample_id_y',         # 合并后产生的文本列
    'sample_id_num',       # 只是索引
    'fault_target',        # 原始标签（文本）
    'fault_target_3class', # 合并后的标签（文本）
]

# 获取所有列名
all_cols = merged_df.columns.tolist()

# 特征列 = 所有列 - 排除列
feature_cols = [col for col in all_cols if col not in exclude_cols]

print(f"6. 最终特征数量: {len(feature_cols)}")
print(f"   特征列示例: {feature_cols[:5]}...")

# ===== 关键修复：先转换数据类型，再提取 X =====
for col in feature_cols:
    # 如果是 object 类型（文本），尝试转为数值
    if merged_df[col].dtype == 'object':
        print(f"   转换列 '{col}' 为数值类型...")
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)
    
    # 如果是 bool 类型，转为 int
    if merged_df[col].dtype == 'bool':
        merged_df[col] = merged_df[col].astype(int)

# 再次检查所有特征列的数据类型
print("\n7. 检查所有特征列的数据类型:")
for col in feature_cols:
    dtype = merged_df[col].dtype
    if dtype == 'object':
        print(f"   ⚠️ 警告：列 '{col}' 仍然是 object 类型！强制转换...")
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)

# 提取 X（确保全部是数值）
X = merged_df[feature_cols].values.astype(np.float64)
y = merged_df['fault_target_3class'].values

# 检查是否有 NaN，填充为 0
if np.isnan(X).any():
    print("⚠️ 发现 NaN 值，填充为 0")
    X = np.nan_to_num(X)

print(f"8. X 形状: {X.shape}, 数据类型: {X.dtype}")

# 9. 编码标签
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 10. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 11. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"9. 训练集: {X_train_scaled.shape}, 测试集: {X_test_scaled.shape}")

# 12. 训练模型
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

print("10. 正在训练新模型...")
model.fit(X_train_scaled, y_train)

# 13. 评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ 新模型准确率: {accuracy*100:.2f}%")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 14. 保存新模型和新特征列表
model_path = os.path.join(code_folder, 'fault_classifier_model.pkl')
joblib.dump(model, model_path)
print(f"✅ 新模型已保存: {model_path}")

scaler_path = os.path.join(code_folder, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"✅ 标准化器已保存: {scaler_path}")

label_encoder_path = os.path.join(code_folder, 'label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_path)
print(f"✅ 标签编码器已保存: {label_encoder_path}")

feature_cols_path = os.path.join(code_folder, 'feature_columns.pkl')
joblib.dump(feature_cols, feature_cols_path)
print(f"✅ 新特征列表已保存: {feature_cols_path}")

print("\n" + "=" * 60)
print("🎉 结构特征融合完成！")
print(f"原特征数: 43, 新特征数: {len(feature_cols)}")
print(f"新准确率: {accuracy*100:.2f}%")
print("=" * 60)