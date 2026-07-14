import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("=" * 60)
print("修复 label_encoder.pkl（3分类版本）")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
code_folder = os.path.join(project_folder, '02_Code')

# 加载数据
data_path = os.path.join(raw_data_folder, 'full_dataset.csv')
df = pd.read_csv(data_path)
print(f"加载数据: {df.shape[0]} 行, {df.shape[1]} 列")

# 合并类别（3分类）
df['fault_target_3class'] = df['fault_target'].replace({
    'Line_2_3_a': 'Line_2_3',
    'Line_2_3_b': 'Line_2_3'
})

print("\n合并后类别分布:")
print(df['fault_target_3class'].value_counts())

# 创建 3 分类的 label_encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['fault_target_3class'].values)

print(f"\n类别映射 (3分类):")
for i, name in enumerate(label_encoder.classes_):
    print(f"  {i} -> {name}")

# 保存
label_encoder_path = os.path.join(code_folder, 'label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_path)
print(f"\n✅ label_encoder.pkl 已更新为3分类版本")

# 验证
print("\n验证加载:")
loaded = joblib.load(label_encoder_path)
print(f"  类别数量: {len(loaded.classes_)}")
print(f"  类别: {loaded.classes_}")