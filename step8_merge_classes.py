import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("加载完整数据集")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
data_path = os.path.join(raw_data_folder, 'full_dataset.csv')

df = pd.read_csv(data_path)

print("\n合并类别: Line_2_3_a 和 Line_2_3_b -> Line_2_3")
print("=" * 60)

df['fault_target_merged'] = df['fault_target'].replace({
    'Line_2_3_a': 'Line_2_3',
    'Line_2_3_b': 'Line_2_3'
})

print("合并前的类别分布:")
print(df['fault_target'].value_counts())
print("\n合并后的类别分布:")
print(df['fault_target_merged'].value_counts())

print("\n构造新特征...")
voltage_cols = [f'voltage_L{i+1}_rms' for i in range(3)]
current_cols = [f'current_L{i+1}_rms' for i in range(3)]

df['voltage_imbalance'] = df[voltage_cols].std(axis=1)
df['current_imbalance'] = df[current_cols].std(axis=1)
df['v_i_ratio_L1'] = df['voltage_L1_rms'] / (df['current_L1_rms'] + 1e-6)
df['v_i_ratio_L2'] = df['voltage_L2_rms'] / (df['current_L2_rms'] + 1e-6)
df['v_i_ratio_L3'] = df['voltage_L3_rms'] / (df['current_L3_rms'] + 1e-6)
for i in range(3):
    df[f'crest_factor_L{i+1}'] = df[f'voltage_L{i+1}_peak'] / (df[f'voltage_L{i+1}_rms'] + 1e-6)
print("✅ 新增 8 个特征")

exclude_cols = ['sample_id', 'sample_id_num', 'fault_target', 'fault_target_merged', 'fault_resistance']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['fault_target_merged']

print(f"\n特征数量: {len(feature_cols)}")
print(f"各类别样本数:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("训练随机森林模型 (3分类)")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
print("✅ 模型训练完成！")

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "=" * 60)
print("模型评估 (3分类)")
print("=" * 60)
print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数 (F1-Score): {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
classes = rf.classes_
print("\n混淆矩阵:")
print(cm)

print("\n" + "=" * 60)
print("详细分类报告")
print("=" * 60)
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (3 Classes)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(project_folder, '02_Code', 'confusion_matrix_3class.png'), dpi=300)
print("\n✅ 混淆矩阵图已保存为: confusion_matrix_3class.png")
plt.show()

print("\n" + "=" * 60)
print("🎉 3分类模型训练完成！")
print("=" * 60)