import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("优化版：加载完整数据集")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
data_path = os.path.join(raw_data_folder, 'full_dataset.csv')

df = pd.read_csv(data_path)
print(f"✅ 数据加载成功！样本数: {df.shape[0]}, 特征数: {df.shape[1]}")

exclude_cols = ['sample_id', 'sample_id_num', 'fault_target', 'fault_resistance']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df['fault_target']

print(f"特征数量: {len(feature_cols)}")
print(f"\n优化前各类别样本数:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集大小: {len(X_train)} 个样本")
print(f"测试集大小: {len(X_test)} 个样本")

print("\n" + "=" * 60)
print("应用 SMOTE 进行数据平衡...")
print("=" * 60)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"SMOTE 后训练集大小: {len(X_train_resampled)} 个样本")
print(f"\nSMOTE 后各类别样本数:")
print(pd.Series(y_train_resampled).value_counts())

print("\n" + "=" * 60)
print("训练优化后的随机森林模型")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=150,        # 更多树
    max_depth=12,            # 调小一点防止过拟合
    min_samples_split=6,
    min_samples_leaf=3,
    class_weight='balanced', # 自动给少数类更高权重
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)
print("✅ 模型训练完成！")

print("\n" + "=" * 60)
print("模型评估（优化后）")
print("=" * 60)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数 (F1-Score): {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
classes = model.classes_
print("\n混淆矩阵:")
print(cm)

print("\n" + "=" * 60)
print("详细分类报告")
print("=" * 60)
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Optimized Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(project_folder, '02_Code', 'confusion_matrix_optimized.png'), dpi=300)
print("✅ 优化版混淆矩阵图已保存为: confusion_matrix_optimized.png")
plt.show()

print("\n🎉 优化完成！")