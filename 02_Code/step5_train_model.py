import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("第1步：加载完整数据集")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
data_path = os.path.join(raw_data_folder, 'full_dataset.csv')

df = pd.read_csv(data_path)
print(f"✅ 数据加载成功！样本数: {df.shape[0]}, 特征数: {df.shape[1]}")

print("\n" + "=" * 60)
print("第2步：分离特征和标签")
print("=" * 60)

exclude_cols = ['sample_id', 'sample_id_num', 'fault_target', 'fault_resistance']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]  # 特征
y = df['fault_target']  # 标签

print(f"特征数量: {len(feature_cols)}")
print(f"标签类别: {y.unique().tolist()}")

print("\n" + "=" * 60)
print("第3步：划分训练集（80%）和测试集（20%）")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {len(X_train)} 个样本")
print(f"测试集大小: {len(X_test)} 个样本")

print("\n" + "=" * 60)
print("第4步：训练随机森林模型")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=100,        # 100棵决策树
    max_depth=15,            # 每棵树的最大深度
    min_samples_split=5,     # 内部节点再划分所需最小样本数
    min_samples_leaf=2,      # 叶子节点最少样本数
    random_state=42,
    n_jobs=-1                # 使用所有CPU核心
)

model.fit(X_train, y_train)
print("✅ 模型训练完成！")

print("\n" + "=" * 60)
print("第5步：模型评估")
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

print("\n" + "=" * 60)
print("第6步：混淆矩阵（展示模型在各类故障上的表现）")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)

classes = model.classes_
print(f"\n类别顺序: {classes}")

print("\n" + "=" * 60)
print("第7步：详细分类报告")
print("=" * 60)

print(classification_report(y_test, y_pred))

print("\n" + "=" * 60)
print("第8步：特征重要性 Top 10")
print("=" * 60)

feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)

print(feature_importance.head(10))

print("\n" + "=" * 60)
print("第9步：保存混淆矩阵图")
print("=" * 60)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(project_folder, '02_Code', 'confusion_matrix.png'), dpi=300)
print("✅ 混淆矩阵图已保存为: confusion_matrix.png")
plt.show()

print("\n" + "=" * 60)
print("第10步：保存特征重要性图")
print("=" * 60)

plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(top_features['特征'], top_features['重要性'], color='steelblue')
plt.xlabel('重要性')
plt.title('Top 10 特征重要性')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(project_folder, '02_Code', 'feature_importance.png'), dpi=300)
print("✅ 特征重要性图已保存为: feature_importance.png")
plt.show()

print("\n" + "=" * 60)
print("🎉 模型训练和评估全部完成！")
print("=" * 60)