import pandas as pd
import numpy as np
import os
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("第2周 - 模型对比：Random Forest vs XGBoost vs SVM")
print("=" * 60)

# ========== 1. 加载数据 ==========
project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
code_folder = os.path.join(project_folder, '02_Code')

data_path = os.path.join(raw_data_folder, 'full_dataset.csv')
df = pd.read_csv(data_path)
print(f"加载数据: {df.shape[0]} 行, {df.shape[1]} 列")

# 查看有哪些列
print(f"列名: {df.columns.tolist()}")

# ========== 2. 准备特征和标签 ==========
# 修复：使用 fault_target（原始标签，4分类）
exclude_cols = ['sample_id', 'sample_id_num', 'fault_target', 'fault_resistance']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y = df['fault_target'].values  # 直接用原始标签

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

print(f"特征数量: {len(feature_cols)}")
print(f"类别: {class_names}")
print(f"各类别样本数:")
print(df['fault_target'].value_counts())

# ========== 3. 划分 + 标准化 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

# ========== 4. 定义模型 ==========
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, use_label_encoder=False,
        eval_metric='mlogloss', n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf', C=1.0, gamma='scale',
        random_state=42, probability=True
    )
}

# ========== 5. 训练并评估每个模型 ==========
results = []
confusion_matrices = {}
class_reports = {}

print("\n" + "=" * 60)
print("开始训练和评估...")
print("=" * 60)

for name, model in models.items():
    print(f"\n▶ 训练 {name}...")
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Train Time (s)': train_time
    })
    
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    class_reports[name] = classification_report(y_test, y_pred, target_names=class_names)
    
    print(f"  准确率: {accuracy*100:.2f}%")
    print(f"  训练时间: {train_time:.2f} 秒")

# ========== 6. 显示结果对比表 ==========
print("\n" + "=" * 60)
print("模型对比结果")
print("=" * 60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 找出最佳模型
best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_accuracy = results_df['Accuracy'].max()
print(f"\n🏆 最佳模型: {best_model_name} (准确率: {best_accuracy*100:.2f}%)")

# ========== 7. 保存最佳模型 ==========
best_model = models[best_model_name]

model_path = os.path.join(code_folder, 'fault_classifier_model.pkl')
joblib.dump(best_model, model_path)
print(f"✅ 最佳模型已保存到: {model_path}")

scaler_path = os.path.join(code_folder, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"✅ 标准化器已保存到: {scaler_path}")

label_encoder_path = os.path.join(code_folder, 'label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_path)
print(f"✅ 标签编码器已保存到: {label_encoder_path}")

feature_cols_path = os.path.join(code_folder, 'feature_columns.pkl')
joblib.dump(feature_cols, feature_cols_path)
print(f"✅ 特征列名已保存到: {feature_cols_path}")

# ========== 8. 绘制对比图 ==========
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25

for i, (name, group) in enumerate(results_df.groupby('Model')):
    values = [group[m].values[0] for m in metrics]
    plt.bar(x + i*width, values, width, label=name)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Comparison: Random Forest vs XGBoost vs SVM')
plt.xticks(x + width, metrics)
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(code_folder, 'model_comparison_week2.png'), dpi=300)
print("✅ 模型对比图已保存为: model_comparison_week2.png")
plt.show()

# ========== 9. 显示最佳模型的混淆矩阵 ==========
best_cm = confusion_matrices[best_model_name]
plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - Best Model ({best_model_name})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(code_folder, 'confusion_matrix_best_week2.png'), dpi=300)
print("✅ 最佳模型混淆矩阵已保存为: confusion_matrix_best_week2.png")
plt.show()

# ========== 10. 保存完整对比报告 ==========
report_path = os.path.join(code_folder, 'model_comparison_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("第2周 - 模型对比报告\n")
    f.write("=" * 60 + "\n\n")
    f.write("数据信息:\n")
    f.write(f"  - 总样本数: {len(df)}\n")
    f.write(f"  - 特征数: {len(feature_cols)}\n")
    f.write(f"  - 类别: {list(class_names)}\n\n")
    f.write("模型对比结果:\n")
    f.write(results_df.to_string(index=False) + "\n\n")
    f.write(f"🏆 最佳模型: {best_model_name} (准确率: {best_accuracy*100:.2f}%)\n\n")
    f.write("=" * 60 + "\n")
    f.write(f"最佳模型详细分类报告:\n")
    f.write(class_reports[best_model_name])

print(f"✅ 完整报告已保存到: {report_path}")

print("\n" + "=" * 60)
print("🎉 第2周任务1完成！")
print("=" * 60)
print(f"   - 最佳模型: {best_model_name}")
print(f"   - 准确率: {best_accuracy*100:.2f}%")
print(f"   - 报告位置: {report_path}")
print("=" * 60)