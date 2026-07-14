import pandas as pd
import numpy as np
import os
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("第2周 - 3分类模型对比 + GridSearchCV 调参")
print("=" * 60)

# ========== 1. 加载数据 ==========
project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
code_folder = os.path.join(project_folder, '02_Code')

data_path = os.path.join(raw_data_folder, 'full_dataset.csv')
df = pd.read_csv(data_path)
print(f"加载数据: {df.shape[0]} 行, {df.shape[1]} 列")

# ========== 2. 合并类别（3分类） ==========
print("\n合并类别: Line_2_3_a + Line_2_3_b → Line_2_3")
df['fault_target_3class'] = df['fault_target'].replace({
    'Line_2_3_a': 'Line_2_3',
    'Line_2_3_b': 'Line_2_3'
})

print("\n合并后类别分布:")
print(df['fault_target_3class'].value_counts())

# ========== 3. 准备特征和标签 ==========
exclude_cols = ['sample_id', 'sample_id_num', 'fault_target', 'fault_target_3class', 'fault_resistance']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y = df['fault_target_3class'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

print(f"\n特征数量: {len(feature_cols)}")
print(f"类别: {class_names}")

# ========== 4. 划分 + 标准化 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

# ========== 5. 定义模型（基础版本） ==========
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

# ========== 6. 训练并评估 ==========
results = []
confusion_matrices = {}
class_reports = {}

print("\n" + "=" * 60)
print("基础模型对比（3分类）")
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

# ========== 7. 显示结果 ==========
print("\n" + "=" * 60)
print("基础模型对比结果（3分类）")
print("=" * 60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_base_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_base_acc = results_df['Accuracy'].max()
print(f"\n🏆 最佳基础模型: {best_base_name} (准确率: {best_base_acc*100:.2f}%)")

# ========== 8. GridSearchCV 调参（针对最佳模型） ==========
print("\n" + "=" * 60)
print(f"GridSearchCV 调参 - 优化 {best_base_name}")
print("=" * 60)

if best_base_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
elif best_base_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9]
    }
    base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
else:  # SVM
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1],
        'kernel': ['rbf']
    }
    base_model = SVC(random_state=42, probability=True)

print(f"参数网格: {param_grid}")
print("开始 GridSearchCV（可能需要 3-5 分钟）...")

grid_search = GridSearchCV(
    base_model, param_grid, cv=3, scoring='accuracy',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print(f"\n✅ 最优参数: {grid_search.best_params_}")
print(f"交叉验证准确率: {grid_search.best_score_:.4f}")

# ========== 9. 用最优参数重新训练 ==========
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best, average='weighted')
best_recall = recall_score(y_test, y_pred_best, average='weighted')
best_f1 = f1_score(y_test, y_pred_best, average='weighted')

print("\n" + "=" * 60)
print(f"优化后 {best_base_name} 评估（3分类）")
print("=" * 60)
print(f"准确率 (Accuracy): {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"加权精确率 (Precision): {best_precision:.4f}")
print(f"加权召回率 (Recall): {best_recall:.4f}")
print(f"加权F1分数 (F1-Score): {best_f1:.4f}")

cm_best = confusion_matrix(y_test, y_pred_best)
print("\n混淆矩阵:")
print(cm_best)

print("\n详细分类报告:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

# ========== 10. 保存最终模型 ==========
model_path = os.path.join(code_folder, 'fault_classifier_model.pkl')
joblib.dump(best_model, model_path)
print(f"\n✅ 最终模型已保存到: {model_path}")

scaler_path = os.path.join(code_folder, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"✅ 标准化器已保存到: {scaler_path}")

label_encoder_path = os.path.join(code_folder, 'label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_path)
print(f"✅ 标签编码器已保存到: {label_encoder_path}")

feature_cols_path = os.path.join(code_folder, 'feature_columns.pkl')
joblib.dump(feature_cols, feature_cols_path)
print(f"✅ 特征列名已保存到: {feature_cols_path}")

# ========== 11. 混淆矩阵图 ==========
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - Optimized {best_base_name} (3 Classes)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(code_folder, 'confusion_matrix_optimized_week2.png'), dpi=300)
print("✅ 混淆矩阵图已保存为: confusion_matrix_optimized_week2.png")
plt.show()

# ========== 12. 保存报告 ==========
report_path = os.path.join(code_folder, 'week2_final_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("第2周 - 最终报告（3分类 + GridSearchCV）\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"最佳模型: {best_base_name}\n")
    f.write(f"最优参数: {grid_search.best_params_}\n")
    f.write(f"准确率: {best_accuracy*100:.2f}%\n")
    f.write(f"加权精确率: {best_precision:.4f}\n")
    f.write(f"加权召回率: {best_recall:.4f}\n")
    f.write(f"加权F1分数: {best_f1:.4f}\n\n")
    f.write("混淆矩阵:\n")
    f.write(str(cm_best) + "\n\n")
    f.write("详细分类报告:\n")
    f.write(classification_report(y_test, y_pred_best, target_names=class_names))

print(f"✅ 完整报告已保存到: {report_path}")

print("\n" + "=" * 60)
print("🎉 第2周任务完成！")
print("=" * 60)
print(f"   - 最佳模型: {best_base_name}")
print(f"   - 准确率: {best_accuracy*100:.2f}%")
print(f"   - 报告位置: {report_path}")
print("=" * 60)