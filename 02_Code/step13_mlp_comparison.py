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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("第2周 - MLP 神经网络对比实验")
print("=" * 60)

# ========== 1. 加载数据 ==========
project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
code_folder = os.path.join(project_folder, '02_Code')

data_path = os.path.join(raw_data_folder, 'full_dataset.csv')
df = pd.read_csv(data_path)
print(f"加载数据: {df.shape[0]} 行, {df.shape[1]} 列")

# ========== 2. 合并类别（3分类） ==========
print("\n合并类别: Line_2_3_a + Line_2_3_b -> Line_2_3")
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

# ========== 5. 训练 Random Forest（作为基准） ==========
print("\n" + "=" * 60)
print("训练 Random Forest（基准模型）")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
rf_model.fit(X_train_scaled, y_train)
rf_time = time.time() - start_time

rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest 准确率: {rf_accuracy*100:.2f}%")
print(f"训练时间: {rf_time:.2f} 秒")

# ========== 6. 训练 MLP 神经网络 ==========
print("\n" + "=" * 60)
print("训练 MLP 神经网络")
print("=" * 60)

# 尝试不同的网络结构
mlp_models = {
    'MLP (50, 25)': MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    ),
    'MLP (100, 50)': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    ),
    'MLP (64, 32, 16)': MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
}

mlp_results = []

for name, model in mlp_models.items():
    print(f"\n训练 {name}...")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    mlp_results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Train Time (s)': train_time
    })
    
    print(f"  准确率: {accuracy*100:.2f}%")
    print(f"  训练时间: {train_time:.2f} 秒")

# ========== 7. 汇总对比结果 ==========
print("\n" + "=" * 60)
print("所有模型对比结果")
print("=" * 60)

# 把 Random Forest 结果也加进去
all_results = [{'Model': 'Random Forest (Optimized)', 'Accuracy': rf_accuracy, 'Train Time (s)': rf_time}] + mlp_results
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# 找出最佳模型
best_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_acc = results_df['Accuracy'].max()
print(f"\n最佳模型: {best_name} (准确率: {best_acc*100:.2f}%)")

# ========== 8. 绘制对比图 ==========
plt.figure(figsize=(10, 6))
model_names = results_df['Model'].tolist()
accuracies = results_df['Accuracy'].tolist()
times = results_df['Train Time (s)'].tolist()

# 准确率柱状图
bars = plt.bar(model_names, accuracies, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison: Random Forest vs MLP Variants')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(code_folder, 'mlp_comparison_week2.png'), dpi=300)
print("\n✅ MLP对比图已保存为: mlp_comparison_week2.png")
plt.show()

# ========== 9. 保存最佳 MLP 模型 ==========
best_mlp_name = results_df.loc[results_df['Model'].str.contains('MLP'), :]
if not best_mlp_name.empty:
    best_mlp_name = best_mlp_name.loc[best_mlp_name['Accuracy'].idxmax(), 'Model']
    best_mlp_model = mlp_models[best_mlp_name]
    
    mlp_model_path = os.path.join(code_folder, 'mlp_model.pkl')
    joblib.dump(best_mlp_model, mlp_model_path)
    print(f"✅ 最佳 MLP 模型已保存到: {mlp_model_path}")

# ========== 10. 保存报告 ==========
report_path = os.path.join(code_folder, 'mlp_comparison_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("第2周 - MLP 神经网络对比报告\n")
    f.write("=" * 60 + "\n\n")
    f.write("数据信息:\n")
    f.write(f"  - 总样本数: {len(df)}\n")
    f.write(f"  - 特征数: {len(feature_cols)}\n")
    f.write(f"  - 类别: {list(class_names)}\n\n")
    f.write("模型对比结果:\n")
    f.write(results_df.to_string(index=False) + "\n\n")
    f.write(f"最佳模型: {best_name} (准确率: {best_acc*100:.2f}%)\n")

print(f"✅ 对比报告已保存到: {report_path}")

print("\n" + "=" * 60)
print("MLP 对比实验完成！")
print("=" * 60)
print(f"   - Random Forest: {rf_accuracy*100:.2f}%")
for r in mlp_results:
    print(f"   - {r['Model']}: {r['Accuracy']*100:.2f}%")
print(f"   - 最佳: {best_name} ({best_acc*100:.2f}%)")
print("=" * 60)