import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ========== 配置 ==========
project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
code_folder = os.path.join(project_folder, '02_Code')
data_path = os.path.join(raw_data_folder, 'full_dataset.csv')
output_folder = code_folder

print("=" * 60)
print("模型评估与图片生成")
print("=" * 60)

# ========== 1. 加载数据并准备 ==========
print("\n1. 加载数据...")
df = pd.read_csv(data_path)
df['fault_target_merged'] = df['fault_target'].replace({
    'Line_2_3_a': 'Line_2_3',
    'Line_2_3_b': 'Line_2_3'
})

# 构造特征
voltage_cols = [f'voltage_L{i+1}_rms' for i in range(3)]
current_cols = [f'current_L{i+1}_rms' for i in range(3)]
df['voltage_imbalance'] = df[voltage_cols].std(axis=1)
df['current_imbalance'] = df[current_cols].std(axis=1)
df['v_i_ratio_L1'] = df['voltage_L1_rms'] / (df['current_L1_rms'] + 1e-6)
df['v_i_ratio_L2'] = df['voltage_L2_rms'] / (df['current_L2_rms'] + 1e-6)
df['v_i_ratio_L3'] = df['voltage_L3_rms'] / (df['current_L3_rms'] + 1e-6)
for i in range(3):
    df[f'crest_factor_L{i+1}'] = df[f'voltage_L{i+1}_peak'] / (df[f'voltage_L{i+1}_rms'] + 1e-6)

exclude_cols = ['sample_id', 'sample_id_num', 'fault_target', 'fault_target_merged', 'fault_resistance']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y = df['fault_target_merged']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ========== 2. 加载模型并预测 ==========
print("\n2. 加载模型并预测...")
model_path = os.path.join(code_folder, 'fault_classifier_model.pkl')
model = joblib.load(model_path)
y_pred = model.predict(X_test)
classes = model.classes_
cm = confusion_matrix(y_test, y_pred)

# ========== 3. 打印评估指标 ==========
print("\n" + "=" * 60)
print("模型评估报告")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"加权精确率 (Weighted Precision): {precision:.4f}")
print(f"加权召回率 (Weighted Recall): {recall:.4f}")
print(f"加权F1分数 (Weighted F1-Score): {f1:.4f}")

print("\n混淆矩阵:")
print(cm)

print("\n详细分类报告:")
print(classification_report(y_test, y_pred))

# ========== 4. 保存评估报告到文本文件 ==========
print("\n3. 保存评估报告...")
with open(os.path.join(code_folder, 'evaluation_report.txt'), 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("模型评估报告\n")
    f.write("=" * 60 + "\n")
    f.write(f"\n准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"加权精确率: {precision:.4f}\n")
    f.write(f"加权召回率: {recall:.4f}\n")
    f.write(f"加权F1分数: {f1:.4f}\n")
    f.write("\n混淆矩阵:\n")
    f.write(str(cm) + "\n")
    f.write("\n详细分类报告:\n")
    f.write(classification_report(y_test, y_pred))
print("✅ evaluation_report.txt")

# ========== 5. 生成混淆矩阵图 ==========
print("\n4. 生成混淆矩阵图...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Random Forest (3 Classes)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'confusion_matrix_3class.png'), dpi=300)
print("✅ confusion_matrix_3class.png")
plt.close()

# ========== 6. 生成特征重要性图 ==========
print("\n5. 生成特征重要性图...")
feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)

plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(top_features['特征'], top_features['重要性'], color='steelblue')
plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'feature_importance.png'), dpi=300)
print("✅ feature_importance.png")
plt.close()

# ========== 7. 生成波形图 ==========
print("\n6. 生成波形图...")
sample_path = os.path.join(raw_data_folder, 'hv_double_line_90kv_preprocessed_data', '1_sample_hv_double_line_90kv.pkl')
with open(sample_path, 'rb') as f:
    sample_df = pickle.load(f)

voltage_cols_sample = [col for col in sample_df.columns if 'vol' in col.lower() and 'L1' in col]
if len(voltage_cols_sample) >= 3:
    v_cols = voltage_cols_sample[:3]
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(v_cols):
        plt.plot(sample_df['time_s'], sample_df[col], label=f'Phase {i+1}', linewidth=1)
    plt.title('Three-Phase Voltage Waveform (Sample Fault Event)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'voltage_waveform.png'), dpi=300)
    print("✅ voltage_waveform.png")
    plt.close()

current_cols_sample = [col for col in sample_df.columns if 'cur' in col.lower() and 'L1' in col]
if len(current_cols_sample) >= 3:
    c_cols = current_cols_sample[:3]
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(c_cols):
        plt.plot(sample_df['time_s'], sample_df[col], label=f'Phase {i+1}', linewidth=1)
    plt.title('Three-Phase Current Waveform (Sample Fault Event)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Current (A)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'current_waveform.png'), dpi=300)
    print("✅ current_waveform.png")
    plt.close()

# ========== 8. 生成性能对比图（基线 vs 优化） ==========
print("\n7. 生成性能对比图...")
labels = ['Baseline\n(4-class)', 'Optimized\n(3-class)']
accuracies = [0.60, 0.82]

plt.figure(figsize=(6, 5))
bars = plt.bar(labels, accuracies, color=['#ff9999', '#66b3ff'], edgecolor='black', linewidth=1)
plt.ylim(0, 1.0)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc*100:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'performance_comparison.png'), dpi=300)
print("✅ performance_comparison.png")
plt.close()

print("\n" + "=" * 60)
print("🎉 全部完成！生成的文件:")
print("=" * 60)
print("  📄 evaluation_report.txt")
print("  📊 confusion_matrix_3class.png")
print("  📊 feature_importance.png")
print("  📊 voltage_waveform.png")
print("  📊 current_waveform.png")
print("  📊 performance_comparison.png")
print(f"\n保存位置: {output_folder}")