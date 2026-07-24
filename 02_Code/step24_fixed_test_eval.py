import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("Step 24: 加载固定测试集评估模型")
print("=" * 60)

code_folder = 'D:\\Fault_Diagnosis_Project\\02_Code'

# ========== 1. 加载 ==========
model = joblib.load(os.path.join(code_folder, 'fault_classifier_model.pkl'))
scaler = joblib.load(os.path.join(code_folder, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(code_folder, 'label_encoder.pkl'))
test_data = joblib.load(os.path.join(code_folder, 'fixed_test_set.pkl'))

X_test = test_data['X_test']
y_test = test_data['y_test']
print(f"测试集样本数: {len(X_test)}")

# ========== 2. 评估 ==========
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(f"准确率: {accuracy*100:.2f}%")
print("=" * 60)
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))