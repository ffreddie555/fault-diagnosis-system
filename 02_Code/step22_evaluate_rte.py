import pandas as pd
import numpy as np
import os
import pickle
import joblib
from sklearn.metrics import accuracy_score

print("=" * 60)
print("评估模型在 RTE 数据集上的表现")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
code_folder = os.path.join(project_folder, '02_Code')
rte_folder = os.path.join(project_folder, '01_Raw_Data', 'rte_preprocessed')

# 加载模型
model = joblib.load(os.path.join(code_folder, 'fault_classifier_model.pkl'))
scaler = joblib.load(os.path.join(code_folder, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(code_folder, 'label_encoder.pkl'))
feature_cols = joblib.load(os.path.join(code_folder, 'feature_columns.pkl'))

# 测试前 100 个 RTE 样本（RTE 没有标签，只看模型能输出什么）
print("测试前 100 个 RTE 样本...")
predictions = []
for i in range(100):
    with open(os.path.join(rte_folder, f'rte_sample_{i}.pkl'), 'rb') as f:
        df = pickle.load(f)
    
    # 用你的特征提取函数（这里简化，只演示）
    # 实际需要提取特征...
    pass