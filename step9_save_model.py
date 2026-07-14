import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
data_path = os.path.join(raw_data_folder, 'full_dataset.csv')

df = pd.read_csv(data_path)

df['fault_target_merged'] = df['fault_target'].replace({
    'Line_2_3_a': 'Line_2_3',
    'Line_2_3_b': 'Line_2_3'
})

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ 最终模型训练完成！")

model_path = os.path.join(project_folder, '02_Code', 'fault_classifier_model.pkl')
joblib.dump(model, model_path)
print(f"✅ 模型已保存到: {model_path}")

feature_cols_path = os.path.join(project_folder, '02_Code', 'feature_columns.pkl')
joblib.dump(feature_cols, feature_cols_path)
print(f"✅ 特征列名已保存到: {feature_cols_path}")

print("\n🎉 模型保存完成！可以开始搭建网页界面了。")