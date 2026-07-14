import pandas as pd
import os

project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')

features_path = os.path.join(raw_data_folder, 'extracted_features.csv')
print(f"正在读取特征表格: {features_path}")
features_df = pd.read_csv(features_path)
print(f"特征表格形状: {features_df.shape}")

labels_path = os.path.join(raw_data_folder, 'hv_double_line_90kv_labels.csv')
print(f"\n正在读取标注文件: {labels_path}")
labels_df = pd.read_csv(labels_path)
print(f"标注文件形状: {labels_df.shape}")
print(f"\n标注文件的列名: {labels_df.columns.tolist()}")

features_df['sample_id'] = features_df['sample_id'].astype(str)
labels_df['sample_id'] = labels_df['sample_id'].astype(str)
print("\n✅ sample_id 列已统一转换为字符串类型")

print("\n=== 故障类型分布 ===")
print(labels_df['fault_target'].value_counts())

labels_keep = labels_df[['sample_id', 'fault_target', 'fault_resistance']].copy()

merged_df = features_df.merge(labels_keep, on='sample_id', how='left')
print(f"\n合并后数据形状: {merged_df.shape}")

missing_labels = merged_df['fault_target'].isna().sum()
if missing_labels > 0:
    print(f"⚠️ 有 {missing_labels} 个样本没有匹配到标签，这些样本将被删除")
    merged_df = merged_df.dropna(subset=['fault_target'])
    print(f"删除后数据形状: {merged_df.shape}")

output_path = os.path.join(raw_data_folder, 'full_dataset.csv')
merged_df.to_csv(output_path, index=False)
print(f"\n✅ 完整数据集已保存到: {output_path}")
print(f"最终数据集形状: {merged_df.shape} (行数, 列数)")

print("\n=== 数据预览（前5行）===")
print(merged_df.head())

print("\n=== 故障类型统计 ===")
print(merged_df['fault_target'].value_counts())

print("\n🎉 合并完成！你现在有了带标签的完整训练数据集 full_dataset.csv")