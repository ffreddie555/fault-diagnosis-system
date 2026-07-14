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

features_df['sample_id_num'] = features_df['sample_id'].str.extract(r'(\d+)')[0].astype(int)
print("\n✅ 已从特征表的 sample_id 中提取数字")

labels_df['sample_id_num'] = labels_df['sample_id'].astype(int)

print("\n特征表 sample_id 和提取的数字（前10个）:")
print(features_df[['sample_id', 'sample_id_num']].head(10))

print("\n标注表 sample_id（前10个）:")
print(labels_df[['sample_id']].head(10))

labels_keep = labels_df[['sample_id_num', 'fault_target', 'fault_resistance']].copy()

merged_df = features_df.merge(labels_keep, on='sample_id_num', how='inner')
print(f"\n✅ 合并完成！合并后数据形状: {merged_df.shape}")

if merged_df.shape[0] == 0:
    print("\n⚠️ 合并后为空！请检查 sample_id_num 的数据类型或内容。")
    print("\n特征表 sample_id_num 示例:", features_df['sample_id_num'].head(10).tolist())
    print("标注表 sample_id_num 示例:", labels_df['sample_id_num'].head(10).tolist())
else:
    output_path = os.path.join(raw_data_folder, 'full_dataset.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"\n✅ 完整数据集已保存到: {output_path}")
    
    print(f"\n=== 故障类型分布 ===")
    print(merged_df['fault_target'].value_counts())
    
    print("\n=== 数据预览（前5行）===")
    print(merged_df.head())
    
    print("\n🎉 合并完成！你现在有了带标签的完整训练数据集 full_dataset.csv")