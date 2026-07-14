import numpy as np
import pandas as pd
import pickle
import os

print("=" * 60)
print("RTE 数据集适配脚本")
print("=" * 60)

project_folder = 'D:\\Fault_Diagnosis_Project'
raw_data_folder = os.path.join(project_folder, '01_Raw_Data')
rte_data_path = os.path.join(raw_data_folder, 'DATA_S.npz')

# 1. 加载 RTE 数据
print("\n1. 加载 DATA_S.npz...")
data = np.load(rte_data_path)
print(f"   文件中的键: {list(data.keys())}")

# 提取主数组（通常是 'arr_0' 或直接就是 DATA_S）
if 'DATA_S' in data:
    rte_array = data['DATA_S']
else:
    rte_array = data['arr_0']

print(f"   数组形状: {rte_array.shape}")
# 预期: (12053, 6, 21000)

# 2. 创建输出文件夹
output_folder = os.path.join(raw_data_folder, 'rte_preprocessed')
os.makedirs(output_folder, exist_ok=True)

print("\n2. 开始转换格式...")

# 3. 逐个样本转换
n_samples, n_channels, n_timesteps = rte_array.shape

for i in range(n_samples):
    if i % 500 == 0:
        print(f"   转换进度: {i}/{n_samples}")
    
    # 提取第 i 个样本 (6个通道)
    sample_data = rte_array[i]  # shape: (6, 21000)
    
    # 创建 DataFrame，与 PROTECT-90 格式对齐
    # 列名: time_s, 电压3相, 电流3相
    df = pd.DataFrame({
        'time_s': np.arange(n_timesteps) / 6400.0,  # 6400 Hz 采样率
        'voltage_L1': sample_data[0],
        'voltage_L2': sample_data[1],
        'voltage_L3': sample_data[2],
        'current_L1': sample_data[3],
        'current_L2': sample_data[4],
        'current_L3': sample_data[5],
    })
    
    # 保存为 .pkl 文件 (和 PROTECT-90 一样的格式)
    output_path = os.path.join(output_folder, f'rte_sample_{i}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(df, f)

print(f"\n3. 转换完成！共 {n_samples} 个样本保存到: {output_folder}")

# 4. 验证
print("\n4. 验证转换结果...")
sample_path = os.path.join(output_folder, 'rte_sample_0.pkl')
with open(sample_path, 'rb') as f:
    test_df = pickle.load(f)
print(f"   第一个样本: {test_df.shape[0]} 行 × {test_df.shape[1]} 列")
print(f"   列名: {test_df.columns.tolist()}")
print(f"   时间范围: {test_df['time_s'].min():.4f}s ~ {test_df['time_s'].max():.4f}s")
print("\n✅ 适配完成！")