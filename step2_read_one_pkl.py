import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

data_folder = 'D:\\Fault_Diagnosis_Project\\01_Raw_Data\\hv_double_line_90kv_preprocessed_data'

pkl_file_name = '1_sample_hv_double_line_90kv.pkl'

pkl_path = os.path.join(data_folder, pkl_file_name)

print(f"正在读取: {pkl_path}")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("\n=== 数据类型 ===")
print(type(data))

if isinstance(data, dict):
    print("\n=== 字典的键（keys）===")
    for key in data.keys():
        print(f"  - {key}")
    
    print("\n=== 每个键对应的数据 ===")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape = {value.shape}, dtype = {value.dtype if hasattr(value, 'dtype') else 'N/A'}")
        elif isinstance(value, list):
            print(f"  {key}: list, 长度 = {len(value)}")
        else:
            print(f"  {key}: {type(value)}")

elif isinstance(data, np.ndarray):
    print(f"\n=== 数据形状 ===")
    print(data.shape)
    print(f"数据类型: {data.dtype}")

elif isinstance(data, list):
    print(f"\n=== 列表长度 ===")
    print(len(data))
    if len(data) > 0:
        print(f"第一个元素的类型: {type(data[0])}")

elif 'pandas' in str(type(data)):
    print("\n=== DataFrame 列名 ===")
    print(data.columns.tolist())

if isinstance(data, dict):
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim == 1:
            print(f"\n正在画图: {key}")
            plt.figure(figsize=(15, 5))
            plt.plot(value, linewidth=1)
            plt.title(f'故障样本波形: {key}')
            plt.xlabel('采样点')
            plt.ylabel('幅值')
            plt.grid(True, alpha=0.3)
            plt.savefig('first_waveform.png', dpi=300, bbox_inches='tight')
            print("✅ 图片已保存为: first_waveform.png")
            plt.show()
            break
        elif isinstance(value, np.ndarray) and value.ndim == 2:
            print(f"\n正在画图: {key} (二维数组，画第1列)")
            plt.figure(figsize=(15, 5))
            plt.plot(value[:, 0], linewidth=1)
            plt.title(f'故障样本波形: {key} (第1列)')
            plt.xlabel('采样点')
            plt.ylabel('幅值')
            plt.grid(True, alpha=0.3)
            plt.savefig('first_waveform.png', dpi=300, bbox_inches='tight')
            print("✅ 图片已保存为: first_waveform.png")
            plt.show()
            break

elif isinstance(data, np.ndarray) and data.ndim == 1:
    plt.figure(figsize=(15, 5))
    plt.plot(data, linewidth=1)
    plt.title('故障样本波形')
    plt.xlabel('采样点')
    plt.ylabel('幅值')
    plt.grid(True, alpha=0.3)
    plt.savefig('first_waveform.png', dpi=300, bbox_inches='tight')
    print("✅ 图片已保存为: first_waveform.png")
    plt.show()

elif isinstance(data, np.ndarray) and data.ndim == 2:
    plt.figure(figsize=(15, 5))
    for i in range(min(data.shape[1], 3)):  
        plt.plot(data[:, i], label=f'列{i}', linewidth=1)
    plt.title('故障样本波形 (多通道)')
    plt.xlabel('采样点')
    plt.ylabel('幅值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('first_waveform.png', dpi=300, bbox_inches='tight')
    print("✅ 图片已保存为: first_waveform.png")
    plt.show()

else:
    print("\n⚠️ 数据格式暂不支持自动画图，请把上面打印的信息发给我。")