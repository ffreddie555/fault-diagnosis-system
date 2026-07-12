import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os

data_folder = 'D:\\Fault_Diagnosis_Project\\01_Raw_Data\\hv_double_line_90kv_preprocessed_data'
pkl_file_name = '1_sample_hv_double_line_90kv.pkl'  
pkl_path = os.path.join(data_folder, pkl_file_name)

print(f"正在读取: {pkl_path}")

with open(pkl_path, 'rb') as f:
    df = pickle.load(f)

print(f"\n数据形状: {df.shape}  (行数, 列数)")
print(f"时间范围: {df['time_s'].min():.4f}s ~ {df['time_s'].max():.4f}s")

voltage_cols = [
    'Bus_1_Line_01_02A_vol_L1_V',
    'Bus_1_Line_01_02A_vol_L2_V', 
    'Bus_1_Line_01_02A_vol_L3_V'
]

plt.figure(figsize=(15, 6))

for col in voltage_cols:
    plt.plot(df['time_s'], df[col], label=col, linewidth=1)

plt.title(f'输电线路三相电压波形 - 样本 {pkl_file_name}')
plt.xlabel('时间 (秒)')
plt.ylabel('电压 (V)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('voltage_waveform.png', dpi=300, bbox_inches='tight')
print("\n✅ 电压波形图已保存为: voltage_waveform.png")

plt.show()

current_cols = [
    'Bus_1_Line_01_02A_cur_L1_A',
    'Bus_1_Line_01_02A_cur_L2_A',
    'Bus_1_Line_01_02A_cur_L3_A'
]

plt.figure(figsize=(15, 6))

for col in current_cols:
    plt.plot(df['time_s'], df[col], label=col, linewidth=1)

plt.title(f'输电线路三相电流波形 - 样本 {pkl_file_name}')
plt.xlabel('时间 (秒)')
plt.ylabel('电流 (A)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('current_waveform.png', dpi=300, bbox_inches='tight')
print("✅ 电流波形图已保存为: current_waveform.png")

plt.show()

print("\n🎉 完成！你现在有 voltage_waveform.png 和 current_waveform.png 两张图了。")