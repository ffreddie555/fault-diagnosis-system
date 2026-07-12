import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import matplotlib.pyplot as plt

# ========== 页面配置 ==========
st.set_page_config(
    page_title="AI电力故障诊断系统",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI 智能电网故障诊断系统")
st.markdown("---")

# ========== 加载模型 ==========
@st.cache_resource
def load_model():
    project_folder = 'D:\\Fault_Diagnosis_Project'
    code_folder = os.path.join(project_folder, '02_Code')
    
    model_path = os.path.join(code_folder, 'fault_classifier_model.pkl')
    feature_cols_path = os.path.join(code_folder, 'feature_columns.pkl')
    
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_cols_path)
    
    return model, feature_cols

model, feature_cols = load_model()
st.sidebar.success("✅ 模型加载成功！")
st.sidebar.write(f"模型需要 {len(feature_cols)} 个特征")

# ========== 侧边栏：上传文件 ==========
st.sidebar.header("📂 上传数据")
uploaded_file = st.sidebar.file_uploader(
    "选择一个 .pkl 文件（故障波形数据）",
    type=['pkl']
)

# ========== 主界面 ==========
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 波形数据预览")
    
    if uploaded_file is not None:
        # 读取上传的 .pkl 文件
        data = pickle.load(uploaded_file)
        
        # 如果是 DataFrame，显示数据信息
        if isinstance(data, pd.DataFrame):
            st.write(f"数据形状: {data.shape[0]} 行 × {data.shape[1]} 列")
            
            # 画波形图
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # 找电压列
            voltage_cols = [col for col in data.columns if 'vol' in col.lower() and 'L1' in col]
            if voltage_cols:
                ax.plot(data['time_s'], data[voltage_cols[0]], label='L1电压', linewidth=1)
            if len(voltage_cols) > 1:
                ax.plot(data['time_s'], data[voltage_cols[1]], label='L2电压', linewidth=1)
            if len(voltage_cols) > 2:
                ax.plot(data['time_s'], data[voltage_cols[2]], label='L3电压', linewidth=1)
            
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('电压 (V)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # ========== 提取特征并预测 ==========
            if st.button("🔍 开始故障诊断", type="primary"):
                with st.spinner("正在分析..."):
                    try:
                        # 提取特征（与训练时完全一致）
                        features = {}
                        
                        # 找电压列和电流列
                        all_vol_cols = [col for col in data.columns if 'vol' in col.lower()]
                        all_cur_cols = [col for col in data.columns if 'cur' in col.lower()]
                        
                        # 精确匹配三相
                        v_cols = []
                        for phase in ['L1', 'L2', 'L3']:
                            for col in all_vol_cols:
                                if phase in col:
                                    v_cols.append(col)
                                    break
                        v_cols = v_cols[:3]
                        
                        c_cols = []
                        for phase in ['L1', 'L2', 'L3']:
                            for col in all_cur_cols:
                                if phase in col:
                                    c_cols.append(col)
                                    break
                        c_cols = c_cols[:3]
                        
                        # 电压特征
                        for i, col in enumerate(v_cols):
                            vals = data[col].values
                            features[f'voltage_L{i+1}_rms'] = float(np.sqrt(np.mean(vals**2)))
                            features[f'voltage_L{i+1}_peak'] = float(np.max(np.abs(vals)))
                            features[f'voltage_L{i+1}_std'] = float(np.std(vals))
                            features[f'voltage_L{i+1}_max'] = float(np.max(vals))
                            features[f'voltage_L{i+1}_min'] = float(np.min(vals))
                            features[f'voltage_L{i+1}_peak_to_peak'] = float(np.max(vals) - np.min(vals))
                        
                        # 电流特征
                        for i, col in enumerate(c_cols):
                            vals = data[col].values
                            features[f'current_L{i+1}_rms'] = float(np.sqrt(np.mean(vals**2)))
                            features[f'current_L{i+1}_peak'] = float(np.max(np.abs(vals)))
                            features[f'current_L{i+1}_std'] = float(np.std(vals))
                            features[f'current_L{i+1}_max'] = float(np.max(vals))
                            features[f'current_L{i+1}_min'] = float(np.min(vals))
                            features[f'current_L{i+1}_peak_to_peak'] = float(np.max(vals) - np.min(vals))
                        
                        # 补充：如果某些特征缺失，补0
                        # 构建完整的特征字典
                        full_features = {}
                        for col in feature_cols:
                            if col in features:
                                full_features[col] = features[col]
                            else:
                                full_features[col] = 0.0
                        
                        # 转成DataFrame
                        input_df = pd.DataFrame([full_features])
                        
                        # 确保列顺序与训练时一致
                        input_df = input_df[feature_cols]
                        
                        # 预测
                        pred = model.predict(input_df)[0]
                        proba = model.predict_proba(input_df)[0]
                        max_prob = np.max(proba)
                        
                        # 显示结果
                        with col2:
                            st.subheader("🔍 诊断结果")
                            
                            # 颜色
                            if max_prob > 0.8:
                                color = "green"
                            elif max_prob > 0.6:
                                color = "orange"
                            else:
                                color = "red"
                            
                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                                <h3 style="color: {color}; text-align: center;">{pred}</h3>
                                <p style="text-align: center; font-size: 20px;">置信度: {max_prob*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 显示各类概率
                            st.write("各类别概率:")
                            proba_df = pd.DataFrame({
                                '故障类型': model.classes_,
                                '概率': proba
                            }).sort_values('概率', ascending=False)
                            st.bar_chart(proba_df.set_index('故障类型'))
                    except Exception as e:
                        st.error(f"诊断出错: {e}")
        else:
            st.warning("⚠️ 上传的文件不是 DataFrame 格式，请上传 .pkl 格式的波形数据。")
    else:
        st.info("👈 请在左侧上传一个 .pkl 波形文件开始诊断")

# ========== 页脚 ==========
st.markdown("---")
st.caption("AI 智能电网故障诊断系统 | 基于随机森林 + Streamlit")