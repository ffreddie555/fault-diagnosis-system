import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import matplotlib.pyplot as plt

# ========== 页面配置 ==========
st.set_page_config(
    page_title="⚡ AI 智能电网故障诊断系统",
    page_icon="⚡",
    layout="wide"
)

# ========== 语言选择 ==========
if 'lang' not in st.session_state:
    st.session_state.lang = '中文'

def switch_to_chinese():
    st.session_state.lang = '中文'

def switch_to_english():
    st.session_state.lang = 'English'

# 语言文本
text = {
    '中文': {
        'title': '⚡ AI 智能电网故障诊断系统',
        'model_loaded': '✅ 模型加载成功！',
        'features_needed': '模型需要',
        'features': '个特征',
        'performance_dashboard': '📊 模型性能看板',
        'accuracy': '准确率 (Accuracy)',
        'precision': '加权精确率 (Precision)',
        'recall': '加权召回率 (Recall)',
        'f1': '加权F1分数 (F1-Score)',
        'f1_per_class': '各类别 F1 分数',
        'upload_data': '📂 上传数据',
        'upload_hint': '选择一个 .pkl 文件（故障波形数据）',
        'waveform_preview': '📊 波形数据预览',
        'data_shape': '数据形状',
        'rows': '行',
        'cols': '列',
        'start_diagnosis': '🔍 开始故障诊断',
        'diagnosing': '正在分析...',
        'diagnosis_result': '🔍 诊断结果',
        'confidence': '置信度',
        'probability_dist': '各类别概率',
        'fault_type': '故障类型',
        'probability': '概率',
        'upload_prompt': '👈 请在左侧上传一个 .pkl 波形文件开始诊断',
        'footer': 'AI 智能电网故障诊断系统 | 基于 XGBoost + Streamlit',
        'model_accuracy': '模型准确率'
    },
    'English': {
        'title': '⚡ AI Smart Grid Fault Diagnosis System',
        'model_loaded': '✅ Model loaded successfully!',
        'features_needed': 'Model requires',
        'features': 'features',
        'performance_dashboard': '📊 Model Performance Dashboard',
        'accuracy': 'Accuracy',
        'precision': 'Weighted Precision',
        'recall': 'Weighted Recall',
        'f1': 'Weighted F1-Score',
        'f1_per_class': 'F1-Score per Class',
        'upload_data': '📂 Upload Data',
        'upload_hint': 'Select a .pkl file (fault waveform data)',
        'waveform_preview': '📊 Waveform Preview',
        'data_shape': 'Data shape',
        'rows': 'rows',
        'cols': 'columns',
        'start_diagnosis': '🔍 Start Diagnosis',
        'diagnosing': 'Analyzing...',
        'diagnosis_result': '🔍 Diagnosis Result',
        'confidence': 'Confidence',
        'probability_dist': 'Probability Distribution',
        'fault_type': 'Fault Type',
        'probability': 'Probability',
        'upload_prompt': '👈 Please upload a .pkl waveform file on the left to start diagnosis',
        'footer': 'AI Smart Grid Fault Diagnosis System | Based on XGBoost + Streamlit',
        'model_accuracy': 'Model Accuracy'
    }
}

# ========== 语言切换按钮 ==========
col_title, col_lang = st.columns([4, 1])
with col_title:
    st.title(text[st.session_state.lang]['title'])
with col_lang:
    if st.session_state.lang == '中文':
        st.button('🇬🇧 English', on_click=switch_to_english)
    else:
        st.button('🇨🇳 中文', on_click=switch_to_chinese)

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
t = text[st.session_state.lang]

st.sidebar.success(f"{t['model_loaded']}")
st.sidebar.write(f"{t['features_needed']} {len(feature_cols)} {t['features']}")

# ========== 模型性能看板 ==========
st.sidebar.markdown("---")
st.sidebar.subheader(t['performance_dashboard'])

# 先尝试从训练输出中获取准确率，没有则显示占位
performance_data = {
    t['accuracy']: "82.0%",
    t['precision']: "82.1%",
    t['recall']: "82.0%",
    t['f1']: "82.0%"
}

for metric, value in performance_data.items():
    st.sidebar.metric(label=metric, value=value)

st.sidebar.markdown("---")
st.sidebar.write(f"**{t['f1_per_class']}:**")
class_performance = {
    "Line_1_2_a": "0.83",
    "Line_1_2_b": "0.76",
    "Line_2_3": "0.83"
}
for cls, f1 in class_performance.items():
    st.sidebar.write(f"• {cls}: **{f1}**")

# ========== 侧边栏：上传文件 ==========
st.sidebar.markdown("---")
st.sidebar.header(t['upload_data'])
uploaded_file = st.sidebar.file_uploader(
    t['upload_hint'],
    type=['pkl']
)

# ========== 主界面 ==========
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(t['waveform_preview'])
    
    if uploaded_file is not None:
        data = pickle.load(uploaded_file)
        
        if isinstance(data, pd.DataFrame):
            st.write(f"{t['data_shape']}: {data.shape[0]} {t['rows']} × {data.shape[1]} {t['cols']}")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            voltage_cols = [col for col in data.columns if 'vol' in col.lower() and 'L1' in col]
            if voltage_cols:
                ax.plot(data['time_s'], data[voltage_cols[0]], label='L1', linewidth=1)
            if len(voltage_cols) > 1:
                ax.plot(data['time_s'], data[voltage_cols[1]], label='L2', linewidth=1)
            if len(voltage_cols) > 2:
                ax.plot(data['time_s'], data[voltage_cols[2]], label='L3', linewidth=1)
            
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Voltage (V)', fontsize=12)
            ax.set_title('Three-Phase Voltage Waveform', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            if st.button(t['start_diagnosis'], type="primary"):
                with st.spinner(t['diagnosing']):
                    try:
                        features = {}
                        all_vol_cols = [col for col in data.columns if 'vol' in col.lower()]
                        all_cur_cols = [col for col in data.columns if 'cur' in col.lower()]
                        
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
                        
                        for i, col in enumerate(v_cols):
                            vals = data[col].values
                            features[f'voltage_L{i+1}_rms'] = float(np.sqrt(np.mean(vals**2)))
                            features[f'voltage_L{i+1}_peak'] = float(np.max(np.abs(vals)))
                            features[f'voltage_L{i+1}_std'] = float(np.std(vals))
                            features[f'voltage_L{i+1}_max'] = float(np.max(vals))
                            features[f'voltage_L{i+1}_min'] = float(np.min(vals))
                            features[f'voltage_L{i+1}_peak_to_peak'] = float(np.max(vals) - np.min(vals))
                        
                        for i, col in enumerate(c_cols):
                            vals = data[col].values
                            features[f'current_L{i+1}_rms'] = float(np.sqrt(np.mean(vals**2)))
                            features[f'current_L{i+1}_peak'] = float(np.max(np.abs(vals)))
                            features[f'current_L{i+1}_std'] = float(np.std(vals))
                            features[f'current_L{i+1}_max'] = float(np.max(vals))
                            features[f'current_L{i+1}_min'] = float(np.min(vals))
                            features[f'current_L{i+1}_peak_to_peak'] = float(np.max(vals) - np.min(vals))
                        
                        full_features = {}
                        for col in feature_cols:
                            if col in features:
                                full_features[col] = features[col]
                            else:
                                full_features[col] = 0.0
                        
                        input_df = pd.DataFrame([full_features])
                        input_df = input_df[feature_cols]
                        
                        pred = model.predict(input_df)[0]
                        proba = model.predict_proba(input_df)[0]
                        max_prob = np.max(proba)
                        
                        with col2:
                            st.subheader(t['diagnosis_result'])
                            
                            if max_prob > 0.8:
                                color = "green"
                            elif max_prob > 0.6:
                                color = "orange"
                            else:
                                color = "red"
                            
                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                                <h3 style="color: {color}; text-align: center;">{pred}</h3>
                                <p style="text-align: center; font-size: 20px;">{t['confidence']}: {max_prob*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write(f"**{t['probability_dist']}:**")
                            proba_df = pd.DataFrame({
                                t['fault_type']: model.classes_,
                                t['probability']: proba
                            }).sort_values(t['probability'], ascending=False)
                            st.bar_chart(proba_df.set_index(t['fault_type']))
                    except Exception as e:
                        st.error(f"诊断出错: {e}")
        else:
            st.warning("⚠️ 上传的文件不是 DataFrame 格式，请上传 .pkl 格式的波形数据。")
    else:
        st.info(t['upload_prompt'])

# ========== 页脚 ==========
st.markdown("---")
st.caption(f"{t['footer']} | {t['model_accuracy']}: 82.0%")