import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import matplotlib.pyplot as plt
from openai import OpenAI

# ========== 页面配置 ==========
st.set_page_config(
    page_title="⚡ AI 智能电网故障诊断系统",
    page_icon="⚡",
    layout="wide"
)

# ========== 初始化 session_state ==========
if 'lang' not in st.session_state:
    st.session_state.lang = '中文'
if 'diagnosis_result' not in st.session_state:
    st.session_state.diagnosis_result = None
if 'report_content' not in st.session_state:
    st.session_state.report_content = None
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

def switch_to_chinese():
    st.session_state.lang = '中文'

def switch_to_english():
    st.session_state.lang = 'English'

# ========== 语言文本 ==========
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
        'footer': 'AI 智能电网故障诊断系统 | 基于 XGBoost + SMOTE',
        'model_accuracy': '模型准确率',
        'operation_status': '⚙️ 运行状态',
        'report_section': '📋 故障分析报告',
        'generate_report': '📝 生成诊断报告',
        'cause_analysis': '🔍 可能原因分析',
        'maintenance_advice': '🛠️ 运维建议',
        'generating': '正在生成报告...',
        'no_api_key': '⚠️ 请先在侧边栏配置 DeepSeek API Key',
        'fault_confirmation': '故障类型确认',
        'confidence_assessment': '置信度评估',
        'debug_info': '🔍 调试：置信度原始值 = {:.4f}，百分比 = {:.1f}%',
        'status_critical': '🚨 严重故障',
        'status_critical_desc': '检测到严重故障，需要立即处理，建议紧急停机检修。',
        'status_fault': '🔴 故障状态',
        'status_fault_desc': '已检测到明确故障，建议安排检修，避免故障扩大。',
        'status_warning': '⚠️ 预警状态',
        'status_warning_desc': '检测到轻微异常，建议加强监测，关注参数变化趋势。',
        'status_normal': '✅ 正常运行',
        'status_normal_desc': '系统运行平稳，电压/电流参数正常，未检测到故障迹象。'
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
        'footer': 'AI Smart Grid Fault Diagnosis System | Based on XGBoost + SMOTE',
        'model_accuracy': 'Model Accuracy',
        'operation_status': '⚙️ Operation Status',
        'report_section': '📋 Fault Analysis Report',
        'generate_report': '📝 Generate Report',
        'cause_analysis': '🔍 Possible Cause Analysis',
        'maintenance_advice': '🛠️ Maintenance Advice',
        'generating': 'Generating report...',
        'no_api_key': '⚠️ Please configure DeepSeek API Key in the sidebar',
        'fault_confirmation': 'Fault Type Confirmation',
        'confidence_assessment': 'Confidence Assessment',
        'debug_info': '🔍 Debug: Raw confidence = {:.4f}, Percentage = {:.1f}%',
        'status_critical': '🚨 Critical Fault',
        'status_critical_desc': 'Critical fault detected. Immediate action required. Recommend emergency shutdown and repair.',
        'status_fault': '🔴 Fault Detected',
        'status_fault_desc': 'Fault detected. Recommend scheduling maintenance to prevent escalation.',
        'status_warning': '⚠️ Warning',
        'status_warning_desc': 'Minor anomalies detected. Recommend enhanced monitoring and trend analysis.',
        'status_normal': '✅ Normal Operation',
        'status_normal_desc': 'System is operating normally. All voltage/current parameters are within normal ranges.'
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
    scaler_path = os.path.join(code_folder, 'scaler.pkl')
    label_encoder_path = os.path.join(code_folder, 'label_encoder.pkl')
    feature_cols_path = os.path.join(code_folder, 'feature_columns.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    feature_cols = joblib.load(feature_cols_path)
    
    return model, scaler, label_encoder, feature_cols

model, scaler, label_encoder, feature_cols = load_model()
t = text[st.session_state.lang]

st.sidebar.success(t['model_loaded'])
st.sidebar.write(f"{t['features_needed']} {len(feature_cols)} {t['features']}")

# ========== 模型性能看板 ==========
st.sidebar.markdown("---")
st.sidebar.subheader(t['performance_dashboard'])

performance_data = {
    t['accuracy']: "74.4%",
    t['precision']: "0.743",
    t['recall']: "0.744",
    t['f1']: "0.743"
}

for metric, value in performance_data.items():
    st.sidebar.metric(label=metric, value=value)

st.sidebar.markdown("---")
st.sidebar.write(f"**{t['f1_per_class']}:**")
class_performance = {
    "Line_1_2_a": "0.77",
    "Line_1_2_b": "0.59",
    "Line_2_3": "0.77"
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

# ========== DeepSeek API Key ==========
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 LLM 配置")
api_key = st.sidebar.text_input(
    "DeepSeek API Key",
    type="password",
    help="在 https://platform.deepseek.com/ 获取"
)
if api_key:
    st.sidebar.success("✅ API Key 已配置")
else:
    st.sidebar.info("⚠️ 请输入 API Key 以启用报告生成")


# ========== 定义特征提取函数 ==========
def extract_features_from_df(df):
    features = {}
    
    all_vol_cols = [col for col in df.columns if 'vol' in col.lower()]
    all_cur_cols = [col for col in df.columns if 'cur' in col.lower()]
    
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
        vals = df[col].values
        features[f'voltage_L{i+1}_rms'] = float(np.sqrt(np.mean(vals**2)))
        features[f'voltage_L{i+1}_peak'] = float(np.max(np.abs(vals)))
        features[f'voltage_L{i+1}_std'] = float(np.std(vals))
        features[f'voltage_L{i+1}_max'] = float(np.max(vals))
        features[f'voltage_L{i+1}_min'] = float(np.min(vals))
        features[f'voltage_L{i+1}_peak_to_peak'] = float(np.max(vals) - np.min(vals))
        zero_crossings = np.where(np.diff(np.sign(vals)))[0]
        if len(zero_crossings) > 1:
            time_span = df['time_s'].iloc[-1] - df['time_s'].iloc[0]
            features[f'voltage_L{i+1}_freq_est'] = float(len(zero_crossings) / (2 * time_span)) if time_span > 0 else 0.0
        else:
            features[f'voltage_L{i+1}_freq_est'] = 0.0
    
    for i, col in enumerate(c_cols):
        vals = df[col].values
        features[f'current_L{i+1}_rms'] = float(np.sqrt(np.mean(vals**2)))
        features[f'current_L{i+1}_peak'] = float(np.max(np.abs(vals)))
        features[f'current_L{i+1}_std'] = float(np.std(vals))
        features[f'current_L{i+1}_max'] = float(np.max(vals))
        features[f'current_L{i+1}_min'] = float(np.min(vals))
        features[f'current_L{i+1}_peak_to_peak'] = float(np.max(vals) - np.min(vals))
    
    v_rms = [features.get(f'voltage_L{i+1}_rms', 0) for i in range(3)]
    c_rms = [features.get(f'current_L{i+1}_rms', 0) for i in range(3)]
    
    features['total_voltage_rms'] = float(np.sqrt(np.mean([v**2 for v in v_rms])))
    features['total_current_rms'] = float(np.sqrt(np.mean([c**2 for c in c_rms])))
    features['voltage_unbalance'] = float(np.std(v_rms))
    features['current_unbalance'] = float(np.std(c_rms))
    
    for col in feature_cols:
        if col not in features:
            features[col] = 0.0
    
    return features


# ========== 生成 LLM 故障分析报告 ==========
def generate_fault_report(fault_type, confidence, proba_dict, api_key, lang='中文'):
    if lang == '中文':
        prompt = """你是一位资深的电力系统故障诊断专家。请根据以下诊断结果，生成一份简洁、专业的故障分析报告。

诊断结果：
- 故障类型：""" + fault_type + """
- 置信度：""" + str(round(confidence*100, 1)) + """%
- 各类别概率：""" + str(proba_dict) + """

请用中文输出，严格按以下格式：

【故障类型确认】
（确认诊断出的故障类型）

【置信度评估】
（评估该置信度是否可靠，分析可能的情况）

【可能原因分析】
（列出3条可能的原因）

【运维建议】
（列出3条具体的运维建议）

报告要求：语言专业、简洁、直接可用。"""
    else:
        prompt = """You are a senior power system fault diagnosis expert. Based on the following diagnostic results, generate a concise and professional fault analysis report.

Diagnostic Results:
- Fault Type: """ + fault_type + """
- Confidence: """ + str(round(confidence*100, 1)) + """%
- Class Probabilities: """ + str(proba_dict) + """

Please output in English, strictly in the following format:

[Fault Type Confirmation]
(Confirm the diagnosed fault type)

[Confidence Assessment]
(Assess whether the confidence is reliable and analyze possible scenarios)

[Possible Cause Analysis]
(List 3 possible causes)

[Maintenance Advice]
(List 3 specific maintenance recommendations)

Requirements: Professional, concise, directly usable."""
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的电力系统故障诊断专家。" if lang == '中文' else "You are a professional power system fault diagnosis expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return "报告生成失败: " + str(e)


# ========== 获取运行状态 ==========
def get_status_info(confidence, lang='中文'):
    if confidence >= 0.8:
        if lang == '中文':
            return '🚨 严重故障', '检测到严重故障，需要立即处理，建议紧急停机检修。'
        else:
            return '🚨 Critical Fault', 'Critical fault detected. Immediate action required. Recommend emergency shutdown and repair.'
    elif confidence >= 0.6:
        if lang == '中文':
            return '🔴 故障状态', '已检测到明确故障，建议安排检修，避免故障扩大。'
        else:
            return '🔴 Fault Detected', 'Fault detected. Recommend scheduling maintenance to prevent escalation.'
    elif confidence >= 0.4:
        if lang == '中文':
            return '⚠️ 预警状态', '检测到轻微异常，建议加强监测，关注参数变化趋势。'
        else:
            return '⚠️ Warning', 'Minor anomalies detected. Recommend enhanced monitoring and trend analysis.'
    else:
        if lang == '中文':
            return '✅ 正常运行', '系统运行平稳，电压/电流参数正常，未检测到故障迹象。'
        else:
            return '✅ Normal Operation', 'System is operating normally. All voltage/current parameters are within normal ranges.'


# ========== 解析 LLM 报告 ==========
def parse_report(report_text):
    lines = report_text.split('\n')
    sections = {
        'confirmation': '',
        'assessment': '',
        'causes': [],
        'advice': []
    }
    
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if '故障类型确认' in line or 'Fault Type Confirmation' in line:
            current_section = 'confirmation'
        elif '置信度评估' in line or 'Confidence Assessment' in line:
            current_section = 'assessment'
        elif '可能原因分析' in line or 'Possible Cause Analysis' in line:
            current_section = 'causes'
        elif '运维建议' in line or 'Maintenance Advice' in line:
            current_section = 'advice'
        elif current_section == 'confirmation' and not line.startswith('【') and not line.startswith('['):
            if sections['confirmation']:
                sections['confirmation'] += ' ' + line
            else:
                sections['confirmation'] = line
        elif current_section == 'assessment' and not line.startswith('【') and not line.startswith('['):
            if sections['assessment']:
                sections['assessment'] += ' ' + line
            else:
                sections['assessment'] = line
        elif current_section == 'causes' and line and not line.startswith('【') and not line.startswith('['):
            if line[0].isdigit() and '.' in line[:3]:
                sections['causes'].append(line)
            elif line.startswith('-') or line.startswith('•'):
                sections['causes'].append(line)
        elif current_section == 'advice' and line and not line.startswith('【') and not line.startswith('['):
            if line[0].isdigit() and '.' in line[:3]:
                sections['advice'].append(line)
            elif line.startswith('-') or line.startswith('•'):
                sections['advice'].append(line)
    
    return sections


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
                        features = extract_features_from_df(data)
                        
                        input_df = pd.DataFrame([features])
                        input_df = input_df[feature_cols]
                        
                        input_scaled = scaler.transform(input_df.values)
                        
                        pred_encoded = model.predict(input_scaled)[0]
                        pred = label_encoder.inverse_transform([pred_encoded])[0]
                        proba = model.predict_proba(input_scaled)[0]
                        max_prob = np.max(proba)
                        
                        st.session_state.diagnosis_result = {
                            'fault_type': pred,
                            'confidence': max_prob,
                            'proba': proba,
                            'classes': label_encoder.classes_
                        }
                        st.session_state.report_content = None
                        st.session_state.report_generated = False
                        
                    except Exception as e:
                        st.error(f"诊断出错: {e}")
                        print(f"错误详情: {e}")
                        import traceback
                        traceback.print_exc()
        else:
            st.warning("⚠️ 上传的文件不是 DataFrame 格式，请上传 .pkl 格式的波形数据。")
    else:
        st.info(t['upload_prompt'])

# ========== 右侧：诊断结果 + 报告 ==========
with col2:
    result = st.session_state.diagnosis_result
    
    if result is not None:
        # ---- 诊断结果 ----
        st.subheader(t['diagnosis_result'])
        
        pred = result['fault_type']
        max_prob = result['confidence']
        proba = result['proba']
        classes = result['classes']
        
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
        
        # ---- 调试信息 ----
        lang = st.session_state.lang
        st.info(t['debug_info'].format(max_prob, max_prob*100))
        
        # ---- 运行状态 ----
        st.markdown("---")
        
        status_text, status_desc = get_status_info(max_prob, lang)
        
        if max_prob >= 0.8:
            bg_color = "#ffebee"
            border_color = "#e53935"
        elif max_prob >= 0.6:
            bg_color = "#fff3e0"
            border_color = "#ff9800"
        elif max_prob >= 0.4:
            bg_color = "#fffde7"
            border_color = "#ff9800"
        else:
            bg_color = "#e8f5e9"
            border_color = "#4caf50"
        
        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 12px; border-radius: 10px; border-left: 5px solid {border_color};">
            <p style="font-size: 14px; font-weight: bold; margin: 0;">{t['operation_status']}</p>
            <p style="font-size: 18px; margin: 5px 0; font-weight: bold;">{status_text}</p>
            <p style="font-size: 13px; color: #555; margin: 3px 0;">{status_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ---- 概率分布 ----
        st.write(f"**{t['probability_dist']}:**")
        if len(classes) == len(proba):
            proba_df = pd.DataFrame({
                t['fault_type']: classes,
                t['probability']: proba
            }).sort_values(t['probability'], ascending=False)
            st.bar_chart(proba_df.set_index(t['fault_type']))
        
        # ---- 故障分析报告 ----
        st.markdown("---")
        st.subheader(t['report_section'])
        
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        with col_summary1:
            st.write(f"**{t['fault_type']}**")
            st.write(f":blue[{pred}]")
        with col_summary2:
            st.write(f"**{t['confidence']}**")
            st.write(f":blue[{max_prob*100:.1f}%]")
        with col_summary3:
            st.write(f"**{t['probability']}**")
            top_idx = np.argmax(proba)
            st.write(f":blue[{classes[top_idx]}: {proba[top_idx]*100:.1f}%]")
        
        # ---- 生成报告按钮 ----
        if api_key:
            if st.button(t['generate_report'], type="primary", key="generate_report_btn"):
                with st.spinner(t['generating']):
                    proba_dict = {classes[i]: proba[i] for i in range(len(classes))}
                    report = generate_fault_report(
                        pred,
                        max_prob,
                        proba_dict,
                        api_key,
                        st.session_state.lang
                    )
                    st.session_state.report_content = report
                    st.session_state.report_generated = True
            
            if st.session_state.report_generated and st.session_state.report_content is not None:
                report = st.session_state.report_content
                parsed = parse_report(report)
                
                st.markdown("---")
                
                if parsed['confirmation']:
                    st.markdown(f"**{t['fault_confirmation']}:** {parsed['confirmation']}")
                else:
                    st.markdown(f"**{t['fault_type']}:** {pred}")
                
                if parsed['assessment']:
                    st.markdown(f"**{t['confidence_assessment']}:** {parsed['assessment']}")
                else:
                    if st.session_state.lang == '中文':
                        st.markdown(f"**{t['confidence_assessment']}:** 置信度为 {max_prob*100:.1f}%，建议结合现场情况综合判断。")
                    else:
                        st.markdown(f"**{t['confidence_assessment']}:** Confidence is {max_prob*100:.1f}%. Recommend comprehensive assessment with on-site conditions.")
                
                if parsed['causes']:
                    st.markdown(f"**{t['cause_analysis']}**")
                    for cause in parsed['causes']:
                        st.markdown(f"- {cause}")
                else:
                    st.markdown(f"**{t['cause_analysis']}**")
                    if st.session_state.lang == '中文':
                        st.markdown("- 线路参数异常")
                        st.markdown("- 设备老化或绝缘性能下降")
                        st.markdown("- 外部环境因素影响")
                    else:
                        st.markdown("- Abnormal line parameters")
                        st.markdown("- Equipment aging or insulation degradation")
                        st.markdown("- External environmental factors")
                
                if parsed['advice']:
                    st.markdown(f"**{t['maintenance_advice']}**")
                    for advice in parsed['advice']:
                        st.markdown(f"- {advice}")
                else:
                    st.markdown(f"**{t['maintenance_advice']}**")
                    if st.session_state.lang == '中文':
                        st.markdown("- 建议巡检该线路段")
                        st.markdown("- 检查设备运行状态")
                        st.markdown("- 关注参数变化趋势")
                    else:
                        st.markdown("- Recommend inspecting this line section")
                        st.markdown("- Check equipment operating status")
                        st.markdown("- Monitor parameter trends")
        else:
            st.warning(t['no_api_key'])
    else:
        st.info("👈 请先上传文件并点击「开始故障诊断」")

# ========== 页脚 ==========
st.markdown("---")
st.caption(f"{t['footer']} | {t['model_accuracy']}: 74.4%")