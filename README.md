\# AI-Driven Fault Diagnosis System for Power Transmission Lines



\*\*Author\*\*: HONG XUANYUE 

\*\*Date\*\*: 2026年7月  

\*\*Project for\*\*: MSc Application (NUS/NTU/HKU)



\## Project Overview

This project develops a machine learning-based fault diagnosis system for 90kV double-circuit transmission lines. The system automatically identifies fault types from waveform data and provides interpretable diagnostic reports.



\## Key Features

\- ⚡ \*\*Data Processing\*\*: Automated feature extraction from .pkl waveform files

\- 🤖 \*\*AI Model\*\*: Random Forest classifier with 82% accuracy on 3-class fault diagnosis

\- 🌐 \*\*Web Interface\*\*: Interactive Streamlit dashboard for real-time fault diagnosis

\- 📊 \*\*Visualization\*\*: Waveform display, confidence scores, and probability distribution



\## System Architecture



\## Dataset

\- \*\*Source\*\*: PROTECT-90 public dataset

\- \*\*Samples\*\*: 1,000 fault events

\- \*\*Fault Types\*\*: Line\_1\_2\_a, Line\_1\_2\_b, Line\_2\_3 (merged)

\- \*\*Features\*\*: 51 features (RMS, peak, standard deviation, crest factor, imbalance ratio, etc.)



\## Model Performance

| Class | Precision | Recall | F1-Score |

|-------|-----------|--------|----------|

| Line\_1\_2\_a | 0.86 | 0.80 | 0.83 |

| Line\_1\_2\_b | 0.79 | 0.73 | 0.76 |

| Line\_2\_3 | 0.81 | 0.86 | 0.83 |

| \*\*Overall\*\* | \*\*0.82\*\* | \*\*0.82\*\* | \*\*0.82\*\* |



\## Project Structure



\## How to Run

1\. Install dependencies: `pip install -r requirements.txt`

2\. Launch web interface: `streamlit run app.py`

3\. Upload a .pkl file and click "Start Diagnosis"



\## Future Work

\- Apply deep learning (CNN/LSTM) for end-to-end fault diagnosis

\- Deploy on edge devices for real-time monitoring

\- Integrate with SCADA systems for automated response



\## Contact

1330774638@qq.com

