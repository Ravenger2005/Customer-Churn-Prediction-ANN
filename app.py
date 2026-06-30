import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Enterprise Churn AI Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-container {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #eef2f5;
    }
    .metric-title { font-size: 14px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .metric-value { font-size: 42px; font-weight: 800; margin-top: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- Load Assets safely ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)  
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

try:
    model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()
except Exception as e:
    st.error(f"Asset Load Error: {e}. Ensure weights are in the directory.")
    st.stop()

# --- Header ---
st.title('🔮 Enterprise Customer Churn Predictor')
st.markdown("Predict consumer defection risk in real-time using Deep Learning. Adjust the parameters below to run the simulation.")
st.divider()

# --- Layout: 3 Columns for User Input ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📋 Demographics")
    geography = st.selectbox('Geography Country', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender Identity', label_encoder_gender.classes_)
    age = st.slider('Age (Years)', 18, 92, 38)

with col2:
    st.markdown("### 💰 Financial Profile")
    balance = st.number_input('Account Balance ($)', min_value=0.0, value=45000.0, step=1000.0)
    estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=75000.0, step=1000.0)
    credit_score = st.slider('Credit Bureau Score', 300, 850, 640)

with col3:
    st.markdown("### 🤝 Engagement Metrics")
    tenure = st.slider('Tenure with Firm (Years)', 0, 10, 4)
    num_of_products = st.slider('Active Products Registered', 1, 4, 2)
    has_cr_card = st.radio('Holds Corporate Credit Card?', ['No', 'Yes'], index=1)
    is_active_member = st.radio('Is Active Engagement Member?', ['No', 'Yes'], index=1)

# Mapping variables
cr_card_val = 1 if has_cr_card == 'Yes' else 0
active_member_val = 1 if is_active_member == 'Yes' else 0

st.divider()

# --- Execution Button ---
if st.button('🚀 Run Neural Network Assessment Pipeline', use_container_width=True):
    
    # Animated Pipeline Loading UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    for percent in range(0, 101, 25):
        time.sleep(0.12)
        progress_bar.progress(percent)
        if percent == 25: status_text.text("Transforming categorical strings with One-Hot encoding matrices...")
        elif percent == 50: status_text.text("Applying StandardScaler mathematical vector scaling...")
        elif percent == 75: status_text.text("Passing array tensors to TensorFlow ANN graph layers...")
        elif percent == 100: status_text.text("Inference Complete.")
    time.sleep(0.2)
    progress_bar.empty()
    status_text.empty()

    # Preprocessing Pipeline Execution
    input_data = pd.DataFrame({
        'CreditScore': [credit_score], 'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age], 'Tenure': [tenure], 'Balance': [balance], 'NumOfProducts': [num_of_products],
        'HasCrCard': [cr_card_val], 'IsActiveMember': [active_member_val], 'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)
    prediction_prob = float(prediction[0][0])
    
    # Display Analytics Dashboard
    st.markdown("## 📊 Model Inference Results")
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        color_theme = '#e63946' if prediction_prob > 0.5 else '#2a9d8f'
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-title">RISK PROBABILITY SCORE</div>
                <div class="metric-value" style="color: {color_theme};">{prediction_prob * 100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        if prediction_prob > 0.5:
            st.error("⚠️ **High Defection Risk Alert**\n\nThis account's metrics strongly correlate with historic customer churn profiles. High risk configuration detected.")
        else:
            st.success("✅ **Stable Account Status**\n\nThis account falls safely within nominal retention boundaries. No risk response required.")

    # Dynamic Dashboard Analytics Chart
    st.markdown("### 📈 Financial Ratio Context")
    chart_data = pd.DataFrame({
        "Financial Vector": ["Available Balance", "Estimated Salary"],
        "Amount ($)": [balance, estimated_salary]
    })
    st.bar_chart(data=chart_data, x="Financial Vector", y="Amount ($)", use_container_width=True)

st.divider()

