import streamlit as st
import pickle
import numpy as np

with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page config
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="wide")

# ---- Custom CSS ----
st.markdown("""
<style>
.main-header {
    background-color: #15573C;
    padding: 20px 30px;
    border-radius: 12px;
    color: white;
    margin-bottom: 25px;
}
.card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e0e0e0;
    margin-bottom: 15px;
}
.result-card {
    background-color: #15573C;
    padding: 25px;
    border-radius: 16px;
    color: white;
}
.badge {
    background-color: #2ecc71;
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    display: inline-block;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="main-header">
    <h2>🌱 AI-Based Crop & Fertilizer Recommendation System</h2>
    <p>Smarter Farming | Higher Yield | Better Tomorrow</p>
</div>
""", unsafe_allow_html=True)

# ---- Layout: Form (left) + Result (right) ----
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🧪 Enter Soil & Weather Details")
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Nitrogen (N)", 0, 200, 90)
            P = st.number_input("Phosphorus (P)", 0, 200, 42)
            K = st.number_input("Potassium (K)", 0, 200, 43)
            temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
        with c2:
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
            ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

        predict = st.button("🔍 Predict Best Crop", use_container_width=True)

with col2:
    st.subheader("📊 Recommendation")
    if predict:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]

        st.markdown(f"""
        <div class="result-card">
            <p style="color:#a8e6b0; margin-bottom:0;">RECOMMENDED CROP</p>
            <h1>🌾 {prediction.upper()}</h1>
            <span class="badge">Suitable for your soil & climate</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card" style="margin-top:15px;">
            <b>Soil Details</b><br><br>
            🔵 Nitrogen (N): <b>{N}</b> &nbsp;&nbsp;
            🟣 Phosphorus (P): <b>{P}</b> &nbsp;&nbsp;
            🟠 Potassium (K): <b>{K}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Enter values and click 'Predict Best Crop'")