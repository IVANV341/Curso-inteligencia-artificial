import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Cargar modelo y scaler
model = joblib.load("modelo_svm.joblib")
scaler = joblib.load("scaler.joblib")

# Función para escalar valores de entrada
def scale_input(values):
    scaled = scaler.transform([values])
    return np.clip(scaled, 0, 1)  # Asegurar que los valores estén en [0,1]

# Configuración de la app
st.set_page_config(layout="wide")
st.markdown("# 🌍 Predicción de Calidad del Aire")

# Sidebar para entrada de datos
st.sidebar.markdown("### 🏡 Ingrese los valores:")
PM10 = st.sidebar.slider("PM10", 0.0, 300.0, 50.0)
NO2 = st.sidebar.slider("NO2", 0.0, 200.0, 30.0)
Humedad = st.sidebar.slider("Humedad (%)", 0.0, 100.0, 50.0)
PM2_5 = st.sidebar.slider("PM2.5", 0.0, 200.0, 25.0)
O3 = st.sidebar.slider("O3", 0.0, 300.0, 100.0)
Lluvia = st.sidebar.slider("Lluvia (mm)", 0.0, 100.0, 5.0)
Temperatura = st.sidebar.slider("Temperatura (°C)", -10.0, 50.0, 25.0)
Vel_Viento = st.sidebar.slider("Vel. Viento (m/s)", 0.0, 20.0, 5.0)
Dir_Viento = st.sidebar.slider("Dir. Viento (°)", 0.0, 360.0, 180.0)

# Botón de predicción
if st.sidebar.button("🔍 Predecir Calidad del Aire"):
    input_values = np.array([PM10, NO2, Humedad, PM2_5, O3, Lluvia, Temperatura, Vel_Viento, Dir_Viento])
    input_scaled = scale_input(input_values)
    
    # Mostrar valores escalados
    st.markdown("### 🎨 Valores escalados:")
    st.dataframe({"value": input_scaled.flatten()})
    
    # Realizar predicción
    prediction = model.predict(input_scaled)[0]
    
    # Mostrar resultado
    if prediction == "Buena":
        color, message = "#4CAF50", "🟢 La calidad del aire es: Buena"
    elif prediction == "Regular":
        color, message = "#FFC107", "🟡 La calidad del aire es: Regular"
    else:
        color, message = "#D32F2F", "🔴 La calidad del aire es: Mala"
    
    st.markdown(f'<div style="background-color:{color}; padding: 15px; border-radius: 10px; text-align: center;">{message}</div>', unsafe_allow_html=True)
    
    # Recomendaciones
    if prediction == "Mala":
        st.markdown("💨 Se recomienda evitar actividades al aire libre. Grupos vulnerables deben permanecer en interiores con ventanas cerradas.")
