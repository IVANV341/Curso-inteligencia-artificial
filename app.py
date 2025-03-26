import streamlit as st
import joblib
import numpy as np

# Cargar el modelo SVM y el escalador
modelo_svm = joblib.load("modelo_svm.joblib")
scaler = joblib.load("scaler.joblib")

# Título de la aplicación
st.title("🌍 Predicción de Calidad del Aire")

# Agregar imagen
st.image("https://www.mexicosocial.org/wp-content/uploads/2022/05/DEFENDER-EL-AIRE-LIMPIO.jpg", 
         caption="Defender el Aire Limpio", 
         use_container_width=True)

# Sección de entrada de datos en columnas
st.sidebar.header("Ingrese los valores:")

col1, col2, col3 = st.sidebar.columns(3)

with col1:
    PM10 = st.number_input("PM10", min_value=0.0, format="%.2f")
    NO2 = st.number_input("NO2", min_value=0.0, format="%.2f")
    HUMEDAD = st.number_input("Humedad (%)", min_value=0.0, max_value=100.0, format="%.2f")

with col2:
    PM25 = st.number_input("PM2.5", min_value=0.0, format="%.2f")
    O3 = st.number_input("O3", min_value=0.0, format="%.2f")
    LLUVIA = st.number_input("Lluvia (mm)", min_value=0.0, format="%.2f")

with col3:
    TEMPERATURA = st.number_input("Temperatura (°C)", min_value=-10.0, max_value=50.0, format="%.2f")
    VEL_VIENTO = st.number_input("Vel. Viento (m/s)", min_value=0.0, format="%.2f")
    DIR_VIENTO = st.number_input("Dir. Viento (°)", min_value=0.0, max_value=360.0, format="%.2f")

# Botón para predecir
if st.sidebar.button("🔍 Predecir Calidad del Aire"):
    # Convertir datos a array (SOLO 9 características)
    datos = np.array([[PM10, PM25, NO2, O3, HUMEDAD, LLUVIA, TEMPERATURA, VEL_VIENTO, DIR_VIENTO]])

    # Escalar los datos
    datos_escalados = scaler.transform(datos)

    # Mostrar los valores escalados en una sola línea
    st.write("🔎 Valores escalados:", ", ".join(map(lambda x: f"{x:.4f}", datos_escalados[0])))

    # Realizar la predicción
    prediccion = modelo_svm.predict(datos_escalados)[0]

    # Asignar color según la calidad del aire
    colores = {"Buena": "🟢", "Regular": "🟡", "Mala": "🔴"}
    st.subheader("🌤️ Resultado de la Predicción")
    st.markdown(f"### {colores.get(prediccion, '⚪')} La calidad del aire es: **{prediccion}**")

    # Mensajes de recomendación según la calidad del aire
    recomendaciones = {
        "Buena": "✅ El aire es saludable. Puede realizar actividades al aire libre sin restricciones.",
        "Regular": "⚠️ Personas sensibles (niños, ancianos y personas con enfermedades respiratorias) deben limitar actividades al aire libre.",
        "Mala": "🚨 Se recomienda evitar actividades al aire libre. Grupos vulnerables deben permanecer en interiores con ventanas cerradas."
    }
    
    st.info(recomendaciones.get(prediccion, "No hay recomendaciones disponibles."))