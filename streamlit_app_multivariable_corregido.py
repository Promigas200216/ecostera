
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📡 Monitoreo y predicción de erosión - Múltiples variables (X, Y, A)")

# Carga de archivos CSV por variable
col1, col2, col3 = st.columns(3)
with col1:
    archivo_y = st.file_uploader("📄 Sube archivo de datos Y", type=["csv"])
with col2:
    archivo_x = st.file_uploader("📄 Sube archivo de datos X", type=["csv"])
with col3:
    archivo_a = st.file_uploader("📄 Sube archivo de datos A", type=["csv"])

# Carga condicional de cada dataset
df_y, df_x, df_a = None, None, None

if archivo_y:
    df_y = pd.read_csv(archivo_y, encoding="latin1")
    df_y = df_y.dropna(subset=["Abscisa"])

if archivo_x:
    df_x = pd.read_csv(archivo_x, encoding="latin1")
    df_x = df_x.dropna(subset=["Abscisa"])

if archivo_a:
    df_a = pd.read_csv(archivo_a, encoding="latin1")
    df_a = df_a.dropna(subset=["Abscisa"])

# Función de predicción por regresión lineal
def prediccion_regresion(df, umbral_minimo, variable_nombre):
    fechas = [col for col in df.columns if "/" in col]
    fechas_dt = [datetime.strptime(f, "%m/%d/%Y") for f in fechas]
    dias = np.array([(f - fechas_dt[0]).days for f in fechas_dt]).reshape(-1, 1)
    resultados = []

    for _, row in df.iterrows():
        abscisa = row["Abscisa"]
        try:
            valores = pd.to_numeric(row[fechas], errors='coerce').values.reshape(-1, 1)
            if np.isnan(valores).any():
                continue
            modelo = LinearRegression()
            modelo.fit(dias, valores)
            pendiente = modelo.coef_[0][0]
            intercepto = modelo.intercept_[0]
            actual = valores[-1][0]
            estado = "Sí" if actual < umbral_minimo else "No"
            if pendiente < 0:
                dias_cruce = (umbral_minimo - intercepto) / pendiente
                fecha_cruce = fechas_dt[0] + timedelta(days=dias_cruce)
                fecha_cruce_str = fecha_cruce.strftime("%Y-%m-%d")
            else:
                fecha_cruce_str = "No aplica"
            resultados.append({
                "Abscisa": abscisa,
                "Pendiente": round(pendiente, 4),
                "Actual (m)": round(actual, 3),
                f"¿Bajo {umbral_minimo}m?": estado,
                f"Cruce estimado de {umbral_minimo}m": fecha_cruce_str
            })
        except:
            continue
    return pd.DataFrame(resultados)

# Crear pestañas dinámicas
tabs = []
if df_y is not None:
    tabs.append("🔹 Variable Y")
if df_x is not None:
    tabs.append("🔹 Variable X")
if df_a is not None:
    tabs.append("🔹 Variable A")
if any([df_y is not None, df_x is not None, df_a is not None]):
    tabs.append("🔀 Análisis combinado")

if tabs:
    seleccion = st.tabs(tabs)
    tab_index = 0

    if df_y is not None:
        with seleccion[tab_index]:
            st.header("📘 Análisis de variable Y")
            fechas_y = [col for col in df_y.columns if "/" in col]
            if fechas_y:
                columna_y = st.selectbox("Fecha a evaluar (Y):", fechas_y)
                umbral_y = st.number_input("Umbral mínimo para Y:", value=0.6)
                df_y["Margen"] = (pd.to_numeric(df_y[columna_y], errors='coerce') - umbral_y).round(3)
                df_y["Alerta"] = df_y["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
                st.dataframe(df_y[["Abscisa", columna_y, "Margen", "Alerta"]])
                pred_y = prediccion_regresion(df_y, umbral_y, "Y")
                st.subheader("🔮 Predicción (Y)")
                st.dataframe(pred_y)
            else:
                st.warning("No se detectaron columnas con formato de fecha en Y.")
        tab_index += 1

    if df_x is not None:
        with seleccion[tab_index]:
            st.header("📗 Análisis de variable X")
            fechas_x = [col for col in df_x.columns if "/" in col]
            if fechas_x:
                columna_x = st.selectbox("Fecha a evaluar (X):", fechas_x)
                umbral_x = st.number_input("Umbral mínimo para X:", value=2.0)
                df_x["Margen"] = (pd.to_numeric(df_x[columna_x], errors='coerce') - umbral_x).round(3)
                df_x["Alerta"] = df_x["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
                st.dataframe(df_x[["Abscisa", columna_x, "Margen", "Alerta"]])
                pred_x = prediccion_regresion(df_x, umbral_x, "X")
                st.subheader("🔮 Predicción (X)")
                st.dataframe(pred_x)
            else:
                st.warning("No se detectaron columnas con formato de fecha en X.")
        tab_index += 1

    if df_a is not None:
        with seleccion[tab_index]:
            st.header("📙 Análisis de variable A")
            fechas_a = [col for col in df_a.columns if "/" in col]
            if fechas_a:
                columna_a = st.selectbox("Fecha a evaluar (A):", fechas_a)
                umbral_a = st.number_input("Umbral mínimo para A:", value=0.3)
                df_a["Margen"] = (pd.to_numeric(df_a[columna_a], errors='coerce') - umbral_a).round(3)
                df_a["Alerta"] = df_a["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
                st.dataframe(df_a[["Abscisa", columna_a, "Margen", "Alerta"]])
                pred_a = prediccion_regresion(df_a, umbral_a, "A")
                st.subheader("🔮 Predicción (A)")
                st.dataframe(pred_a)
            else:
                st.warning("No se detectaron columnas con formato de fecha en A.")
        tab_index += 1

    with seleccion[tab_index]:
        st.header("🔀 Análisis combinado")
        if all([df_y is not None, df_x is not None, df_a is not None]):
            pred_y = prediccion_regresion(df_y, umbral_y, "Y")
            pred_x = prediccion_regresion(df_x, umbral_x, "X")
            pred_a = prediccion_regresion(df_a, umbral_a, "A")
            merged = df_y[["Abscisa"]].copy()
            merged = merged.merge(pred_y, on="Abscisa", how="left", suffixes=('', '_Y'))
            merged = merged.merge(pred_x, on="Abscisa", how="left", suffixes=('', '_X'))
            merged = merged.merge(pred_a, on="Abscisa", how="left", suffixes=('', '_A'))

            def semaforo(row):
                alertas = sum([
                    row.get(f"¿Bajo {umbral_y}m?", "") == "Sí",
                    row.get(f"¿Bajo {umbral_x}m?", "") == "Sí",
                    row.get(f"¿Bajo {umbral_a}m?", "") == "Sí"
                ])
                if alertas >= 2:
                    return "🔴 CRÍTICO"
                elif alertas == 1:
                    return "🟡 RIESGO"
                return "🟢 ESTABLE"

            merged["Estado combinado"] = merged.apply(semaforo, axis=1)
            st.dataframe(merged)
        else:
            st.warning("Debes subir los tres archivos para el análisis combinado.")
else:
    st.info("Sube al menos un archivo para comenzar.")
