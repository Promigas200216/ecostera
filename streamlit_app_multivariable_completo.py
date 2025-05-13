
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
    opciones_fechas = df.columns[2:]
    fechas = [datetime.strptime(f, "%m/%d/%Y") for f in opciones_fechas]
    dias = np.array([(f - fechas[0]).days for f in fechas]).reshape(-1, 1)

    resultados_pred = []

    for _, row in df.iterrows():
        abscisa = row["Abscisa"]
        try:
            y_vals = pd.to_numeric(row[opciones_fechas], errors='coerce').values.reshape(-1, 1)
            if np.isnan(y_vals).any():
                continue
            modelo = LinearRegression()
            modelo.fit(dias, y_vals)
            pendiente = modelo.coef_[0][0]
            intercepto = modelo.intercept_[0]
            actual = y_vals[-1][0]
            estado = "Sí" if actual < umbral_minimo else "No"
            if pendiente < 0:
                dias_cruce = (umbral_minimo - intercepto) / pendiente
                fecha_cruce = fechas[0] + timedelta(days=dias_cruce)
                fecha_cruce_str = fecha_cruce.strftime("%Y-%m-%d")
            else:
                fecha_cruce_str = "No aplica"
            resultados_pred.append({
                "Abscisa": abscisa,
                "Pendiente": round(pendiente, 4),
                "Actual (m)": round(actual, 3),
                f"¿Bajo {umbral_minimo}m?": estado,
                f"Cruce estimado de {umbral_minimo}m": fecha_cruce_str
            })
        except:
            continue
    return pd.DataFrame(resultados_pred)

# Definir pestañas
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

    # TAB Y
    if df_y is not None:
        with seleccion[tab_index]:
            st.header("📘 Análisis de variable Y")
            opciones_fechas = df_y.columns[2:]
            columna_a_evaluar = st.selectbox("Selecciona una fecha para evaluar (Y):", opciones_fechas)
            umbral_y = st.number_input("Umbral mínimo para Y:", value=0.6)

            df_y["Margen"] = (df_y[columna_a_evaluar] - umbral_y).round(3)
            df_y["Alerta"] = df_y["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
            st.dataframe(df_y[["Abscisa", columna_a_evaluar, "Margen", "Alerta"]])

            pred_y = prediccion_regresion(df_y, umbral_y, "Y")
            st.subheader("🔮 Predicción por regresión lineal (Y)")
            st.dataframe(pred_y)
        tab_index += 1

    # TAB X
    if df_x is not None:
        with seleccion[tab_index]:
            st.header("📗 Análisis de variable X")
            opciones_fechas = df_x.columns[2:]
            columna_a_evaluar = st.selectbox("Selecciona una fecha para evaluar (X):", opciones_fechas)
            umbral_x = st.number_input("Umbral mínimo para X:", value=2.0)

            df_x["Margen"] = (df_x[columna_a_evaluar] - umbral_x).round(3)
            df_x["Alerta"] = df_x["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
            st.dataframe(df_x[["Abscisa", columna_a_evaluar, "Margen", "Alerta"]])

            pred_x = prediccion_regresion(df_x, umbral_x, "X")
            st.subheader("🔮 Predicción por regresión lineal (X)")
            st.dataframe(pred_x)
        tab_index += 1

    # TAB A
    if df_a is not None:
        with seleccion[tab_index]:
            st.header("📙 Análisis de variable A")
            opciones_fechas = df_a.columns[2:]
            columna_a_evaluar = st.selectbox("Selecciona una fecha para evaluar (A):", opciones_fechas)
            umbral_a = st.number_input("Umbral mínimo para A:", value=0.3)

            df_a["Margen"] = (df_a[columna_a_evaluar] - umbral_a).round(3)
            df_a["Alerta"] = df_a["Margen"].apply(lambda x: "ALERTA" if x < 0 else "OK")
            st.dataframe(df_a[["Abscisa", columna_a_evaluar, "Margen", "Alerta"]])

            pred_a = prediccion_regresion(df_a, umbral_a, "A")
            st.subheader("🔮 Predicción por regresión lineal (A)")
            st.dataframe(pred_a)
        tab_index += 1

    # TAB COMBINADO
    with seleccion[tab_index]:
        st.header("🔀 Análisis combinado multivariable")
        st.write("Cruce de alertas y predicciones en X, Y y A")

        if all([df_y is not None, df_x is not None, df_a is not None]):
            merged = df_y[["Abscisa"]].copy()
            pred_y = prediccion_regresion(df_y, umbral_y, "Y")
            pred_x = prediccion_regresion(df_x, umbral_x, "X")
            pred_a = prediccion_regresion(df_a, umbral_a, "A")

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
            st.warning("Sube los tres archivos para activar el análisis combinado.")
else:
    st.info("💡 Sube al menos uno de los archivos (X, Y, A) para comenzar.")
