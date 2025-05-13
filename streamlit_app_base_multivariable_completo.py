
import streamlit as st
import pandas as pd

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

# Definición de pestañas activas
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
            st.write("Aquí irá toda la lógica de la variable Y.")
        tab_index += 1

    if df_x is not None:
        with seleccion[tab_index]:
            st.header("📗 Análisis de variable X")
            st.write("Aquí irá toda la lógica de la variable X.")
        tab_index += 1

    if df_a is not None:
        with seleccion[tab_index]:
            st.header("📙 Análisis de variable A")
            st.write("Aquí irá toda la lógica de la variable A.")
        tab_index += 1

    with seleccion[tab_index]:
        st.header("🔀 Análisis combinado multivariable")
        st.write("Aquí se cruzarán Y, X y A con semáforos y alertas integradas.")
else:
    st.info("💡 Sube al menos uno de los archivos (X, Y, A) para comenzar.")
