import streamlit as st
import pandas as pd
from surprise import dump

# ---------- Configuración y carga ----------
MODELO_PATH   = "modelo_general/modelo_svd_general.pkl"
METRICAS_PATH = "modelo_general/metricas_modelo_general.txt"
CATALOGO_PATH = "data/aisles.csv"          # <-- tu catálogo de islas

@st.cache_resource
def load_model():
    _, algo = dump.load(MODELO_PATH)       # devuelve (rating_scale, algoritmo)
    return algo

@st.cache_data
def load_data():
    aisles = pd.read_csv(CATALOGO_PATH)    # id, nombre, etc.
    return aisles

@st.cache_data
def load_metrics():
    with open(METRICAS_PATH, "r", encoding="utf-8") as f:
        return f.read()

model   = load_model()
aisles  = load_data()
metrics = load_metrics()

# ---------- Interfaz ----------
st.title("Recomendador SVD general")

with st.expander("Ver métricas del modelo"):
    st.text(metrics)

uid = st.number_input("ID de usuario", min_value=1, step=1)
k   = st.slider("Número de recomendaciones", 1, 30, 10)

if st.button("Recomendar"):
    # Predecir todas las islas no visitadas por el usuario
    preds = [
        (iid, model.predict(uid, iid).est)
        for iid in aisles["aisle_id"]        # columna con los IDs
    ]
    top_k = sorted(preds, key=lambda x: x[1], reverse=True)[:k]

    resultados = (
        aisles
        .loc[aisles["aisle_id"].isin([i for i, _ in top_k])]
        .assign(puntuacion=[round(s, 3) for _, s in top_k])
        .sort_values("puntuacion", ascending=False)
        .reset_index(drop=True)
    )
    st.subheader("Recomendaciones")
    st.dataframe(resultados, use_container_width=True)
