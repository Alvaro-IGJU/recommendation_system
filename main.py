# app.py
import streamlit as st
import pandas as pd
import time
from recomendar_usuario_clusterizado import recomendar_usuario_completo
from nlp import buscar_productos_semanticos

st.set_page_config(page_title="Instacart Recommender", layout="wide")
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Helvetica Neue', sans-serif;
            background-color: #ffffff;
        }
        .card {
            border-radius: 14px;
            background-color: #f9f9f9;
            padding: 1.5rem;
            box-shadow: 0px 1px 5px rgba(0, 0, 0, 0.04);
            transition: all 0.25s ease;
            border: 1px solid #eee;
            margin-bottom: 1.5rem;
        }
        .card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            background-color: #ffffff;
        }
        .card h4 {
            font-size: 1.1rem;
            color: #222;
            margin-bottom: 0.3rem;
        }
        .card p {
            font-size: 0.9rem;
            color: #444;
        }
        .card small {
            font-size: 0.75rem;
            color: #888;
        }
        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #222;
            border-bottom: 2px solid #1e90ff;
            margin: 2rem 0 1rem;
        }
        .stButton > button {
            background: linear-gradient(90deg, #1e90ff, #0077cc);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1.2rem;
            font-size: 1rem;
            transition: background 0.3s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #0077cc, #005fa3);
        }
    </style>
""", unsafe_allow_html=True)

st.title("🛒 Instacart Recommender")
st.markdown("Introduce el ID de un usuario para obtener recomendaciones personalizadas, o utiliza el buscador para explorar el catálogo.")

# Inputs lado a lado
col1, col2 = st.columns(2)
with col1:
    user_id = st.number_input("ID de usuario", min_value=1, step=1)

with col2:
    consulta = st.text_input("Buscar producto por descripción", key="consulta")

# Control del debounce para búsqueda NLP
if "last_input_time" not in st.session_state:
    st.session_state.last_input_time = time.time()
    st.session_state.last_text = ""
    st.session_state.resultado_nlp = None

current_time = time.time()
if consulta != st.session_state.last_text:
    st.session_state.last_input_time = current_time
    st.session_state.last_text = consulta

elif current_time - st.session_state.last_input_time >= 1 and consulta:
    with st.spinner("Buscando productos similares..."):
        st.session_state.resultado_nlp = buscar_productos_semanticos(consulta, top_n=10)

# Resultado del recomendador de usuario
resultado_usuario = None
if st.button("Obtener recomendaciones"):
    with st.spinner("Generando recomendaciones..."):
        resultado_usuario = recomendar_usuario_completo(user_id, n=10)

# Mostrar recomendaciones combinadas
resultado_nlp = st.session_state.resultado_nlp
if resultado_usuario or resultado_nlp is not None:
    st.markdown("<div class='section-title'>Productos recomendados</div>", unsafe_allow_html=True)
    all_recommendations = []

    if resultado_usuario and "error" not in resultado_usuario:
        svd_df = pd.DataFrame(resultado_usuario["recomendaciones_svd"])
        mba_df = pd.DataFrame({"product_name": resultado_usuario["recomendaciones_mba"]})
        svd_df["source"] = "SVD"
        mba_df["source"] = "MBA"
        mba_df["aisle"] = ""
        all_recommendations.append(pd.concat([svd_df, mba_df], ignore_index=True))

    if resultado_nlp is not None and not resultado_nlp.empty:
        resultado_nlp = resultado_nlp.copy()
        resultado_nlp["aisle"] = ""
        resultado_nlp["source"] = "NLP"
        resultado_nlp = resultado_nlp.rename(columns={"product_name": "product_name"})
        all_recommendations.append(resultado_nlp)

    if all_recommendations:
        combined_df = pd.concat(all_recommendations, ignore_index=True)
        cols = st.columns(2)
        for i, row in combined_df.iterrows():
            with cols[i % 2]:
                st.markdown(f"""
                    <div class='card'>
                        <h4>{row['product_name']}</h4>
                        <p>Categoría: <b>{row['aisle']}</b></p>
                        <small>Fuente: {row['source']}</small>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No se encontraron recomendaciones.")

if resultado_usuario and "error" in resultado_usuario:
    st.error(resultado_usuario["error"])
