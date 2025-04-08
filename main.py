import streamlit as st
import pandas as pd
from recomendar_usuario_clusterizado import recomendar_usuario_completo

# Configuración general
st.set_page_config(page_title="Recomendador Instacart", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .card {
            border: 1px solid #d9d9d9;
            padding: 1.2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            background-color: #ffffff;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
        }
        .card h4 {
            margin: 0 0 0.3rem 0;
            font-size: 1.1rem;
            color: #222;
        }
        .card p {
            margin: 0;
            font-size: 0.9rem;
            color: #555;
        }
        .section-title {
            border-left: 4px solid #0e76a8;
            padding-left: 10px;
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-size: 1.3rem;
            font-weight: 600;
            color: #0e76a8;
        }
        .stButton>button {
            background-color: #0e76a8;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            border: none;
            font-size: 1rem;
        }
        .stButton>button:hover {
            background-color: #0b5e87;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("🛍️ Sistema de recomendación Instacart")
st.markdown("Introduce el ID de un usuario para recibir recomendaciones personalizadas basadas en sus preferencias de compra.")

# Input del usuario
user_id = st.number_input("ID de usuario", min_value=1, step=1)

# Botón de recomendación
if st.button("🔎 Obtener recomendaciones"):
    with st.spinner("🔄 Procesando recomendaciones..."):
        resultado = recomendar_usuario_completo(user_id, n=10)

    if "error" in resultado:
        st.error(resultado["error"])
    else:
        st.success(f"🎉 Recomendaciones generadas para el usuario {resultado['usuario']}")

        # Islas favoritas
        st.markdown("<div class='section-title'>🏝️ Islas favoritas del usuario</div>", unsafe_allow_html=True)
        st.markdown(" → ".join([f"`{isla}`" for isla in resultado["top_islas_svd"]]))

        # Recomendaciones SVD
        st.markdown("<div class='section-title'>🛒 Productos sugeridos por SVD</div>", unsafe_allow_html=True)
        svd_df = pd.DataFrame(resultado["recomendaciones_svd"])
        cols = st.columns(2)

        for i, row in svd_df.iterrows():
            with cols[i % 2]:
                st.markdown(f"""
                    <div class='card'>
                        <h4>{row['product_name']}</h4>
                        <p>🧭 Categoría: <b>{row['aisle']}</b></p>
                    </div>
                """, unsafe_allow_html=True)

        # Recomendaciones MBA
        st.markdown("<div class='section-title'>💡 También te podría interesar</div>", unsafe_allow_html=True)
        mba_cols = st.columns(2)

        for i, prod in enumerate(resultado["recomendaciones_mba"]):
            with mba_cols[i % 2]:
                st.markdown(f"""
                    <div class='card'>
                        <h4>{prod}</h4>
                        <p>📦 Recomendado por patrones de compra</p>
                    </div>
                """, unsafe_allow_html=True)
