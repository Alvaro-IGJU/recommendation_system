import pandas as pd
import sqlite3
from surprise import dump
import os
import gdown

# ================================
# 📥 Descargar modelo si no existe
# ================================
def descargar_modelo_si_no_existe():
    modelo_path = "modelo_usuario_producto/modelo_svd_usuario_producto.pkl"
    if not os.path.exists(modelo_path):
        print("📦 Descargando modelo desde Google Drive...")
        os.makedirs(os.path.dirname(modelo_path), exist_ok=True)
        url = "https://drive.google.com/uc?id=1pwdepBBNQLVA1300gAvtmB0vwjcc9ZY3"
        gdown.download(url, modelo_path, quiet=False)

descargar_modelo_si_no_existe()

# ================================
# 📂 Cargar modelo y datos
# ================================
modelo_path = "modelo_usuario_producto/modelo_svd_usuario_producto.pkl"
# _, algo = dump.load(modelo_path)

interacciones = pd.read_csv("data/interacciones_usuario_producto_limpio.csv")
conn = sqlite3.connect("instacart.db")
productos = pd.read_sql("SELECT product_id, product_name FROM products", conn)
clusters = pd.read_csv("data/usuarios_clusters.csv")
conn.close()

# ================================
# 🛠️ Funciones auxiliares
# ================================
def parse_frozenset_string(fz_string):
    clean = fz_string.replace("frozenset({", "").replace("})", "").replace("'", "")
    return [item.strip() for item in clean.split(",") if item.strip()]

# ================================
# 🔁 Recomendador principal
# ================================
def recomendar_usuario_completo_usuario_producto(user_id, n=10):
    productos_comprados = interacciones[interacciones['user_id'] == user_id]['product_id'].unique()
    productos_no_comprados = [pid for pid in productos['product_id'].unique() if pid not in productos_comprados]

    frecuencia = interacciones['product_id'].value_counts(normalize=True).to_dict()

    # 🧠 Recomendaciones SVD
    predicciones = []
    for product_id in productos_no_comprados:
        pred = algo.predict(user_id, product_id)
        freq = frecuencia.get(product_id, 0)
        adjusted_score = pred.est * (1 - freq)
        predicciones.append((product_id, pred.est, adjusted_score))

    top_predicciones = sorted(predicciones, key=lambda x: x[2], reverse=True)[:n]
    recomendaciones_svd = [
        {'product_name': productos.loc[productos['product_id'] == product_id, 'product_name'].values[0],
         'estimation': est, 'adjusted_score': adjusted}
        for product_id, est, adjusted in top_predicciones
    ]

    # 📦 Recomendaciones MBA
    fila_cluster = clusters[clusters['user_id'] == user_id]
    recomendaciones_mba = []

    if not fila_cluster.empty:
        cluster_id = int(fila_cluster['cluster'].values[0])
        mba_path = f"mba_rules_clusters/cluster{cluster_id}_rules.csv"
        fallback_path = f"mba_fallbacks/cluster{cluster_id}_fallback.csv"

        productos_comprados_nombres = productos[productos['product_id'].isin(productos_comprados)]['product_name'].tolist()

        if os.path.exists(mba_path):
            reglas_mba = pd.read_csv(mba_path)
            for _, regla in reglas_mba.iterrows():
                antecedente = parse_frozenset_string(regla['antecedents'])
                consecuente = parse_frozenset_string(regla['consequents'])
                if any(p in productos_comprados_nombres for p in antecedente):
                    recomendaciones_mba.extend([p for p in consecuente if p not in productos_comprados_nombres])
            recomendaciones_mba = list(set(recomendaciones_mba))[:n]
        elif os.path.exists(fallback_path):
            fallback_df = pd.read_csv(fallback_path)
            recomendaciones_mba = fallback_df[
                ~fallback_df['product_name'].isin(productos_comprados_nombres)
            ]['product_name'].head(n).tolist()

    return {
        "usuario": user_id,
        "recomendaciones_svd": recomendaciones_svd,
        "recomendaciones_mba": recomendaciones_mba
    }
