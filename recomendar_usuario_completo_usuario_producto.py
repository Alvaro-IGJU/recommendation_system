# recomendar_usuario_completo_usuario_producto.py
import pandas as pd
import sqlite3
from surprise import dump

# Cargar el modelo entrenado
modelo_path = "modelo_usuario_producto/modelo_svd_usuario_producto.pkl"
_, algo = dump.load(modelo_path)

# Cargar interacciones limpias
interacciones = pd.read_csv("data/interacciones_usuario_producto_limpio.csv")
productos = pd.read_sql("SELECT product_id, product_name FROM products", sqlite3.connect("instacart.db"))

def recomendar_usuario_completo_usuario_producto(user_id, n=10):
    productos_comprados = interacciones[interacciones['user_id'] == user_id]['product_id'].unique()
    productos_no_comprados = [pid for pid in productos['product_id'].unique() if pid not in productos_comprados]

    frecuencia = interacciones['product_id'].value_counts(normalize=True).to_dict()

    predicciones = []
    for product_id in productos_no_comprados:
        pred = algo.predict(user_id, product_id)
        freq = frecuencia.get(product_id, 0)
        adjusted_score = pred.est * (1 - freq)
        predicciones.append((product_id, pred.est, adjusted_score))

    top_predicciones = sorted(predicciones, key=lambda x: x[2], reverse=True)[:n]

    recomendaciones = []
    for product_id, est, adjusted in top_predicciones:
        nombre_producto = productos.loc[productos['product_id'] == product_id, 'product_name'].values[0]
        recomendaciones.append({'product_name': nombre_producto, 'estimation': est, 'adjusted_score': adjusted})

    return {"usuario": user_id, "recomendaciones_svd": recomendaciones, "recomendaciones_mba": []}
