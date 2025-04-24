# recomendar_usuario_completo_usuario_producto.py
import pandas as pd
import os
from surprise import dump

# Cargar las interacciones limpias
interacciones = pd.read_csv("data/interacciones_usuario_producto_limpio.csv")
productos = pd.read_csv("data/products.csv")

# Cargar el modelo SVD entrenado
modelo_path = "modelo_usuario_producto/modelo_svd_usuario_producto.pkl"
_, algo = dump.load(modelo_path)

def recomendar_usuario_completo_usuario_producto(user_id, n=10):
    # Productos ya comprados por el usuario
    productos_comprados = interacciones[interacciones['user_id'] == user_id]['product_id'].unique()

    # Todos los productos posibles
    todos_los_productos = productos['product_id'].unique()
    
    # Productos que NO ha comprado
    productos_no_comprados = [p for p in todos_los_productos if p not in productos_comprados]
    
    if not productos_no_comprados:
        return {"error": "El usuario ya ha comprado todos los productos."}
    
    # Calcular popularidad global de los productos (frecuencia de compra)
    frecuencia = interacciones['product_id'].value_counts(normalize=True).to_dict()

    # Predecir las puntuaciones para los productos no comprados
    predicciones = []
    for product_id in productos_no_comprados:
        pred = algo.predict(user_id, product_id)
        freq = frecuencia.get(product_id, 0)
        pred.adjusted_score = pred.est * (1 - freq)  # Penalización por popularidad
        predicciones.append(pred)
    
    # Ordenar las predicciones por score ajustado y quedarnos con las top N
    top_predicciones = sorted(predicciones, key=lambda x: x.adjusted_score, reverse=True)[:n]
    
    # Preparar la respuesta con nombres de los productos
    recomendaciones = []
    for pred in top_predicciones:
        nombre_producto = productos[productos['product_id'] == pred.iid]['product_name'].values[0]
        recomendaciones.append({
            'product_name': nombre_producto,
            'estimacion': pred.est,
            'ajustado': pred.adjusted_score
        })
    
    return {
        "usuario": user_id,
        "recomendaciones_svd": recomendaciones
    }
