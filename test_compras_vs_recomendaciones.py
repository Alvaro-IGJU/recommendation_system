import sqlite3
import pandas as pd
import random
from recomendar_usuario_completo_usuario_producto import recomendar_usuario_completo_usuario_producto

def obtener_productos_comprados(user_id):
    conn = sqlite3.connect("instacart.db")
    query = """
    SELECT DISTINCT LOWER(TRIM(p.product_name)) AS product_name
    FROM order_products_prior opp
    JOIN orders o ON opp.order_id = o.order_id
    JOIN products p ON opp.product_id = p.product_id
    WHERE o.user_id = ?
    """
    productos = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return sorted(set(productos['product_name'].tolist()))

def obtener_usuarios_random(n=50):
    conn = sqlite3.connect("instacart.db")
    user_ids = pd.read_sql("SELECT DISTINCT user_id FROM orders", conn)['user_id'].tolist()
    conn.close()
    return random.sample(user_ids, min(n, len(user_ids)))

def analizar_usuarios(user_ids, n_recomendaciones=10):
    resultados = []

    for user_id in user_ids:
        comprados = obtener_productos_comprados(user_id)
        resultado = recomendar_usuario_completo_usuario_producto(user_id, n=n_recomendaciones)
        recomendaciones_svd = [rec['product_name'] for rec in resultado.get("recomendaciones_svd", [])]
        recomendaciones_mba = resultado.get("recomendaciones_mba", [])

        # Alineamos para impresión
        max_len = max(len(comprados), len(recomendaciones_svd), len(recomendaciones_mba))
        comprados += [""] * (max_len - len(comprados))
        recomendaciones_svd += [""] * (max_len - len(recomendaciones_svd))
        recomendaciones_mba += [""] * (max_len - len(recomendaciones_mba))

        tabla = [f"\n🧑 Usuario ID: {user_id}", "-" * 105]
        tabla.append(f"{'🛒 Comprado':<35} | {'🎯 Recomendado (SVD)':<35} | {'📦 Recomendado (MBA)':<35}")
        tabla.append("-" * 105)
        for c, svd, mba in zip(comprados, recomendaciones_svd, recomendaciones_mba):
            tabla.append(f"{c:<35} | {svd:<35} | {mba:<35}")
        tabla.append("\n")

        resultados.append("\n".join(tabla))

    with open("reporte_recomendaciones.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(resultados))

    print("\n✅ Informe en columnas guardado como 'reporte_recomendaciones.txt'")

if __name__ == "__main__":
    usuarios = obtener_usuarios_random(10)
    analizar_usuarios(usuarios)
