# test_compras_vs_recomendaciones.py
import sqlite3
import pandas as pd
from recomendar_usuario_completo_usuario_producto import recomendar_usuario_completo_usuario_producto

def obtener_productos_comprados(user_id):
    conn = sqlite3.connect("instacart.db")
    query = """
    SELECT DISTINCT p.product_name
    FROM order_products_prior opp
    JOIN orders o ON opp.order_id = o.order_id
    JOIN products p ON opp.product_id = p.product_id
    WHERE o.user_id = ?
    """
    productos = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return set(productos['product_name'].tolist())  # Devuelvo set para comprobar rápido si está o no

def comparar_compras_recomendaciones(user_id, n_recomendaciones=10):
    print(f"\n🧑 Usuario ID: {user_id}")

    # Productos comprados (sin duplicados)
    productos_comprados = obtener_productos_comprados(user_id)
    print("\n✅ Productos comprados por el usuario (sin duplicados):")
    if productos_comprados:
        for prod in list(productos_comprados)[:20]:
            print(f"   - {prod}")
        if len(productos_comprados) > 20:
            print(f"   ... y {len(productos_comprados) - 20} más")
    else:
        print("   ⚠️ No se encontraron compras.")

    # Recomendaciones
    resultado = recomendar_usuario_completo_usuario_producto(user_id, n=n_recomendaciones)
    recomendaciones = resultado.get("recomendaciones_svd", [])

    print("\n🎯 Recomendaciones SVD (✔️ = ya comprado):")
    if recomendaciones:
        for rec in recomendaciones:
            comprado = "✔️" if rec['product_name'] in productos_comprados else "❌"
            print(f"   {comprado} {rec['product_name']} (estimación: {rec['estimation']:.4f} | ajustado: {rec['adjusted_score']:.4f})")
    else:
        print("   ⚠️ No se encontraron recomendaciones.")

if __name__ == "__main__":
    user_id = int(input("Introduce el ID del usuario a analizar: "))
    comparar_compras_recomendaciones(user_id)
