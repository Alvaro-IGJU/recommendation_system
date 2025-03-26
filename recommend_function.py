import pandas as pd
import sqlite3
from surprise import dump

# Cargar el modelo SVD entrenado
modelo_path = "modelo_svd.pkl"
_, algo = dump.load(modelo_path)

# Cargar interacciones y clusters
interacciones = pd.read_csv("interacciones_usuario_isla.csv")
df_clusters = pd.read_csv("clusters_usuarios_svd.csv")

# Conectar a la base de datos
conn = sqlite3.connect("instacart.db")
orders = pd.read_sql("SELECT * FROM orders", conn)
products = pd.read_sql("SELECT * FROM products", conn)
aisles = pd.read_sql("SELECT * FROM aisles", conn)  # Asegúrate de que esta tabla esté en la DB
order_products_prior = pd.read_sql("SELECT * FROM order_products_prior", conn)

# Añadir nombre de la isla al DataFrame de productos
products = products.merge(aisles, on='aisle_id')  # Añade 'aisle' (nombre)

# ✅ Recomendación personalizada con SVD + clustering que devuelve JSON
def recomendar_productos_por_usuario(user_id, n=10):
    # 1. Productos que ya ha comprado el usuario
    pedidos_usuario = orders[orders['user_id'] == user_id]
    productos_usuario = order_products_prior[order_products_prior['order_id'].isin(pedidos_usuario['order_id'])]
    productos_comprados = productos_usuario['product_id'].unique()

    # 2. Islas en las que ha comprado
    productos_con_isla = products[['product_id', 'aisle_id', 'aisle', 'product_name']]
    productos_usuario = productos_usuario.merge(productos_con_isla, on='product_id')
    islas_usuario = productos_usuario['aisle_id'].unique()

    # 3. Predecir puntuación del modelo SVD para islas que ha visitado
    scored_islas = [(aisle_id, algo.predict(user_id, aisle_id).est) for aisle_id in islas_usuario]
    scored_islas.sort(key=lambda x: x[1], reverse=True)
    top_islas = [aisle_id for aisle_id, _ in scored_islas[:3]]

    print(f"\n🏝️ Islas favoritas estimadas por SVD para el usuario {user_id}: {top_islas}\n")

    # 4. Cluster al que pertenece el usuario
    cluster_usuario = df_clusters[df_clusters['user_id'] == user_id]['cluster'].values[0]
    usuarios_mismo_cluster = df_clusters[df_clusters['cluster'] == cluster_usuario]['user_id'].tolist()

    # 5. Obtener compras del mismo cluster
    pedidos_cluster = orders[orders['user_id'].isin(usuarios_mismo_cluster)]
    compras_cluster = order_products_prior[order_products_prior['order_id'].isin(pedidos_cluster['order_id'])]
    compras_cluster = compras_cluster.merge(products, on='product_id')

    # 6. Filtrar productos en las islas favoritas
    compras_filtradas = compras_cluster[compras_cluster['aisle_id'].isin(top_islas)]

    # 7. Productos populares en esas islas
    productos_populares = compras_filtradas.groupby(['product_id', 'product_name', 'aisle']) \
        .size().reset_index(name='n_compras') \
        .sort_values(by='n_compras', ascending=False)

    # 8. Eliminar productos ya comprados
    recomendaciones = productos_populares[~productos_populares['product_id'].isin(productos_comprados)]

    # 9. Devolver resultados como lista de diccionarios (JSON serializable)
    resultado_json = [
        {
            "producto": row['product_name'],
            "isla": row['aisle']
        }
        for _, row in recomendaciones.head(n).iterrows()
    ]

    return resultado_json
