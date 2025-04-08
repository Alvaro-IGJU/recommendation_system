import pandas as pd
import os
from surprise import dump

# 📂 Cargar datos necesarios
interacciones = pd.read_csv("data/interacciones_usuario_isla.csv")
clusters = pd.read_csv("data/usuarios_clusters.csv")
orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")
order_products_prior = pd.read_csv("data/order_products__prior.csv")

# Añadir nombres de productos
datos_productos = order_products_prior.merge(products[['product_id', 'product_name']], on='product_id')

# Añadir cluster y subcluster
datos_orders = orders[['order_id', 'user_id']].merge(clusters, on='user_id')
datos = datos_productos.merge(datos_orders, on='order_id')

def recomendar_usuario_completo(user_id, n=10):
    fila = clusters[clusters['user_id'] == user_id]
    if fila.empty:
        return {"error": "Usuario no encontrado en los clusters."}
    
    cluster, subcluster = fila[['cluster', 'subcluster']].values[0]
    print(f"Este usuario pertenece al cluster {cluster} y al subcluster {subcluster}")
    
    modelo_path = "modelo_general/modelo_svd_general.pkl"
    if not os.path.exists(modelo_path):
        return {"error": f"Modelo general no encontrado: {modelo_path}"}
    
    _, algo = dump.load(modelo_path)

    pedidos_usuario = orders[orders['user_id'] == user_id]['order_id']
    productos_usuario = order_products_prior[order_products_prior['order_id'].isin(pedidos_usuario)]
    productos_comprados = productos_usuario['product_id'].unique()

    productos_usuario = productos_usuario.merge(products[['product_id', 'aisle_id']], on='product_id')
    # islas_usuario = productos_usuario['aisle_id'].unique()

    # scored_islas = [(aisle_id, algo.predict(user_id, aisle_id).est) for aisle_id in islas_usuario]
    # ✅ Mejor:
    todos_los_aisles = interacciones['aisle_id'].unique()
    scored_islas = [(aisle_id, algo.predict(user_id, aisle_id).est) for aisle_id in todos_los_aisles]
    top_islas = sorted(scored_islas, key=lambda x: x[1], reverse=True)[:5]
    top_isla_ids = [a for a, _ in top_islas]

    user_ids_cluster = clusters[(clusters['cluster'] == cluster) & (clusters['subcluster'] == subcluster)]['user_id']
    pedidos_cluster = orders[orders['user_id'].isin(user_ids_cluster)]
    compras_cluster = order_products_prior[order_products_prior['order_id'].isin(pedidos_cluster['order_id'])]
    compras_cluster = compras_cluster.merge(products[['product_id', 'aisle_id', 'product_name']], on='product_id')
    compras_filtradas = compras_cluster[compras_cluster['aisle_id'].isin(top_isla_ids)]

    productos_populares = compras_filtradas.groupby(['product_id', 'product_name', 'aisle_id']) \
        .size().reset_index(name='n_compras') \
        .sort_values(by='n_compras', ascending=False)

    recomendaciones_svd_df = productos_populares[~productos_populares['product_id'].isin(productos_comprados)].head(n)
    nombres_svd = set(recomendaciones_svd_df['product_name'])

    # 📦 Reglas MBA por subcluster
    mba_path = f"mba_rules/cluster{cluster}_sub{subcluster}_rules.csv"
    recomendaciones_mba = []
    if os.path.exists(mba_path):
        reglas_mba = pd.read_csv(mba_path)
        productos_comprados_nombres = products[products['product_id'].isin(productos_comprados)]['product_name'].tolist()
        for _, regla in reglas_mba.iterrows():
            antecedente = eval(regla['antecedents'])
            consecuente = eval(regla['consequents'])
            if any(p in productos_comprados_nombres for p in antecedente):
                recomendaciones_mba.extend([p for p in consecuente if p not in productos_comprados_nombres])
        recomendaciones_mba = list(set(recomendaciones_mba) - nombres_svd)
        recomendaciones_mba = list(recomendaciones_mba)[:n]

    # 📉 Fallback si MBA está vacío
    if not recomendaciones_mba:
        fallback_path = f"mba_fallbacks/populares_cluster{cluster}_sub{subcluster}.csv"
        if os.path.exists(fallback_path):
            fallback_df = pd.read_csv(fallback_path)
            recomendaciones_mba = fallback_df[
                (~fallback_df['product_name'].isin(productos_comprados)) &
                (~fallback_df['product_name'].isin(nombres_svd))
            ]['product_name'].head(n).tolist()

    return {
        "usuario": user_id,
        "top_islas_svd": top_isla_ids,
        "recomendaciones_svd": recomendaciones_svd_df[['product_name', 'aisle_id', 'n_compras']].to_dict(orient='records'),
        "recomendaciones_mba": recomendaciones_mba
    }
