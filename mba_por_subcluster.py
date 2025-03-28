import pandas as pd
import os
from mlxtend.frequent_patterns import apriori, association_rules

# 📂 Cargar datos
orders = pd.read_csv("orders.csv")
products = pd.read_csv("products.csv")
clusters = pd.read_csv("usuarios_clusters.csv")
interacciones = pd.read_csv("interacciones_usuario_isla.csv")
order_products_prior = pd.read_csv("order_products__prior.csv")  # asegúrate de tenerlo

# 🔗 Merge con productos para añadir nombre
order_products = order_products_prior.merge(products[['product_id', 'product_name']], on='product_id')

# Añadir user_id a cada compra
orders_cluster = orders[['order_id', 'user_id']].merge(clusters, on='user_id')
datos = order_products.merge(orders_cluster, on='order_id')

# 📁 Crear carpeta para guardar reglas y fallback
os.makedirs("mba_rules", exist_ok=True)
os.makedirs("mba_fallbacks", exist_ok=True)

# 🧠 Por cada subcluster
for (cluster_id, subcluster_id), grupo in clusters.groupby(['cluster', 'subcluster']):
    print(f"📊 Generando reglas para Cluster {cluster_id} - Subcluster {subcluster_id}")

    user_ids = grupo['user_id'].unique()
    datos_sub = datos[datos['user_id'].isin(user_ids)]

    # 🔽 Filtrar productos más frecuentes
    top_n = 1000000
    productos_frecuentes = datos_sub['product_name'].value_counts().nlargest(top_n).index
    datos_filtrados = datos_sub[datos_sub['product_name'].isin(productos_frecuentes)]

    # 🔽 Filtrar número de pedidos
    pedidos_limitados = datos_filtrados['order_id'].unique()[:10000]
    datos_filtrados = datos_filtrados[datos_filtrados['order_id'].isin(pedidos_limitados)]

    # 🧺 Matriz de transacciones
    basket = datos_filtrados.groupby(['order_id', 'product_name'])['product_id'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # 🪄 Apriori
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    if not frequent_itemsets.empty:
        reglas = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        nombre_archivo = f"mba_rules/rules_cluster{cluster_id}_sub{subcluster_id}.csv"
        reglas.to_csv(nombre_archivo, index=False)
        print(f"✅ Reglas guardadas en {nombre_archivo}\n")
    else:
        print(f"⚠️ No se generaron reglas para cluster {cluster_id} - subcluster {subcluster_id}. Creando fallback...")

        productos_populares = (
            datos_sub['product_name']
            .value_counts()
            .reset_index()
            .rename(columns={'index': 'product_name', 'product_name': 'n_compras'})
        )

        fallback_path = f"mba_fallbacks/populares_cluster{cluster_id}_sub{subcluster_id}.csv"
        productos_populares.to_csv(fallback_path, index=False)
        print(f"📦 Fallback de productos populares guardado en {fallback_path}\n")
