import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ================================
# ⚙️ CONFIGURACIÓN
# ================================
MIN_SUPPORT = 0.007    # Ajusta según el tamaño del dataset
MIN_LIFT = 1          # Para reglas más útiles
TOP_N_ORDERS = 10000   # Limitar número de pedidos
TOP_N_PRODUCTS = 500    # Productos más frecuentes a considerar
USE_SUBCLUSTERS = True # Cambiar a True si quieres reglas por subcluster

# ================================
# 📥 Cargar datos
# ================================
orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")
order_products = pd.read_csv("data/order_products__prior.csv")
clusters = pd.read_csv("data/usuarios_clusters.csv")

# Añadir nombres de producto
order_products = order_products.merge(products[['product_id', 'product_name']], on='product_id')

# Añadir user_id y cluster/subcluster
orders_clusters = orders[['order_id', 'user_id']].merge(clusters, on='user_id')
datos = order_products.merge(orders_clusters, on='order_id')

# ================================
# 📁 Carpetas de salida
# ================================
os.makedirs("mba_rules", exist_ok=True)
os.makedirs("mba_fallbacks", exist_ok=True)

# ================================
# 🧠 Función para generar reglas
# ================================
def generar_mba_para(grupo_datos, nombre_archivo_prefix):
    # Filtrar productos más frecuentes
    top_productos = grupo_datos['product_name'].value_counts().nlargest(TOP_N_PRODUCTS).index
    grupo_datos = grupo_datos[grupo_datos['product_name'].isin(top_productos)]

    # Limitar a los primeros N pedidos
    pedidos = grupo_datos['order_id'].unique()[:TOP_N_ORDERS]
    grupo_datos = grupo_datos[grupo_datos['order_id'].isin(pedidos)]

    # 🧺 Cesta: one-hot
    basket = grupo_datos.groupby(['order_id', 'product_name'])['product_id'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Apriori
    frequent_itemsets = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True)

    if frequent_itemsets.empty:
        # Fallback: productos más populares
        fallback = grupo_datos['product_name'].value_counts().reset_index()
        fallback.columns = ['product_name', 'n_compras']
        fallback.to_csv(f"mba_fallbacks/{nombre_archivo_prefix}_fallback.csv", index=False)
        print(f"⚠️ No se generaron reglas. Fallback guardado como {nombre_archivo_prefix}_fallback.csv")
        return

    reglas = association_rules(frequent_itemsets, metric="lift", min_threshold=MIN_LIFT)
    reglas.to_csv(f"mba_rules/{nombre_archivo_prefix}_rules.csv", index=False)
    print(f"✅ Reglas guardadas: mba_rules/{nombre_archivo_prefix}_rules.csv")

# ================================
# 🔁 Ejecutar análisis
# ================================

if USE_SUBCLUSTERS:
    for (cluster, sub), grupo in clusters.groupby(['cluster', 'subcluster']):
        print(f"📊 Cluster {cluster} - Subcluster {sub}")
        user_ids = grupo['user_id'].unique()
        grupo_datos = datos[datos['user_id'].isin(user_ids)]
        nombre = f"cluster{cluster}_sub{sub}"
        generar_mba_para(grupo_datos, nombre)
else:
    print("📊 Generando reglas globales...")
    generar_mba_para(datos, "global")
