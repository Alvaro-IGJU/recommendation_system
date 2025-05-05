import os
import ast
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import precision_score, recall_score, f1_score

# ================================
# ⚙️ CONFIGURACIÓN
# ================================
MIN_SUPPORT = 0.002
MIN_CONFIDENCE = 0.2
MIN_LIFT = 1
TOP_N_ORDERS = 15000
TOP_N_PRODUCTS = 1200
TOP_N_RECOMENDACIONES = 25
MIN_PRODUCTOS_TRAIN = 5
MIN_PRODUCTOS_TEST = 3

# ================================
# 📥 Cargar datos
# ================================
orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")
order_products = pd.read_csv("data/order_products__prior.csv")
clusters = pd.read_csv("data/usuarios_clusters.csv")

# Añadir nombres
order_products = order_products.merge(products[['product_id', 'product_name']], on='product_id')
orders_clusters = orders[['order_id', 'user_id']].merge(clusters, on='user_id')
datos = order_products.merge(orders_clusters, on='order_id')

# ================================
# 📁 Carpetas de salida
# ================================
os.makedirs("mba_rules_clusters", exist_ok=True)
os.makedirs("mba_fallbacks_clusters", exist_ok=True)

# ================================
# 🧠 Función MBA + evaluación
# ================================
def generar_mba_y_metricas(grupo_datos, nombre_archivo_prefix, user_ids):
    orders_unicos = grupo_datos['order_id'].unique()
    corte = int(len(orders_unicos) * 0.8)
    pedidos_train = orders_unicos[:corte]
    pedidos_test = orders_unicos[corte:]

    grupo_train = grupo_datos[grupo_datos['order_id'].isin(pedidos_train)]
    grupo_test = grupo_datos[grupo_datos['order_id'].isin(pedidos_test)]

    top_productos = grupo_train['product_name'].value_counts().nlargest(TOP_N_PRODUCTS).index
    grupo_train = grupo_train[grupo_train['product_name'].isin(top_productos)]

    pedidos = grupo_train['order_id'].unique()[:TOP_N_ORDERS]
    grupo_train = grupo_train[grupo_train['order_id'].isin(pedidos)]

    basket = grupo_train.groupby(['order_id', 'product_name'])['product_id'].count().unstack().fillna(0)
    basket = basket.astype(bool)

    frequent_itemsets = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True)
    if frequent_itemsets.empty:
        return 0.0, 0.0, 0.0

    reglas = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    reglas = reglas[reglas['lift'] >= MIN_LIFT]
    if reglas.empty:
        return 0.0, 0.0, 0.0

    reglas = reglas.sort_values(by=["confidence", "lift"], ascending=False)
    reglas.to_csv(f"mba_rules_clusters/{nombre_archivo_prefix}_rules.csv", index=False)

    y_true_all, y_pred_all = [], []
    for user_id in user_ids:
        productos_test = grupo_test[grupo_test['user_id'] == user_id]['product_name'].unique().tolist()
        productos_train = grupo_train[grupo_train['user_id'] == user_id]['product_name'].unique().tolist()
        if len(productos_test) < MIN_PRODUCTOS_TEST or len(productos_train) < MIN_PRODUCTOS_TRAIN:
            continue

        recomendados = []
        for _, regla in reglas.iterrows():
            antecedente = ast.literal_eval(regla['antecedents']) if isinstance(regla['antecedents'], str) else regla['antecedents']
            consecuente = ast.literal_eval(regla['consequents']) if isinstance(regla['consequents'], str) else regla['consequents']
            if any(p in productos_train for p in antecedente):
                recomendados.extend([p for p in consecuente if p not in productos_train])

        recomendados = list(dict.fromkeys(recomendados))[:TOP_N_RECOMENDACIONES]
        if not recomendados:
            continue

        y_true = [1 if prod in productos_test else 0 for prod in recomendados]
        y_pred = [1] * len(recomendados)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    if y_true_all:
        precision = precision_score(y_true_all, y_pred_all, zero_division=0)
        recall = recall_score(y_true_all, y_pred_all, zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    else:
        precision = recall = f1 = 0.0

    return precision, recall, f1

# ================================
# 🔁 Ejecutar análisis por cluster
# ================================
metricas_globales = []

for cluster, grupo in clusters.groupby('cluster'):
    print(f"\n📊 Procesando cluster {cluster}...")
    user_ids = grupo['user_id'].unique()
    grupo_datos = datos[datos['user_id'].isin(user_ids)]
    nombre = f"cluster{cluster}"

    p, r, f = generar_mba_y_metricas(grupo_datos, nombre, user_ids)
    print(f"cluster{cluster}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
    metricas_globales.append((cluster, p, r, f))

# ================================
# 💾 Guardar métricas
# ================================
with open("mba_metricas_por_cluster.txt", "w") as f:
    for cluster, p, r, f1 in sorted(metricas_globales):
        f.write(f"cluster{cluster}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}\n")

print("\n✅ Evaluación completa. Métricas guardadas en mba_metricas_por_cluster.txt")
