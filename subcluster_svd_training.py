import os
import pandas as pd
from surprise import Dataset, Reader, SVD, dump, accuracy
from surprise.model_selection import train_test_split
import sqlite3
from sklearn.metrics import precision_score, recall_score, f1_score

# 📥 Conectar a la base de datos
conn = sqlite3.connect("instacart.db")

# Crear carpeta para guardar los modelos
carpeta_modelos = "modelos_subclusters"
os.makedirs(carpeta_modelos, exist_ok=True)

# Crear archivo de métricas
archivo_metricas = "metricas_modelos.txt"
with open(archivo_metricas, "w") as f:
    f.write("📊 Métricas de modelos SVD por subcluster\n")
    f.write("=" * 50 + "\n\n")

# Cargar datos
interacciones = pd.read_sql("SELECT * FROM interacciones_usuario_isla", conn)
usuarios_clusters = pd.read_sql("SELECT * FROM usuarios_clusters", conn)

# Entrenar un modelo SVD por subcluster
for (cluster_id, subcluster_id), grupo in usuarios_clusters.groupby(['cluster', 'subcluster']):
    print(f"\n🚀 Entrenando modelo para cluster {cluster_id} - subcluster {subcluster_id}")

    user_ids = grupo['user_id'].unique()
    subset = interacciones[interacciones['user_id'].isin(user_ids)]

    if len(subset) < 100:
        print("⚠️ Subcluster con muy pocas interacciones. Se omite.")
        continue

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(subset[['user_id', 'aisle_id', 'comprado']], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # ✅ Modelo mejorado
    algo = SVD(
        n_factors=150,
        n_epochs=35,
        lr_all=0.005,
        reg_all=0.04,
        verbose=False
    )
    algo.fit(trainset)

    # Evaluar
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    # 🔍 Convertir predicciones a binario para precisión y recall
    y_true = [int(pred.r_ui >= 0.5) for pred in predictions]
    y_pred = [int(pred.est >= 0.5) for pred in predictions]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Guardar modelo
    modelo_path = os.path.join(carpeta_modelos, f"model_svd_cluster{cluster_id}_sub{subcluster_id}.pkl")
    dump.dump(modelo_path, algo=algo)

    # Guardar métricas
    with open(archivo_metricas, "a") as f:
        f.write(f"Modelo cluster {cluster_id} - subcluster {subcluster_id}\n")
        f.write(f" - N interacciones: {len(subset)}\n")
        f.write(f" - RMSE      : {rmse:.4f}\n")
        f.write(f" - MAE       : {mae:.4f}\n")
        f.write(f" - Precision : {precision:.4f}\n")
        f.write(f" - Recall    : {recall:.4f}\n")
        f.write(f" - F1-score  : {f1:.4f}\n")
        f.write("-" * 40 + "\n")

print("\n✅ Entrenamiento finalizado. Métricas completas en 'metricas_modelos.txt'.")
