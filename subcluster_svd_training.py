import os
import pandas as pd
from surprise import Dataset, Reader, SVD, dump

# Crear carpeta para guardar los modelos
carpeta_modelos = "modelos_subclusters"
os.makedirs(carpeta_modelos, exist_ok=True)

# Cargar datos
interacciones = pd.read_csv("interacciones_usuario_isla.csv")
usuarios_clusters = pd.read_csv("usuarios_clusters.csv")

# Entrenar un modelo SVD por subcluster
for (cluster_id, subcluster_id), grupo in usuarios_clusters.groupby(['cluster', 'subcluster']):
    print(f"Entrenando modelo para cluster {cluster_id}, subcluster {subcluster_id}...")

    user_ids = grupo['user_id'].unique()
    subset = interacciones[interacciones['user_id'].isin(user_ids)]

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(subset[['user_id', 'aisle_id', 'comprado']], reader)
    trainset = data.build_full_trainset()

    algo = SVD()
    algo.fit(trainset)

    modelo_path = os.path.join(carpeta_modelos, f"model_svd_cluster{cluster_id}_sub{subcluster_id}.pkl")
    dump.dump(modelo_path, algo=algo)
