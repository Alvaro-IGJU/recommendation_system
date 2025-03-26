import pandas as pd
import matplotlib.pyplot as plt
from surprise import dump
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns

# 📁 Ruta al modelo entrenado
modelo_path = "modelo_svd.pkl"

# 📥 Cargar el modelo SVD guardado
_, algo = dump.load(modelo_path)

# 🧠 Obtener los factores latentes del usuario y el trainset
user_factors = algo.pu
trainset = algo.trainset

# 🔢 Número de clusters
n_clusters = 3

# 🎯 Clustering con KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(user_factors)

# 🔄 Obtener user_ids reales
user_ids = [trainset.to_raw_uid(i) for i in range(user_factors.shape[0])]
df_clusters = pd.DataFrame({
    'user_id': user_ids,
    'cluster': clusters
})

# 💾 Guardar en CSV
df_clusters.to_csv("clusters_usuarios_svd.csv", index=False)

# 🎨 Visualización PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(user_factors)

plt.figure(figsize=(10, 7))
palette = sns.color_palette("tab10", n_clusters)

for cluster_id in range(n_clusters):
    mask = df_clusters['cluster'] == cluster_id
    plt.scatter(coords[mask, 0], coords[mask, 1],
                label=f"Cluster {cluster_id}", alpha=0.6, s=15,
                color=palette[cluster_id])

plt.title("👥 Clusters de usuarios según factores latentes del modelo SVD")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("clusters_svd_usuarios.png")
print("✅ Imagen guardada como 'clusters_svd_usuarios.png'")
