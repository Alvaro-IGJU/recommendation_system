import pandas as pd
from recomendar_usuario_clusterizado import recomendar_usuario_completo

# 📥 Cargar datos
interacciones = pd.read_csv("interacciones_usuario_isla.csv")
usuarios_clusters = pd.read_csv("usuarios_clusters.csv")

# 👤 Usuario a probar
usuario_objetivo = 5  # ⬅️ Sustituye por un ID válido
print(recomendar_usuario_completo(2789, n=10))
