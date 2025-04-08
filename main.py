import pandas as pd
from recomendar_usuario_clusterizado import recomendar_usuario_completo

# 📥 Cargar datos
interacciones = pd.read_csv("data/interacciones_usuario_isla.csv")
usuarios_clusters = pd.read_csv("data/usuarios_clusters.csv")

# 👤 Usuario a probar
usuario_objetivo = 543  # ⬅️ Sustituye por un ID válido
print(recomendar_usuario_completo(usuario_objetivo, n=10))
print(recomendar_usuario_completo(145879, n=10))
