from surprise import Dataset, Reader, SVD, dump
import pandas as pd
import os

# 📥 Leer el CSV con interacciones
interacciones = pd.read_csv("interacciones_usuario_isla.csv")

# 📐 Preparar los datos para Surprise
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(interacciones[['user_id', 'aisle_id', 'comprado']], reader)
trainset = data.build_full_trainset()

# 📁 Ruta del modelo guardado
modelo_path = "modelo_svd.pkl"

# ✅ Usar SVD: cargar si existe, entrenar si no
if os.path.exists(modelo_path):
    print("✅ Cargando modelo SVD guardado...")
    _, algo = dump.load(modelo_path)
else:
    print("🚀 Entrenando modelo SVD...")
    algo = SVD(n_factors=150, random_state=42)
    algo.fit(trainset)
    dump.dump(modelo_path, algo=algo)
    print("💾 Modelo guardado como 'modelo_svd.pkl'")


