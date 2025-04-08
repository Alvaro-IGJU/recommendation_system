import os
import sqlite3
import pandas as pd
from surprise import Dataset, Reader, SVD, dump, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# 📥 Conectar a la base de datos
conn = sqlite3.connect("instacart.db")

# Crear carpeta para guardar el modelo general
carpeta_modelo_general = "modelo_general"
os.makedirs(carpeta_modelo_general, exist_ok=True)

# Crear archivo de métricas
archivo_metricas = os.path.join(carpeta_modelo_general, "metricas_modelo_general.txt")
with open(archivo_metricas, "w") as f:
    f.write("📊 Métricas del modelo SVD general\n")
    f.write("=" * 50 + "\n\n")

# Cargar datos de interacciones completas
interacciones = pd.read_sql("SELECT * FROM interacciones_usuario_isla", conn)

# Preparar datos para Surprise
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(interacciones[['user_id', 'aisle_id', 'comprado']], reader)

# Separar datos en entrenamiento y prueba
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Entrenar el modelo SVD general
algo = SVD(
    n_factors=150,
    n_epochs=35,
    lr_all=0.005,
    reg_all=0.04,
    verbose=False
)
algo.fit(trainset)

# Evaluar el modelo
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)

# Calcular métricas binarias
y_true = [int(pred.r_ui >= 0.5) for pred in predictions]
y_pred = [int(pred.est >= 0.5) for pred in predictions]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Guardar modelo
modelo_path = os.path.join(carpeta_modelo_general, "modelo_svd_general.pkl")
dump.dump(modelo_path, algo=algo)

# Guardar métricas
with open(archivo_metricas, "a") as f:
    f.write(f" - N interacciones: {len(interacciones)}\n")
    f.write(f" - RMSE      : {rmse:.4f}\n")
    f.write(f" - MAE       : {mae:.4f}\n")
    f.write(f" - Precision : {precision:.4f}\n")
    f.write(f" - Recall    : {recall:.4f}\n")
    f.write(f" - F1-score  : {f1:.4f}\n")
    f.write("=" * 50 + "\n")

conn.close()

"✅ Modelo general entrenado y guardado con métricas."
