import os
import sqlite3
import pandas as pd
from surprise import Dataset, Reader, SVD, dump, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Conectar a la base de datos
conn = sqlite3.connect("instacart.db")

# Crear carpeta para guardar el modelo general
carpeta_modelo_general = "modelo_general"
os.makedirs(carpeta_modelo_general, exist_ok=True)

# Crear archivo de métricas
archivo_metricas = os.path.join(carpeta_modelo_general, "metricas_modelo_general.txt")
with open(archivo_metricas, "w") as f:
    f.write("Métricas del modelo SVD general\n")
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

# Calcular métricas binarias con threshold más bajo
threshold = 0.5
y_true = [int(pred.r_ui >= threshold) for pred in predictions]
y_pred = [int(pred.est >= threshold) for pred in predictions]

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

# Generar recomendaciones personalizadas para un usuario
user_id = 12345  # Cambia por el ID del usuario deseado

# Obtener todas las islas posibles
all_aisles = interacciones['aisle_id'].unique()

# Islas ya visitadas por el usuario
aisles_visitadas = interacciones[interacciones['user_id'] == user_id]['aisle_id'].unique()

# Islas no visitadas
aisles_no_visitadas = [a for a in all_aisles if a not in aisles_visitadas]

# Calcular frecuencia global de cada isla
frecuencia = interacciones[interacciones['comprado'] == 1]['aisle_id'].value_counts(normalize=True).to_dict()

# Predecir para cada isla no visitada y penalizar por popularidad
predicciones = []
for aisle_id in aisles_no_visitadas:
    pred = algo.predict(user_id, aisle_id)
    freq = frecuencia.get(aisle_id, 0)
    pred.adjusted_score = pred.est * (1 - freq)  # penalización por frecuencia
    predicciones.append(pred)

# Ordenar por score ajustado
top_N = 10
top_predicciones = sorted(predicciones, key=lambda x: x.adjusted_score, reverse=True)[:top_N]

# Mostrar recomendaciones
print(f"\nRecomendaciones personalizadas para el usuario {user_id} (Top {top_N}):\n")
for pred in top_predicciones:
    print(f"Aisle {int(pred.iid)} → Estimación SVD: {pred.est:.4f} | Score ajustado: {pred.adjusted_score:.4f}")

conn.close()
