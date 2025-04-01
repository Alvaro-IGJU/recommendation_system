import sqlite3
import pandas as pd

# 📥 Conectar a la base de datos
conn = sqlite3.connect("instacart.db")
cursor = conn.cursor()

# 📤 Leer la tabla de características por usuario
df = pd.read_sql("SELECT * FROM user_aisle_features", conn)

# 🔍 Seleccionar las columnas de islas (a partir de la columna 6)
aisle_columns = df.columns[6:]

# 🔄 Transformar a formato largo: cada fila es (usuario, isla, comprado)
interacciones = df.melt(
    id_vars=["user_id"],
    value_vars=aisle_columns,
    var_name="aisle_id",
    value_name="comprado"
)

# Asegurar tipos correctos
interacciones['aisle_id'] = interacciones['aisle_id'].astype(int)
interacciones['comprado'] = interacciones['comprado'].astype(int)

# 💾 Guardar en CSV
interacciones.to_csv("data/interacciones_usuario_isla.csv", index=False)
print("✅ Archivo 'interacciones_usuario_isla.csv' guardado correctamente.")

# 🧱 Crear tabla en la base de datos
cursor.execute("DROP TABLE IF EXISTS interacciones_usuario_isla")

cursor.execute("""
CREATE TABLE interacciones_usuario_isla (
    user_id INTEGER,
    aisle_id INTEGER,
    comprado INTEGER,
    FOREIGN KEY (user_id) REFERENCES orders(user_id),
    FOREIGN KEY (aisle_id) REFERENCES aisles(aisle_id)
)
""")

# Insertar los datos
interacciones.to_sql("interacciones_usuario_isla", conn, if_exists="append", index=False)
conn.commit()
conn.close()

print("✅ Tabla 'interacciones_usuario_isla' creada e insertada en 'instacart.db'.")
