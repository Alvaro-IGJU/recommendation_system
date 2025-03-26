import pandas as pd

# 📥 Cargar el dataset
df = pd.read_csv("user_aisle_features.csv")

# 🔍 Seleccionar las columnas de islas
aisle_columns = df.columns[6:]  # A partir de la columna 6 están las islas

# 🔄 Transformar a formato largo: cada fila es (usuario, isla, comprado)
interacciones = df.melt(
    id_vars=["user_id"],
    value_vars=aisle_columns,
    var_name="aisle_id",
    value_name="comprado"
)

# Asegurar tipo entero en aisle_id y comprado
interacciones['aisle_id'] = interacciones['aisle_id'].astype(int)
interacciones['comprado'] = interacciones['comprado'].astype(int)

# 💾 Guardar en CSV
interacciones.to_csv("interacciones_usuario_isla.csv", index=False)
print("✅ Archivo 'interacciones_usuario_isla.csv' generado correctamente sin aleatoriedad.")
