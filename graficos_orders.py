import os

# 🧼 LIMPIEZA antes de importar matplotlib
os.environ.pop("MPLBACKEND", None)

import matplotlib
matplotlib.use("TkAgg")  # Backend gráfico clásico (asegúrate de tener tkinter instalado)

import matplotlib.pyplot as plt
import pandas as pd

# 📂 Carga del CSV
orders = pd.read_csv("orders.csv")
orders['days_since_prior_order'].fillna(0, inplace=True)

# 📅 Día de la semana
plt.figure(figsize=(8, 4))
orders['order_dow'].value_counts().sort_index().plot(kind='bar')
plt.title("Distribución de pedidos por día de la semana (0=Domingo)")
plt.xlabel("Día")
plt.ylabel("Número de pedidos")
plt.tight_layout()
plt.show()

# 🕒 Hora del día
plt.figure(figsize=(10, 4))
orders['order_hour_of_day'].value_counts().sort_index().plot(kind='bar')
plt.title("Distribución por hora del día")
plt.xlabel("Hora")
plt.ylabel("Número de pedidos")
plt.tight_layout()
plt.show()

# ⏳ Días desde pedido anterior
plt.figure(figsize=(10, 4))
orders['days_since_prior_order'].value_counts().sort_index().plot(kind='bar')
plt.title("Días desde el pedido anterior")
plt.xlabel("Días")
plt.ylabel("Número de pedidos")
plt.tight_layout()
plt.show()

# 📊 Estadísticas
print("\n📊 Estadísticas descriptivas:\n")
print(orders[['order_number', 'order_hour_of_day', 'order_dow', 'days_since_prior_order']].describe())
