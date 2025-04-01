import os
from eralchemy import render_er
import subprocess

# 1. Exportar el esquema a formato .er
print("🧱 Extrayendo esquema desde instacart.db...")
subprocess.run([
    "eralchemy",
    "-i", "sqlite:///instacart.db",
    "-o", "esquema.er"
])

# 2. Renderizar a imagen PNG
print("🖼️ Generando imagen del diagrama ER...")
render_er("esquema.er", "diagrama_er.png")

print("✅ Diagrama generado como 'diagrama_er.png'")
