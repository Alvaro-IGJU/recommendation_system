# 🛒 Recommendation System

Sistema de recomendación de productos alimentarios usando **SVD**, **Market Basket Analysis** y **búsqueda semántica**.

---

## ⚙️ Instalación

```bash
pip install -r requirements.txt
```

---

## 📁 Datos necesarios

1. Descarga los datos desde este enlace de Google Drive:  
   👉 [Dataset del proyecto](https://drive.google.com/drive/folders/1FXWafNbEtMYv7n_tHdKi_TzqcTw6BdMT?usp=sharing)

2. Copia los siguientes directorios en la raíz del proyecto:

```
/data
/mba_rules_clusters
```

---

## 🚀 Ejecución (modo usuario)

Lanza la aplicación web en dos terminales:

### 1️⃣ Terminal 1: iniciar el servidor FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2️⃣ Terminal 2: abrir túnel público con ngrok

```bash
ngrok http 8000
```

🔗 **Haz clic en el enlace público que te dará ngrok** (ej. `https://abcd1234.ngrok-free.app`) para acceder directamente al simulador web funcionando.

> ⚠️ No necesitas cambiar `API_BASE_URL` a mano, el sistema lo detecta automáticamente y lo inserta en el HTML.

---

## 🧠 Entrenamiento de modelos

Si deseas entrenar los modelos desde cero, ejecuta los siguientes scripts en orden:

1. `create_db.ipynb`
2. `main.ipynb`
3. `dataframe_for_svd.py`
4. `subcluster_svd_training.py`
5. `mba_por_subcluster.py`

---

## ✅ Requisitos previos

- Python 3.10 o superior
- FastAPI
- Uvicorn
- scikit-learn
- sentence-transformers
- pandas, numpy, etc. (consultar `requirements.txt`)

---

## 📬 Contacto

> Proyecto desarrollado por Alvaro Iglesias Jusmet  
> © 2025 · Todos los derechos reservados
