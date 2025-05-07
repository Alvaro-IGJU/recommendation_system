from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests

from recomendar_usuario_completo_usuario_producto import recomendar_usuario_completo_usuario_producto
from nlp import buscar_productos_semanticos

app = FastAPI()

# Servir archivos estáticos (logo, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# Modelos de entrada
# =====================
class RecomendacionRequest(BaseModel):
    user_id: int
    n: int = 10

class BusquedaRequest(BaseModel):
    consulta: str
    top_n: int = 10

# =====================
# Obtener la URL pública de ngrok
# =====================
def get_ngrok_url():
    try:
        r = requests.get("http://127.0.0.1:4040/api/tunnels")
        tunnels = r.json()["tunnels"]
        for t in tunnels:
            if t["proto"] == "https":
                return t["public_url"]
    except Exception as e:
        print("⚠️ No se pudo obtener la URL de ngrok:", e)
        return "http://localhost:8000"

# =====================
# Endpoints
# =====================
@app.get("/", response_class=HTMLResponse)
async def inicio():
    api_base = get_ngrok_url()
    with open("index.html", encoding="utf-8") as f:
        html = f.read().replace("API_BASE_PLACEHOLDER", api_base)
    return HTMLResponse(content=html)

@app.post("/recomendar/")
async def recomendar(req: RecomendacionRequest):
    return recomendar_usuario_completo_usuario_producto(req.user_id, req.n)

@app.post("/buscar/")
async def buscar(req: BusquedaRequest):
    resultados_df = buscar_productos_semanticos(req.consulta, top_n=req.top_n)
    resultados = resultados_df[["product_name"]].to_dict(orient="records")
    return {"resultados": resultados}
