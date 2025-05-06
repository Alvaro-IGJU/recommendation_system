from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from recomendar_usuario_completo_usuario_producto import recomendar_usuario_completo_usuario_producto
from nlp import buscar_productos_semanticos

app = FastAPI()

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
# Endpoints
# =====================
@app.get("/")
async def inicio():
    return {"mensaje": "🛒 API de recomendaciones funcionando correctamente"}

@app.post("/recomendar/")
async def recomendar(req: RecomendacionRequest):
    return recomendar_usuario_completo_usuario_producto(req.user_id, req.n)

@app.post("/buscar/")
async def buscar(req: BusquedaRequest):
    resultados_df = buscar_productos_semanticos(req.consulta, top_n=req.top_n)
    resultados = resultados_df[["product_name"]].to_dict(orient="records")
    return {"resultados": resultados}
