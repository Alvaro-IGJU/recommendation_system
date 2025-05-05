# buscador_semantico.py
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Cargar modelo multilingüe
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Cargar productos desde un archivo CSV
products = pd.read_csv("data/products.csv")[['product_id', 'product_name']]
products['product_name'] = products['product_name'].str.lower()

# Generar embeddings semánticos para los nombres
product_embeddings = model.encode(products['product_name'].tolist(), convert_to_tensor=True)

# Función de búsqueda
def buscar_productos_semanticos(consulta, top_n=10):
    consulta = consulta.lower()
    consulta_embedding = model.encode(consulta, convert_to_tensor=True)
    similitudes = util.cos_sim(consulta_embedding, product_embeddings)[0]
    top_indices = torch.topk(similitudes, k=top_n).indices
    resultados = products.iloc[top_indices].copy()
    resultados['score'] = similitudes[top_indices].cpu().numpy()
    return resultados.reset_index(drop=True)
