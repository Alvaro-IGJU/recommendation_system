#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrena un modelo SVD general a partir de la tabla
`interacciones_usuario_isla` en instacart.db y
guarda:
  • modelo_general/modelo_svd_general.pkl
  • modelo_general/metricas_modelo_general.txt
"""

import os
import sqlite3
import pandas as pd
from surprise import Dataset, Reader, SVD, dump, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


def main() -> None:
    # ---------- Conexión y datos ----------
    with sqlite3.connect("instacart.db") as conn:
        interacciones = pd.read_sql(
            "SELECT * FROM interacciones_usuario_isla", conn
        )

    # ---------- Carpeta de salida ----------
    out_dir = "modelo_general"
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metricas_modelo_general.txt")

    # Crear/limpiar archivo de métricas (UTF‑8)
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        f.write("📊 Métricas del modelo SVD general\n")
        f.write("=" * 50 + "\n\n")

    # ---------- Preparar datos para Surprise ----------
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(
        interacciones[["user_id", "aisle_id", "comprado"]],
        reader,
    )

    trainset, testset = train_test_split(
        data, test_size=0.20, random_state=42
    )

    # ---------- Entrenamiento ----------
    algo = SVD(
        n_factors=150,
        n_epochs=35,
        lr_all=0.005,
        reg_all=0.04,
        random_state=42,
        verbose=False,
    )
    algo.fit(trainset)

    # ---------- Evaluación ----------
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    y_true = [int(p.r_ui >= 0.5) for p in predictions]
    y_pred = [int(p.est >= 0.5) for p in predictions]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ---------- Guardar modelo ----------
    model_path = os.path.join(out_dir, "modelo_svd_general.pkl")
    dump.dump(model_path, algo=algo)

    # ---------- Guardar métricas ----------
    with open(metrics_path, "a", encoding="utf-8", newline="") as f:
        f.write(f"- N.º de interacciones: {len(interacciones)}\n")
        f.write(f"- RMSE      : {rmse:.4f}\n")
        f.write(f"- MAE       : {mae:.4f}\n")
        f.write(f"- Precision : {precision:.4f}\n")
        f.write(f"- Recall    : {recall:.4f}\n")
        f.write(f"- F1‑score  : {f1:.4f}\n")
        f.write("=" * 50 + "\n")

    print("✅ Modelo general entrenado y guardado con métricas.")


if __name__ == "__main__":
    main()
