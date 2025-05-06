# recommendation_system

Execution order:

- create_db.ipynb
- main.ipynb
- dataframe_for_svd.py
- subcluster_svd_training.py
- mba_por_subcluster.py
- streamlit run main.py


- uvicorn main:app --host 0.0.0.0 --port 8501
- ngrok http 8501