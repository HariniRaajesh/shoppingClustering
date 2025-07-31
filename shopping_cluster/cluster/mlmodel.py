import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'shopping_data.csv')

def perform_clustering(n_clusters=5):
    df = pd.read_csv(CSV_PATH)
    data = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = model.fit_predict(data)
    return {
        "x": data[:, 0].tolist(),
        "y": data[:, 1].tolist(),
        "labels": labels.tolist()
    }

def get_customer_cluster(customer_id, n_clusters=5):
    df = pd.read_csv(CSV_PATH)
    if customer_id not in df["CustomerID"].values:
        return None
    data = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = model.fit_predict(data)
    idx = df[df["CustomerID"] == customer_id].index[0]
    return {
        "customer_id": int(df.loc[idx, "CustomerID"]),
        "annual_income": float(data[idx][0]),
        "spending_score": float(data[idx][1]),
        "cluster": int(labels[idx])
    }
