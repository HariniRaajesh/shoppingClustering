from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import os

# CSV file path (same directory as views.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'shopping_data.csv'   )

# 1. Cluster Visualization View
@require_GET
def cluster_view(request):
    try:
        n_clusters = int(request.GET.get('n_clusters', 5))
        col1 = int(request.GET.get('col1', 3))  # e.g., Annual Income
        col2 = int(request.GET.get('col2', 4))  # e.g., Spending Score

        df = pd.read_csv(csv_path)
        data = df.iloc[:, [col1, col2]].dropna().values

        model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        labels = model.fit_predict(data)

        return JsonResponse({
            "x": data[:, 0].tolist(),
            "y": data[:, 1].tolist(),
            "labels": labels.tolist()
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# 2. Get List of Customer IDs
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


# ---- Customer ID List API ----
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'shopping_data.csv')

@csrf_exempt
def customer_ids_api(request):
    try:
        df = pd.read_csv(csv_path)
        if 'CustomerID' not in df.columns:
            return JsonResponse({"error": "CustomerID column missing in CSV"}, status=500)

        customer_ids = df['CustomerID'].dropna().unique().tolist()
        return JsonResponse({'customer_ids': customer_ids})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

@csrf_exempt
def customer_cluster_lookup_api(request):
    customer_id = request.GET.get('customer_id')
    if not customer_id:
        return JsonResponse({'error': 'Missing customer_id parameter'}, status=400)

    # Set correct CSV path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, 'shopping_data.csv')
    try:
        df = pd.read_csv(csv_path)

        # Ensure CustomerID column exists
        if 'CustomerID' not in df.columns:
            return JsonResponse({'error': 'CustomerID column missing in CSV'}, status=500)

        # Clean the data
        df = df.dropna()
        df['CustomerID'] = df['CustomerID'].astype(str)  # Convert to string for matching

        # Save original CustomerID for lookup
        customer_ids = df['CustomerID'].values

        # ✅ Drop non-numeric columns like Gender
        if 'Gender' in df.columns:
            df = df.drop(columns=['Gender'])

        features = df.drop(columns=['CustomerID'])

        # ✅ Standardize the numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # ✅ Cluster the data
        model = AgglomerativeClustering(n_clusters=5)
        clusters = model.fit_predict(X_scaled)

        # ✅ Find the cluster for the selected customer
        if customer_id not in customer_ids:
            return JsonResponse({'error': 'Customer ID not found'}, status=404)

        index = list(customer_ids).index(customer_id)
        cluster_num = int(clusters[index])

        # ✅ Define cluster labels
        cluster_labels = {
            0: "Luxury Shoppers 💎",
            1: "Budget Buyers 💰",
            2: "Average Spenders 🛒",
            3: "Discount Seekers 🏷️",
            4: "High Value Loyalists 🔁",
        }

        cluster_name = cluster_labels.get(cluster_num, f"Cluster {cluster_num}")

        return JsonResponse({
            'customer_id': customer_id,
            'cluster': cluster_num,
            'cluster_name': cluster_name
        })

    except FileNotFoundError:
        return JsonResponse({'error': 'shopping_data.csv not found'}, status=500)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import json

@csrf_exempt
def predict_gender(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    try:
        # Load CSV
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, 'shopping_data.csv')
        df = pd.read_csv(csv_path).dropna()

        # Features & Target
        X = df[['Age', 'annual_income', 'spending_score']]
        y = df['Gender']

        # Encode Gender
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)  # Male = 1, Female = 0

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Model
        model = LogisticRegression()
        model.fit(X_scaled, y_encoded)

        # Parse input JSON
        body = json.loads(request.body.decode('utf-8'))
        age = float(body.get('age'))
        income = float(body.get('income'))
        score = float(body.get('score'))

        # Predict
        input_scaled = scaler.transform([[age, income, score]])
        pred = model.predict(input_scaled)[0]
        pred_label = le.inverse_transform([pred])[0]

        return JsonResponse({'predicted_gender': pred_label})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
