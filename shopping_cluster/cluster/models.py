from django.db import models

# Create your models here.
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def perform_clustering():
    customer_data = pd.read_csv('shopping_data.csv')
    data = customer_data.iloc[:, 3:5].values

    model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
    labels = model.fit_predict(data)

    return {
        "x": data[:, 0].tolist(),
        "y": data[:, 1].tolist(),
        "labels": labels.tolist()
    }
from django.db import models

class Customer(models.Model):
    customer_id = models.IntegerField(unique=True)
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    annual_income = models.FloatField()
    spending_score = models.FloatField()

    def __str__(self):
        return f'Customer {self.customer_id}'

