from django.urls import path

from . import views
from .views import predict_gender

from .views import cluster_view, customer_ids_api, customer_cluster_lookup_api

urlpatterns = [
    path('api/cluster/',  views.cluster_view),
    path('api/customers/', views.customer_ids_api, name='customer_ids_api'),
    path('api/customer-cluster/', views.customer_cluster_lookup_api, name='customer_cluster_lookup_api'),
    path('api/predict-gender/', predict_gender),

]
