from django.urls import path
from . import views

urlpatterns = [
    path('measure-object-size/', views.measure_object_size, name='measure_object_size'),
]