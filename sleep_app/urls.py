from django.urls import path, include
from . import views

urlpatterns = [
    
    path('',views.upload_view,name='upload_page'),
    
    path('dashboard/',views.dashboard_view,name='dashboard_page'),
]
