from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='upload_file'),  # Home route for file upload
    #path('download_csv/', views.download_csv, name='download_csv'), # New URL for downloading the CSV
    path('download_preprocessed_csv/', views.download_preprocessed_csv, name='download_preprocessed_csv'),  # New route for downloading preprocessed data
    # path("", views.home, name='home'),
    # path("todos/", views.todos, name="Todos")
]