from django.urls import path
from .views import (
    inference, 
    admin_retrain, 
    admin_get_datasets,
    admin_add_dataset,
    admin_get_jobs,
    admin_get_models,
    admin_getCurrent_deployment,
    admin_model_deployment,
    admin_get_training_log,
    admin_management
)


urlpatterns = [
    path("inference", inference, name="Inference"), 
    path("admin/retrain", admin_retrain, name="Retrain"), 
    path("admin/get_datasets", admin_get_datasets, name="GetDataset"), 
    path("admin/add_dataset", admin_add_dataset, name="AddDataset"), 
    path("admin/get_job", admin_get_jobs, name="GetJobs"), 
    path("admin/get_model", admin_get_models, name="GetModels"), 
    path("admin/get_deployment", admin_getCurrent_deployment, name="GetCurrentDeployment"), 
    path("admin/get_traininglog", admin_get_training_log, name="GetTrainingLog"), 
    path("admin/deploy", admin_model_deployment, name="Deploy"), 
    path("admin/manage", admin_management, name="Management")
]


# CCoTzkZDnUtYalmoHefQFCuedaTibcrnUywr
