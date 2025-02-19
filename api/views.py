import os 
import json
import time
import datetime
import psutil
import subprocess
import pandas as pd
from .models import User, Admin, Models, Dataset, TrainingJob, Deployment
from .Model import SpamDetection, labelEncoding, generate_token, get_Deployment_Config
from django.http import HttpResponseForbidden, JsonResponse, FileResponse




with open('config.json', 'r') as f:
    master_token = json.load(f)['authorization']['master-token']


deployment_config = get_Deployment_Config()
model = SpamDetection(mode=deployment_config['inference-type'], model_id=deployment_config['deployed-model'])



def inference(request):
    token = request.headers["Authorization"].split(' ')[1]
    if authUser(token):
        data = load_JSONBody(request) 
        if data:
            message = data['message']
            res = model.inference(message)
            if isinstance(res, str):
                return JsonResponse({
                    'status': 'failed',
                    'message': res
                })
            else:
                return JsonResponse({
                    'status': 'success',
                    'spam': int(res)
                })
        else: 
            return JsonResponse({
                "status": "failed", 
                "message": "No JSON found."
            })
    else: 
        return HttpResponseForbidden("Invalid authorization token.")




# ADMIN VIEWS
# Training
last_training_job = 999999
def admin_retrain(request):
    token = request.headers["Authorization"].split(' ')[1]
    admin = authAdmin(token)
    if admin:
        try:
            data = load_JSONBody(request)
            if data:
                dataset_id = data['dataset_id']
                job_id = generate_token(n=20) 
                global last_training_job
                if not psutil.pid_exists(last_training_job):
                    DETACHED_PROCESS = 0x00000008
                    process = subprocess.Popen(
                        ["python", "train.py", "--admin_id", str(admin), "--dataset_id", dataset_id, "--job_id", job_id, "--test_size", str(0.1)], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL, 
                        cwd="func/", 
                        creationflags=DETACHED_PROCESS
                    )
                    last_training_job = process.pid
                    print(f"Training Job started with Process ID: {process.pid}.")

                    time.sleep(5)
                    if process.poll() is None:
                        print(f"Training job ID:{job_id} is running...")
                        return JsonResponse({
                            'status': 'success', 
                            'job_id': job_id,
                            'message': 'Model training initiated.'
                        })
                    else:
                        print(f"Training Job ID:{job_id} has exited.")
                        return JsonResponse({
                            'status': 'failed', 
                            'message': 'Error while creating a Training Job.'
                        })
                else: 
                    return JsonResponse({
                        "status": "failed", 
                        "message": "Compute is already busy running a training job. Try again later."
                    })
            else: 
                return JsonResponse({
                    "status": "failed", 
                    "message": "No JSON found."
                })
        except Exception as e: 
            print(e)
            return JsonResponse({
                "status": "failed", 
                "message": "Invalid Json Structure. Example: {'dataset_id': 'jdAdutWbnX'}"
            })
    else: 
        return HttpResponseForbidden("Invalid authorization token.")



def admin_get_datasets(request):
    token = request.headers["Authorization"].split(' ')[1]
    data = load_JSONBody(request)
    admin = authAdmin(token)
    if admin:
        if data and "dataset_id" in data:
            try:
                dataset = Dataset.objects.filter(dataset_id=data["dataset_id"]).first()
                if dataset:
                    d = {
                        "dataset_id": dataset.dataset_id,
                        "row_count": dataset.row_count,
                        "value_ratio": dataset.value_ratio,
                        "created_by": dataset.created_by,
                        "created_at": dataset.created_at
                    }
                    return JsonResponse({
                        "status": "success", 
                        "dataset": d
                    })
                else: 
                    return JsonResponse({
                        "status": "failed", 
                        "message": f"Dataset with ID:{data['dataset_id']} not found."
                    })
            except:
                return JsonResponse({
                    "status": "failed", 
                    "message": "Error while fetching dataset."
                })
        else: 
            try:
                datasets = Dataset.objects.all()
                d = []
                for dataset in datasets:
                    d.append({
                        "dataset_id": dataset.dataset_id,
                        "row_count": dataset.row_count,
                        "value_ratio": dataset.value_ratio,
                        "created_by": dataset.created_by,
                        "created_at": dataset.created_at
                    })
                return JsonResponse({
                    "status": "success", 
                    "datasets": d
                })
            except:
                return JsonResponse({
                    "status": "failed", 
                    "message": "Error while fetching datasets."
                })
    else: 
        return HttpResponseForbidden("Invalid authorization token.")

def admin_add_dataset(request):
    token = request.headers["Authorization"].split(' ')[1]
    admin = authAdmin(token)
    if admin:
        try:
            dataset = request.FILES['dataset']
        except:
            return JsonResponse({
                "status": "failed", 
                "message": "File not found in request."
            })
        if not str(dataset).endswith('.csv'):
            return JsonResponse({
                "error": "Invalid file type."
            })
        dataset_id = generate_token(n=10)
        dataset_info = validate_dataset(dataset, dataset_id)
        if dataset_info:
            d = Dataset(
                dataset_id=dataset_id,
                row_count = dataset_info[0],
                value_ratio = f"{dataset_info[1]}:{dataset_info[2]}",
                created_by=admin,
                created_at=datetime.datetime.now()
            )
            d.save()
            return JsonResponse({
                "status": "success", 
                "message": f"Dataset created with ID:{dataset_id}."
            })
        else: 
            try:
                os.remove(f"data/{dataset_id}.csv")
                return JsonResponse({
                    "status": "failed", 
                    "message": "Dataset validation Failed. Dataset should only have 2 columns['message', 'label']. Label should only have two unique values."
                })
            except:
                pass
    else: 
        return HttpResponseForbidden("Invalid authorization token.")



def admin_get_jobs(request):
    token = request.headers["Authorization"].split(' ')[1]
    data = load_JSONBody(request)
    admin = authAdmin(token)
    if admin:
        if data and "job_id" in data:
            try:
                job = TrainingJob.objects.filter(job_id=data["job_id"]).first()
                if job:
                    j = {
                        "job_id": job.job_id,
                        "dataset_id": job.dataset_id,
                        "status": job.status,
                        "created_by": job.created_by,
                        "created_at": job.created_at
                    }
                    return JsonResponse({
                        "status": "success", 
                        "dataset": j
                    })
                else: 
                    return JsonResponse({
                        "status": "failed", 
                        "message": f"Training Job with ID:{data['job_id']} not found."
                    })
            except:
                return JsonResponse({
                    "status": "failed", 
                    "message": "Error while fetching jobs."
                })
        else: 
            try:
                jobs = TrainingJob.objects.all()
                j = []
                for job in jobs:
                    j.append({
                        "job_id": job.job_id,
                        "dataset_id": job.dataset_id,
                        "status": job.status,
                        "created_by": job.created_by,
                        "created_at": job.created_at
                    })
                return JsonResponse({
                    "status": "success", 
                    "datasets": j
                })
            except:
                return JsonResponse({
                    "status": "failed", 
                    "message": "Error while fetching jobs."
                })
    else: 
        return HttpResponseForbidden("Invalid authorization token.")


def admin_get_models(request):
    token = request.headers["Authorization"].split(' ')[1]
    data = load_JSONBody(request)
    admin = authAdmin(token)
    if admin:
        if data and "model_id" in data:
            try:
                model = Models.objects.filter(model_id=data["model_id"]).first()
                if model:
                    m = {
                        "model_id": model.model_id,
                        "naive_accuracy": model.naive_accuracy,
                        "naive_f1_score": model.naive_f1_score,
                        "network_bce_loss": model.network_bce_loss,
                        "created_by": model.created_by,
                        "created_at": model.created_at
                    }
                    return JsonResponse({
                        "status": "success", 
                        "dataset": m
                    })
                else: 
                    return JsonResponse({
                        "status": "failed", 
                        "message": f"Model with ID:{data['model_id']} not found."
                    })
            except Exception as e: 
                print(e)
                return JsonResponse({
                    "status": "failed", 
                    "message": "Error while fetching dataset."
                })
        else: 
            try:
                models = Models.objects.all()
                m = []
                for model in models:
                    m.append({
                        "model_id": model.model_id,
                        "dataset_id": model.dataset_id,
                        "naive_accuracy": model.naive_accuracy,
                        "naive_f1_score": model.naive_f1_score,
                        "network_bce_loss": model.network_bce_loss,
                        "created_by": model.created_by,
                        "created_at": model.created_at
                    })
                return JsonResponse({
                    "status": "success", 
                    "datasets": m
                })
            except Exception as e: 
                print(e, 299)
                return JsonResponse({
                    "status": "failed", 
                    "message": "Error while fetching models."
                })
    else: 
        return HttpResponseForbidden("Invalid authorization token.")



def admin_getCurrent_deployment(request):
    token = request.headers["Authorization"].split(' ')[1]
    admin = authAdmin(token)
    if admin:
        try:
            deployment = Deployment.objects.all().last()
            if deployment:
                d_model = Models.objects.filter(model_id=deployment.model_id).first()
                if deployment.type == "naive-bayes":
                    m = "Naive-Bayes"
                elif deployment.type == "nn": 
                    m = "Neural Network"
                else: 
                    m = "Hybrid"
                return JsonResponse({
                    "status": "success",  
                    "deployment": {
                        "deployment_id": deployment.id, 
                        "model": {
                            "model_id": d_model.model_id,
                            "naive_accuracy": d_model.naive_accuracy,
                            "naive_f1_score": d_model.naive_f1_score,
                            "network_bce_loss": d_model.network_bce_loss,
                            "created_by": d_model.created_by,
                            "created_at": d_model.created_at
                        }, 
                        "type": m, 
                        "created_by": deployment.created_by, 
                        "created_at": deployment.created_at
                    }
                })
            else: 
                return JsonResponse({
                    "status": "success",
                    "message": "No model is deployed."
                })
        except Exception as e: 
            print(e)
            return JsonResponse({
                "status": "failed", 
                "message": "Error while fetching current deployment."
            })
    else: 
        return HttpResponseForbidden("Invalid authorization token.")


def admin_model_deployment(request):
    token = request.headers["Authorization"].split(' ')[1]
    admin = authAdmin(token)
    if admin:
        try:
            data = load_JSONBody(request)
            if data:
                model_id = data['model_id']
                deployment_type = data['deployment-type'].lower()
                global model 
                del model 
                model = SpamDetection(mode=deployment_type, model_id=model_id)
                deployment = Deployment(
                    model_id=model_id,
                    type=deployment_type,
                    created_by=admin,
                    created_at=datetime.datetime.now()
                )
                deployment.save()
                global deployment_config
                deployment_config = {
                    "deployed-model": model_id,
                    "inference-type": type
                }
                return JsonResponse({
                    'status': 'success',  
                    'message': f'Model with ID:{model_id} is deployed.'
                })
            else: 
                return JsonResponse({
                    "status": "failed", 
                    "message": "No JSON found."
                })
        except Exception as e: 
            print(e)
            return JsonResponse({
                "status": "failed", 
                "message": "Invalid Json Structure. Example: {'model_id': 'jdAdutWbnX', 'deployment-type': 'naive-bayes'}"
            })
    else: 
        return HttpResponseForbidden("Invalid authorization token.")


def admin_get_training_log(request):
    token = request.headers["Authorization"].split(' ')[1]
    admin = authAdmin(token)
    if admin:
        try:
            data = load_JSONBody(request)
            if data:
                job_id = data["job_id"]
                file = open(f"func/training_logs/{job_id}_training.log", "rb")
                return FileResponse(file)
            else: 
                return JsonResponse({
                    "status": "failed", 
                    "message": "No JSON data found."
                })
        except:
            return JsonResponse({
                "status": "failed", 
                "message": "Error while fetching training log."
            })
                
    else: 
        return HttpResponseForbidden("Invalid authorization token.")


def admin_management(request):
    token = request.headers["Authorization"].split(' ')[1]
    if token == master_token:
        try:
            data = load_JSONBody(request)
            if data:
                if data['command'] == 'create':
                    try:
                        if data['user-type'] == 'user':
                            args = data['args']
                            key = generate_token(n=24)
                            user = User(
                                name=args['name'], 
                                key=key,
                                created_at=datetime.datetime.now()
                            )
                            user.save()
                            return JsonResponse({
                                "status": "success", 
                                "message": "User creation successfully.",
                                "key": key
                            })
                        elif data['user-type'] == 'admin':
                            args = data['args']
                            key = generate_token(n=36)
                            user = Admin(
                                name=args['name'], 
                                key=key,
                                created_at=datetime.datetime.now()
                            )
                            user.save()
                            return JsonResponse({
                                "status": "success", 
                                "message": "Admin creation successfully.",
                                "key": key
                            })
                        else:
                            return JsonResponse({
                                "status": "failed", 
                                "message": "Invalid 'user-type'"
                            })
                    except Exception as e: 
                        print(e)
                        return JsonResponse({
                            "status": "failed", 
                            "message": "Invalid Json Structure. Example: {'command':'create', 'user-type':'user', 'args':{'name':'Vicky'}}"
                        })
                elif data['command'] == 'delete':
                    try:
                        if data['user-type'] == 'user':
                            id = int(data['args']['id'])
                            user = User.objects.filter(id=id).first()
                            if user:
                                user = User.objects.get(id=id)
                                user.delete()
                                return JsonResponse({
                                    "status": "success", 
                                    "message": "User deleted successfully."
                                })
                            else: 
                                return JsonResponse({
                                    "status": "failed", 
                                    "message": f"No user found with ID:{id}."
                                })
                        elif data['user-type'] == 'admin':
                            id = int(data['args']['id'])
                            admin = Admin.objects.filter(id=id).first()
                            if admin:
                                admin.delete()
                                return JsonResponse({
                                    "status": "success", 
                                    "message": "Admin deleted successfully."
                                })
                            else: 
                                return JsonResponse({
                                    "status": "failed", 
                                    "message": f"No admin found with ID:{id}."
                                })
                    except Exception as e: 
                        print(e)
                        return JsonResponse({
                            "status": "failed", 
                            "message": "Invalid Json Structure. Example: {'command':'delete', 'user-type':'user', 'args':{'id':'120'}}"
                        })
                elif data['command'] == 'list':
                    try:
                        if data['user-type'] == 'user':
                            try:
                                users = []
                                for user in User.objects.all():
                                    users.append({
                                        "id": user.id,
                                        "name": user.name,
                                        "key": user.key,
                                        "created_at": user.created_at
                                    })
                                return JsonResponse({
                                    "status": "success", 
                                    "users": users
                                })
                            except: 
                                return JsonResponse({
                                    "status": "failed", 
                                    "message": "Error while fetching users."
                                })
                        elif data['user-type'] == 'admin':
                            try:
                                admins = []
                                for admin in Admin.objects.all():
                                    admins.append({
                                        "id": admin.id,
                                        "name": admin.name,
                                        "key": admin.key,
                                        "created_at": admin.created_at
                                    })
                                return JsonResponse({
                                    "status": "success", 
                                    "users": admins
                                })
                            except: 
                                return JsonResponse({
                                    "status": "failed", 
                                    "message": "Error while fetching admins."
                                })
                    except Exception as e: 
                        print(e)
                        return JsonResponse({
                            "status": "failed", 
                            "message": "Invalid Json Structure. Example: {'command':'list', 'user-type':'user'}"
                        })
                elif data['command'] == 'reset-token':
                    try:
                        if data['user-type'] == 'user':
                            id = int(data['args']['id'])
                            user = User.objects.filter(id=id).first()
                            if user:
                                key = generate_token(n=24)
                                user.key = key
                                user.save()
                                return JsonResponse({
                                    "status": "success", 
                                    "message": "User token reset successfully.",
                                    "key": key
                                })
                            else: 
                                return JsonResponse({
                                    "status": "failed", 
                                    "message": f"No user found with ID:{id}."
                                })
                        elif data['user-type'] == 'admin':
                            id = int(data['args']['id'])
                            admin = Admin.objects.filter(id=id).first()
                            if admin:
                                key = generate_token(n=36)
                                admin.key = key
                                admin.save()
                                return JsonResponse({
                                    "status": "success", 
                                    "message": "Admin token reset successfully.", 
                                    "key": key
                                })
                            else: 
                                return JsonResponse({
                                    "status": "failed", 
                                    "message": f"No admin found with ID:{id}."
                                })
                    except Exception as e: 
                        print(e)
                        return JsonResponse({
                            "status": "failed", 
                            "message": "Invalid Json Structure. Example: {'command':'reset', 'user-type':'user', 'args':{'id':'120'}}"
                        })
                else:
                    return JsonResponse({
                        "status": "failed", 
                        "message": "Invalid command. Available commands create, delete, list, reset-token"
                    })
            else: 
                return JsonResponse({
                    "status": "failed", 
                    "message": "No JSON found."
                })
        except Exception as e: 
            print(e)
            return JsonResponse({
                "status": "failed", 
                "message": "Invalid data structure. Example: {'command':'create', 'user-type':'user', 'args':{'name':'XXXXX'}}"
            })
    else: 
        return HttpResponseForbidden("Invalid authorization token.")





# FUNCTIONS
def authUser(token):
    user = User.objects.filter(key=token).first()
    if user:
        return True
    else: 
        return False
    
def authAdmin(token):
    admin = Admin.objects.filter(key=token).first()
    if admin:
        return admin.id
    else: 
        return False
    
def validate_dataset(dataset, dataset_id):
    try:
        df = pd.read_csv(dataset)
        if ('message' in df.columns) and ('label' in df.columns) and (len(df.columns) == 2) and (len(df['label'].unique()) == 2):
            if df['label'].dtype.name == 'object':
                encoded = labelEncoding(df['label'].values.reshape(-1,1))
                df.drop(columns=['label'])
                df['label'] = pd.Series(encoded)

                df.to_csv(f'data/{dataset_id}.csv' ,sep=',', index=False)
                count = df['label'].count()
                return (
                    count,
                    round((df['label'].value_counts()['ham'] / count)*100), 
                    round((df['label'].value_counts()['spam'] / count)*100) 
                )
            elif df['label'].dtype.name == 'int64':
                df.to_csv(f'data/{dataset_id}.csv' ,sep=',', index=False)
                count = df['label'].count()
                return (
                    count,
                    round((df['label'].value_counts()[0] / count)*100),
                    round((df['label'].value_counts()[1] / count)*100)
                )
        else: 
            return False
    except Exception as e:
        print(e)
        return False


def load_JSONBody(request):
    try:
        data = json.loads(request.body)
        return data 
    except:
        return False
    