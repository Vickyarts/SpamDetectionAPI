# 📌 Spam Detection API  

This is a Django-based API application for spam message detection, supporting two types of models:  

- **Naive Bayes (Sklearn)**  
- **Neural Network (PyTorch)**  

The system provides a **token-based API** for spam message classification, dataset management, model training, and deployment.  

---

## 🔥 Key Features  

### ✅ **1. Token-Based Authentication (Bearer Token)**  
- All API requests **must include the token as a Bearer token in the Authorization header**.  
- **User Token:** Required for making inference requests.  
- **Admin Token:** Required for dataset upload, training, and model deployment.  
- **Master Token:** Special token (editable only on the server) used for creating, deleting, and resetting user/admin tokens.  

**🔹 How to send the token in requests:**  
```
Authorization: Bearer your_token_here
```

---

### ✅ **2. Model Training & Deployment**  
- Admins can upload datasets (**CSV format with "message" and "label" columns**).  
- Models are trained **on the server** using the uploaded dataset.  
- Admins can **deploy trained models** for real-time inference.  

### ✅ **3. Inference Methods**  
- **Naive Bayes (NB)** – Uses Sklearn's Naive Bayes model.  
- **Neural Network (NN)** – Uses a PyTorch-based neural network model.  
- **Hybrid** – Combines both models with weighted probabilities:  
  - **NB Weight = 1.2**  
  - **NN Weight = 1.7**  
  - Final prediction is a weighted sum of both models' probabilities.  

---

## 🔗 API Endpoints  

### 📌 **1. Inference**  
🔹 `POST /inference` – Perform spam detection on a given message.  

#### **Request Format**  
```json
{
    "message": "Win a free iPhone now!"
}
```
#### **Headers:**  
```
Authorization: Bearer user_token_here
```
#### **Response Format**  
```json
{
    "status": "success",
    "spam": 1
}
```

---

### 📌 **2. Admin: Train New Model**  
🔹 `POST /admin/retrain` – Train a new model using an uploaded dataset.  

#### **Request Format**  
```json
{
    "dataset_id": "DNXWUpygGg"
}
```
#### **Headers:**  
```
Authorization: Bearer admin_token_here
```
#### **Response Format**  
```json
{
    "status": "success",
    "job_id": "JfRQuiOupKJJkGhrwJHA",
    "message": "Model training initiated."
}
```

---

### 📌 **3. Admin: Get Dataset List**  
🔹 `GET /admin/get_datasets` – Retrieve the list of datasets stored on the server.  

#### **Headers:**  
```
Authorization: Bearer admin_token_here
```
#### **Response Format**  
```json
{
    "status": "success",
    "datasets": [
        {
            "dataset_id": "DNXWUpygGg",
            "row_count": 5572,
            "value_ratio": "87:13",
            "created_by": 6,
            "created_at": "2025-02-18T17:41:58.262Z"
        }
    ]
}
```

---

### 📌 **4. Admin: Add Dataset**  
🔹 `POST /admin/add_dataset` – Upload a new dataset.  

#### **Headers:**  
```
Authorization: Bearer admin_token_here
```
#### **Response Format**  
```json
{
    "status": "success",
    "message": "Dataset created with ID:hdCZKJcoVu."
}
```

---

### 📌 **5. Admin: Get Job History**  
🔹 `GET /admin/get_job` – Retrieve the history of all training jobs.  

#### **Headers:**  
```
Authorization: Bearer admin_token_here
```
#### **Response Format**  
```json
{
    "status": "success",
    "datasets": [
        {
            "job_id": "LmPAauQspnytXNZYKLzt",
            "dataset_id": "DNXWUpygGg",
            "status": "Completed",
            "created_by": 6,
            "created_at": "2025-02-18T17:43:03.606Z"
        },
        {
            "job_id": "JfRQuiOupKJJkGhrwJHA",
            "dataset_id": "DNXWUpygGg",
            "status": "Running",
            "created_by": 6,
            "created_at": "2025-02-18T22:41:29.192Z"
        }
    ]
}
```

---

### 📌 **6. Admin: Get Trained Models**  
🔹 `GET /admin/get_model` – Retrieve all trained models available on the server.  

#### **Headers:**  
```
Authorization: Bearer admin_token_here
```
#### **Response Format**  
```json
{
    "status": "success",
    "datasets": [
        {
            "model_id": "HrUrnHZePSgp",
            "dataset_id": "DNXWUpygGg",
            "naive_accuracy": 0.9659498207885304,
            "naive_f1_score": 0.8741721854304636,
            "network_bce_loss": 0.3045429587364197,
            "created_by": 6,
            "created_at": "2025-02-18T17:47:12.746Z"
        }
    ]
}
```

---

### 📌 **7. Admin: Get Deployed Model**  
🔹 `GET /admin/get_deployment` – Retrieve details of the currently deployed model.  

#### **Headers:**  
```
Authorization: Bearer admin_token_here
```
#### **Response Format**  
```json
{
    "status": "success",
    "deployment": {
        "deployment_id": 2,
        "model": {
            "model_id": "HrUrnHZePSgp",
            "naive_accuracy": 0.9659498207885304,
            "naive_f1_score": 0.8741721854304636,
            "network_bce_loss": 0.3045429587364197,
            "created_by": 6,
            "created_at": "2025-02-18T17:47:12.746Z"
        },
        "type": "Hybrid",
        "created_by": 6,
        "created_at": "2025-02-18T17:49:02.395Z"
    }
}
```

---

### 📌 **8. Admin: Get Training Log**  
🔹 `GET /admin/get_traininglog` – Retrieve training logs of a particular job.  

#### **Request Format**  
```json
{
    "job_id": "LmPAauQspnytXNZYKLzt"
}
```
#### **Headers:**  
```
Authorization: Bearer admin_token_here
```
#### **Response**  
```text
2025-02-18 17:43:03: Data Embedding Started.
2025-02-18 17:46:58: Data Embedding completed.
2025-02-18 17:46:58: Naive Bayes Model training started.
2025-02-18 17:46:58: Naive Bayes Model training completed.
2025-02-18 17:46:58: Torch Model training started.
2025-02-18 17:46:58: Epoch: 1/64  Epoch Loss: 0.7086177468299866
2025-02-18 17:46:58: Epoch: 2/64  Epoch Loss: 0.7064809799194336
2025-02-18 17:46:59: Epoch: 3/64  Epoch Loss: 0.7043536901473999
...
2025-02-18 17:47:12: Epoch: 62/64  Epoch Loss: 0.3049781322479248
2025-02-18 17:47:12: Epoch: 63/64  Epoch Loss: 0.2984835207462311
2025-02-18 17:47:12: Epoch: 64/64  Epoch Loss: 0.2922666072845459
2025-02-18 17:47:12: Torch Model training completed.
2025-02-18 17:47:12: Torch Model evaluation started.
2025-02-18 17:47:12: Torch Model evaluation completed.
2025-02-18 17:47:12: Naive Bayes Model evaluation started.
2025-02-18 17:47:12: Naive Bayes Model evaluation completed.
2025-02-18 17:47:12: Saving models....
2025-02-18 17:47:12: Model Saved.
2025-02-18 17:47:12: Training job ID:LmPAauQspnytXNZYKLzt completed.

```

---

### 📌 **9. Admin: Deploy New Model**  
🔹 `POST /admin/deploy` – Deploy a trained model for inference.  

#### **Request Format**  
```json
{
    "model_id": "HrUrnHZePSgp",
    "deployment-type": "naive-bayes"
}
```
#### **Headers:**  
```
Authorization: Bearer admin_token_here
```
#### **Response Format**  
```json
{
    "status": "success",
    "message": "Model with ID:HrUrnHZePSgp is deployed."
}
```

---

### 📌 **10. Admin: Manage Users & Admins**  
🔹 `POST /admin/manage` – Create, delete, or reset tokens for users/admins (**requires master token**).  

#### **Request Format**  
```json
{
    "action": "create",  // Options: "create", "delete", "reset"
    "role": "user",  // Options: "user", "admin"
    "args": {
        "id": 1, 
        "name": "Vicky"  // Only needed with creating user.
    }
}
```
#### **Headers:**  
```
Authorization: Bearer master_token_here
```
#### **Response Format**  
```json
{
    "message": "User token created successfully",
    "token": "new_user_token_here"
}
```

---

## 🔑 **Authentication & Tokens**  
- **All tokens must be sent as Bearer tokens in the `Authorization` header**.  
- **Users** can only perform **inference** using their **user token**.  
- **Admins** can manage datasets, train, and deploy models using an **admin token**.  
- **Master Token** can only be edited on the **server-side** and is used for **managing user/admin tokens**.  

---

## 🚀 **Deployment**  

### **1️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **2️⃣ Apply Migrations**  
```bash
python manage.py migrate
```

### **3️⃣ Run Server**  
```bash
python manage.py runserver
```

---

## 📜 License  
This project is licensed under [LICENSE](LICENSE).  