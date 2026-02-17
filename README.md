# Heart Disease Classification – ML Deployment (Capstone)

This project deploys a trained Machine Learning pipeline for heart disease risk classification as a REST API.
It includes data preprocessing, model training, inference API (FastAPI), Docker containerization, and deployment on Google Cloud Run.

## 1) Project Structure

```
.
├─ artifacts/
│  ├─ models/
│  │  ├─ heart_disease_pipeline.joblib
│  │  └─ metadata.json
│  └─ sample_payload.json
├─ data/
│  ├─ raw/
│  │  └─ heart.csv
│  └─ processed/
│     └─ heart_clean.csv
├─ src/
│  ├─ api/
│  │  ├─ main.py
│  │  ├─ schemas.py
│  │  └─ security.py
│  ├─ data/
│  │  └─ make_dataset.py
│  └─ model/
│     └─ train_pipeline.py
├─ Dockerfile
├─ .dockerignore
└─ requirements.txt
```

## 2) Requirements

- Python 3.11 recommended
- Conda (optional, for local environment)
- Docker Desktop
- Deployed Docker image on Google Cloud Run

## 3) Local Setup (Conda)

### 3.1 Create and activate environment
```powershell
conda create -n mlhub python=3.11 -y
conda activate mlhub
pip install -r requirements.txt
```

### 3.2 Run the API locally
Set an API key (PowerShell):
```powershell
$env:API_KEY="devkey123"
```

Run the server:
```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open:
- Health: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs

## 4) Using the API

### 4.1 Health check
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" | Select-Object -ExpandProperty Content
```

### 4.2 Prediction endpoint
The `/predict` endpoint expects JSON input:

```json
{
  "features": {
    "HighBP": 1.0,
    "HighChol": 1.0,
    "CholCheck": 1.0,
    "BMI": 40.0,
    "Smoker": 1.0,
    "Stroke": 0.0,
    "Diabetes": 0.0,
    "PhysActivity": 0.0,
    "Fruits": 0.0,
    "Veggies": 1.0,
    "HvyAlcoholConsump": 0.0,
    "AnyHealthcare": 1.0,
    "NoDocbcCost": 0.0,
    "GenHlth": 5.0,
    "MentHlth": 18.0,
    "PhysHlth": 15.0,
    "DiffWalk": 1.0,
    "Sex": 0.0,
    "Age": 9.0,
    "Education": 4.0,
    "Income": 3.0
  }
}
```

Example:
```powershell
$headers = @{
  "Content-Type" = "application/json"
  "X-API-Key" = "devkey123"
}
$body = Get-Content -Raw "artifacts\sample_payload.json"

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Headers $headers -Body $body
```

## 5) Docker

### 5.1 Build image
```powershell
docker build -t heart-api:0.2 .
```

### 5.2 Run container
```powershell
docker run --rm -p 8000:8000 -e API_KEY="devkey123" heart-api:0.2
```

## 6) Cloud Deployment (GCP Cloud Run)

- Container port: 8000
- Environment variable: API_KEY

Test:
```
https://https://heart-disease-api-800105443947.us-central1.run.app/health
```

## 7) Reproducibility Note

Dependencies are pinned in requirements.txt to avoid scikit-learn version mismatch issues during deployment.

## 8) Model Training (Optional)

```powershell
python src/data/make_dataset.py
python src/model/train_pipeline.py
```

## 9) Academic Use

This repository is submitted as part of a capstone project for academic purposes.
