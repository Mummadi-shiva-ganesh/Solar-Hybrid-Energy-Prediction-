# MLOps & Cloud Deployment Guide

## Section 10.5 — Cloud Deployment (AWS / Azure / GCP)

This guide shows how to deploy the Solar Energy Prediction API to each major cloud platform.

---

## Prerequisites

```bash
# Build the Docker image locally first
docker build -t solar-yield-api .
docker run -p 5000:5000 solar-yield-api   # verify it works
```

---

## Option A — AWS Elastic Beanstalk

```bash
# 1. Install EB CLI
pip install awsebcli

# 2. Initialise project (choose Python 3.9 platform)
eb init solar-yield-api --platform python-3.9 --region ap-south-1

# 3. Create environment & deploy
eb create solar-yield-env
eb deploy

# 4. Open in browser
eb open

# 5. Monitor logs
eb logs
```

**Alternatively — AWS via Docker (EC2)**
```bash
# Push image to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.ap-south-1.amazonaws.com
docker tag solar-yield-api:latest <account>.dkr.ecr.ap-south-1.amazonaws.com/solar-yield-api:latest
docker push <account>.dkr.ecr.ap-south-1.amazonaws.com/solar-yield-api:latest

# Run on EC2 (after SSH into instance)
docker pull <account>.dkr.ecr.ap-south-1.amazonaws.com/solar-yield-api:latest
docker run -d -p 80:5000 <account>.dkr.ecr.ap-south-1.amazonaws.com/solar-yield-api:latest
```

---

## Option B — Microsoft Azure Container Apps

```bash
# 1. Login
az login

# 2. Create resource group
az group create --name solar-yield-rg --location eastus

# 3. Create Azure Container Registry
az acr create --name solaryieldacr --resource-group solar-yield-rg --sku Basic
az acr login --name solaryieldacr

# 4. Push image
docker tag solar-yield-api solaryieldacr.azurecr.io/solar-yield-api:latest
docker push solaryieldacr.azurecr.io/solar-yield-api:latest

# 5. Deploy as Container App
az containerapp create \
  --name solar-yield-api \
  --resource-group solar-yield-rg \
  --image solaryieldacr.azurecr.io/solar-yield-api:latest \
  --target-port 5000 \
  --ingress external \
  --env-vars API_KEY=solar-yield-secret-2026

# 6. Get URL
az containerapp show --name solar-yield-api --resource-group solar-yield-rg \
  --query properties.configuration.ingress.fqdn -o tsv
```

---

## Option C — Google Cloud Run (Recommended for Simplicity)

```bash
# 1. Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build and push via Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/solar-yield-api

# 3. Deploy to Cloud Run (auto-scales to zero when idle — free tier friendly)
gcloud run deploy solar-yield-api \
  --image gcr.io/YOUR_PROJECT_ID/solar-yield-api \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --port 5000 \
  --set-env-vars API_KEY=solar-yield-secret-2026

# 4. Get service URL
gcloud run services describe solar-yield-api --region asia-south1 \
  --format 'value(status.url)'
```

---

## Option D — Render (Easiest — Free Tier Available)

1. Push project to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repository
4. Set:
   - **Environment**: Docker
   - **Port**: 5000
   - **Environment Variable**: `API_KEY = solar-yield-secret-2026`
5. Click **Deploy** — Render builds and deploys automatically

---

## Environment Variables for Production

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API key for prediction endpoints | `solar-yield-secret-2026` |
| `FLASK_ENV` | Flask environment | `production` |
| `PORT` | Server port | `5000` |

> ⚠️ **Security**: Always override `API_KEY` with a strong secret in production.
> Use your cloud platform's secret manager (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager).

---

## Model Versioning with MLflow (10.2 — Optional Enhancement)

```bash
pip install mlflow

# Log experiment during training
import mlflow
mlflow.set_experiment("solar-yield")
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metrics({"r2": val_r2, "rmse": val_rmse, "mae": val_mae})
    mlflow.sklearn.log_model(model, "random_forest_model")

# View experiments UI
mlflow ui   # open http://localhost:5001
```

## DVC for Dataset Versioning (10.2 — Optional Enhancement)

```bash
pip install dvc

# Initialize DVC
dvc init
dvc add data/solar_generation.csv
dvc add data/solar_weather.csv

# Push to remote storage (e.g. AWS S3)
dvc remote add -d myremote s3://my-bucket/solar-yield-dvc
dvc push

# On another machine: reproduce exact dataset version
dvc pull
```
