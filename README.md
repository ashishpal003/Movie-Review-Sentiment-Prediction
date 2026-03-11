# Movie Review Sentiment Identification

# 🚀 End-to-End MLOps Project

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)]()
[![Kubernetes](https://img.shields.io/badge/Kubernetes-EKS-blue)]()
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)]()
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-blue)]()

A **production-style MLOps pipeline** demonstrating the complete
lifecycle of a machine learning system --- from experimentation to
deployment and monitoring.

This project integrates modern tools used in real-world ML
infrastructure including:

-   Experiment tracking with **MLflow + Dagshub**
-   Data & model versioning with **DVC + AWS S3**
-   CI/CD automation with **GitHub Actions**
-   Containerization using **Docker**
-   Deployment on **AWS EKS (Kubernetes)**
-   Monitoring using **Prometheus + Grafana**

------------------------------------------------------------------------

# 📌 Architecture Overview

    Developer
       │
       ▼
    GitHub Repository
       │
       ▼
    GitHub Actions (CI/CD)
       │
       ▼
    Docker Image
       │
       ▼
    AWS ECR (Container Registry)
       │
       ▼
    AWS EKS (Kubernetes Cluster)
       │
       ▼
    Flask API (Model Serving)
       │
       ▼
    Monitoring Stack
       ├── Prometheus
       └── Grafana

------------------------------------------------------------------------

# 🧰 Tech Stack

  Category              Tools
  --------------------- ----------------------
  Language              Python 3.10
  Experiment Tracking   MLflow, Dagshub
  Data Versioning       DVC
  Storage               AWS S3
  API Framework         Flask
  Containerization      Docker
  CI/CD                 GitHub Actions
  Orchestration         Kubernetes (AWS EKS)
  Monitoring            Prometheus, Grafana

------------------------------------------------------------------------

# 📂 Project Structure

    project/
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── notebooks/
    │
    ├── src/
    │   ├── logger/
    │   ├── data_ingestion.py
    │   ├── data_preprocessing.py
    │   ├── feature_engineering.py
    │   ├── model_building.py
    │   ├── model_evaluation.py
    │   └── register_model.py
    │
    ├── flask_app/
    │
    ├── tests/
    ├── scripts/
    │
    ├── dvc.yaml
    ├── params.yaml
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# ⚙️ Local Development Setup

## 1️⃣ Clone Repository

    git clone <repo-url>
    cd <repo>

## 2️⃣ Create Environment

    conda create -n atlas python=3.10
    conda activate atlas

## 3️⃣ Install Dependencies

    pip install cookiecutter

Generate project template:

    cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science

------------------------------------------------------------------------

# 🔬 Experiment Tracking (MLflow + Dagshub)

1.  Create a repository on Dagshub
2.  Connect your GitHub repository
3.  Copy the **MLflow tracking URI**

Install dependencies:

    pip install mlflow dagshub

MLflow tracks:

-   hyperparameters
-   model metrics
-   artifacts
-   model versions

------------------------------------------------------------------------

# 📦 Data Versioning with DVC

Initialize DVC:

    dvc init

Add remote storage:

    mkdir local_s3
    dvc remote add -d mylocal local_s3

Push data to remote:

    dvc push

Reproduce pipeline:

    dvc repro

------------------------------------------------------------------------

# 🔄 ML Pipeline

Pipeline stages include:

    data_ingestion
          │
          ▼
    data_preprocessing
          │
          ▼
    feature_engineering
          │
          ▼
    model_training
          │
          ▼
    model_evaluation

Each stage is defined in:

    dvc.yaml

------------------------------------------------------------------------

# 🌐 Flask Model API

The trained model is exposed as a **REST API**.

Example endpoint:

    POST /predict

Example request:

``` json
{
 "feature1": 10,
 "feature2": 20
}
```

------------------------------------------------------------------------

# 🐳 Docker Containerization

Build Docker image:

    docker build -t capstone-app:latest .

Run container:

    docker run -p 8000:8080 capstone-app:latest

------------------------------------------------------------------------

# 🔁 CI/CD Pipeline

GitHub Actions pipeline automatically:

1.  Runs tests
2.  Builds Docker image
3.  Pushes image to **AWS ECR**
4.  Deploys to **EKS cluster**

Secrets required:

    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION
    AWS_ACCOUNT_ID
    ECR_REPOSITORY

------------------------------------------------------------------------

# ☁️ Deployment on AWS EKS

Create cluster:

    eksctl create cluster \
    --name flask-app-cluster \
    --region us-east-1 \
    --nodegroup-name flask-app-nodes \
    --node-type t3.small \
    --nodes 1 \
    --managed

Check cluster:

    kubectl get nodes

Check services:

    kubectl get svc

Access deployed API:

    http://<external-ip>:5000

------------------------------------------------------------------------

# 📊 Monitoring Stack

Monitoring pipeline:

    Flask API
       │
       ▼
    Prometheus
       │
       ▼
    Grafana Dashboard

Metrics monitored:

-   API request rate
-   latency
-   error rate
-   CPU usage
-   memory usage

------------------------------------------------------------------------

# 📈 Prometheus Setup

Download Prometheus:

    wget https://github.com/prometheus/prometheus/releases/download/v2.46.0/prometheus-2.46.0.linux-amd64.tar.gz

Run server:

    /usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml

Access UI:

    http://<ec2-ip>:9090

------------------------------------------------------------------------

# 📊 Grafana Setup

Install Grafana:

    wget https://dl.grafana.com/oss/release/grafana_10.1.5_amd64.deb
    sudo apt install ./grafana_10.1.5_amd64.deb -y

Start service:

    sudo systemctl start grafana-server

Access dashboard:

    http://<ec2-ip>:3000

Default credentials:

    username: admin
    password: admin

------------------------------------------------------------------------

# 🎯 Key Features

✔ Reproducible ML pipelines\
✔ Experiment tracking with MLflow\
✔ Dataset versioning with DVC\
✔ Automated CI/CD pipeline\
✔ Containerized ML API\
✔ Kubernetes deployment on AWS\
✔ Production monitoring with Prometheus + Grafana

------------------------------------------------------------------------

# 📌 Future Improvements

-   Add **model drift detection**
-   Integrate **feature store**
-   Implement **A/B model deployment**
-   Add **auto-scaling policies in Kubernetes**

------------------------------------------------------------------------

# 👨‍💻 Author

Ashish Pal

------------------------------------------------------------------------

⭐ If you found this project useful, consider giving it a star!
