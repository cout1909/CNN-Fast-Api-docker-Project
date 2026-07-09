# CIFAR-10 Image Classifier API

A CNN-based image classification service built with TensorFlow, served through a FastAPI backend, containerized with Docker, and deployed using an automated CI/CD pipeline.

## 🚀 Overview

This project trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and serves it as a real-time prediction API. The entire pipeline — from model training to deployment — is automated and containerized, making it easy to run consistently on any machine.

## 🧠 What It Does

- Classifies images into one of 10 categories: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
- Exposes a `/predict` endpoint that accepts an image and returns the predicted class with a confidence score
- Runs identically on any machine via Docker
- Automatically tests and builds on every code push via GitHub Actions

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | TensorFlow / Keras (CNN) |
| Dataset | CIFAR-10 |
| API Framework | FastAPI |
| Server | Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |

## 📁 Project Structure

```
.
├── main.py              # FastAPI app with prediction endpoint
├── train_model.py        # CNN training script
├── cnn_model.keras       # Trained model file
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build instructions
├── .dockerignore
└── .github/
    └── workflows/
        └── ci-cd.yml     # GitHub Actions pipeline
```

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Docker Desktop

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

### Option 1: Run locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Visit `http://localhost:8000/docs` for the interactive API documentation.

### Option 2: Run with Docker
```bash
docker build -t my-cnn-app .
docker run -p 8000:8000 my-cnn-app
```
Visit `http://localhost:8000/docs` — same result, running inside an isolated container.

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check / welcome message |
| GET | `/health` | Server status check |
| POST | `/predict` | Upload an image, get back a predicted class + confidence |

## 🔄 CI/CD Pipeline

Every push to the repository automatically triggers a GitHub Actions workflow that:
1. Runs tests against the codebase
2. Builds the Docker image
3. Pushes the image to Docker Hub (if tests pass)

This removes the need to manually build and deploy after every change.

## 📚 What This Project Demonstrates

- Training and serving a deep learning model (TensorFlow/Keras)
- Building a production-style REST API (FastAPI)
- Containerization and environment consistency (Docker)
- Automated testing and deployment pipelines (CI/CD)

## 🔮 Future Improvements

- Add input validation and error handling for `/predict`
- Add automated unit tests with pytest
- Wire this API up as a callable tool for an LLM agent (agentic AI integration)

## 👤 Author

Pratik Bitke — Final Year Computer Engineering Student, SKNCOE, Savitribai Phule Pune University
