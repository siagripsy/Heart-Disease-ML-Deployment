# Heart Disease Classification – ML Deployment Project

This project is developed as part of a Capstone / Machine Learning Deployment assignment.

The goal is to build, evaluate, and deploy a machine learning model that predicts the presence of heart disease based on patient clinical features.
The final solution is exposed as a REST API and deployed on a cloud platform using Docker.

---

## Project Overview

Heart disease is one of the leading causes of death worldwide.
In this project, machine learning techniques are used to classify whether a patient is likely to have heart disease based on structured medical data.

The project follows an end-to-end machine learning workflow:
- Dataset exploration and analysis
- Model training and evaluation
- Model selection
- Pipeline creation
- Model deployment as an API

---

## Machine Learning Models

Two classification models are trained and compared:

- Random Forest Classifier
- Multi-Layer Perceptron (MLP)

Model performance is evaluated using appropriate classification metrics, and the best-performing model is selected for deployment.

---

## Project Structure

.
├─ notebooks/
│   └─ heart_disease_eda_training.ipynb
│
├─ src/
│   ├─ api/
│   ├─ model/
│   └─ utils/
│
├─ artifacts/
│   └─ models/
│
├─ docker/
│
├─ requirements.txt
├─ README.md
└─ .gitignore

---

## Deployment Workflow

1. Train and evaluate machine learning models
2. Select the best-performing model
3. Create a preprocessing + model pipeline
4. Save the trained pipeline as an artifact
5. Build a REST API using FastAPI
6. Dockerize the application
7. Deploy the container to a cloud platform
8. Expose a public API endpoint for inference

---

## API Description (Planned)

- Endpoint: /predict
- Method: POST
- Input: JSON containing patient clinical features
- Output: Prediction indicating presence or absence of heart disease, along with confidence score

---

## Technologies Used

- Python
- Scikit-learn
- Pandas & NumPy
- FastAPI
- Docker
- Cloud Platform (AWS / GCP / Azure)

---

## Notes

- The model is loaded once at application startup and is not retrained during inference.
- The focus of this project is on real-world deployment rather than model accuracy alone.
