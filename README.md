# DL Rides Karachi Demand Prediction

## Karachi Taxi Demand Forecasting & Monitoring Platform

A deep learning based demand prediction system for ride-sharing and taxi activity in Karachi.
The project focuses on forecasting ride demand across city regions using time-based patterns, location features, H3 geospatial indexing, GRU neural networks, and monitoring tools for data drift and model explainability.

This is not just a notebook-based ML project. It includes a complete workflow with data processing, feature engineering, model training, evaluation, API support, and an interactive Streamlit dashboard.

---

## Project Overview

Ride-sharing platforms depend heavily on demand forecasting. If demand is high in one area but drivers are not available, customers face delays and the platform loses revenue. This project solves that problem by predicting expected ride demand for different Karachi regions and time periods.

The system helps answer questions like:

* Which areas may have high ride demand?
* What time of day has stronger activity?
* How does location affect predicted demand?
* Can the model explain why a prediction was made?
* Has the data changed enough to affect model reliability?

---

## Core Idea

The project uses historical ride data and transforms pickup locations into H3 hexagonal regions. These regions allow the city to be divided into smaller geographic zones, making demand prediction more structured and location-aware.

A GRU-based deep learning model is then trained to learn demand patterns over time.

```text
Ride Data
   ↓
Cleaning & Preprocessing
   ↓
H3 Geospatial Feature Extraction
   ↓
Train/Test Feature Processing
   ↓
GRU Deep Learning Model
   ↓
Demand Forecasting
   ↓
Monitoring + Explainability Dashboard
```

---

## Key Features

### Demand Prediction

* Predicts ride demand for selected date and time
* Supports region-based forecasting
* Uses deep learning for time-series style demand patterns
* Designed around Karachi ride activity

### Geospatial Processing

* Converts latitude and longitude into H3 hexagonal regions
* Groups ride demand by location zones
* Helps analyze demand distribution across the city

### Deep Learning Model

* GRU-based neural network architecture
* Configurable model parameters through `params.yaml`
* Uses dropout, dense layers, and training callbacks
* Saves trained model and metadata for reuse

### Streamlit Dashboard

* Interactive dashboard for demand forecasting
* Date and time selection
* Location-based demand view
* Visual statistics and user-friendly output

### FastAPI Backend

* API support for prediction requests
* Structured request and response models
* Backend endpoints for prediction, drift monitoring, and explainability

### Data Drift Monitoring

* Uses Evidently AI to monitor data changes
* Helps detect when new data behaves differently from training data
* Useful for checking whether the model may need retraining

### Explainable AI

* SHAP-based model explanations
* LIME-based local explanations
* Helps understand which features influenced a prediction
* Makes the model easier to interpret and trust

### DVC Pipeline

* Reproducible ML workflow using DVC
* Pipeline stages for ingestion, feature extraction, training, evaluation, and model registration
* Tracks experiment flow in a clean project structure

---

## Tech Stack

### Machine Learning & Deep Learning

* Python
* TensorFlow / Keras
* GRU Neural Network
* Scikit-learn
* Joblib

### Data Processing

* Pandas
* NumPy
* Dask
* H3 Geospatial Indexing

### App & API

* Streamlit
* FastAPI
* Pydantic

### Monitoring & Explainability

* Evidently AI
* SHAP
* LIME

### MLOps / Project Workflow

* DVC
* YAML configuration
* Modular source-code structure

---

## Project Structure

```text
DL-Rides-Karachi-Demand-Prediction/
│
├── app.py                         # Streamlit dashboard
├── backend.py                     # FastAPI backend
├── params.yaml                    # Model and feature configuration
├── dvc.yaml                       # DVC pipeline stages
├── dvc.lock                       # DVC pipeline lock file
├── requirements.txt               # Project dependencies
│
├── data/
│   ├── external/                  # External/raw data placeholder
│   └── processed/                 # Processed dataset files
│
├── models/                        # Saved ML/DL models
│
├── notebooks/
│   ├── EDA-Demand-Prediction.ipynb
│   ├── Model-Selection.ipynb
│   └── Plot-Map.ipynb
│
├── src/
│   ├── data/
│   │   └── data_ingestion.py
│   │
│   ├── features/
│   │   ├── extract_features.py
│   │   └── feature_processing.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── register_model.py
│   │
│   └── visualization/
│       └── visualize.py
│
├── reports/
├── docs/
└── README.md
```

---

## DVC Pipeline Stages

The machine learning workflow is organized into multiple DVC stages:

| Stage              | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| Data Ingestion     | Loads and prepares the initial dataset               |
| Feature Extraction | Converts location and time data into useful features |
| Feature Processing | Creates train/test datasets                          |
| Training           | Trains the GRU deep learning model                   |
| Evaluation         | Tests model performance                              |
| Model Registration | Stores model metadata and deployment information     |

---

## Model Configuration

The model settings are controlled from `params.yaml`, making it easier to change training behavior without editing the main training code.

Example configuration areas include:

* H3 resolution
* EWMA smoothing alpha
* GRU layers
* GRU units
* Dense layers
* Dropout rate
* Learning rate
* Batch size
* Epochs
* Loss function

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Ahab-ali/DL-Rides-Karachi-Demand-Prediction.git
cd DL-Rides-Karachi-Demand-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

### 4. Run the FastAPI Backend

```bash
uvicorn backend:app --reload
```

---

## Main Use Cases

* Forecasting ride demand in Karachi
* Understanding location-based ride patterns
* Building a deep learning pipeline for demand prediction
* Demonstrating geospatial feature engineering using H3
* Showing model monitoring using Evidently AI
* Explaining model predictions using SHAP and LIME
* Practicing MLOps workflow with DVC

---

## What I Learned

This project helped me practice and understand:

* End-to-end machine learning project structure
* Deep learning model training with GRU
* Time and location-based feature engineering
* Geospatial indexing using H3
* Building dashboards with Streamlit
* Creating API endpoints with FastAPI
* Tracking ML workflows using DVC
* Adding explainability and monitoring to ML systems

---

## Future Improvements

* Add live ride data integration
* Deploy the dashboard online
* Add user authentication
* Improve map-based visualization
* Add more advanced forecasting models
* Compare GRU with LSTM and Transformer-based models
* Improve model retraining automation
* Add Docker support for easier deployment

---

## Contributors

* **Baziq Khan**
* * **Noman Ahmed**
* * **Insha Khan**
* **Ahab Ali**
