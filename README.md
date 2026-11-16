# AutoValue AI: End-to-End AIML Valuation & Market Intelligence System

**AutoValue AI** is a complete, production-grade Machine Learning system that models used-vehicle prices, evaluates listing deals (Overpriced, Fair, Undervalued), and simulates empirical market depreciation curves.

The project features a **13-stage Jupyter Research Notebook**, a high-performance **FastAPI Backend**, an interactive **Glassmorphic Web Dashboard**, automated **Unit & Integration Tests**, and **Docker containerization**.

---

## Architecture Overview

```
                                      ┌──────────────────────────────────────────┐
                                      │     Interactive Web Frontend (SPA)       │
                                      │  (Tailwind CSS, Glassmorphism, Chart.js) │
                                      └────────────────────┬─────────────────────┘
                                                           │ HTTP / JSON
                                                           ▼
                                      ┌──────────────────────────────────────────┐
                                      │        FastAPI Serving Backend           │
                                      │   (/api/predict, /api/simulate, /api/..) │
                                      └────────────────────┬─────────────────────┘
                                                           │
                                                           ▼
                                      ┌──────────────────────────────────────────┐
                                      │       Trained Inference Pipeline         │
                                      │  (ColumnTransformer + Ridge Regressor)   │
                                      └────────────────────┬─────────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────────────┐
│  Research & Exploration (13 Stg)│──▶│  Serialized Production Assets (Joblib)   │
│   (Vatsal_DS_Project.ipynb)     │   │  (preprocessor, ridge_model, mfr_counts) │
└─────────────────────────────────┘   └──────────────────────────────────────────┘
```

---

## Project Structure

```
d:\CLG\Vatsal\AIML\Project\
├── Vatsal_DS_Project.ipynb          # Cleaned & structured 13-stage ML research notebook
├── Vatsal_DS_Project_backup.ipynb   # Backup of original research notebook
├── app.py                           # Application launcher (FastAPI + Web UI on port 8000)
├── train_model.py                   # Automated ML pipeline training & artifact exporter
├── predict.py                       # Standalone CLI inference module
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Multi-stage production container build
├── README.md                        # Documentation & setup guide
│
├── models/                          # Serialized ML artifacts
│   ├── preprocessor.joblib          # Scikit-learn ColumnTransformer
│   ├── ridge_model.joblib           # Trained Ridge Regression Model (L2)
│   └── mfr_counts.joblib            # Brand frequency map
│
├── backend/                         # FastAPI Application
│   ├── __init__.py
│   ├── main.py                      # App routing, CORS, static mounting
│   ├── inference.py                 # In-memory ModelService & explainability
│   ├── schemas.py                   # Pydantic v2 validation models
│   └── config.py                    # Environment & path configuration
│
├── frontend/                        # Web Dashboard (Single-Page App)
│   ├── index.html                   # Responsive glassmorphism dashboard
│   ├── css/
│   │   └── style.css                # Custom styling & neon glow themes
│   └── js/
│       └── app.js                   # Client-side API interactions & Chart.js logic
│
└── tests/                           # Automated Test Suite
    ├── __init__.py
    ├── test_model.py                # Model pipeline & numerical tests
    └── test_api.py                  # FastAPI route integration tests
```

---

## Features

### 1. Interactive Web Dashboard
* **Dynamic Valuation Engine:** Real-time fair market value prediction with high/low statistical confidence intervals ($\pm 15\%$).
* **Intelligent Deal Evaluator:** Classifies any user-provided asking price into **Undervalued (Great Deal!)**, **Fair Price**, or **Overpriced**.
* **Key Valuation Drivers:** Feature attribution breakdown detailing the positive and negative contributors (e.g. brand equity, mileage depreciation, vehicle age).
* **Interactive What-If Simulation:**
  * **Mileage Depreciation Curve:** Chart.js spline charting price decay across odometer readings ($10\text{k} - 200\text{k}$ miles).
  * **Condition Impact Bar Chart:** Visualizing market value across condition tiers (*New*, *Like New*, *Excellent*, *Good*, *Fair*, *Salvage*).
* **Preloaded Vehicle Presets:** One-click presets for popular vehicles ('21 Toyota RAV4, '19 Ford F-150, '18 BMW M3).

### 2. FastAPI Backend
* `GET /`: Serves the Single-Page Web Dashboard.
* `GET /api/health`: Health status & service telemetry.
* `GET /api/metadata`: Dynamic category options (manufacturers, conditions, fuels, states).
* `GET /api/model-info`: Architectural metadata and model performance metrics ($R^2$, MAE, RMSE).
* `POST /api/predict`: High-throughput valuation and deal evaluation endpoint.
* `POST /api/simulate`: Generates mileage and condition sensitivity curves.
* Auto-generated Interactive Documentation at `http://localhost:8000/docs` (Swagger UI).

### 3. Model Pipeline Details
* **Target Transformation:** Models $\log(1 + \text{price})$ to stabilize right-skewed price variance and prevent negative predictions.
* **Feature Schema:**
  * **Numerical (StandardScaler):** `year`, `odometer`, `lat`, `long`, `vehicle_age`, `mileage_per_year`, `manufacturer_popularity`.
  * **Categorical (OneHotEncoder):** `manufacturer`, `condition`, `cylinders`, `fuel`, `title_status`, `transmission`, `drive`, `type`, `state`, `age_group`.
* **Performance:** $R^2 = 0.9729$, $\text{MAE} = \$972.41$, $\text{RMSE} = \$1,628.48$.

---

## Quickstart Guide

### 1. Installation
Clone or navigate to the project directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch Full-Stack Application
Start the unified FastAPI server and web dashboard:
```bash
python app.py
```
Open your browser and navigate to:
* **Web UI Dashboard:** [http://localhost:8000](http://localhost:8000)
* **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Run Automated Tests
Execute the automated unit and integration test suite:
```bash
python tests/test_model.py
python tests/test_api.py
```

### 4. CLI Inference
Test predictions directly from terminal:
```bash
python predict.py
```

### 5. Retrain Model
To retrain the pipeline and update `.joblib` assets:
```bash
python train_model.py
```

### 6. Docker Deployment (Optional)
Build and run the containerized application:
```bash
docker build -t autovalue-ai .
docker run -p 8000:8000 autovalue-ai
```
