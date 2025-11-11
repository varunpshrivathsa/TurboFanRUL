# TurboFanRUL 🛠️  
End-to-end machine learning backend for predicting Remaining Useful Life (RUL) of turbofan engines using NASA CMAPSS dataset.  
Includes data preprocessing, model training with Optuna tuning, MLflow experiment tracking, and a FastAPI inference service ready for Docker deployment.

---

## Features
- Data cleaning and preprocessing for CMAPSS
- Baseline and tuned XGBoost models
- Optuna hyperparameter optimization
- MLflow experiment tracking
- FastAPI service for model inference (`/predict`)
- Dockerized deployment (API + MLflow tracking)
- Compatible with AWS EC2 or local setup

---

## Project Structure
TurboFanRUL/
│
├── data/ # Raw & processed data
├── models/ # Saved model files (.pkl, .json)
├── notebooks/ # EDA and RUL analysis
├── results/ # Metrics, plots, Optuna logs
├── src/ # Core training + API logic
│ ├── preprocess.py
│ ├── train_baseline.py
│ ├── trainOptuna.py
│ ├── api_service.py
│ └── utils.py
├── docker-compose.yml
├── Dockerfile.api
├── requirements.txt
└── README.md


---

## Setup & Installation

### 1 Clone the repository
```bash
git clone https://github.com/varunpshrivathsa/TurboFanRUL.git
cd TurboFanRUL

### 2 Create environment

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 3 Run API locally

uvicorn src.api_service:app --host 0.0.0.0 --port 8000
Then open : http://localhost:8000/docs

### 4 Training and Tuning
python src/train_baseline.py
python src/trainOptuna.py
- Best parameters are logged in models/xgb_best_params.json
- MLflow logs experiments automatically under mlruns/

### 5 Docker Deployment
Build and run:
docker-compose up --build -d
API runs at http://<EC2-IP>:8000
MLflow UI (if enabled) runs at http://<EC2-IP>:5000

### 6 API Usage
Send a JSON input to /predict:
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d @sample_valid_input.json

🧩 Related Project

🔗 Streamlit Dashboard Frontend:
Turbofan_RUL-Dashboard

MIT License © 2025 Varun Phanindra Shrivathsa
