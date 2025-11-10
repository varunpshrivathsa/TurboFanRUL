from fastapi import FastAPI
import joblib, numpy as np, os

app = FastAPI(title="TurboFan RUL Prediction API")

MODEL_PATH = os.path.join("models", "xgb_tuned.pkl")

@app.get("/")
def root():
    return {"message": "TurboFAN RUL API is running"}

@app.post("/predict")
def predict(data: dict):
    """
    Expects JSON input:
    {
      "features": [[x1, x2, x3, ...], [x1, x2, x3, ...]]
    }
    """
    features = np.array(data.get("features", []))
    if features.ndim == 1:
        features = features.reshape(1, -1)

    if not os.path.exists(MODEL_PATH):
        return {"error": "Model file not found"}

    model = joblib.load(MODEL_PATH)
    preds = model.predict(features)
    return {"predictions": preds.tolist()}
