# src/trainOptuna.py
import os
import json
import numpy as np
import optuna
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
DATA_DIR = "../data/processed"
MODEL_DIR = "../models"
BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "xgb_best_params.json")
TUNED_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_tuned.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
mlflow.set_experiment("Turbofan_XGB_Optuna")

# ---------- LOAD DATA ----------
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test

# ---------- OBJECTIVE FUNCTION ----------
def objective(trial):
    X_train, X_test, y_train, y_test = load_data()

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

    return rmse  # minimize RMSE

# ---------- MAIN ----------
def main():
    study = optuna.create_study(direction="minimize", study_name="XGB_Tuning")
    study.optimize(objective, n_trials=20, n_jobs=1)

    print(f"\nBest Trial RMSE: {study.best_value:.3f}")
    print(f"Best Params: {study.best_params}")

    # Save best params
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=4)

    # Retrain final tuned model
    X_train, X_test, y_train, y_test = load_data()
    best_model = XGBRegressor(**study.best_params)
    best_model.fit(X_train, y_train)
    np.save(os.path.join(MODEL_DIR, "X_test_predictions.npy"), best_model.predict(X_test))

    import joblib
    joblib.dump(best_model, TUNED_MODEL_PATH)
    print(f"Tuned model saved to {TUNED_MODEL_PATH}")

    with mlflow.start_run(run_name="Best_XGB_Model"):
        mlflow.log_params(study.best_params)
        mlflow.sklearn.log_model(best_model, "best_xgb_model")
        mlflow.log_metric("Best_RMSE", study.best_value)

if __name__ == "__main__":
    main()