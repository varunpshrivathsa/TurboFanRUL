import os 
import numpy as np 
import pandas as pd 
import joblib 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
import shutil


DATA_DIR = "../data/processed"
MODEL_DIR = "../models"
RESULTS_DIR = "../results/baseline_results.csv"

os.makedirs(MODEL_DIR,exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_DIR),exist_ok=True)

#step1 -> Load Data
def load_data():
    X_train = np.load(os.path.join(DATA_DIR,"X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR,"X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR,"y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR,"y_test.npy"))

    col_means = np.nanmean(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_means, inds[1])

    col_means_test = np.nanmean(X_test, axis=0)
    inds = np.where(np.isnan(X_test))
    X_test[inds] = np.take(col_means_test, inds[1])

    print(f"Data Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"NaNs handled: {np.isnan(X_train).sum()} remaining in X_train, {np.isnan(X_test).sum()} in X_test")

    return X_train, X_test, y_train, y_test

#step2 -> Evaluate the model with metrics
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    mae = mean_absolute_error(y_test,preds)
    r2 = r2_score(y_test,preds)
    return rmse, mae, r2 

#step3 -> Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = []
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators = 200,
            learning_rate = 0.1, 
            random_state=42,
            n_jobs = -1,
            verbosity = 0)}
    
    for name, model in models.items():
        print(f"Training {name}")
        model.fit(X_train,y_train)
        rmse, mae, r2 = evaluate_model(model,X_test,y_test)
        results.append({
            "Model":name,
            "RMSE": round(rmse,3),
            "MAE": round(mae, 3),
            "R2": round(r2,3)
        })

        model_path = os.path.join(MODEL_DIR, f"{name.replace(' ','_').lower()}.pkl")
        joblib.dump(model,model_path)
        print(f"{name} done. RMSE = {rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
        print(f"Saved model to {model_path}")

    return pd.DataFrame(results)

#step4 -> Save results
def save_results(df_results):
    df_results.to_csv(RESULTS_DIR,index = False)
    print(df_results)

#step5 -> Select Best Model 
def select_best_model(df_results):
    best_row = df_results.loc[df_results["RMSE"].idxmin()]
    best_model_name = best_row["Model"].replace(' ','_').lower()
    src_path = os.path.join(MODEL_DIR, f"{best_model_name}.pkl")
    dst_path = os.path.join(MODEL_DIR, "best_model.pkl")
    shutil.copy(src_path, dst_path)
    print(f"Best model: {best_row['Model']} (RMSE={best_row['RMSE']})")
    print(f"Saved best model as {dst_path}")

#step6 -> Main pipeline
def main():
    X_train, X_test, y_train, y_test = load_data()
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    save_results(results_df)
    select_best_model(results_df)
    print("\nBaseline training complete")

if __name__ == "__main__":
    main()
    




