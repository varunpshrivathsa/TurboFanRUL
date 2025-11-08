import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import joblib #for saving the pkl file of scaler 

DATA_PATH = "../data/processed/train_FD001_RUL.csv"
OUTPUT_DIR = "../data/processed/"
TEST_SIZE = 0.2 
RANDOM_STATE = 42
SCALER_PATH = "../models/scaler.pkl"

#Step1 -> Load data
def load_processed_data(path:str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found at path {path}")
    df = pd.read_csv(path)
    print(f"Loaded Data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df 


#Step2 -> Split Features and Target
def get_feature_target(df:pd.DataFrame):
    if "RUL" not in df.columns:
        raise ValueError("Column 'RUL' not found in dataframe")
    
    X = df.drop(columns=["unit_number","RUL","time_in_cycles"],errors="ignore")
    y = df["RUL"]

    print(f"Feature matrix shape {X.shape}, Target Shape: {y.shape}")
    return X,y

#step3 -> Standard Scaling
def apply_standard_scaler(X_train: pd.DataFrame, X_test:pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Applied Standard Scaler")
    return X_train_scaled,X_test_scaled, scaler

#step4 -> train and test split
def split_data(X,y, test_size = TEST_SIZE, random_state = RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

    print(f"Train/Test Split : {X_train.shape[0]} / {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test 

#step5 -> saving everything
def save_artifacts(X_train, X_test, y_train, y_test,scaler):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok= True)

    np.save(os.path.join(OUTPUT_DIR,"X_train.npy"),X_train)
    np.save(os.path.join(OUTPUT_DIR,"X_test.npy"),X_test)
    np.save(os.path.join(OUTPUT_DIR,"y_train.npy"),y_train)
    np.save(os.path.join(OUTPUT_DIR,"y_test.npy"),y_test)

    joblib.dump(scaler, SCALER_PATH)

    print(f"Saved processed arrays to {OUTPUT_DIR}")
    print(f"Saved Scaler to {SCALER_PATH}")


#step6 -> Main Pipeline
def process_data():
    df = load_processed_data(DATA_PATH)
    X,y = get_feature_target(df)
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_scaled,X_test_scaled, scaler = apply_standard_scaler(X_train,X_test)
    save_artifacts(X_train_scaled, X_test_scaled, y_train, y_test, scaler)

    print("Processing Complete")


if __name__ == "__main__":
    process_data()








