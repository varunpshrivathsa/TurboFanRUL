import numpy as np
import pandas as pd

X = np.load("data/processed/X_test.npy")
df = pd.DataFrame(X[:10])  # take first 10 rows
df.to_csv("sample_valid_input.csv", index=False)
print("Saved sample_valid_input.csv with shape:", df.shape)
