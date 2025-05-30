import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("output_energy.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)

target_col = "price actual"
window_size = 5

for i in range(1, window_size + 1):
    df[f"{target_col}_t-{i}"] = df[target_col].shift(i)


df = df.dropna().reset_index(drop=True)

y = df[target_col]

feature_cols = [col for col in df.columns if col not in [target_col, "timestamp"]]
X = df[feature_cols]

split_ratio = 0.8
split_idx = int(len(df) * split_ratio)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Check the correlation bwtween the variables
for col in X.columns:
    if np.issubdtype(X[col].dtype, np.datetime64):
        continue  # 跳过 datetime 列
    corr = np.corrcoef(X[col], y)[0, 1]
    if abs(corr) > 0.95:
        print(f"⚠️ WARNING: High correlation between {col} and target ({corr:.3f}) — possible leakage.")

print(f"✅ Data ready: X_train = {X_train.shape}, X_test = {X_test.shape}")
