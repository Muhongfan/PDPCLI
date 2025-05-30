import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)
def convert_non_numeric_to_numeric(df):
    df_numeric = df.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object' or df_numeric[col].dtype.name == 'category':
            df_numeric[col] = LabelEncoder().fit_transform(df_numeric[col].astype(str))
    return df_numeric
def prepare_training_data(df: pd.DataFrame, 
                          target_col: str, 
                          time_col: str,
                          test_ratio: float = 0.2):
    df = df.copy()
    try:
        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors="raise")
    except Exception:
        df[time_col] = pd.to_datetime(df[time_col], format="%d.%m.%Y-%H:%M", errors="coerce")

    # df[time_col] = pd.to_datetime(df[time_col])
    df.sort_values(by=time_col, inplace=True)
    if not is_numeric_series(df[target_col]):
        print(f"[INFO] Converting target column '{target_col}' to numeric...")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    drop_cols = [target_col, time_col]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols]
    y = df[target_col]

    split_idx = int(len(df) * (1 - test_ratio))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def train_and_evaluate(name, X_train, X_test, y_train, y_test):
    assert not y_train.isnull().any(), f"{name}: y_train contains NaNs"
    assert not y_test.isnull().any(), f"{name}: y_test contains NaNs"

    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    nonzero = preds != 0
    mape = np.mean(np.abs((preds[nonzero] - preds[nonzero]) / preds[nonzero])) * 100

    print(f"\n📊 {name} Evaluation:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE : {mae:.4f}")
    print(f"   R²  : {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    return rmse


def main(raw_file, processed_file, target_col, time_col):
    print("🔍 Loading data...")

    # raw data
    df_raw = pd.read_csv(raw_file)

    X_raw_train, X_raw_test, y_raw_train, y_raw_test = prepare_training_data(
        df_raw, target_col, time_col)
    X_raw_train = convert_non_numeric_to_numeric(X_raw_train)
    X_raw_test = convert_non_numeric_to_numeric(X_raw_test)
    train_and_evaluate("Raw Data", X_raw_train, X_raw_test, y_raw_train, y_raw_test)
    X_raw_test = convert_non_numeric_to_numeric(X_raw_test)


    # processed data
    df_proc = pd.read_csv(processed_file)
    X_proc_train, X_proc_test, y_proc_train, y_proc_test = prepare_training_data(
        df_proc, target_col, time_col)
    
    X_proc_train = convert_non_numeric_to_numeric(X_proc_train)
    X_proc_test = convert_non_numeric_to_numeric(X_proc_test)
    train_and_evaluate("Preprocessed Data", X_proc_train, X_proc_test, y_proc_train, y_proc_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate forecasting models on raw and processed data.")

    parser.add_argument(
        "--raw_file",
        type=str,
        required=True,
        help="Path to the raw CSV file (e.g., without preprocessing)."
    )
    parser.add_argument(
        "--processed_file",
        type=str,
        required=True,
        help="Path to the processed CSV file (with preprocessing applied)."
    )
    parser.add_argument(
        "--target_col",
        type=str,
        required=True,
        help="Name of the target column to forecast."
    )
    parser.add_argument(
        "--time_col",
        type=str,
        default="timestamp",
        help="Name of the timestamp column (default: 'timestamp')."
    )

    args = parser.parse_args()

    main(
        raw_file=args.raw_file,
        processed_file=args.processed_file,
        target_col=args.target_col,
        time_col=args.time_col
    )
