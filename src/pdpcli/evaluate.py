import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

def print_normalized_importance(model, X_train):
    importance = model.feature_importance()
    feat_names = X_train.columns
    df_imp = pd.DataFrame(dict(zip(feat_names, importance)).items(), columns=["feature", "importance"])
    df_imp["importance_norm"] = df_imp["importance"] / df_imp["importance"].max()
    df_imp = df_imp.sort_values("importance_norm", ascending=False)
    print("🧠 Normalized Top Features:")
    for _, row in df_imp.iterrows():
        if row['importance_norm']>0:
            print(f"{row['feature']:>30s} : {row['importance_norm']:.3f}")

def save_plot(fig, filename, save_dir="plots", show=False):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def plot_learning_curve(model, metric='rmse', title="Learning Curve", name="Model", save_dir="plots", show=False):
    """
    Plot training and validation metrics from LightGBM model and optionally save the plot.

    Parameters:
    - model: trained LGBMRegressor with evals_result_
    - metric: str, evaluation metric used during training (e.g., 'rmse')
    - title: str, title of the plot
    - name: str, name prefix
    - save_dir: str, directory to save the plot
    - show: bool, whether to display the plot
    """
    if hasattr(model, 'evals_result_'):
        results = model.evals_result_
        if 'train' in results and metric in results['train']:
            fig = plt.figure(figsize=(8, 4))
            plt.plot(results['train'][metric], label='Train')
            if 'valid' in results and metric in results['valid']:
                plt.plot(results['valid'][metric], label='Validation')
            plt.title(f"{name} - {title} ({metric.upper()})")
            plt.xlabel("Boosting Rounds")
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            save_plot(fig, f"{name}_learning_curve.png", save_dir, show)
        else:
            print(f"[WARN] Metric '{metric}' not found in model's evals_result_.")
    else:
        print("[WARN] Model does not have evals_result_.")

def plot_actual_vs_pred(y_true, y_pred, title="Actual vs Predicted", name="Model", save_dir="plots", show=False):
    """
    Plot actual vs predicted values and optionally save the plot.

    Parameters:
    - y_true: pd.Series or array-like of actual values
    - y_pred: array-like of predicted values
    - title: str, plot title
    - name: str, used for filename
    - save_dir: str, folder to save plots
    - show: bool, whether to display the plot
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(pd.Series(y_true).reset_index(drop=True), label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linestyle='--')
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot(fig, f"{name}_actual_vs_pred.png", save_dir, show)

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

def lr_scheduler(iteration):
    if iteration < 10:
        return 0.2
    elif iteration < 50:
        return 0.1
    else:
        return 0.05

def train_and_evaluate(name, X_train, X_test, y_train, y_test, plot=True, save_dir="plots"):
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "random_state": 42,
        "min_data_in_leaf": 10,
        "min_gain_to_split": 0.0,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_set=lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(10),
            lgb.log_evaluation(1),
            lgb.reset_parameter(learning_rate=lr_scheduler) 
        ]
    )

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    nonzero_true = y_test != 0
    mape = np.mean(np.abs((y_test[nonzero_true] - preds[nonzero_true]) / y_test[nonzero_true])) * 100

    print(f"\n📊 {name} Evaluation:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE : {mae:.4f}")
    print(f"   R²  : {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    plot_actual_vs_pred(y_test, preds, title=f"{name} - Actual vs Predicted")
    plot_learning_curve(model, metric='rmse', name=name)
    print_normalized_importance(model, X_train)

    return rmse


def evaluate_processing(data_path, processed_file, target_col, time_col):
    print("🔍 Loading data...")

    # raw data
    df_raw = pd.read_csv(data_path)

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

    evaluate_processing(
        raw_file=args.raw_file,
        processed_file=args.processed_file,
        target_col=args.target_col,
        time_col=args.time_col
    )
