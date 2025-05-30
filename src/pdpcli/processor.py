# pdpcli/processor.py

import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller

step_map = {
    "R1": "Fill missing values",
    "R2": "Standardize numeric features",
    "R3": "Generate sliding window (for supervised learning)",
    "R4": "Extract time features (hour, weekday, etc.)",
    "R5": "Remove outliers",
    "R6": "Remove trend/seasonality",
    "R7": "Create lag features",
    "R8": "Differencing (make stationary)",
    "R9": "Smoothing (e.g., moving average)",
    "R10": "FFT transform (frequency domain analysis)"
}

def parse_datetime_column(df, time_col):
    if time_col in df.columns:
        # 强制使用 utc=True，确保兼容 mixed timezone
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)

        # 只有当成功转换为 datetimelike 时才移除 tz
        if pd.api.types.is_datetime64tz_dtype(df[time_col]):
            df[time_col] = df[time_col].dt.tz_convert(None)

    return df
def apply_r5_outlier_removal(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    exclude_cols = exclude_cols or []
    numeric_cols = [col for col in df.select_dtypes(include=["number"]).columns if col not in exclude_cols]
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

def get_columns_excluding_target(df, target):
    return [col for col in df.columns if col != target]

def recommend_preprocessing(data_path, time_col, target_col=None):
    print(f"\n📊 Analyzing dataset: {data_path}")
    df = pd.read_csv(data_path)
    # df = recommend_preprocessing(df, target_col)

    if time_col in df.columns:
        try:
            df = parse_datetime_column(df, time_col)

            # df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        except Exception as e:
            print(f"⚠️ Could not parse {time_col} as datetime: {e}")

    numeric_cols = df.select_dtypes(include=[np.number])

    print("\n🔎 Preprocessing Recommendations:\n")
    # R1 - Missing values
    if df.isnull().values.any():
        print("✅ R1:", step_map["R1"])
    else:
        print("❌ R1:", step_map["R1"], "- No missing values")

    # R2 - Standardization
    std_devs = numeric_cols.std()
    if (std_devs > 10).any():
        print("✅ R2:", step_map["R2"])
    else:
        print("❌ R2:", step_map["R2"], "- Numeric columns are already scaled")

    # R3 - Sliding window
    if target_col:
        print("✅ R3:", step_map["R3"])
    else:
        print("❌ R3:", step_map["R3"], "- No target column provided")

    # R4 - Time features
    if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        print("✅ R4:", step_map["R4"])
    else:
        print("❌ R4:", step_map["R4"], "- Timestamp column missing or invalid")

    # R5 - Outlier detection (basic IQR test)
    has_outliers = False
    for col in numeric_cols.columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        if ((df[col] < lower) | (df[col] > upper)).any():
            has_outliers = True
            break
    if has_outliers:
        print("✅ R5:", step_map["R5"])
    else:
        print("❌ R5:", step_map["R5"], "- No significant outliers detected")

    # R6 - Trend/seasonality removal (simple trend detection via ADF test)
    if target_col and target_col in df.columns:
        try:
            result = adfuller(df[target_col].dropna())
            if result[1] > 0.05:
                print("✅ R6:", step_map["R6"], "- Trend detected (non-stationary)")
            else:
                print("❌ R6:", step_map["R6"], "- Stationary series (ADF p <= 0.05)")
        except Exception as e:
            print("❌ R6:", step_map["R6"], f"- Could not test ADF: {e}")
    else:
        print("❌ R6:", step_map["R6"], "- No target column provided")

    # R7 - Lag features
    if target_col:
        print("✅ R7:", step_map["R7"])
    else:
        print("❌ R7:", step_map["R7"], "- No target column provided")

    # R8 - Differencing
    if target_col and target_col in df.columns:
        try:
            result = adfuller(df[target_col].dropna())
            if result[1] > 0.05:
                print("✅ R8:", step_map["R8"], "- Non-stationary detected")
            else:
                print("❌ R8:", step_map["R8"], "- Already stationary")
        except Exception as e:
            print("❌ R8:", step_map["R8"], f"- ADF test error: {e}")
    else:
        print("❌ R8:", step_map["R8"], "- No target column")

    # R9 - Smoothing (high noise level)
    if target_col and target_col in numeric_cols.columns:
        series = df[target_col].dropna()
        noise = (series - series.rolling(3, min_periods=1).mean()).std()
        if noise > 1.0:  # heuristic threshold
            print("✅ R9:", step_map["R9"], "- Noise detected in target")
        else:
            print("❌ R9:", step_map["R9"], "- Low noise")
    else:
        print("❌ R9:", step_map["R9"], "- No target column")

    # R10 - FFT transform (always optional)
    print("⚠️  R10:", step_map["R10"], "- Consider if frequency analysis is relevant")

    print("\n📌 End of recommendations.")

STEP_ORDER = ["R1", "R5", "R2", "R6", "R8", "R9", "R7", "R3", "R4", "R10"]

def apply_preprocessing(data_path, steps, output_path, time_col, target_col=None, auto_order=False):
    df = pd.read_csv(data_path)
    df = parse_datetime_column(df, time_col)
    df.sort_values(by=time_col, inplace=True)

    if time_col in df.columns:
        try:
            df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors="raise")
        except Exception:
            df[time_col] = pd.to_datetime(df[time_col], format="%d.%m.%Y-%H:%M", errors="coerce")


        # df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    if auto_order:
        print("⚙️  Sorting steps in recommended execution order.")
        steps = sorted(set(steps), key=lambda s: STEP_ORDER.index(s))

    for step in steps:
        print(f"\n🔧 Applying {step}: {step_map.get(step, 'Unknown step')}")

        if step == "R1":
            # df = df.fillna(method="ffill").fillna(method="bfill")
            df = df.dropna(axis=1, how="all")
            df = df.ffill().bfill()

            print(f"📐 After {step}, shape = {df.shape}")


        elif step == "R2":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore')
            existing_numeric_cols = [col for col in numeric_cols if col in df.columns]

            if len(existing_numeric_cols) > 0 and not df[existing_numeric_cols].empty:
                scaler = StandardScaler()
                df[existing_numeric_cols] = scaler.fit_transform(df[existing_numeric_cols])
            else:
                print("⚠️ Skipping standardization: no valid numeric columns with data.")
            print(f"📐 After {step}, shape = {df.shape}")

            # scaler = StandardScaler()
            # numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore')
            # df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        elif step == "R3" and target_col:
            window_size = 5
            for i in range(1, window_size + 1):
                df[f"{target_col}_t-{i}"] = df[target_col].shift(i)
            print(f"📐 After {step}, shape = {df.shape}")

        elif step == "R4" and time_col in df.columns:
            df["hour"] = df[time_col].dt.hour
            df["weekday"] = df[time_col].dt.weekday
            df["month"] = df[time_col].dt.month
            print(f"📐 After {step}, shape = {df.shape}")

        elif step == "R5":
            df = apply_r5_outlier_removal(df, exclude_cols=[target_col])
            print(f"📐 After {step}, shape = {df.shape}")

        elif step == "R6" and target_col:
            trend = df[target_col].rolling(window=12, min_periods=1).mean()
            df[target_col] = df[target_col] - trend
            print(f"📐 After {step}, shape = {df.shape}")

        elif step == "R7" and target_col:
            df[f"{target_col}_lag1"] = df[target_col].shift(1)
            print(f"📐 After {step}, shape = {df.shape}")

        elif step == "R8" and target_col:
            df[f"{target_col}_diff"] = df[target_col].diff()
            print(f"📐 After {step}, shape = {df.shape}")

        elif step == "R9" and target_col:
            df[f"{target_col}_smooth"] = df[target_col].rolling(window=3, min_periods=1).mean()
            print(f"📐 After {step}, shape = {df.shape}")

        elif step == "R10" and target_col:
            y = df[target_col].dropna().values
            n = len(y)
            if n == 0:
                raise ValueError("No data points in target column for FFT.")
            yf = fft(y)
            xf = fftfreq(n, 1)[:n // 2]
            amplitudes = 2.0 / n * np.abs(yf[:n // 2])

            df.loc[0, "fft_max_amplitude"] = np.max(amplitudes)
            df.loc[0, "fft_main_frequency"] = xf[np.argmax(amplitudes)]
            df.loc[0, "fft_mean_amplitude"] = np.mean(amplitudes)
            df.loc[0, "fft_std_amplitude"] = np.std(amplitudes)
            print("📈 FFT features added to output.")
    if df.empty:
        raise ValueError("❌ Preprocessing resulted in an empty dataframe. Check your steps.")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Output saved to {output_path}")


def validate_processing(data_path, time_col):

    df = pd.read_csv(data_path, parse_dates=["time"])
    df = df.sort_values(time_col).reset_index(drop=True)

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

    # Check the correlation bwtween the variables
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.datetime64):
            continue  
        corr = np.corrcoef(X[col], y)[0, 1]
        if abs(corr) > 0.95:
            print(f"⚠️ WARNING: High correlation between {col} and target ({corr:.3f}) — possible leakage.")

    print(f"✅ Data ready: X_train = {X_train.shape}, X_test = {X_test.shape}")



