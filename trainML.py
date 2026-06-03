from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_CSV_PATH = "solar_data.csv"
DEFAULT_MODEL_PATH = "solar_model.joblib"

REQUIRED_COLUMNS = (
    "hour_ts",
    "cloud_cover",
    "sunshine_duration",
    "temperature_2m",
    "is_day",
    "generation_kwh",
)
FEATURE_COLUMNS = (
    "cloud_cover",
    "sunshine_duration",
    "temperature_2m",
    "is_day",
    "hour",
)
TARGET_COLUMN = "generation_kwh"


def loadAndPreprocessData(csv_path: str | Path = DEFAULT_CSV_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[list(REQUIRED_COLUMNS)].copy()

    df["hour_ts"] = pd.to_datetime(df["hour_ts"], errors="coerce") #error coerece makes an NaT
    invalid_ts = df["hour_ts"].isna() #is not available
    if invalid_ts.any():
        df = df.loc[~invalid_ts].copy()

    df["hour"] = df["hour_ts"].dt.hour.astype(float)

    numeric_cols = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(lower=0.0) #clipping is used to prevent negative values

    before_drop = len(df)
    df = df.dropna(subset=list(FEATURE_COLUMNS) + [TARGET_COLUMN])
    dropped_na = before_drop - len(df)
    if dropped_na:
        print(f"Dropped {dropped_na} row(s) with missing feature or target values.")

    if df.empty:
        raise ValueError("No valid rows remain after preprocessing.")

    X = df[list(FEATURE_COLUMNS)].reset_index(drop=True)
    y = df[TARGET_COLUMN].reset_index(drop=True)
    return X, y


def trainRandomForest(X: pd.DataFrame,y: pd.Series,*,test_size: float = 0.2,random_state: int = 42) -> Tuple[
    RandomForestRegressor,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(n_estimators=100,random_state=random_state,n_jobs=-1) #100 arbori de decizie, seed, jobs -1 = toti procesorii
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluateAndSaveModel(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series, model_path: str | Path = DEFAULT_MODEL_PATH) -> Tuple[
    float, float]:
    predictions = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    print("--- Evaluation ---")
    print(f"Test samples : {len(y_test)}")
    print(f"MAE (kWh)    : {mae:.4f}")
    print(f"R2 score     : {r2:.4f}")
    print("-----------------------------------------")

    model_path = Path(model_path)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    return mae, r2


def main() -> int:
    X, y = loadAndPreprocessData()
    model, _x_train, X_test, _y_train, y_test = trainRandomForest(X, y)
    evaluateAndSaveModel(model, X_test, y_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
