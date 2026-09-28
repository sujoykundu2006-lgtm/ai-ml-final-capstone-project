"""
Module 6: Final Capstone Project - Model Training, Benchmarking & Inference Pipeline
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "House Price Prediction Dataset.csv"
MODEL_PATH = "best_model.pkl"
FEATURES_PATH = "model_features.pkl"


def load_and_preprocess_data(filepath=DATA_PATH):
    """Load dataset, handle missing data, and encode categorical variables."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)

    # Drop non-predictive identifiers
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)

    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=["Location", "Condition", "Garage"], drop_first=True)

    X = df_encoded.drop(columns=["Price"])
    y = df_encoded["Price"]

    return X, y


def train_and_benchmark(X, y):
    """Benchmark multiple regression algorithms and persist the top performer."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
    }

    results = []
    trained_models = {}

    print("\n" + "=" * 65)
    print("           MODEL BENCHMARKING & EVALUATION RESULTS")
    print("=" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({
            "Algorithm": name,
            "MAE ($)": round(mae, 2),
            "RMSE ($)": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(by="MAE ($)")
    print(results_df.to_string(index=False))
    print("=" * 65)

    # Save the Random Forest Regressor and feature columns
    best_model_name = "Random Forest Regressor"
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)
    print(f"\n[Artifact Saved] Successfully exported {best_model_name} to '{MODEL_PATH}'")

    return results_df


def predict_price(custom_input: dict):
    """Run inference for a single house profile using the saved model artifact."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("Model artifacts missing. Run pipeline first.")

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    # Construct input dataframe matching model feature schema
    row = {col: 0 for col in feature_cols}
    for k, v in custom_input.items():
        if k in row:
            row[k] = v

    # Set categorical flags
    if f"Location_{custom_input.get('Location')}" in row:
        row[f"Location_{custom_input.get('Location')}"] = 1
    if f"Condition_{custom_input.get('Condition')}" in row:
        row[f"Condition_{custom_input.get('Condition')}"] = 1
    if custom_input.get("Garage") == "Yes" and "Garage_Yes" in row:
        row["Garage_Yes"] = 1

    input_df = pd.DataFrame([row])[feature_cols]
    prediction = model.predict(input_df)[0]
    return prediction


if __name__ == "__main__":
    # Execute training and benchmarking
    X, y = load_and_preprocess_data()
    train_and_benchmark(X, y)

    # Test single-record inference
    sample_property = {
        "Area": 3200,
        "Bedrooms": 4,
        "Bathrooms": 3,
        "Floors": 2,
        "YearBuilt": 2012,
        "Location": "Downtown",
        "Condition": "Excellent",
        "Garage": "Yes"
    }
    est_price = predict_price(sample_property)
    print("\n--- Sample Inference Test ---")
    print(f"Property: {sample_property}")
    print(f"Estimated Market Value: ${est_price:,.2f}\n")