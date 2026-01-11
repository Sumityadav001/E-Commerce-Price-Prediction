# models/train.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import numpy as np
import os

def load_data(path="data/processed/flipkart_laptops_clean.csv"):
    df = pd.read_csv(path)
    # Target and features
    y = df["price"]
    X = df.drop(columns=["price"])
    # Choose categorical columns (if any)
    cat_cols = ["reviews_bucket"]  # add brand/category if scraped
    num_cols = [c for c in X.columns if c not in cat_cols and X[c].dtype != "object"]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    return X, y, pre, cat_cols, num_cols

def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    # Use sqrt of MSE for RMSE for compatibility with older sklearn versions
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{label} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

def train_models():
    X, y, pre, cat_cols, num_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ridge regression
    ridge = Pipeline(steps=[("pre", pre), ("model", Ridge())])
    ridge_params = {"model__alpha": [0.1, 1.0, 10.0]}
    ridge_gs = GridSearchCV(ridge, ridge_params, cv=3, n_jobs=-1)
    ridge_gs.fit(X_train, y_train)
    ridge_pred = ridge_gs.predict(X_test)
    evaluate(y_test, ridge_pred, "Ridge (best)")

    # Random Forest
    rf = Pipeline(steps=[("pre", pre), ("model", RandomForestRegressor(random_state=42))])
    rf_params = {"model__n_estimators": [200, 400], "model__max_depth": [None, 10, 20]}
    rf_gs = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1)
    rf_gs.fit(X_train, y_train)
    rf_pred = rf_gs.predict(X_test)
    evaluate(y_test, rf_pred, "RandomForest (best)")

    # XGBoost
    xgb_model = Pipeline(steps=[("pre", pre), ("model", xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        random_state=42, objective="reg:squarederror"
    ))])
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    evaluate(y_test, xgb_pred, "XGBoost")

    # Pick best by RMSE
    scores = [
        ("ridge", ridge_gs, ridge_pred),
        ("rf", rf_gs, rf_pred),
        ("xgb", xgb_model, xgb_pred),
    ]
    # pick best by RMSE (use sqrt of MSE for wider sklearn compatibility)
    best = min(scores, key=lambda s: np.sqrt(mean_squared_error(y_test, s[2])))
    best_name, best_pipe, _ = best

    # ensure artifacts directory exists
    os.makedirs(os.path.dirname("models/artifacts/best_price_model.joblib"), exist_ok=True)
    joblib.dump(best_pipe, "models/artifacts/best_price_model.joblib")
    print(f"Saved best model: {best_name}")

if __name__ == "__main__":
    train_models()