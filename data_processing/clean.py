# data_processing/clean.py
import pandas as pd
import numpy as np

def clean_prices(df):
    df = df.copy()
    df = df[df["price"].notna()]
    df["price"] = df["price"].astype(int)
    # Remove extreme outliers (IQR rule)
    q1, q3 = df["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    df = df[(df["price"] >= lower) & (df["price"] <= upper)]
    return df

def fill_missing(df):
    df = df.copy()
    # Ratings: fill missing with median
    if "rating" in df.columns:
        df["rating"] = df["rating"].fillna(df["rating"].median())
    # Reviews: fill missing with 0
    if "reviews_count" in df.columns:
        df["reviews_count"] = df["reviews_count"].fillna(0).astype(int)
    return df

def add_features(df):
    df = df.copy()
    # Text features
    df["name_len"] = df["product_name"].str.len()
    df["has_pro"] = df["product_name"].str.contains(r"\bPro\b", case=False, regex=True).astype(int)
    df["has_ultra"] = df["product_name"].str.contains(r"\bUltra\b", case=False, regex=True).astype(int)
    # Bucketize reviews
    df["reviews_bucket"] = pd.cut(df["reviews_count"], bins=[-1, 50, 200, 1000, np.inf],
                                  labels=["very_low", "low", "medium", "high"])
    return df

def clean_pipeline(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    df = df.drop_duplicates(subset=["product_name", "price"])
    df = fill_missing(df)
    df = clean_prices(df)
    df = add_features(df)
    df.to_csv(csv_out, index=False)
    print(f"Processed: {df.shape[0]} rows → {csv_out}")

if __name__ == "__main__":
    clean_pipeline("data/raw/flipkart_laptops.csv", "data/processed/flipkart_laptops_clean.csv")