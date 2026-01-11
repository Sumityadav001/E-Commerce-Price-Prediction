# app/app.py
import streamlit as st
import pandas as pd
import joblib
import argparse
import sys


from pathlib import Path

def load_resources(data_path="data/processed/flipkart_laptops_clean.csv", model_path="models/artifacts/best_price_model.joblib"):
    """Load data and model; paths are resolved relative to the repository root (two folders above this file).
    Returns (df, model) or (None, None) on failure, and adds helpful error messages."""
    repo_root = Path(__file__).resolve().parents[1]
    data_path = (repo_root / data_path).resolve()
    model_path = (repo_root / model_path).resolve()

    df = None
    model = None

    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
    else:
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Could not load data from {data_path}: {e}")

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
    else:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            # show the exception and stack trace to help debugging in the Streamlit UI
            st.error(f"Could not load model from {model_path}: {e}")
            try:
                import traceback

                st.exception(traceback.format_exc())
            except Exception:
                pass

    return df, model


def run_streamlit_app():
    st.set_page_config(page_title="E‑commerce Price Insights & Prediction")
    st.title("E‑commerce Price Insights & Prediction")

    df, model = load_resources()
    if df is None:
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    st.subheader("Price distribution")
    st.bar_chart(df["price"].value_counts().sort_index())

    st.subheader("Model prediction")

    if model is None:
        st.warning("Model not available — predictions disabled.")
        return

    # Simple input form (align with features used in training)
    rating = st.slider("Rating", 0.0, 5.0, 4.0, 0.1)
    reviews_count = st.number_input("Reviews count", 0, 5000, 100)
    name_len = st.number_input("Name length", 1, 200, 30)
    has_pro = st.checkbox("Contains 'Pro'")
    has_ultra = st.checkbox("Contains 'Ultra'")
    reviews_bucket = st.selectbox("Reviews bucket", ["very_low", "low", "medium", "high"])

    input_df = pd.DataFrame([{
        "product_name": "custom",
        "rating": rating,
        "reviews_count": reviews_count,
        "name_len": name_len,
        "has_pro": int(has_pro),
        "has_ultra": int(has_ultra),
        "reviews_bucket": reviews_bucket,
        # include default page field because pipeline expects it
        "page": 1
    }])

    try:
        pred = model.predict(input_df)[0]
        st.markdown(f"**Predicted price:** ₹{int(pred):,}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


def run_cli_test():
    """Quick CLI check that loads data and model and prints a sample prediction."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    data_path = (repo_root / "data/processed/flipkart_laptops_clean.csv").resolve()
    model_path = (repo_root / "models/artifacts/best_price_model.joblib").resolve()

    if not data_path.exists():
        print(f"Error: data file not found: {data_path}")
        sys.exit(1)
    print(f"Data found: {data_path} (size={data_path.stat().st_size})")

    if not model_path.exists():
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)
    print(f"Model found: {model_path} (size={model_path.stat().st_size})")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error: could not load data '{data_path}': {e}")
        sys.exit(1)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error: could not load model '{model_path}': {e}")
        sys.exit(1)

    sample = pd.DataFrame([{
        "product_name": "cli-sample",
        "rating": 4.0,
        "reviews_count": 100,
        "name_len": 30,
        "has_pro": 0,
        "has_ultra": 0,
        "reviews_bucket": "low",
        "page": 1
    }])

    try:
        pred = model.predict(sample)[0]
        # Use ASCII 'Rs' in console to avoid encode issues on Windows
        print(f"Predicted price: Rs {int(pred):,}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run a quick CLI check (no Streamlit server)")
    args = parser.parse_args()

    if args.cli:
        run_cli_test()
    else:
        # If user invoked with `python app.py` we display a short note and start Streamlit when run with `streamlit run`
        print("Note: this app is intended to be started with: `streamlit run app.py`. Use --cli to run a quick check.")
        run_streamlit_app()