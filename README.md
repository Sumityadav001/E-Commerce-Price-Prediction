# E-commerce Price Prediction ✅

A small, end-to-end project that scrapes laptop listings from Flipkart, performs exploratory data analysis (EDA), trains price prediction models, and exposes a demo Streamlit app for predictions.

## 🔧 Project structure
- `scraper/` — scraping scripts (Flipkart search pages)
- `data/raw/` — raw scraped CSVs
- `data/processed/` — cleaned datasets used for modeling
- `data_processing/` — data cleaning helpers
- `notebooks/` — EDA and analysis notebooks (e.g., `eda.ipynb`)
- `models/` — training scripts and saved artifacts
  - `models/artifacts/best_price_model.joblib`
- `app/` — Streamlit app and CLI test mode
- `reports/` — generated EDA outputs (plots, HTML snippets)

## 🚀 Quick start
1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate    # Windows PowerShell
   source .venv/bin/activate     # macOS / Linux
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the scraper to collect raw data (network required):

   ```bash
   python scraper/flipkart_scraper.py
   # output: data/raw/flipkart_laptops.csv
   ```

4. Clean the data:

   ```bash
   python data_processing/clean.py
   # output: data/processed/flipkart_laptops_clean.csv
   ```

5. Train models and save the best artifact:

   ```bash
   python models/train.py
   # output: models/artifacts/best_price_model.joblib
   ```

6. Run quick CLI prediction (for smoke test):

   ```bash
   python app/app.py --cli
   ```

7. Start the Streamlit demo:

   ```bash
   streamlit run app/app.py --server.port 8502
   ```

## 🔬 Notebooks & EDA
Open `notebooks/eda.ipynb` to explore data diagnostics, distributions, and feature correlations. The notebook includes generated plot images and a short insights summary saved to `reports/eda_outputs/` when exporting.


## 📁 Artifacts & Outputs
- Raw data: `data/raw/flipkart_laptops.csv`
- Cleaned data: `data/processed/flipkart_laptops_clean.csv`
- Model: `models/artifacts/best_price_model.joblib`
- Notebook exports: `reports/eda_outputs/` (PNG, CSV, HTML snippets)


## License
This repository is available under the MIT License. See `LICENSE` for details.

---

If you'd like, I can also: add badges, generate a one-page project summary, or create a CONTRIBUTING.md. Let me know which you'd prefer.
