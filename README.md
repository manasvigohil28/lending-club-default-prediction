# Loan Default Prediction (Lending Club)

Predicts whether a Lending Club borrower will default (Charged Off) or fully pay their loan,
comparing Logistic Regression, Random Forest and XGBoost, with a Streamlit dashboard for the results.

## Files

| File / folder | What it is |
|---|---|
| `loan_default_prediction.ipynb` | The whole project, run top to bottom (cleaning, EDA, statistics, models, approval strategy) |
| `app.py` | Streamlit dashboard (5 pages) |
| `images/` | Graphs saved by the notebook (used in the report) |
| `models/loan_models.pkl` | Trained models, scaler and test results saved by the notebook (used by the dashboard) |
| `data/` | The dataset, the 100,000-loan sample and the cleaned data (not included, see below) |
| `.streamlit/config.toml` | Dashboard theme |

## How to run

1. Download `accepted_2007_to_2018Q4.csv` from
   [Kaggle: Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
   and put it in the `data/` folder.
2. Install the libraries: `pip install -r requirements.txt`
3. Open `loan_default_prediction.ipynb` in Jupyter and choose **Run All**.
   The first run creates a 100,000-loan sample (about a minute); later runs take about a minute in total.
   Section 11 saves `data/loans_clean.csv` and `models/loan_models.pkl` for the dashboard.
4. Start the dashboard: `streamlit run app.py` (it opens in the browser at http://localhost:8501).

## Dashboard pages

- **Overview**: filters, key numbers and default-rate charts
- **Explore the Data**: default rate for any category, box plots for any numeric feature
- **Model Performance**: comparison table, ROC curves, confusion matrix, feature importance
- **Predict a Loan**: enter a new application and get a default risk score
- **Approval Strategy**: choose how many of the riskiest applications to reject and see the effect
