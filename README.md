# EV Charging Demand Prediction

## Project Overview
This project analyzes electric vehicle (EV) charging data to understand charging demand and predict demand levels using Machine Learning. An interactive Streamlit dashboard allows users to explore charging data, view visualizations, and make predictions.

## Features
- Interactive filters for location, charger type, company, and date
- EV charging demand visualizations
- Demand prediction using XGBoost Regression
- High/Normal demand classification using Random Forest
- Comparison of classification models
- Regression evaluation metrics
- Random Forest feature importance
- SHAP model explanations
- LIME explanations for individual predictions

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Random Forest
- Streamlit
- Plotly
- SHAP
- LIME

## Machine Learning Models
**Regression:** XGBoost

**Classification:** Random Forest

The classification model categorizes demand as High Demand or Normal Demand based on the project's selected threshold.

## Evaluation Metrics
Regression:
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Classification:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Run the Project

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly shap lime
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

## Project Structure
```text
EV-Charging-Demand-Prediction/
├── app.py
├── cleaned_ev_charging_data.csv
├── 2_classification_ev_charging_data.csv
├── random_forest_model.pkl
├── xgboost_demand_model.pkl
└── README.md
```

*The filenames above should match the files actually included in the repository.*

## Future Improvements
- Improve prediction performance with additional data
- Deploy the dashboard online
- Add more charging-demand forecasting features

## Author
Sreejith R S

## Disclaimer
Predictions are based on the available dataset and trained models. Results may vary with new charging data.
