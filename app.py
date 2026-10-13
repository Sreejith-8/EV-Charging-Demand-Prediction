import streamlit as st
import pandas as pd
import plotly.express as px
import shap
from lime.lime_tabular import LimeTabularExplainer
import pickle
import os

st.set_page_config(page_title="EV Charging Dashboard", layout="wide")

df = pd.read_csv("cleaned_ev_charging_data.csv")

df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])

st.sidebar.subheader("Date Filter")

min_date = df["StartDatetime"].min().date()
max_date = df["StartDatetime"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

st.sidebar.header("Filters")
with open("xgboost_demand_model.pkl", "rb") as file:
    regression_model = pickle.load(file)
    
st.success("Regression model loaded successfully!")
with open("random_forest_model.pkl", "rb") as file:
    classification_model = pickle.load(file)

st.success("Classification model loaded successfully!")

location_options = ["All"] + sorted(df["Location"].dropna().unique().tolist())
selected_location = st.sidebar.selectbox("Select Location", location_options)

charger_options = ["All"] + sorted(df["ChargerType"].dropna().unique().tolist())
selected_charger = st.sidebar.selectbox("Select Charger Type", charger_options)

company_options = ["All"] + sorted(df["ChargerCompany"].dropna().unique().tolist())
selected_company = st.sidebar.selectbox("Select Charger Company", company_options)


filtered_df = df.copy()

if len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered_df = filtered_df[
        (filtered_df["StartDatetime"].dt.date >= start_date) &
        (filtered_df["StartDatetime"].dt.date <= end_date)
    ]


if selected_location != "All":
    filtered_df = filtered_df[filtered_df["Location"] == selected_location]

if selected_charger != "All":
    filtered_df = filtered_df[filtered_df["ChargerType"] == selected_charger]

if selected_company != "All":
    filtered_df = filtered_df[filtered_df["ChargerCompany"] == selected_company]

st.title("EV Charging Demand Analysis Dashboard")
st.write("Explore EV charging data and demand patterns.")

st.subheader("Overall EV Charging Usage")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sessions", len(df))

with col2:
    st.metric("Average Demand", round(df["Demand"].mean(), 2))

with col3:
    st.metric("Average Duration", round(df["Duration"].mean(), 2))

with col4:
    st.metric("Total Locations", df["Location"].nunique())

st.subheader("Average Demand by Location")

location_demand = df.groupby("Location", as_index=False)["Demand"].mean()

fig = px.bar(
    location_demand,
    x="Location",
    y="Demand",
    title="Average Charging Demand by Location"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Average Demand by Charger Type")

charger_demand = df.groupby("ChargerType", as_index=False)["Demand"].mean()

fig = px.bar(
    charger_demand,
    x="ChargerType",
    y="Demand",
    title="Average Charging Demand by Charger Type"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Average Demand by Charger Company")

company_demand = df.groupby("ChargerCompany", as_index=False)["Demand"].mean()

fig = px.bar(
    company_demand,
    x="ChargerCompany",
    y="Demand",
    title="Average Charging Demand by Company"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("Demand Trend Over Time")

filtered_df["StartDatetime"] = pd.to_datetime(filtered_df["StartDatetime"])

daily_demand = (
    filtered_df.groupby(filtered_df["StartDatetime"].dt.date)["Demand"]
    .mean()
    .reset_index()
)

daily_demand.columns = ["Date", "Average Demand"]

fig = px.line(
    daily_demand,
    x="Date",
    y="Average Demand",
    title="Daily Average Charging Demand"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("Regression: Predict Charging Demand")

st.write("Enter charging details to predict the expected demand.")
st.subheader("Enter Details for Demand Prediction")

with st.form("regression_form"):
    company = st.selectbox(
        "Charger Company",
        sorted(df["ChargerCompany"].dropna().unique())
    )

    charger_type = st.selectbox(
        "Charger Type",
        sorted(df["ChargerType"].dropna().unique())
    )

    location = st.selectbox(
        "Location",
        sorted(df["Location"].dropna().unique())
    )

    duration = st.number_input(
        "Duration",
        min_value=0.0,
        value=60.0
    )

    hour = st.slider("Hour", 0, 23, 12)

    day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"]
    )

    day_type = st.selectbox("Day Type", sorted(df["DayType"].dropna().unique()))

    predict_button = st.form_submit_button("Predict Demand")
if predict_button:
    input_data = pd.DataFrame([{
        "ChargerCompany": company,
        "ChargerType": charger_type,
        "DayOfWeek": day_of_week,
        "DayType": day_type,
        "Duration": duration,
        "Hour": hour,
        "Location": location
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=["DayOfWeek", "Location"]
    )

    input_data = input_data.reindex(
        columns=regression_model.feature_names_in_,
        fill_value=0
    )

    prediction = regression_model.predict(input_data)

    st.success(f"Predicted Charging Demand: {prediction[0]:.2f}")
    

st.subheader("Classification: Predict Demand Level")
st.write("Predict whether a charging session has High Demand or Normal Demand.")

with st.form("classification_form"):
    duration = st.number_input(
        "Session Duration",
        min_value=0.0,
        value=60.0,
        key="class_duration"
    )

    hour = st.slider(
        "Session Hour",
        0, 23, 12,
        key="class_hour"
    )

    company = st.selectbox(
        "Charger Company",
        sorted(df["ChargerCompany"].dropna().unique()),
        key="class_company"
    )

    charger_type = st.selectbox(
        "Charger Type",
        sorted(df["ChargerType"].dropna().unique()),
        key="class_charger_type"
    )

    location = st.selectbox(
        "Location",
        sorted(df["Location"].dropna().unique()),
        key="class_location"
    )

    day_type = st.selectbox(
        "Day Type",
        sorted(df["DayType"].dropna().unique()),
        key="class_day_type"
    )

    month = st.slider(
        "Month",
        1, 12, 6,
        key="class_month"
    )

    day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"],
        key="class_day_of_week"
    )

    time_of_day = st.selectbox(
        "Time of Day",
        ["Morning", "Afternoon", "Evening", "Night"],
        key="class_time_of_day"
    )

    classify_button = st.form_submit_button("Predict Demand Level")

if classify_button:
    input_data = pd.DataFrame([{
        "ChargerCompany": company,
        "ChargerType": charger_type,
        "Duration": duration,
        "Hour": hour,
        "DayType": day_type,
        "Month": month,
        "Location": location,
        "DayOfWeek": day_of_week,
        "TimeOfDay": time_of_day
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=["Location", "DayOfWeek", "TimeOfDay"]
    )

    input_data = input_data.reindex(
        columns=classification_model.feature_names_in_,
        fill_value=0
    )

    prediction = classification_model.predict(input_data)[0]

    if prediction == 1:
        st.success("Predicted Demand Level: High Demand")
    else:
        st.info("Predicted Demand Level: Normal Demand")

st.subheader("Classification Model Performance Comparison")

performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "SVM"
    ],
    "Accuracy": [
        0.8398,
        0.8540,
        0.8880,
        0.8310,
        0.8045
    ]
})

fig = px.bar(
    performance,
    x="Model",
    y="Accuracy",
    title="Accuracy Comparison of Classification Models",
    text="Accuracy"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Regression Model Performance")

regression_metrics = pd.DataFrame({
    "Metric": ["R² Score", "MAE", "MSE", "RMSE"],
    "Value": [0.8049, 3.3307, 34.7238, 5.8927]
})

st.dataframe(regression_metrics, use_container_width=True)

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": classification_model.feature_names_in_,
    "Importance": classification_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(10)

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Features Affecting Demand Classification"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("SHAP Model Explanation")

if st.button("Explain Random Forest Model", key="shap_explanation_button"):
    st.info("SHAP calculation started...")
    try:
        sample_data = pd.get_dummies(
            df[[
                "ChargerCompany",
                "ChargerType",
                "Duration",
                "Hour",
                "DayType",
                "Month",
                "Location",
                "DayOfWeek",
                "TimeOfDay"
            ]],
            columns=["Location", "DayOfWeek", "TimeOfDay"]
        )

        sample_data = sample_data.reindex(
            columns=classification_model.feature_names_in_,
            fill_value=0
        ).head(100)

       
        explainer = shap.TreeExplainer(classification_model)
        st.info("Calculating SHAP values...")

        shap_values = explainer.shap_values(sample_data)
        st.success("SHAP values calculated!")


        
        import matplotlib.pyplot as plt

        if isinstance(shap_values, list):
            values_to_plot = shap_values[1]
        elif len(shap_values.shape) == 3:
            values_to_plot = shap_values[:, :, 1]
        else:
            values_to_plot = shap_values

        shap.summary_plot(
            values_to_plot,
            sample_data,
            show=False
        )

        st.pyplot(plt.gcf())
        plt.close()

        st.write("The SHAP plot shows which features influence predictions.")

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
        st.subheader("LIME Explanation")

if st.button("Explain One Prediction", key="lime_explanation_button"):
    try:
        lime_data = pd.get_dummies(
            df[[
                "ChargerCompany", "ChargerType", "Duration",
                "Hour", "DayType", "Month", "Location",
                "DayOfWeek", "TimeOfDay"
            ]],
            columns=["Location", "DayOfWeek", "TimeOfDay"]
        )

        lime_data = lime_data.reindex(
            columns=classification_model.feature_names_in_,
            fill_value=0
        )

        explainer = LimeTabularExplainer(
            training_data=lime_data.values,
            feature_names=lime_data.columns.tolist(),
            class_names=["Normal Demand", "High Demand"],
            mode="classification"
        )

        explanation = explainer.explain_instance(
            lime_data.iloc[0].values,
            classification_model.predict_proba,
            num_features=10
        )

        lime_list = explanation.as_list()

        lime_df = pd.DataFrame(
            lime_list,
            columns=["Feature", "Contribution"]
        )

        st.dataframe(lime_df, use_container_width=True)

    except Exception as e:
        st.error(f"LIME explanation failed: {e}")
        


