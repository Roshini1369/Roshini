import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Fetch real-time COVID-19 data
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])
st.write("### Current COVID-19 Data in the USA")
st.dataframe(df)

# Bar Chart for Visual Representation
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
st.pyplot(plt)

# Generate Random Historical Data (Last 30 Days)
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Cases per day
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = np.arange(1, 31)

# Show historical data
st.write("### Historical COVID-19 Cases (Last 30 Days)")
st.dataframe(df_historical)

# Prepare Data for SVR Model
X = df_historical[["day"]].values
y = df_historical["cases"].values

# Feature Scaling (Important for SVR)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train the SVR Model
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model.fit(X_train, y_train)

# Streamlit UI for Prediction
st.title("COVID-19 Cases Prediction in the USA")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User Input for Prediction
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    next_day_scaled = scaler_X.transform(np.array([[day_input]]))  # Scale input
    predicted_cases_scaled = model.predict(next_day_scaled)  # Predict
    predicted_cases = scaler_y.inverse_transform(predicted_cases_scaled.reshape(-1, 1))  # Inverse transform output
    st.write(f"### Predicted cases for Day {day_input}: {int(predicted_cases[0][0])}")
