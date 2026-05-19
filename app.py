import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Page title
st.title("Sales & Demand Forecasting")

# Load dataset
df = pd.read_csv("data/train.csv")

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Create daily sales
daily_sales = df.groupby('date')['sales'].sum().reset_index()

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Plot sales trend
st.subheader("Daily Sales Trend")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(daily_sales['date'], daily_sales['sales'])

ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Sales Trend")

st.pyplot(fig)

# Load trained model
with open("models/sales_forecasting_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction section
st.subheader("Predict Sales")

store = st.number_input("Store Number", 1, 54, 1)
family = st.number_input("Family Category", 0, 30, 1)
promotion = st.number_input("On Promotion", 0, 100, 0)
year = st.number_input("Year", 2013, 2030, 2017)
month = st.number_input("Month", 1, 12, 1)
day = st.number_input("Day", 1, 31, 1)
dayofweek = st.number_input("Day of Week", 0, 6, 1)

if st.button("Predict Sales"):

    prediction = model.predict([[
        store,
        family,
        promotion,
        year,
        month,
        day,
        dayofweek
    ]])

    st.success(f"Predicted Sales: {prediction[0]:.2f}")