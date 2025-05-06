
import streamlit as st
import pandas as pd
import joblib

st.title("MLB Home Run Prediction - Top 25 HR Threats Today")

# Load the model
model = joblib.load("hr_model.pkl")

# Load top 25 predictions
try:
    df = pd.read_csv("top_25_hr_threats.csv")
    st.subheader("Top 25 HR Threats")
    st.dataframe(df)
except Exception as e:
    st.error("Predictions not found. Run daily_predictions.py first.")
