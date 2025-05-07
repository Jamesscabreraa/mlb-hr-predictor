
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders trained on real data
model = joblib.load("hr_model_with_real_names.pkl")
le_batter = joblib.load("le_batter_real.pkl")
le_pitcher = joblib.load("le_pitcher_real.pkl")

st.title("💣 MLB Home Run Predictor (Real Names)")

st.sidebar.header("🔍 Predict a Single Matchup")
batter_name = st.sidebar.text_input("Batter Name (e.g., Aaron Judge)")
pitcher_name = st.sidebar.text_input("Pitcher Name (e.g., Gerrit Cole)")
pitch_speed = st.sidebar.slider("Release Speed (MPH)", 70.0, 105.0, 93.0)

if batter_name and pitcher_name:
    try:
        batter_encoded = le_batter.transform([batter_name])[0]
        pitcher_encoded = le_pitcher.transform([pitcher_name])[0]
        X = pd.DataFrame([{
            "release_speed": pitch_speed,
            "batter_name_encoded": batter_encoded,
            "pitcher_name_encoded": pitcher_encoded
        }])
        prob = model.predict_proba(X)[0][1]
        st.success(f"🔥 Predicted HR probability: **{round(prob * 100, 2)}%**")
    except Exception as e:
        st.error("❌ Batter or pitcher not recognized in training data.")

st.markdown("---")
st.header("📂 Bulk Prediction via File Upload")
uploaded_file = st.file_uploader("Upload CSV with columns: batter_name, pitcher_name, release_speed", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df["batter_name_encoded"] = le_batter.transform(df["batter_name"])
        df["pitcher_name_encoded"] = le_pitcher.transform(df["pitcher_name"])
        X = df[["release_speed", "batter_name_encoded", "pitcher_name_encoded"]]
        df["HR_Probability (%)"] = model.predict_proba(X)[:, 1] * 100
        df["HR_Probability (%)"] = df["HR_Probability (%)"].round(2)
        st.dataframe(df[["batter_name", "pitcher_name", "HR_Probability (%)"]])
    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {e}")
