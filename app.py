
import streamlit as st
import pandas as pd
import joblib
from pybaseball import statcast_pitcher, statcast_batter, playerid_lookup
from datetime import datetime, timedelta

# Load model and encoders
model = joblib.load("hr_model_with_names.pkl")
le_batter = joblib.load("le_batter.pkl")
le_pitcher = joblib.load("le_pitcher.pkl")

# UI
st.title("💣 MLB Home Run Predictor")

st.sidebar.header("🔍 Predict a Single Matchup")
batter_name = st.sidebar.text_input("Batter Name (e.g., Aaron Judge)")
pitcher_name = st.sidebar.text_input("Pitcher Name (e.g., Gerrit Cole)")

ballpark_factors = pd.read_csv("ballpark_factors.csv")
ballpark = st.sidebar.selectbox("Ballpark", ballpark_factors["ballpark"])
hr_factor = ballpark_factors.loc[ballpark_factors["ballpark"] == ballpark, "hr_factor"].values[0]

# Helper to get MLBAM ID
def get_mlbam_id(name):
    try:
        last, first = name.split()[-1], " ".join(name.split()[:-1])
        result = playerid_lookup(last, first)
        return int(result.iloc[0]["key_mlbam"]) if not result.empty else None
    except:
        return None

# Single matchup prediction
if batter_name and pitcher_name:
    batter_id = get_mlbam_id(batter_name)
    pitcher_id = get_mlbam_id(pitcher_name)

    if batter_id and pitcher_id:
        start = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")
        end = datetime.today().strftime("%Y-%m-%d")

        try:
            st.write(f"Fetching data for {batter_name} vs {pitcher_name}...")
            batter_df = statcast_batter(start, end, batter_id)
            pitcher_df = statcast_pitcher(start, end, pitcher_id)

            merged = pd.concat([batter_df, pitcher_df]).dropna(subset=["release_speed", "pitch_type"]).head(10)
            merged["pitch_speed"] = merged["release_speed"]
            merged["pitch_location_x"] = 0.0
            merged["pitch_location_y"] = 2.5
            merged["spin_rate"] = 2200
            merged["batter_ISO_vs_pitch"] = 0.25
            merged["batter_SLG_vs_pitch"] = 0.500
            merged["batter_LA_vs_pitch"] = 15.0
            merged["pitcher_HR9_vs_pitch"] = 1.0
            merged["pitcher_Barrel_vs_pitch"] = 7.0
            merged["ballpark_HR_factor"] = hr_factor
            merged["count"] = 2
            merged["pitch_type_FF"] = 1
            merged["handedness_matchup_LR"] = 1

            merged["batter_name_encoded"] = le_batter.transform([batter_name] * len(merged))
            merged["pitcher_name_encoded"] = le_pitcher.transform([pitcher_name] * len(merged))

            features = [
                'pitch_speed', 'pitch_location_x', 'pitch_location_y', 'release_speed', 'spin_rate',
                'batter_ISO_vs_pitch', 'batter_SLG_vs_pitch', 'batter_LA_vs_pitch',
                'pitcher_HR9_vs_pitch', 'pitcher_Barrel_vs_pitch', 'ballpark_HR_factor',
                'count', 'pitch_type_FF', 'handedness_matchup_LR',
                'batter_name_encoded', 'pitcher_name_encoded'
            ]

            prob = model.predict_proba(merged[features])[:, 1].mean()
            st.success(f"🔥 Predicted HR probability: **{round(prob * 100, 2)}%**")
            st.write(f"Batter: {batter_name}")
            st.write(f"Pitcher: {pitcher_name}")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
    else:
        st.error("❌ Could not find batter or pitcher ID.")

# Upload support
st.markdown("---")
st.header("📂 Bulk Prediction via File Upload")
uploaded_file = st.file_uploader("Upload CSV with batter/pitcher names and features", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    try:
        df["batter_name_encoded"] = le_batter.transform(df["batter_name"])
        df["pitcher_name_encoded"] = le_pitcher.transform(df["pitcher_name"])

        features = [
            'pitch_speed', 'release_speed', 'spin_rate', 'batter_ISO_vs_pitch',
            'batter_SLG_vs_pitch', 'batter_LA_vs_pitch', 'pitcher_HR9_vs_pitch',
            'pitcher_Barrel_vs_pitch', 'ballpark_HR_factor', 'count',
            'pitch_type_FF', 'handedness_matchup_LR',
            'batter_name_encoded', 'pitcher_name_encoded'
        ]

        df["HR_Probability (%)"] = model.predict_proba(df[features])[:, 1] * 100
        df["HR_Probability (%)"] = df["HR_Probability (%)"].round(2)

        st.dataframe(df[["batter_name", "pitcher_name", "HR_Probability (%)"]])

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
