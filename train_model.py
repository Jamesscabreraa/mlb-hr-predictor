
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv("simulated_pitch_data.csv")

# Encode categorical variables
df = pd.get_dummies(df, columns=["pitch_type", "handedness_matchup"], drop_first=True)

# Separate features and target
X = df.drop("is_home_run", axis=1)
y = df["is_home_run"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "hr_model.pkl")
print("Model saved to hr_model.pkl")
