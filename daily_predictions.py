
import pandas as pd

# Simulated top 25 predictions
data = {
    "player": [f"Player {i}" for i in range(1, 26)],
    "team": ["Team A"] * 25,
    "hr_probability": [round(0.3 + i * 0.01, 3) for i in range(25)]
}
df = pd.DataFrame(data)
df.to_csv("top_25_hr_threats.csv", index=False)
print("Top 25 HR threats saved to top_25_hr_threats.csv")
