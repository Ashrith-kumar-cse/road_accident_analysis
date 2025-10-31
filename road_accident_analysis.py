# road_accident_analysis.py
# Smart synthetic dataset for realistic accident severity prediction
# Expected accuracy: 80–100%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

# -------------------- Generate Smarter Synthetic Dataset --------------------
n_samples = 400

speed_limit = np.random.choice([20, 30, 40, 50, 60, 70], n_samples)
weather = np.random.choice(["Fine", "Rain", "Fog", "Snow"], n_samples)
light = np.random.choice(["Daylight", "Darkness"], n_samples)
road = np.random.choice(["Dry", "Wet", "Icy"], n_samples)
vehicle = np.random.choice(["Car", "Motorcycle", "Bus", "Truck"], n_samples)

# Logical accident severity rules
severity = []
for s, w, l, r, v in zip(speed_limit, weather, light, road, vehicle):
    score = 0
    if s >= 60: score += 2
    if w in ["Rain", "Fog", "Snow"]: score += 2
    if l == "Darkness": score += 1
    if r in ["Wet", "Icy"]: score += 2
    if v in ["Motorcycle", "Truck"]: score += 1

    if score <= 2:
        severity.append("Slight")
    elif 3 <= score <= 5:
        severity.append("Serious")
    else:
        severity.append("Fatal")

data = pd.DataFrame({
    "Speed_limit": speed_limit,
    "Weather_conditions": weather,
    "Light_conditions": light,
    "Road_surface": road,
    "Vehicle_type": vehicle,
    "Accident_Severity": severity
})

# -------------------- Encode and Train --------------------
label_encoders = {}
for col in ["Weather_conditions", "Light_conditions", "Road_surface", "Vehicle_type", "Accident_Severity"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop("Accident_Severity", axis=1)
y = data["Accident_Severity"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

# -------------------- User Input --------------------
print("\nEnter the accident details below 👇")

speed = int(input("Enter speed limit (20, 30, 40, 50, 60, 70): "))
weather_in = input("Weather condition (Fine, Rain, Fog, Snow): ").capitalize()
light_in = input("Light condition (Daylight, Darkness): ").capitalize()
road_in = input("Road surface (Dry, Wet, Icy): ").capitalize()
vehicle_in = input("Vehicle type (Car, Motorcycle, Bus, Truck): ").capitalize()

input_df = pd.DataFrame({
    "Speed_limit": [speed],
    "Weather_conditions": [weather_in],
    "Light_conditions": [light_in],
    "Road_surface": [road_in],
    "Vehicle_type": [vehicle_in]
})

# Encode categorical columns only
for col in ["Weather_conditions", "Light_conditions", "Road_surface", "Vehicle_type"]:
    le = label_encoders[col]
    # Handle unseen labels safely
    if input_df[col].iloc[0] not in le.classes_:
        input_df[col] = le.transform([le.classes_[0]])
    else:
        input_df[col] = le.transform(input_df[col])

# Scale
input_scaled = scaler.transform(input_df)

pred = model.predict(input_scaled)[0]
severity_label = label_encoders["Accident_Severity"].inverse_transform([pred])[0]

print("\n🚦 Predicted Accident Severity:", severity_label)
