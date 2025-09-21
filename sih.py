# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# -----------------------------
# 1. Crop thresholds (example)
# -----------------------------
crop_thresholds = {
    'rice': {'N': 80, 'P': 40, 'K': 40, 'pH': (6, 7.5), 'soil_moisture': 30},
    'wheat': {'N': 90, 'P': 50, 'K': 40, 'pH': (6, 7), 'soil_moisture': 25},
    'maize': {'N': 100, 'P': 50, 'K': 50, 'pH': (6, 7), 'soil_moisture': 35},
    # Add more crops as needed
}

# -----------------------------
# 2. Load or train Random Forest model
# -----------------------------
# Option 1: Load a pre-trained model
# model = pickle.load(open("fertilizer_yield_model.pkl", "rb"))

# Option 2: Train a small example model if you don't have a pickle
# Example: replace this with your actual dataset and preprocessing
np.random.seed(42)
X_dummy = np.random.randint(10, 100, (50, 5))
y_dummy = np.random.randint(1000, 3000, 50)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_dummy, y_dummy)

# -----------------------------
# 3. Function to suggest improvements
# -----------------------------
def suggest_improvements(crop, inputs):
    thresholds = crop_thresholds[crop]
    suggestions = []

    if inputs['N'] < thresholds['N']:
        suggestions.append(f"Increase Nitrogen (N) to at least {thresholds['N']}")
        inputs['N'] = thresholds['N']

    if inputs['P'] < thresholds['P']:
        suggestions.append(f"Increase Phosphorus (P) to at least {thresholds['P']}")
        inputs['P'] = thresholds['P']

    if inputs['K'] < thresholds['K']:
        suggestions.append(f"Increase Potassium (K) to at least {thresholds['K']}")
        inputs['K'] = thresholds['K']

    if inputs['pH'] < thresholds['pH'][0]:
        suggestions.append(f"Apply lime to increase soil pH to {thresholds['pH'][0]}")
        inputs['pH'] = thresholds['pH'][0]
    elif inputs['pH'] > thresholds['pH'][1]:
        suggestions.append(f"Apply sulfur to decrease soil pH to {thresholds['pH'][1]}")
        inputs['pH'] = thresholds['pH'][1]

    if inputs['soil_moisture'] < thresholds['soil_moisture']:
        suggestions.append(f"Irrigate to increase soil moisture to at least {thresholds['soil_moisture']}%")
        inputs['soil_moisture'] = thresholds['soil_moisture']

    if not suggestions:
        suggestions.append("Your soil nutrient levels are optimal for this crop!")

    return suggestions, inputs

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("Crop Fertilizer Recommendation & Yield Prediction")

crop = st.selectbox("Select Crop", list(crop_thresholds.keys()))
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=30)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=30)
pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=5.5)
soil_moisture = st.number_input("Soil Moisture (%)", min_value=0, max_value=100, value=20)

inputs = {'N': N, 'P': P, 'K': K, 'pH': pH, 'soil_moisture': soil_moisture}

if st.button("Get Recommendations & Predict Yield"):
    suggestions, improved_inputs = suggest_improvements(crop, inputs)
    st.subheader("Recommendations:")
    for s in suggestions:
        st.write("- " + s)

    # Predict yield using improved inputs
    X_input = np.array([improved_inputs['N'], improved_inputs['P'], improved_inputs['K'],
                        improved_inputs['pH'], improved_inputs['soil_moisture']]).reshape(1, -1)
    predicted_yield = model.predict(X_input)[0]
    st.subheader("Predicted Yield (kg/ha):")
    st.write(f"{predicted_yield:.2f}")
