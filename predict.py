import joblib
import numpy as np

# Load the trained model
model = joblib.load("crop_recommender.pkl")

print("\n🌱 AI-based Crop Recommendation System 🌱\n")

# Take inputs from user
N = float(input("Enter Nitrogen content in soil (N): "))
P = float(input("Enter Phosphorus content in soil (P): "))
K = float(input("Enter Potassium content in soil (K): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter Soil pH: "))
rainfall = float(input("Enter Rainfall (mm): "))

# Convert inputs to array
sample_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# Make prediction
prediction = model.predict(sample_input)

print("\n✅ Recommended Crop for you is:", prediction[0])
