import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection")

st.write("Enter 30 feature values (like PCA-transformed V1 to V28, normalizedAmount, normalizedTime):")

input_data = []
for i in range(30):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    input_data.append(val)

if st.button("Predict"):
    prediction = model.predict([input_data])
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction.")
