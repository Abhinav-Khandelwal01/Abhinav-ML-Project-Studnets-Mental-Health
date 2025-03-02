import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
gb_model = joblib.load("final_mental_health_model.pkl")

st.title("Student Mental Health Prediction")

# User input
study_hours = st.slider("Study Hours", 2, 16, 8)
sleep_hours = st.slider("Sleep Hours", 3, 9, 6)
social_activity = st.slider("Social Activity", 0, 5, 2)
parental_pressure = st.slider("Parental Pressure", 1, 10, 5)
exam_stress_level = st.slider("Exam Stress Level", 1, 10, 5)
physical_activity = st.slider("Physical Activity (Days/Week)", 0, 7, 3)
diet_quality = st.slider("Diet Quality", 1, 5, 3)
screen_time = st.slider("Screen Time (Hours/Day)", 1, 10, 5)
academic_performance = st.slider("Academic Performance (%)", 40, 100, 75)
peer_support = st.slider("Peer Support Level", 1, 5, 3)
family_income = st.slider("Family Income Level", 1, 10, 5)
hobbies_leisure = st.slider("Hobbies/Leisure Time (Hours/Week)", 0, 5, 2)
mental_health_history = st.radio("Mental Health History", [0, 1])

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[
        study_hours, sleep_hours, social_activity, parental_pressure, exam_stress_level,
        physical_activity, diet_quality, screen_time, academic_performance, peer_support,
        family_income, hobbies_leisure, mental_health_history
    ]])
    
    # Prediction
    prediction = gb_model.predict(input_data)
    
    # Display result
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Mental Health Issues")
    else:
        st.success("✅ Low Risk of Mental Health Issues")
