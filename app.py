import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("Student Grade Prediction")

attendance = st.slider("Attendance Rate", 0, 100, 80)
study_hours_week = st.slider("Study Hours Per Week", 0, 40, 15)
previous_grade = st.slider("Previous Grade", 0, 100, 75)
extra = st.slider("Extracurricular Activities", 0, 5, 1)

if st.button("Predict Grade"):
    data = pd.DataFrame({
    "AttendanceRate": [attendance],
    "StudyHoursPerWeek": [study_hours_week],
    "PreviousGrade": [previous_grade],
    "ExtracurricularActivities": [extra],
    "Study Hours": [3.0],
    "Attendance (%)": [80],
    "Gender_Male": [1],
    "ParentalSupport_Low": [0],
    "ParentalSupport_Medium": [1],
    "Online Classes Taken_True": [1]
})
    prediction = model.predict(data)

    st.success(f"Predicted Final Grade: {prediction[0]:.2f}")
