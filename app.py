import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Student Grade Prediction")

attendance = st.slider("Attendance Rate", 0, 100, 80)
study_hours_week = st.slider("Study Hours Per Week", 0, 40, 15)
previous_grade = st.slider("Previous Grade", 0, 100, 75)
extra = st.slider("Extracurricular Activities", 0, 5, 1)

# Numerical inputs
study_hours = st.slider(
    "Study Hours",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1
)

attendance_percent = st.slider(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=80
)

# Gender
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

gender_male = 1 if gender == "Male" else 0

# Parental Support
parental_support = st.selectbox(
    "Parental Support",
    ["High", "Medium", "Low"]
)

parental_support_low = 1 if parental_support == "Low" else 0
parental_support_medium = 1 if parental_support == "Medium" else 0

# Online Classes
online_classes = st.selectbox(
    "Online Classes Taken",
    ["No", "Yes"]
)

online_classes_true = 1 if online_classes == "Yes" else 0

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
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    
    st.success(f"Predicted Final Grade: {prediction[0]:.2f}")
