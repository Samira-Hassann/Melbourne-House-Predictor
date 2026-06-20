import streamlit as st
import pandas as pd
import joblib

model = joblib.load("diabetes_model.pkl")


st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


st.title("🩺 Diabetes Prediction App")
st.markdown(
    "Enter the patient's clinical information to estimate diabetes risk:"
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        value=0
    )

with col2:
    glucose = st.number_input(
        "Glucose",
        min_value=0,
        value=120
    )


col1, col2 = st.columns(2)

with col1:
    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        value=70
    )

with col2:
    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        value=20
    )


col1, col2 = st.columns(2)

with col1:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        value=80
    )

with col2:
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0
    )


col1, col2 = st.columns(2)

with col1:
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.5
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=0,
        value=30
    )

st.divider()
_, center, _ = st.columns([1, 2, 1])

with center:
    predict_btn = st.button(
        "🔍 Predict Diabetes Risk",
        use_container_width=True
    )


if predict_btn:

    data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    probability = model.predict_proba(data)[0][1]


    threshold = 0.35

    prediction = int(probability >= threshold)

    st.subheader("Prediction Result")

    st.metric(
        label="Diabetes Probability",
        value=f"{probability:.2%}"
    )

    if prediction == 1:
        st.error(
            f"⚠️ High Risk of Diabetes\n\nProbability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Low Risk of Diabetes\n\nProbability: {probability:.2%}"
        )
        


