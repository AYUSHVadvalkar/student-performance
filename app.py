import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# LOAD SAVED MODEL AND PREPROCESSOR
# ============================================================

model = joblib.load(
    "model/mlp_student_performance.pkl"
)

preprocessor = joblib.load(
    "model/preprocessor.pkl"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Performance Prediction")

st.write(
    "Enter the student's information below to predict "
    "whether the student is likely to PASS or FAIL."
)

st.divider()


# ============================================================
# STUDENT INFORMATION
# ============================================================

st.subheader("📋 Student Information")


# Age
age = st.number_input(
    "Age",
    min_value=15,
    max_value=25,
    value=17,
    step=1
)


# Study time
studytime = st.selectbox(
    "Weekly Study Time",
    options=[1, 2, 3, 4],
    index=1,
    format_func=lambda x: {
        1: "1 - Less than 2 hours",
        2: "2 - 5 hours",
        3: "3 - 10 hours",
        4: "4 - More than 10 hours"
    }[x]
)


# Previous failures
failures = st.number_input(
    "Number of Previous Failures",
    min_value=0,
    max_value=4,
    value=0,
    step=1
)


# School support
schoolsup = st.selectbox(
    "Extra Educational Support from School",
    ["no", "yes"]
)


# Family support
famsup = st.selectbox(
    "Family Educational Support",
    ["no", "yes"]
)


# Paid classes
paid = st.selectbox(
    "Extra Paid Classes",
    ["no", "yes"]
)


# Activities
activities = st.selectbox(
    "Extra-Curricular Activities",
    ["no", "yes"]
)


# Higher education
higher = st.selectbox(
    "Wants to Pursue Higher Education",
    ["no", "yes"]
)


# Internet
internet = st.selectbox(
    "Internet Access at Home",
    ["no", "yes"]
)


# Family relationship
famrel = st.slider(
    "Family Relationship Quality",
    min_value=1,
    max_value=5,
    value=4
)


# Free time
freetime = st.slider(
    "Free Time After School",
    min_value=1,
    max_value=5,
    value=3
)


# Absences
absences = st.number_input(
    "Number of School Absences",
    min_value=0,
    max_value=93,
    value=4,
    step=1
)


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Student Performance",
    use_container_width=True
):

    # Create DataFrame using the same
    # 12 features used during training
    input_data = pd.DataFrame({
        "age": [age],
        "studytime": [studytime],
        "failures": [failures],
        "schoolsup": [schoolsup],
        "famsup": [famsup],
        "paid": [paid],
        "activities": [activities],
        "higher": [higher],
        "internet": [internet],
        "famrel": [famrel],
        "freetime": [freetime],
        "absences": [absences]
    })


    # --------------------------------------------------------
    # Apply the SAME preprocessing used during training
    # --------------------------------------------------------

    input_processed = preprocessor.transform(
        input_data
    )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_processed
    )[0]


    # --------------------------------------------------------
    # Get prediction probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        input_processed
    )[0]

    fail_probability = probabilities[0] * 100
    pass_probability = probabilities[1] * 100


    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.success(
            "🎉 Prediction: PASS"
        )

        st.metric(
            "PASS Probability",
            f"{pass_probability:.2f}%"
        )

        st.write(
            f"FAIL Probability: {fail_probability:.2f}%"
        )

    else:

        st.error(
            "⚠️ Prediction: FAIL"
        )

        st.metric(
            "FAIL Probability",
            f"{fail_probability:.2f}%"
        )

        st.write(
            f"PASS Probability: {pass_probability:.2f}%"
        )


    # --------------------------------------------------------
    # Display entered student data
    # --------------------------------------------------------

    st.subheader("📝 Student Data Used")

    st.dataframe(
        input_data,
        use_container_width=True
    )