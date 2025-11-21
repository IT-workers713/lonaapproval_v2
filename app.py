# app.py

import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load trained pipeline
model = joblib.load("loan_pipeline.joblib")

# Page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

st.title(" Loan Approval Prediction")
st.write("Enter applicant details to check loan approval probability.")

# Form for user input
with st.form("loan_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0, step=500)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, step=10)
    loan_amount_term = st.selectbox("Loan Amount Term (in days)", [360, 180, 480, 300, 240, 120, 84])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    # Create input DataFrame
    input_df = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])

    # Predict probability
    prob = model.predict_proba(input_df)[0][1]
    decision = " Approved" if prob >= 0.5 else " Not Approved"

    st.subheader(" Prediction Result")
    st.write(f"**Decision:** {decision}")
    st.write(f"**Approval Probability:** {prob:.2f}")

    # Show explanation note
    st.info("Prediction is based on historical loan approval data and may not reflect actual bank decisions.")

st.markdown("""
###  Loan Prediction Dataset – Field Details

| Field Name        | Type                 | Meaning                              | Value Range / Units                      | How the Model Considers It |
|-------------------|----------------------|--------------------------------------|-------------------------------------------|----------------------------|
| **Gender**        | Categorical          | Applicant’s gender                   | "Male" or "Female"                        | Encoded as binary; minor influence based on patterns. |
| **Married**       | Categorical          | Marital status                       | "Yes" or "No"                             | Encoded as binary; married sometimes higher approvals. |
| **Dependents**    | Categorical          | Number of dependents                  | "0", "1", "2", "3+"                        | One-hot encoded; more dependents may lower chance slightly. |
| **Education**     | Categorical          | Education level                       | "Graduate" or "Not Graduate"              | Encoded as binary; graduates slightly higher approvals. |
| **Self_Employed** | Categorical          | Self-employment status                | "Yes" or "No"                              | Encoded as binary; self-employed may have small negative weight. |
| **ApplicantIncome** | Numeric             | Applicant’s monthly income         | Positive integers                          | Higher income increases approval chance. |
| **CoapplicantIncome** | Numeric           | Co-applicant’s monthly income      | Positive integers or 0                     | Adds to repayment capacity. |
| **LoanAmount**    | Numeric              | Loan amount requested               | Example: 100 = 100,000                    | Higher amount vs income may lower chance. |
| **Loan_Amount_Term** | Numeric           | Loan term in days                     | e.g., 360, 120, 84                         | Longer terms can slightly increase probability. |
| **Credit_History** | Numeric (binary)    | Past loan repayment record            | 1.0 = good history, 0.0 = bad/no history   | Most important feature for approval. |
| **Property_Area** | Categorical          | Location type of property             | "Urban", "Semiurban", "Rural"              | One-hot encoded; semiurban often has higher odds. |
| **Loan_Status**   | Target Variable      | Approval status                       | "Y" (1) or "N" (0)                         | What the model predicts. |
""")

# Feature importance chart
st.subheader(" Factors Affecting Loan Approval")
try:
    image = Image.open("feature_importance.png")
    st.image(image, caption="Top Features Influencing Approval", use_column_width=True)
except FileNotFoundError:
    st.warning("Feature importance chart not found. Run the training script to generate it.")
