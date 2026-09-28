
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load saved model, scaler, and feature names
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(
    page_title="Credit Wise Loan System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Credit Wise Loan System")
st.write("Loan Approval Prediction")

st.subheader("Applicant Details")

# Numerical inputs
col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Applicant Income", min_value=0.0, value=5000.0
    )
    coapplicant_income = st.number_input(
        "Coapplicant Income", min_value=0.0, value=0.0
    )
    age = st.number_input(
        "Age", min_value=18, max_value=100, value=30
    )
    dependents = st.number_input(
        "Dependents", min_value=0, value=0
    )
    existing_loans = st.number_input(
        "Existing Loans", min_value=0, value=0
    )
    savings = st.number_input(
        "Savings", min_value=0.0, value=50000.0
    )
    collateral_value = st.number_input(
        "Collateral Value", min_value=0.0, value=0.0
    )

with col2:
    loan_amount = st.number_input(
        "Loan Amount", min_value=0.0, value=150.0
    )
    loan_term = st.number_input(
        "Loan Term", min_value=1, value=12
    )
    education_level = st.number_input(
        "Education Level (encoded value)",
        min_value=0, value=1
    )
    credit_score = st.number_input(
        "Credit Score", min_value=0.0, value=700.0
    )
    dti_ratio = st.number_input(
        "DTI Ratio (use the same units as training)",
        min_value=0.0, value=0.3
    )

st.subheader("Personal Details")

gender = st.selectbox(
    "Gender", ["Female", "Male"]
)

marital_status = st.selectbox(
    "Marital Status", ["Married", "Single"]
)

employment_status = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed", "Unemployed"]
)

employer_category = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Car", "Education", "Home", "Personal"]
)

property_area = st.selectbox(
    "Property Area", ["Rural", "Semiurban", "Urban"]
)

if st.button("Predict Loan Status", type="primary"):

    # Create initial input
    data = pd.DataFrame([{
        "Applicant_Income": income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": existing_loans,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education_level,
        "DTI_Ratio_sq": dti_ratio ** 2,
        "Credit_Score_sq": credit_score ** 2,
        "Applicant_Income_log": np.log1p(income)
    }])

    # Employment status encoding
    data["Employment_Status_Salaried"] = int(
        employment_status == "Salaried"
    )
    data["Employment_Status_Self-employed"] = int(
        employment_status == "Self-employed"
    )
    data["Employment_Status_Unemployed"] = int(
        employment_status == "Unemployed"
    )

    # Marital status encoding
    data["Marital_Status_Single"] = int(
        marital_status == "Single"
    )

    # Loan purpose encoding
    data["Loan_Purpose_Car"] = int(
        loan_purpose == "Car"
    )
    data["Loan_Purpose_Education"] = int(
        loan_purpose == "Education"
    )
    data["Loan_Purpose_Home"] = int(
        loan_purpose == "Home"
    )
    data["Loan_Purpose_Personal"] = int(
        loan_purpose == "Personal"
    )

    # Property area encoding
    data["Property_Area_Semiurban"] = int(
        property_area == "Semiurban"
    )
    data["Property_Area_Urban"] = int(
        property_area == "Urban"
    )

    # Gender encoding
    data["Gender_Male"] = int(gender == "Male")

    # Employer category encoding
    data["Employer_Category_Government"] = int(
        employer_category == "Government"
    )
    data["Employer_Category_MNC"] = int(
        employer_category == "MNC"
    )
    data["Employer_Category_Private"] = int(
        employer_category == "Private"
    )
    data["Employer_Category_Unemployed"] = int(
        employer_category == "Unemployed"
    )

    # Arrange columns in exact training order
    data = data.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Apply the saved scaler
    scaled_data = scaler.transform(data)

    # Predict
    prediction = model.predict(scaled_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")