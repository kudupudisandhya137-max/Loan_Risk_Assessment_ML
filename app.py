import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Intelligent Loan Risk Assessment System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================
MODEL_PATH = "Models/final_loan_risk_model.pkl"
FEATURE_PATH = "Models/feature_columns.pkl"
TARGET_ENC_PATH = "Models/target_encoder.pkl"
LABEL_ENC_PATH = "Models/label_encoders.pkl"
IMPORTANCE_PATH = "Models/feature_importance.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model files not found! Please run 'python train.py' first.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)
target_encoder = joblib.load(TARGET_ENC_PATH)
label_encoders = joblib.load(LABEL_ENC_PATH)
feature_importance = joblib.load(IMPORTANCE_PATH)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div style="font-size: 26px; font-weight: 700; color: #1f77b4;">🏦 Intelligent Loan Risk Assessment System</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 13px; color: #555; margin-bottom: 15px;">Binary Risk Evaluation Engine (Low Risk vs High Risk)</div>', unsafe_allow_html=True)
st.divider()

# ============================================================
# SIDEBAR — APPLICANT INPUTS
# ============================================================
st.sidebar.header("📋 Applicant Profile")

applicant_age = st.sidebar.number_input("Applicant Age", 18, 70, 32)
employment_type = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-Employed"])

monthly_income = st.sidebar.number_input("Monthly Income (₹)", 15000, 500000, 80000, step=5000)
existing_monthly_emi = st.sidebar.number_input("Existing Monthly EMI (₹)", 0, 200000, 10000, step=2500)

loan_amount = st.sidebar.number_input("Requested Loan Amount (₹)", 50000, 10000000, 600000, step=25000)
loan_term_months = st.sidebar.slider("Loan Term (Months)", 12, 360, 60, step=12)

credit_score = st.sidebar.slider("Credit Score (CIBIL)", 300, 900, 750)
past_due_payments = st.sidebar.number_input("Past Due Payments (Defaults)", 0, 15, 0)
credit_history_years = st.sidebar.number_input("Credit History (Years)", 0.0, 30.0, 7.5, step=0.5)

assess_button = st.sidebar.button("🔍 Assess Loan Risk", use_container_width=True)

# ============================================================
# MAIN DASHBOARD OVERVIEW
# ============================================================
st.subheader("📊 Applicant Summary & Key Financials")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Monthly Income", f"₹{monthly_income:,.0f}")
c2.metric("Requested Loan", f"₹{loan_amount:,.0f}")
c3.metric("Credit Score", credit_score)
c4.metric("Existing EMI", f"₹{existing_monthly_emi:,.0f}")

# ============================================================
# PREDICTION & CALCULATIONS
# ============================================================
if assess_button:
    debt_to_income_ratio = existing_monthly_emi / (monthly_income + 1e-5)
    loan_to_annual_income = loan_amount / ((12 * monthly_income) + 1e-5)
    
    estimated_new_emi = loan_amount / loan_term_months if loan_term_months > 0 else 0
    total_projected_emi = existing_monthly_emi + estimated_new_emi
    net_disposable_income = monthly_income - total_projected_emi
    total_emi_burden_pct = (total_projected_emi / (monthly_income + 1e-5)) * 100

    try:
        emp_type_encoded = label_encoders["Employment_Type"].transform([employment_type])[0]
    except:
        emp_type_encoded = 0

    input_dict = {
        "Applicant_Age": applicant_age,
        "Employment_Type": emp_type_encoded,
        "Monthly_Income": monthly_income,
        "Existing_Monthly_EMI": existing_monthly_emi,
        "Loan_Amount": loan_amount,
        "Loan_Term_Months": loan_term_months,
        "Credit_Score": credit_score,
        "Past_Due_Payments": past_due_payments,
        "Credit_History_Years": credit_history_years,
        "Debt_to_Income_Ratio": debt_to_income_ratio,
        "Loan_to_Annual_Income": loan_to_annual_income
    }

    input_data = pd.DataFrame([input_dict])[feature_columns]

    pred_num = model.predict(input_data)[0]
    risk_level = target_encoder.inverse_transform([pred_num])[0]
    probabilities = model.predict_proba(input_data)[0]

    st.divider()
    st.subheader("🎯 Assessment Results & Decision Hub")

    r1, r2, r3, r4 = st.columns(4)

    if risk_level == "Low Risk":
        r1.success(f"Risk: {risk_level} (APPROVED)")
    else:
        r1.error(f"Risk: {risk_level} (REJECTED)")

    r2.metric("Debt-to-Income (DTI)", f"{debt_to_income_ratio:.2%}")
    r3.metric("Est. New EMI", f"₹{estimated_new_emi:,.0f}")
    r4.metric("Net Surplus Income", f"₹{net_disposable_income:,.0f}")

    st.write("### 💡 AI Underwriting Recommendation")
    if risk_level == "Low Risk":
        st.success("✅ **Recommendation: Instant Approval.** The applicant has a strong credit profile and manageable debt. Safe for loan disbursement.")
    else:
        st.error("❌ **Recommendation: Rejection Advised.** High default risk detected due to credit score or debt constraints.")

    st.write("### 📈 Risk Probability Distribution")
    for i, class_name in enumerate(target_encoder.classes_):
        st.progress(float(probabilities[i]), text=f"{class_name} Probability: {probabilities[i] * 100:.2f}%")

    # Financial Breakdown Chart
    st.divider()
    st.subheader("📊 Dynamic Financial Breakdown")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### Monthly Income vs Commitments")
        fig_d1, ax_d1 = plt.subplots(figsize=(6, 4))
        categories = ['Monthly Income', 'Existing EMI', 'New Est. EMI', 'Net Surplus']
        values = [monthly_income, existing_monthly_emi, estimated_new_emi, max(0, net_disposable_income)]
        colors = ['#2ca02c', '#ff7f0e', '#d62728', '#1f77b4']
        ax_d1.bar(categories, values, color=colors)
        ax_d1.set_ylabel("Amount (₹)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig_d1)

    with col_d2:
        st.markdown("#### Model Feature Importance Insights")
        top_features = feature_importance.sort_values("Importance", ascending=False).head(6)
        fig_d2, ax_d2 = plt.subplots(figsize=(6, 4))
        ax_d2.barh(top_features["Feature"][::-1], top_features["Importance"][::-1], color="#17becf")
        ax_d2.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig_d2)

else:
    st.info("👈 Enter the applicant details in the sidebar and click **Assess Loan Risk**.")

st.divider()
st.caption("Intelligent Loan Risk Assessment System — Powered by Machine Learning")