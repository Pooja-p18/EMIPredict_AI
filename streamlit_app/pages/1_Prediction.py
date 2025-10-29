"""
Real-time EMI Prediction Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

st.set_page_config(page_title="EMI Prediction", page_icon="🔮", layout="wide")

st.title("🔮 EMI Eligibility Prediction")
st.markdown("### Enter customer details for instant EMI assessment")

# Helper functions
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        classification_model = joblib.load('models/classification/best_model.pkl')
        regression_model = joblib.load('models/regression/best_model.pkl')
        return classification_model, regression_model
    except:
        return None, None

def calculate_derived_features(data):
    """Calculate all derived features matching training pipeline"""
    
    # Financial Ratios
    data['debt_to_income_ratio'] = np.clip((data['current_emi_amount'] / data['monthly_salary']) * 100, 0, 100)
    
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                   'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
    data['total_monthly_expenses'] = sum([data[col] for col in expense_cols])
    data['expense_to_income_ratio'] = np.clip((data['total_monthly_expenses'] / data['monthly_salary']) * 100,0, 150)
    
    data['disposable_income'] = max(0, data['monthly_salary'] - data['total_monthly_expenses'] - data['current_emi_amount'])
    data['affordability_ratio'] = np.clip((data['disposable_income'] / data['monthly_salary']) * 100,0, 100)
    
    data['total_obligations'] = data['current_emi_amount'] + data['total_monthly_expenses']
    data['obligation_ratio'] = np.clip((data['total_obligations'] / data['monthly_salary']) * 100,0, 150)
    
    data['savings_ratio'] = np.clip((data['bank_balance'] + data['emergency_fund']) / data['monthly_salary'], 0, 50)
    data['emi_burden_ratio'] = np.clip((data['requested_amount'] / data['requested_tenure'] / data['monthly_salary']) * 100,0, 100)
    
    # Risk Scores
    data['credit_risk_score'] = np.clip((data['credit_score'] - 300) / (850 - 300) * 100, 0, 100)
    data['employment_stability_score'] = min(data['years_of_employment'] * 10, 100)
    data['financial_cushion_score'] = min(100, (data['savings_ratio'] * 0.6) + ((100 - data['obligation_ratio']) * 0.4))
    data['family_burden_score'] = min(100, (data['dependents'] / data['family_size'] * 100) if data['family_size'] > 0 else 0)
    
    housing_map = {'Own': 100, 'Family': 80, 'Rented': 50}
    data['housing_stability_score'] = housing_map.get(data['house_type'], 50)
    data['has_existing_loans'] = 1 if data['existing_loans'] == 'Yes' else 0
    
    # Interaction Features
    data['income_credit_interaction'] = (data['monthly_salary'] / 100000) * (data['credit_score'] / 100)
    data['age_employment_interaction'] = data['age'] * data['years_of_employment']
    data['savings_credit_interaction'] = (data['bank_balance'] / 10000) * (data['credit_score'] / 100)
    data['per_capita_income'] = data['monthly_salary'] / data['family_size'] if data['family_size'] > 0 else data['monthly_salary']
    data['emergency_fund_months'] = min(24, data['emergency_fund'] / (data['total_monthly_expenses'] + 1))
    
    # Categorical groups
    if data['age'] <= 30:
        data['age_group'] = 'Young'
    elif data['age'] <= 40:
        data['age_group'] = 'Mid'
    elif data['age'] <= 50:
        data['age_group'] = 'Senior'
    else:
        data['age_group'] = 'Elder'
    
    if data['monthly_salary'] <= 30000:
        data['income_bracket'] = 'Low'
    elif data['monthly_salary'] <= 50000:
        data['income_bracket'] = 'Medium'
    elif data['monthly_salary'] <= 75000:
        data['income_bracket'] = 'Upper-Mid'
    elif data['monthly_salary'] <= 100000:
        data['income_bracket'] = 'High'
    else:
        data['income_bracket'] = 'Very-High'
    
    if data['credit_score'] <= 580:
        data['credit_category'] = 'Poor'
    elif data['credit_score'] <= 670:
        data['credit_category'] = 'Fair'
    elif data['credit_score'] <= 740:
        data['credit_category'] = 'Good'
    elif data['credit_score'] <= 800:
        data['credit_category'] = 'Very-Good'
    else:
        data['credit_category'] = 'Excellent'
    
    if data['obligation_ratio'] <= 40:
        data['risk_level'] = 'Low-Risk'
    elif data['obligation_ratio'] <= 60:
        data['risk_level'] = 'Medium-Risk'
    elif data['obligation_ratio'] <= 80:
        data['risk_level'] = 'High-Risk'
    else:
        data['risk_level'] = 'Very-High-Risk'
    
    return data

def encode_categorical(data):
    """Encode categorical variables"""
    # Simple mapping for categorical variables
    gender_map = {'Male': 0, 'Female': 1}
    marital_map = {'Single': 0, 'Married': 1}
    education_map = {'High School': 0, 'Graduate': 1, 'Post Graduate': 2, 'Professional': 3}
    employment_map = {'Private': 0, 'Government': 1, 'Self-employed': 2}
    company_map = {'Small': 0, 'Medium': 1, 'Large': 2, 'MNC': 3}
    house_map = {'Rented': 0, 'Own': 1, 'Family': 2}
    loans_map = {'No': 0, 'Yes': 1}
    scenario_map = {'E-commerce': 0, 'Home Appliances': 1, 'Vehicle': 2, 'Personal Loan': 3, 'Education': 4}
    age_group_map = {'Young': 0, 'Mid': 1, 'Senior': 2, 'Elder': 3}
    income_bracket_map = {'Low': 0, 'Medium': 1, 'Upper-Mid': 2, 'High': 3, 'Very-High': 4}
    credit_category_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very-Good': 3, 'Excellent': 4}
    risk_level_map = {'Low-Risk': 0, 'Medium-Risk': 1, 'High-Risk': 2, 'Very-High-Risk': 3}
    
    data['gender_encoded'] = gender_map.get(data['gender'], 0)
    data['marital_status_encoded'] = marital_map.get(data['marital_status'], 0)
    data['education_encoded'] = education_map.get(data['education'], 0)
    data['employment_type_encoded'] = employment_map.get(data['employment_type'], 0)
    data['company_type_encoded'] = company_map.get(data['company_type'], 0)
    data['house_type_encoded'] = house_map.get(data['house_type'], 0)
    data['existing_loans_encoded'] = loans_map.get(data['existing_loans'], 0)
    data['emi_scenario_encoded'] = scenario_map.get(data['emi_scenario'], 0)
    data['age_group_encoded'] = age_group_map.get(data['age_group'], 0)
    data['income_bracket_encoded'] = income_bracket_map.get(data['income_bracket'], 0)
    data['credit_category_encoded'] = credit_category_map.get(data['credit_category'], 0)
    data['risk_level_encoded'] = risk_level_map.get(data['risk_level'], 0)
    
    return data

# Load models
class_model, reg_model = load_models()

if class_model is None or reg_model is None:
    st.error("⚠️ Models not found! Please train models first by running: python src/model_training.py")
    st.stop()

# Create input form
st.markdown("---")
st.markdown("## 📝 Customer Information")

# Personal Information
st.markdown("### 👤 Personal Details")
col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col3:
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
with col4:
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])

# Employment Details
st.markdown("### 💼 Employment Information")
col1, col2, col3, col4 = st.columns(4)

with col1:
    monthly_salary = st.number_input("Monthly Salary (₹)", min_value=15000, max_value=500000, value=50000, step=1000)
with col2:
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
with col3:
    years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=40, value=5)
with col4:
    company_type = st.selectbox("Company Type", ["Small", "Medium", "Large", "MNC"])

# Family & Housing
st.markdown("### 🏠 Family & Housing")
col1, col2, col3, col4 = st.columns(4)

with col1:
    house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
with col2:
    monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, max_value=100000, value=10000, step=500)
with col3:
    family_size = st.number_input("Family Size", min_value=1, max_value=15, value=4)
with col4:
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=2)

# Monthly Expenses
st.markdown("### 💸 Monthly Expenses")
col1, col2, col3 = st.columns(3)

with col1:
    school_fees = st.number_input("School Fees (₹)", min_value=0, max_value=50000, value=5000, step=500)
    college_fees = st.number_input("College Fees (₹)", min_value=0, max_value=100000, value=0, step=1000)
with col2:
    travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0, max_value=30000, value=3000, step=500)
    groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0, max_value=50000, value=8000, step=500)
with col3:
    other_monthly_expenses = st.number_input("Other Expenses (₹)", min_value=0, max_value=50000, value=2000, step=500)

# Financial Status
st.markdown("### 💰 Financial Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    existing_loans = st.selectbox("Existing Loans?", ["No", "Yes"])
with col2:
    current_emi_amount = st.number_input("Current EMI (₹)", min_value=0, max_value=100000, value=0, step=500)
with col3:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
with col4:
    bank_balance = st.number_input("Bank Balance (₹)", min_value=0, max_value=10000000, value=50000, step=5000)

col1, col2 = st.columns(2)
with col1:
    emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, max_value=5000000, value=25000, step=5000)

# Loan Request
st.markdown("### 📋 Loan Request Details")
col1, col2, col3 = st.columns(3)

with col1:
    emi_scenario = st.selectbox("EMI Scenario", 
                                ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"])
with col2:
    requested_amount = st.number_input("Requested Amount (₹)", min_value=10000, max_value=2000000, value=100000, step=10000)
with col3:
    requested_tenure = st.number_input("Tenure (months)", min_value=3, max_value=84, value=24)

# Predict button
st.markdown("---")
if st.button("🔮 Predict EMI Eligibility", type="primary", use_container_width=True):
    
    # Collect all inputs
    input_data = {
        'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
        'monthly_salary': monthly_salary, 'employment_type': employment_type,
        'years_of_employment': years_of_employment, 'company_type': company_type,
        'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size,
        'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees,
        'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
        'current_emi_amount': current_emi_amount, 'credit_score': credit_score,
        'bank_balance': bank_balance, 'emergency_fund': emergency_fund,
        'emi_scenario': emi_scenario, 'requested_amount': requested_amount,
        'requested_tenure': requested_tenure
    }
    
    # Calculate derived features
    input_data = calculate_derived_features(input_data)
    
    # Encode categorical
    input_data = encode_categorical(input_data)
    
    # Create DataFrame (match training feature order)
    feature_order = [
        'age', 'monthly_salary', 'years_of_employment', 'monthly_rent', 'family_size',
        'dependents', 'school_fees', 'college_fees', 'travel_expenses',
        'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount',
        'credit_score', 'bank_balance', 'emergency_fund', 'requested_amount',
        'requested_tenure', 'debt_to_income_ratio', 'total_monthly_expenses',
        'expense_to_income_ratio', 'disposable_income', 'affordability_ratio',
        'obligation_ratio', 'savings_ratio', 'emi_burden_ratio',
        'credit_risk_score', 'employment_stability_score', 'financial_cushion_score',
        'family_burden_score', 'housing_stability_score', 'has_existing_loans',
        'income_credit_interaction', 'age_employment_interaction',
        'savings_credit_interaction', 'per_capita_income', 'emergency_fund_months',
        'gender_encoded', 'marital_status_encoded', 'education_encoded',
        'employment_type_encoded', 'company_type_encoded', 'house_type_encoded',
        'existing_loans_encoded', 'emi_scenario_encoded', 'age_group_encoded',
        'income_bracket_encoded', 'credit_category_encoded', 'risk_level_encoded'
    ]
    
    X_input = pd.DataFrame([input_data])[feature_order]
    
    # Align with training feature columns before prediction
    try:
        # Load the exact feature names used during training
        train_columns = joblib.load("artifacts/train_columns.pkl")
        
        # Reindex to ensure same structure (missing columns = 0)
        X_input = X_input.reindex(columns=train_columns, fill_value=0)
        
    except FileNotFoundError:
        st.warning("⚠️ 'train_columns.pkl' not found. Please ensure it's saved during training.")
    
    
    # Make predictions
    with st.spinner("🤖 AI is analyzing..."):
        class_pred = class_model.predict(X_input)[0]
        reg_pred = reg_model.predict(X_input)[0]
        
        # Get probability if available
        if hasattr(class_model, 'predict_proba'):
            class_proba = class_model.predict_proba(X_input)[0]
        else:
            class_proba = None
    
    # Display results
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    # Classification result
    class_labels = {0: 'Not Eligible', 1: 'High Risk', 2: 'Eligible'}
    result_label = class_labels[class_pred]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if result_label == 'Eligible':
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success("### ✅ ELIGIBLE")
            st.markdown("Customer meets all eligibility criteria for EMI approval.")
            st.markdown('</div>', unsafe_allow_html=True)
        elif result_label == 'High Risk':
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning("### ⚠️ HIGH RISK")
            st.markdown("Customer shows elevated risk factors. Manual review recommended.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="danger-box">', unsafe_allow_html=True)
            st.error("### ❌ NOT ELIGIBLE")
            st.markdown("Customer does not meet minimum eligibility criteria.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Maximum Affordable EMI", f"₹{reg_pred:,.0f} / month")
        st.metric("Requested EMI", f"₹{requested_amount/requested_tenure:,.0f} / month")
        
        if reg_pred < requested_amount/requested_tenure:
            st.error("⚠️ Requested EMI exceeds recommended amount")
        else:
            st.success("✅ Requested EMI is within safe limits")
    
    # Detailed Analysis
    st.markdown("---")
    st.markdown("### 🔍 Detailed Financial Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Debt-to-Income Ratio", f"{input_data['debt_to_income_ratio']:.1f}%")
        st.metric("Expense Ratio", f"{input_data['expense_to_income_ratio']:.1f}%")
    
    with col2:
        st.metric("Disposable Income", f"₹{input_data['disposable_income']:,.0f}")
        st.metric("Affordability Ratio", f"{input_data['affordability_ratio']:.1f}%")
    
    with col3:
        st.metric("Credit Risk Score", f"{input_data['credit_risk_score']:.0f}/100")
        st.metric("Financial Cushion", f"{input_data['financial_cushion_score']:.0f}/100")
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    if result_label == 'Eligible':
        st.success("""
        - ✅ Proceed with loan processing
        - ✅ Standard interest rates applicable
        - ✅ Maximum EMI: ₹{:,.0f}
        """.format(reg_pred))
    elif result_label == 'High Risk':
        st.warning("""
        - ⚠️ Consider higher interest rate
        - ⚠️ Request additional collateral
        - ⚠️ Reduce loan amount or increase tenure
        - ⚠️ Recommended EMI: ₹{:,.0f} (vs requested: ₹{:,.0f})
        """.format(reg_pred, requested_amount/requested_tenure))
    else:
        st.error("""
        - ❌ Loan not recommended at this time
        - 📋 Suggestions for customer:
          - Improve credit score (current: {})
          - Reduce existing obligations
          - Increase monthly income
          - Build emergency fund
        """.format(credit_score))