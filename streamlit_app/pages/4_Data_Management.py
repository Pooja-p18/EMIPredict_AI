"""
CRUD Operations Page - Data Management
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="Data Management", page_icon="⚙️", layout="wide")

st.title("⚙️ Data Management System")
st.markdown("### Create, Read, Update, and Delete Records")

# File path for demo data
DATA_FILE = 'data/processed/demo_records.csv'

# Initialize demo data if doesn't exist
def init_demo_data():
    if not os.path.exists(DATA_FILE):
        demo_df = pd.DataFrame({
            'id': range(1, 6),
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
            'age': [30, 35, 28, 42, 31],
            'monthly_salary': [50000, 75000, 45000, 90000, 60000],
            'credit_score': [720, 780, 650, 800, 700],
            'emi_eligibility': ['Eligible', 'Eligible', 'High_Risk', 'Eligible', 'Eligible'],
            'max_monthly_emi': [8000, 12000, 5000, 15000, 9000],
            'created_date': [datetime.now().strftime('%Y-%m-%d')] * 5
        })
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        demo_df.to_csv(DATA_FILE, index=False)
    return pd.read_csv(DATA_FILE)

# Load data
@st.cache_data
def load_data():
    return init_demo_data()

# Save data
def save_data(df):
    df.to_csv(DATA_FILE, index=False)
    st.cache_data.clear()

# Tabs for CRUD operations
tab1, tab2, tab3, tab4 = st.tabs(["📖 Read", "➕ Create", "✏️ Update", "🗑️ Delete"])

# ----------------------------------------------------------------
# READ TAB
# ----------------------------------------------------------------
with tab1:
    st.markdown("## 📖 View Records")
    df = load_data()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        search_name = st.text_input("🔍 Search by Name")
    with col2:
        filter_eligibility = st.multiselect(
            "Filter by Eligibility",
            options=['All'] + list(df['emi_eligibility'].unique()),
            default=['All']
        )
    with col3:
        sort_by = st.selectbox("Sort by", 
                              ['id', 'name', 'age', 'monthly_salary', 'credit_score', 'created_date'])

    # Apply filters
    filtered_df = df.copy()
    if search_name:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False, na=False)]
    if 'All' not in filter_eligibility:
        filtered_df = filtered_df[filtered_df['emi_eligibility'].isin(filter_eligibility)]
    filtered_df = filtered_df.sort_values(sort_by)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(filtered_df))
    with col2:
        avg_salary = filtered_df['monthly_salary'].mean()
        st.metric("Avg Salary", f"₹{avg_salary:,.0f}")
    with col3:
        avg_credit = filtered_df['credit_score'].mean()
        st.metric("Avg Credit Score", f"{avg_credit:.0f}")
    with col4:
        eligible_pct = (filtered_df['emi_eligibility'] == 'Eligible').sum() / len(filtered_df) * 100
        st.metric("Eligibility %", f"{eligible_pct:.1f}%")

    # Show data
    st.dataframe(filtered_df, use_container_width=True)

# ----------------------------------------------------------------
# CREATE TAB
# ----------------------------------------------------------------
with tab2:
    st.markdown("## ➕ Add New Record")
    df = load_data()

    with st.form("create_form", clear_on_submit=True):
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        salary = st.number_input("Monthly Salary (₹)", min_value=1000, value=50000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
        eligibility = st.selectbox("EMI Eligibility", ['Eligible', 'High_Risk', 'Not_Eligible'])
        max_emi = st.number_input("Max Monthly EMI (₹)", min_value=1000, value=8000)

        submitted = st.form_submit_button("✅ Add Record")

        if submitted:
            new_id = df['id'].max() + 1 if not df.empty else 1
            new_record = {
                'id': new_id,
                'name': name,
                'age': age,
                'monthly_salary': salary,
                'credit_score': credit_score,
                'emi_eligibility': eligibility,
                'max_monthly_emi': max_emi,
                'created_date': datetime.now().strftime('%Y-%m-%d')
            }
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            save_data(df)
            st.success(f"✅ Record for **{name}** added successfully!")

# ----------------------------------------------------------------
# UPDATE TAB
# ----------------------------------------------------------------
with tab3:
    st.markdown("## ✏️ Update Record")
    df = load_data()

    record_id = st.number_input("Enter Record ID to Update", min_value=1, value=1)
    record = df[df['id'] == record_id]

    if not record.empty:
        st.write("### Current Record:")
        st.dataframe(record)

        with st.form("update_form"):
            new_salary = st.number_input("New Monthly Salary (₹)", value=int(record.iloc[0]['monthly_salary']))
            new_credit = st.number_input("New Credit Score", value=int(record.iloc[0]['credit_score']))
            new_eligibility = st.selectbox("New EMI Eligibility", 
                                           ['Eligible', 'High_Risk', 'Not_Eligible'], 
                                           index=['Eligible', 'High_Risk', 'Not_Eligible'].index(record.iloc[0]['emi_eligibility']))
            submitted = st.form_submit_button("💾 Update Record")

            if submitted:
                df.loc[df['id'] == record_id, ['monthly_salary', 'credit_score', 'emi_eligibility']] = [
                    new_salary, new_credit, new_eligibility
                ]
                save_data(df)
                st.success(f"✅ Record ID {record_id} updated successfully!")
    else:
        st.warning("⚠️ No record found with this ID.")

# ----------------------------------------------------------------
# DELETE TAB
# ----------------------------------------------------------------
with tab4:
    st.markdown("## 🗑️ Delete Record")
    df = load_data()

    record_id = st.number_input("Enter Record ID to Delete", min_value=1, value=1)
    if st.button("🗑️ Delete"):
        if record_id in df['id'].values:
            df = df[df['id'] != record_id]
            save_data(df)
            st.success(f"🗑️ Record ID {record_id} deleted successfully!")
        else:
            st.warning("⚠️ Record not found.")