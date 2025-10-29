"""
Model Performance and MLflow Integration Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("📈 Model Performance Dashboard")
st.markdown("### Compare ML models and track experiments")

# Load comparison results
@st.cache_data
def load_results():
    try:
        class_results = pd.read_csv('reports/classification_comparison.csv')
        reg_results = pd.read_csv('reports/regression_comparison.csv')
        return class_results, reg_results
    except:
        return None, None

class_results, reg_results = load_results()

if class_results is None or reg_results is None:
    st.error("⚠️ Model results not found! Please train models first: `python src/model_training.py`")
    st.stop()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Classification", "📉 Regression", "🔬 MLflow"])

with tab1:
    st.markdown("## 🎯 Model Performance Overview")
    
    # Best models summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### 🏆 Best Classification Model")
        best_class = class_results.iloc[0]
        st.metric("Model", best_class['Model'])
        st.metric("Validation Accuracy", f"{best_class['Val_Acc']:.4f}")
        st.metric("F1-Score", f"{best_class['F1_Score']:.4f}")
        st.metric("ROC-AUC", f"{best_class['ROC_AUC']:.4f}")
    
    with col2:
        st.success("### 🏆 Best Regression Model")
        best_reg = reg_results.iloc[0]
        st.metric("Model", best_reg['Model'])
        st.metric("Validation RMSE", f"₹{best_reg['Val_RMSE']:,.2f}")
        st.metric("R² Score", f"{best_reg['R2_Score']:.4f}")
        st.metric("MAPE", f"{best_reg['MAPE']:.2f}%")
    
    # Quick comparison charts
    st.markdown("---")
    st.markdown("### 📊 Quick Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Classification Models - Accuracy")
        fig = px.bar(class_results, x='Model', y='Val_Acc',
                    color='Val_Acc', color_continuous_scale='Greens',
                    text='Val_Acc')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Validation Accuracy")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Regression Models - RMSE")
        fig = px.bar(reg_results, x='Model', y='Val_RMSE',
                    color='Val_RMSE', color_continuous_scale='Reds_r',
                    text='Val_RMSE')
        fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Validation RMSE (₹)")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## 🎯 Classification Models Analysis")
    
    # Display full comparison table
    st.markdown("### 📋 Detailed Metrics Comparison")
    st.dataframe(class_results.style.highlight_max(axis=0, subset=['Val_Acc', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']),
                use_container_width=True)
    
    # Multiple metrics comparison
    st.markdown("---")
    st.markdown("### 📊 Multi-Metric Comparison")
    
    metrics_to_plot = ['Val_Acc', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    
    fig = go.Figure()
    for metric in metrics_to_plot:
        fig.add_trace(go.Bar(
            name=metric,
            x=class_results['Model'],
            y=class_results[metric],
            text=class_results[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        barmode='group',
        xaxis_title="Model",
        yaxis_title="Score",
        legend_title="Metric",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Training vs Validation comparison
    st.markdown("---")
    st.markdown("### 🔄 Training vs Validation Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Training', x=class_results['Model'], y=class_results['Train_Acc']))
        fig.add_trace(go.Bar(name='Validation', x=class_results['Model'], y=class_results['Val_Acc']))
        fig.update_layout(barmode='group', title='Accuracy Comparison',
                         xaxis_title="Model", yaxis_title="Accuracy")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Overfitting analysis
        class_results['Overfitting'] = (class_results['Train_Acc'] - class_results['Val_Acc']) * 100
        fig = px.bar(class_results, x='Model', y='Overfitting',
                    color='Overfitting', color_continuous_scale='Reds',
                    title='Overfitting Analysis (Train - Val Accuracy)')
        fig.update_layout(yaxis_title="Accuracy Gap (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Metric explanations
    st.markdown("---")
    st.markdown("### 📚 Metric Explanations")
    
    with st.expander("ℹ️ Understanding Classification Metrics"):
        st.markdown("""
        **Accuracy**: Percentage of correct predictions
        - Higher is better
        - Target: > 90%
        
        **Precision**: Of all predicted eligible, how many are actually eligible
        - Important to avoid approving bad loans
        - Target: > 0.85
        
        **Recall**: Of all actually eligible, how many we correctly identified
        - Important to not miss good customers
        - Target: > 0.85
        
        **F1-Score**: Harmonic mean of precision and recall
        - Balanced metric
        - Target: > 0.85
        
        **ROC-AUC**: Area under ROC curve
        - Measures model's ability to distinguish classes
        - Target: > 0.90
        """)

with tab3:
    st.markdown("## 📉 Regression Models Analysis")
    
    # Display full comparison table
    st.markdown("### 📋 Detailed Metrics Comparison")
    st.dataframe(reg_results.style.highlight_min(axis=0, subset=['Val_RMSE', 'MAE', 'MAPE'])
                .highlight_max(axis=0, subset=['R2_Score']),
                use_container_width=True)
    
    # Multiple metrics comparison
    st.markdown("---")
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(reg_results, x='Model', y='Val_RMSE',
                    color='Val_RMSE', color_continuous_scale='Reds_r',
                    text='Val_RMSE', title='RMSE Comparison (Lower is Better)')
        fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="RMSE (₹)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(reg_results, x='Model', y='R2_Score',
                    color='R2_Score', color_continuous_scale='Greens',
                    text='R2_Score', title='R² Score (Higher is Better)')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="R² Score")
        st.plotly_chart(fig, use_container_width=True)
    
    # Error metrics comparison
    st.markdown("---")
    st.markdown("### 📏 Error Metrics Comparison")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='RMSE', x=reg_results['Model'], y=reg_results['Val_RMSE'],
                        text=reg_results['Val_RMSE'].round(0)))
    fig.add_trace(go.Bar(name='MAE', x=reg_results['Model'], y=reg_results['MAE'],
                        text=reg_results['MAE'].round(0)))
    
    fig.update_layout(
        barmode='group',
        title='RMSE vs MAE',
        xaxis_title="Model",
        yaxis_title="Error (₹)",
        height=400
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # MAPE comparison
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(reg_results, x='Model', y='MAPE',
                    color='MAPE', color_continuous_scale='Oranges_r',
                    text='MAPE', title='MAPE - Percentage Error')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="MAPE (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Training vs Validation RMSE
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Training', x=reg_results['Model'], y=reg_results['Train_RMSE']))
        fig.add_trace(go.Bar(name='Validation', x=reg_results['Model'], y=reg_results['Val_RMSE']))
        fig.update_layout(barmode='group', title='Training vs Validation RMSE',
                         xaxis_title="Model", yaxis_title="RMSE (₹)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Metric explanations
    st.markdown("---")
    st.markdown("### 📚 Metric Explanations")
    
    with st.expander("ℹ️ Understanding Regression Metrics"):
        st.markdown("""
        **RMSE (Root Mean Squared Error)**: Average prediction error in rupees
        - Lower is better
        - Penalizes large errors more
        - Target: < ₹2,000
        
        **MAE (Mean Absolute Error)**: Median prediction error
        - Lower is better
        - More robust to outliers
        - Target: < ₹1,500
        
        **R² Score**: Proportion of variance explained by model
        - Range: 0 to 1 (1 is perfect)
        - Target: > 0.85
        
        **MAPE (Mean Absolute Percentage Error)**: Average % error
        - Easier to interpret for business
        - Target: < 10%
        """)

with tab4:
    st.markdown("## 🔬 MLflow Experiment Tracking")
    
    st.info("""
    ### 🚀 Access MLflow Dashboard
    
    MLflow provides comprehensive experiment tracking with:
    - All model parameters and hyperparameters
    - Training metrics and artifacts
    - Model versioning and registry
    - Experiment comparison tools
    
    **To view MLflow dashboard:**
    1. Open terminal in project directory
    2. Run: `mlflow ui`
    3. Open browser at: `http://localhost:5000`
    """)
    
    # MLflow features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ MLflow Features Used")
        st.markdown("""
        - ✅ Experiment tracking for all 10 models
        - ✅ Parameter logging
        - ✅ Metric logging (accuracy, RMSE, etc.)
        - ✅ Model artifact storage
        - ✅ Model registry for best models
        - ✅ Run comparison dashboard
        """)
    
    with col2:
        st.markdown("### 📊 Logged Information")
        st.markdown("""
        **For Each Model:**
        - Model hyperparameters
        - Training metrics
        - Validation metrics
        - Model artifacts (.pkl files)
        - Training timestamp
        - Model version
        """)
    
    # Screenshot placeholder
    st.markdown("---")
    st.markdown("### 🖼️ MLflow Dashboard Preview")
    st.info("Run `mlflow ui` in terminal to access the full dashboard with interactive charts and detailed comparisons.")
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📝 Using MLflow")
    
    with st.expander("💡 Quick Start Guide"):
        st.code("""
# Start MLflow UI
mlflow ui

# Access in browser
http://localhost:5000

# View experiment runs
- Click on "EMI_Prediction" experiment
- Compare runs side-by-side
- Sort by metrics
- Download model artifacts

# Query runs programmatically
import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=['0'])
        """, language='bash')

# Model download section
st.markdown("---")
st.markdown("### 📥 Download Model Results")

col1, col2 = st.columns(2)

with col1:
    csv = class_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Classification Results",
        data=csv,
        file_name="classification_comparison.csv",
        mime="text/csv"
    )

with col2:
    csv = reg_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Regression Results",
        data=csv,
        file_name="regression_comparison.csv",
        mime="text/csv"
    )