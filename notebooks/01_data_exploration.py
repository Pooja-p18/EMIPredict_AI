"""
Exploratory Data Analysis (EDA)
Comprehensive analysis of EMI dataset patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class EDAAnalyzer:
    """
    Complete EDA pipeline for financial risk assessment
    """
    
    def __init__(self, data_path='data/processed/train.csv'):
        """Initialize with training data"""
        self.df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(self.df):,} records for analysis")
        self.convert_numeric_columns()
        
    def convert_numeric_columns(self):
        """Convert numeric-like columns to float safely"""
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check if at least 70% of values look numeric
                numeric_like = self.df[col].str.replace(r'[^0-9.\-]', '', regex=True)
                valid_count = numeric_like.str.match(r'^-?\d+(\.\d+)?$').sum()
                ratio = valid_count / len(self.df)
                
                if ratio > 0.7:
                    try:
                        self.df[col] = pd.to_numeric(numeric_like, errors='coerce')
                        print(f"✓ Converted {col}: numeric ({valid_count}/{len(self.df)})")
                    except Exception as e:
                        print(f"⚠️ Could not convert {col}: {e}")
                else:
                    print(f"🔹 Keeping {col} as categorical (non-numeric values detected)")
                
        
    def target_distribution_analysis(self):
        """Analyze distribution of target variables"""
        print("\n" + "="*60)
        print("1. TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        # Classification Target
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('EMI Eligibility Distribution', 'Max Monthly EMI Distribution'),
            specs=[[{'type': 'bar'}, {'type': 'histogram'}]]
        )

        # EMI Eligibility counts
        eligibility_counts = self.df['emi_eligibility'].value_counts()
        fig.add_trace(
            go.Bar(x=eligibility_counts.index, y=eligibility_counts.values,
                   marker_color=['green', 'orange', 'red']),
            row=1, col=1
        )
        
        # Max EMI distribution
        fig.add_trace(
            go.Histogram(x=self.df['max_monthly_emi'], nbinsx=50,
                        marker_color='steelblue'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, title_text="Target Variables Overview")
        fig.write_html('reports/figures/target_distribution.html')
        print("✓ Saved: reports/figures/target_distribution.html")
        
        # Print statistics
        print("\nClassification Target (emi_eligibility):")
        print(f"  Eligible: {eligibility_counts.get('Eligible', 0):,} ({eligibility_counts.get('Eligible', 0)/len(self.df)*100:.1f}%)")
        print(f"  High Risk: {eligibility_counts.get('High_Risk', 0):,} ({eligibility_counts.get('High_Risk', 0)/len(self.df)*100:.1f}%)")
        print(f"  Not Eligible: {eligibility_counts.get('Not_Eligible', 0):,} ({eligibility_counts.get('Not_Eligible', 0)/len(self.df)*100:.1f}%)")
        
        print("\nRegression Target (max_monthly_emi):")
        print(f"  Mean: ₹{self.df['max_monthly_emi'].mean():,.2f}")
        print(f"  Median: ₹{self.df['max_monthly_emi'].median():,.2f}")
        print(f"  Std Dev: ₹{self.df['max_monthly_emi'].std():,.2f}")
        
    def demographic_analysis(self):
        """Analyze demographic patterns"""
        print("\n" + "="*60)
        print("2. DEMOGRAPHIC ANALYSIS")
        print("="*60)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Age Distribution', 'Gender vs Eligibility',
                          'Education Level Impact', 'Marital Status Impact'),
            specs=[[{'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Age distribution
        fig.add_trace(
            go.Histogram(x=self.df['age'], nbinsx=30, marker_color='lightblue'),
            row=1, col=1
        )
        
        # Gender analysis
        gender_elig = pd.crosstab(self.df['gender'], self.df['emi_eligibility'], normalize='index') * 100
        for col in gender_elig.columns:
            fig.add_trace(
                go.Bar(name=col, x=gender_elig.index, y=gender_elig[col]),
                row=1, col=2
            )
        
        # Education analysis
        edu_elig = pd.crosstab(self.df['education'], self.df['emi_eligibility'], normalize='index') * 100
        for col in edu_elig.columns:
            fig.add_trace(
                go.Bar(name=col, x=edu_elig.index, y=edu_elig[col], showlegend=False),
                row=2, col=1
            )
        
        # Marital status
        marital_elig = pd.crosstab(self.df['marital_status'], self.df['emi_eligibility'], normalize='index') * 100
        for col in marital_elig.columns:
            fig.add_trace(
                go.Bar(name=col, x=marital_elig.index, y=marital_elig[col], showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Demographic Patterns Analysis")
        fig.update_xaxes(title_text="Age", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.write_html('reports/figures/demographic_analysis.html')
        print("✓ Saved: reports/figures/demographic_analysis.html")
        
    def show_columns(self):
        """Print all column names"""
        print("\nColumns in the dataset:\n")
        for col in self.df.columns:
            print(col)
    
    def financial_metrics_analysis(self):
        """Analyze financial characteristics"""
        print("\n" + "="*60)
        print("3. FINANCIAL METRICS ANALYSIS")
        print("="*60)
        
        # Key financial metrics by eligibility
        financial_cols = ['monthly_salary', 'credit_score', 'bank_balance', 
                         'current_emi_amount', 'emergency_fund']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=financial_cols + ['Total Monthly Expenses'],
            specs=[[{'type': 'box'}, {'type': 'box'}, {'type': 'box'}],
                   [{'type': 'box'}, {'type': 'box'}, {'type': 'box'}]]
        )
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
        
        for idx, col in enumerate(financial_cols):
            row, col_pos = positions[idx]
            for elig in self.df['emi_eligibility'].unique():
                fig.add_trace(
                    go.Box(y=self.df[self.df['emi_eligibility']==elig][col],
                          name=elig, showlegend=(idx==0)),
                    row=row, col=col_pos
                )
        
        # Calculate total expenses
        expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                       'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
        self.df['total_expenses'] = self.df[expense_cols].sum(axis=1)
        
        for elig in self.df['emi_eligibility'].unique():
            fig.add_trace(
                go.Box(y=self.df[self.df['emi_eligibility']==elig]['total_expenses'],
                      name=elig, showlegend=False),
                row=2, col=3
            )
        
        fig.update_layout(height=800, title_text="Financial Metrics by Eligibility Status")
        fig.write_html('reports/figures/financial_analysis.html')
        print("✓ Saved: reports/figures/financial_analysis.html")
        
        # Print key insights
        print("\nKey Financial Insights:")
        for col in financial_cols:
            print(f"\n{col}:")
            grouped = self.df.groupby('emi_eligibility')[col].agg(['mean', 'median'])
            print(grouped)
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print("\n" + "="*60)
        print("4. CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = self.df[numerical_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, 
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('reports/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: reports/figures/correlation_heatmap.png")
        
        # Find top correlations with target
        target_corr = correlation_matrix['max_monthly_emi'].sort_values(ascending=False)
        print("\nTop 10 Features Correlated with Max Monthly EMI:")
        print(target_corr.head(11)[1:])  # Exclude self-correlation
        
        print("\nBottom 10 Features (Negative Correlation):")
        print(target_corr.tail(10))
        
    def emi_scenario_analysis(self):
        """Analyze different EMI scenarios"""
        print("\n" + "="*60)
        print("5. EMI SCENARIO ANALYSIS")
        print("="*60)
        
        scenario_stats = self.df.groupby('emi_scenario').agg({
            'requested_amount': ['mean', 'median', 'min', 'max'],
            'requested_tenure': ['mean', 'median'],
            'max_monthly_emi': ['mean', 'median'],
            'emi_eligibility': lambda x: (x == 'Eligible').sum() / len(x) * 100
        }).round(2)
        
        print("\nScenario-wise Statistics:")
        print(scenario_stats)
        
        # Visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Requested Amount by Scenario', 
                          'Tenure Distribution',
                          'Eligibility Rate by Scenario',
                          'Max EMI by Scenario'),
            specs=[[{'type': 'box'}, {'type': 'box'}],
                   [{'type': 'bar'}, {'type': 'box'}]]
        )
        
        scenarios = self.df['emi_scenario'].unique()
        
        # Requested amount
        for scenario in scenarios:
            fig.add_trace(
                go.Box(y=self.df[self.df['emi_scenario']==scenario]['requested_amount'],
                      name=scenario, showlegend=False),
                row=1, col=1
            )
        
        # Tenure
        for scenario in scenarios:
            fig.add_trace(
                go.Box(y=self.df[self.df['emi_scenario']==scenario]['requested_tenure'],
                      name=scenario, showlegend=False),
                row=1, col=2
            )
        
        # Eligibility rate
        elig_rate = self.df.groupby('emi_scenario')['emi_eligibility'].apply(
            lambda x: (x == 'Eligible').sum() / len(x) * 100
        )
        fig.add_trace(
            go.Bar(x=elig_rate.index, y=elig_rate.values, marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Max EMI
        for scenario in scenarios:
            fig.add_trace(
                go.Box(y=self.df[self.df['emi_scenario']==scenario]['max_monthly_emi'],
                      name=scenario, showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="EMI Scenario Comparison")
        fig.write_html('reports/figures/scenario_analysis.html')
        print("✓ Saved: reports/figures/scenario_analysis.html")
        
    def employment_analysis(self):
        """Analyze employment patterns and their impact"""
        print("\n" + "="*60)
        print("6. EMPLOYMENT ANALYSIS")
        print("="*60)
        
        # Employment type impact
        emp_analysis = pd.crosstab(
            self.df['employment_type'], 
            self.df['emi_eligibility'],
            normalize='index'
        ) * 100
        
        print("\nEligibility by Employment Type (%):")
        print(emp_analysis)
        
        # Salary analysis by employment
        print("\nAverage Salary by Employment Type:")
        salary_by_emp = self.df.groupby('employment_type')['monthly_salary'].agg(['mean', 'median', 'std'])
        print(salary_by_emp)
        
        # Years of employment impact
        fig = px.box(self.df, x='emi_eligibility', y='years_of_employment',
                    color='emi_eligibility',
                    title='Years of Employment vs Eligibility',
                    labels={'years_of_employment': 'Years of Employment',
                           'emi_eligibility': 'EMI Eligibility'})
        fig.write_html('reports/figures/employment_analysis.html')
        print("✓ Saved: reports/figures/employment_analysis.html")
        
    def risk_factor_analysis(self):
        """Identify key risk factors"""
        print("\n" + "="*60)
        print("7. RISK FACTOR ANALYSIS")
        print("="*60)
        
        # Calculate debt-to-income ratio
        self.df['debt_to_income'] = (self.df['current_emi_amount'] / self.df['monthly_salary'] * 100).clip(0, 100)
        
        # Calculate expense ratio
        expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                       'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
        self.df['expense_ratio'] = (self.df[expense_cols].sum(axis=1) / self.df['monthly_salary'] * 100).clip(0, 200)
        
        # Analyze by eligibility
        print("\nRisk Metrics by Eligibility:")
        risk_metrics = self.df.groupby('emi_eligibility').agg({
            'debt_to_income': ['mean', 'median'],
            'expense_ratio': ['mean', 'median'],
            'credit_score': ['mean', 'median'],
            'bank_balance': ['mean', 'median']
        }).round(2)
        print(risk_metrics)
        
        # Visualize risk factors
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Debt-to-Income Ratio', 'Expense Ratio',
                          'Credit Score Distribution', 'Bank Balance'),
            specs=[[{'type': 'violin'}, {'type': 'violin'}],
                   [{'type': 'violin'}, {'type': 'violin'}]]
        )
        
        for elig in self.df['emi_eligibility'].unique():
            mask = self.df['emi_eligibility'] == elig
            
            fig.add_trace(go.Violin(y=self.df[mask]['debt_to_income'], name=elig), row=1, col=1)
            fig.add_trace(go.Violin(y=self.df[mask]['expense_ratio'], name=elig, showlegend=False), row=1, col=2)
            fig.add_trace(go.Violin(y=self.df[mask]['credit_score'], name=elig, showlegend=False), row=2, col=1)
            fig.add_trace(go.Violin(y=self.df[mask]['bank_balance'], name=elig, showlegend=False), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Risk Factor Distribution by Eligibility")
        fig.write_html('reports/figures/risk_analysis.html')
        print("✓ Saved: reports/figures/risk_analysis.html")
        
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "="*60)
        print("8. GENERATING INSIGHTS REPORT")
        print("="*60)
        
        insights = []
        
        # Eligibility distribution
        elig_dist = self.df['emi_eligibility'].value_counts(normalize=True) * 100
        insights.append(f"## Overall Statistics\n")
        insights.append(f"- **Eligible Applications**: {elig_dist.get('Eligible', 0):.1f}%")
        insights.append(f"- **High Risk Applications**: {elig_dist.get('High_Risk', 0):.1f}%")
        insights.append(f"- **Not Eligible Applications**: {elig_dist.get('Not_Eligible', 0):.1f}%")
        
        # Salary insights
        eligible_salary = self.df[self.df['emi_eligibility']=='Eligible']['monthly_salary'].mean()
        not_eligible_salary = self.df[self.df['emi_eligibility']=='Not_Eligible']['monthly_salary'].mean()
        insights.append(f"\n## Income Analysis")
        insights.append(f"- **Average salary of eligible applicants**: ₹{eligible_salary:,.2f}")
        insights.append(f"- **Average salary of non-eligible applicants**: ₹{not_eligible_salary:,.2f}")
        insights.append(f"- **Salary gap**: {(eligible_salary - not_eligible_salary)/not_eligible_salary*100:.1f}% higher for eligible")
        
        # Credit score insights
        eligible_credit = self.df[self.df['emi_eligibility']=='Eligible']['credit_score'].mean()
        not_eligible_credit = self.df[self.df['emi_eligibility']=='Not_Eligible']['credit_score'].mean()
        insights.append(f"\n## Credit Score Analysis")
        insights.append(f"- **Average credit score of eligible**: {eligible_credit:.0f}")
        insights.append(f"- **Average credit score of non-eligible**: {not_eligible_credit:.0f}")
        insights.append(f"- **Credit score difference**: {eligible_credit - not_eligible_credit:.0f} points")
        
        # EMI scenario insights
        insights.append(f"\n## EMI Scenario Performance")
        for scenario in self.df['emi_scenario'].unique():
            scenario_data = self.df[self.df['emi_scenario'] == scenario]
            elig_rate = (scenario_data['emi_eligibility'] == 'Eligible').sum() / len(scenario_data) * 100
            avg_amount = scenario_data['requested_amount'].mean()
            insights.append(f"- **{scenario}**: {elig_rate:.1f}% eligible | Avg. Amount: ₹{avg_amount:,.0f}")
        
        # Risk factors
        insights.append(f"\n## Key Risk Factors Identified")
        
        # Debt-to-income
        expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                       'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
        self.df['total_obligations'] = self.df['current_emi_amount'] + self.df[expense_cols].sum(axis=1)
        self.df['obligation_ratio'] = (self.df['total_obligations'] / self.df['monthly_salary'] * 100).clip(0, 150)
        
        eligible_obligation = self.df[self.df['emi_eligibility']=='Eligible']['obligation_ratio'].mean()
        not_eligible_obligation = self.df[self.df['emi_eligibility']=='Not_Eligible']['obligation_ratio'].mean()
        
        insights.append(f"- **Eligible applicants** have {eligible_obligation:.1f}% of income as obligations")
        insights.append(f"- **Non-eligible applicants** have {not_eligible_obligation:.1f}% of income as obligations")
        
        # Business recommendations
        insights.append(f"\n## Business Recommendations")
        insights.append(f"1. **Minimum Credit Score Threshold**: Set at {eligible_credit - 50:.0f} for auto-approval consideration")
        insights.append(f"2. **Income Requirement**: Minimum monthly salary of ₹{not_eligible_salary * 1.2:,.0f} recommended")
        insights.append(f"3. **Obligation Ratio**: Keep total obligations below {eligible_obligation * 1.1:.0f}% of monthly income")
        insights.append(f"4. **High-Risk Segment**: Applications with obligation ratio > {not_eligible_obligation * 0.9:.0f}% need manual review")
        
        # Save report
        report_content = '\n'.join(insights)
        with open("reports/insights_report.txt", "w", encoding="utf-8") as f:
            f.write("# EMI Prediction - Exploratory Data Analysis Insights\n\n")
            f.write(report_content)
        
        print("✓ Saved: reports/eda_insights.md")
        print("\n" + "="*60)
        print("KEY INSIGHTS:")
        print("="*60)
        print(report_content)
        
    def run_complete_analysis(self):
        """Execute complete EDA pipeline"""
        print("\n" + "🔍" + "="*58 + "🔍")
        print("  STARTING COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("🔍" + "="*58 + "🔍" + "\n")
        
        self.target_distribution_analysis()
        self.demographic_analysis()
        self.financial_metrics_analysis()
        self.correlation_analysis()
        self.emi_scenario_analysis()
        self.employment_analysis()
        self.risk_factor_analysis()
        self.generate_insights_report()
        
        print("\n" + "="*60)
        print("✅ EXPLORATORY DATA ANALYSIS COMPLETE!")
        print("="*60)
        print("\n📊 Generated Visualizations:")
        print("  - reports/figures/target_distribution.html")
        print("  - reports/figures/demographic_analysis.html")
        print("  - reports/figures/financial_analysis.html")
        print("  - reports/figures/correlation_heatmap.png")
        print("  - reports/figures/scenario_analysis.html")
        print("  - reports/figures/employment_analysis.html")
        print("  - reports/figures/risk_analysis.html")
        print("\n📄 Insights Report:")
        print("  - reports/eda_insights.md")
        print("\n🎉 Ready for next section: Feature Engineering!")
    
    def show_dtypes(self):
        """Print data types of all columns"""
        print("\nColumn Data Types:\n")
        print(self.df.dtypes)

# Main execution
if __name__ == "__main__":
    # Initialize EDA analyzer
    analyzer = EDAAnalyzer('data/processed/train.csv')
    analyzer.show_dtypes()
    # Run complete analysis
    analyzer.run_complete_analysis()