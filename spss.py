import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, pearsonr
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Statistical Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# Helper functions
def generate_sample_data(analysis_type, n_samples=100):
    """Generate sample data based on analysis type"""
    np.random.seed(42)
    
    if analysis_type == "Frequency Analysis":
        data = {
            'Category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age_Group': np.random.choice(['18-25', '26-35', '36-45', '46+'], n_samples)
        }
    elif analysis_type == "Chi-Square Test":
        data = {
            'Treatment': np.random.choice(['Control', 'Treatment'], n_samples),
            'Outcome': np.random.choice(['Success', 'Failure'], n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples)
        }
    elif analysis_type in ["T-Test", "F-Test"]:
        data = {
            'Group': np.random.choice(['Group1', 'Group2'], n_samples),
            'Score': np.random.normal(50, 10, n_samples),
            'Age': np.random.randint(18, 65, n_samples)
        }
    elif analysis_type in ["ANOVA", "MANOVA"]:
        data = {
            'Group': np.random.choice(['A', 'B', 'C'], n_samples),
            'Score1': np.random.normal(50, 10, n_samples),
            'Score2': np.random.normal(45, 12, n_samples),
            'Score3': np.random.normal(55, 8, n_samples)
        }
    else:  # For other analyses
        data = {
            'Variable1': np.random.normal(50, 10, n_samples),
            'Variable2': np.random.normal(45, 12, n_samples),
            'Variable3': np.random.normal(55, 8, n_samples),
            'Variable4': np.random.normal(60, 15, n_samples),
            'Group': np.random.choice(['A', 'B', 'C'], n_samples)
        }
    
    return pd.DataFrame(data)

def display_descriptive_stats(df):
    """Display descriptive statistics"""
    st.markdown('<div class="sub-header">📈 Descriptive Statistics</div>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe())
    else:
        st.warning("No numeric columns found for descriptive statistics.")

def display_demographic_analysis(df):
    """Display demographic analysis with visualizations"""
    st.markdown('<div class="sub-header">👥 Demographic Analysis</div>', unsafe_allow_html=True)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                # Frequency table
                freq_table = df[col].value_counts().reset_index()
                freq_table.columns = [col, 'Count']
                freq_table['Percentage'] = (freq_table['Count'] / len(df) * 100).round(2)
                st.write(f"**{col} Distribution:**")
                st.dataframe(freq_table)
            
            with col2:
                # Visualization
                fig = px.bar(freq_table, x=col, y='Count', 
                           title=f'{col} Distribution',
                           text='Count')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical variables found for demographic analysis.")

def get_chatgpt_interpretation(analysis_type, results_text):
    """Get ChatGPT interpretation of statistical results"""
    
    # Check if API key is provided
    if not st.session_state.openai_api_key:
        return """
**⚠️ OpenAI API Key Required**

To get AI-powered statistical interpretation, please:
1. Enter your OpenAI API key in the sidebar
2. Click the AI Statistical Analysis button again

**Sample Interpretation (Demo Mode):**
This analysis provides valuable insights into your data. The statistical results indicate significant patterns that warrant further investigation. Consider the practical significance alongside statistical significance when making decisions based on these findings.

**To get full AI interpretation:**
- Obtain an API key from OpenAI (https://platform.openai.com/api-keys)
- Enter it in the sidebar under "AI Settings"
- Enjoy comprehensive statistical interpretations!
        """
    
    try:
        # Try to import openai
        try:
            import openai
        except ImportError:
            return """
**📦 Installation Required**

To use AI interpretation, please install the OpenAI package:

```bash
pip install openai>=1.0.0
```

Then restart the application and enter your API key in the sidebar.
            """
        
        prompt = f"""
        As a professional statistician, please analyze and interpret the following {analysis_type} results:

        {results_text}

        Please provide a comprehensive interpretation including:
        1. Summary of key findings
        2. Statistical significance and meaning
        3. Effect size and practical significance
        4. Assumptions and limitations
        5. Practical implications and recommendations
        6. Suggestions for further analysis if applicable

        Format your response in a clear, professional manner suitable for both academic and business contexts.
        Use markdown formatting for better readability.
        """
        
        # Try new OpenAI API (v1.0+)
        try:
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system", 
                    "content": "You are an expert statistician providing professional statistical analysis interpretations."
                }, {
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except AttributeError:
            # Fallback to old OpenAI API (v0.x)
            openai.api_key = st.session_state.openai_api_key
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system", 
                    "content": "You are an expert statistician providing professional statistical analysis interpretations."
                }, {
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        
    except Exception as e:
        error_message = str(e).lower()
        
        # Check for specific error types based on error message content
        if "invalid api key" in error_message or "authentication" in error_message or "unauthorized" in error_message or "incorrect api key" in error_message:
            return """
**🔑 Authentication Error**

The provided API key is invalid. Please check:
1. Your API key is correct and complete
2. Your API key has sufficient credits
3. Your API key has the necessary permissions
4. Try regenerating your API key from OpenAI dashboard

Please update your API key in the sidebar.
            """
        
        elif "rate limit" in error_message or "quota" in error_message or "billing" in error_message:
            return """
**⏱️ Rate Limit or Billing Error**

You've exceeded your API rate limit or have billing issues. Please:
1. Wait a moment and try again
2. Check your OpenAI usage limits
3. Verify your billing information on OpenAI dashboard
4. Consider upgrading your OpenAI plan if needed
            """
        
        elif "connection" in error_message or "network" in error_message or "timeout" in error_message:
            return """
**🌐 Connection Error**

Unable to connect to OpenAI API. Please check:
1. Your internet connection
2. OpenAI service status
3. Try again in a moment
4. Check if your firewall is blocking the connection
            """
        
        elif "model" in error_message and "not found" in error_message:
            return """
**🤖 Model Error**

The requested model is not available. This might be because:
1. Your API key doesn't have access to GPT-3.5-turbo
2. The model name has changed
3. Try using a different model

Please check your OpenAI dashboard for available models.
            """
        
        else:
            return f"""
**❌ OpenAI API Error**

Error details: {str(e)}

**Troubleshooting Steps:**
1. **Update OpenAI library**: Run `pip install openai --upgrade`
2. **Check API key**: Ensure it's valid and has credits
3. **Try again**: Sometimes it's a temporary issue

**Demo Interpretation for {analysis_type}:**

Your {analysis_type} analysis has been completed successfully. The results provide valuable insights into your data patterns and relationships. The statistical findings suggest significant patterns that warrant careful consideration in your decision-making process.

**Key Points to Consider:**
- Review the statistical significance levels (p-values)
- Consider practical significance alongside statistical significance
- Examine effect sizes to understand the magnitude of relationships
- Validate findings with domain knowledge and additional data if available

**Recommendations:**
- Document your methodology and findings
- Consider additional analyses to explore relationships further
- Use visualizations to communicate results effectively
- Implement findings in your decision-making process with appropriate caution

**Note:** To resolve the API issue, try updating the OpenAI library: `pip install openai --upgrade`
            """

# Main application
def main():
    st.markdown('<div class="main-header">📊 Statistical Analysis Dashboard for SumSET Lecture by Sky Kang</div>', unsafe_allow_html=True)
    
    # Sidebar for analysis selection
    st.sidebar.title("Analysis Methods")
    analysis_options = [
        "Frequency Analysis",
        "Chi-Square Test", 
        "T-Test",
        "F-Test",
        "ANOVA",
        "MANOVA",
        "Correlation Analysis",
        "Reliability Analysis",
        "Factor Analysis",
        "Principal Component Analysis",
        "Cluster Analysis",
        "Regression Analysis",
        "Structural Equation Modeling",
        "AHP Analysis"
    ]
    
    selected_analysis = st.sidebar.selectbox("Select Analysis Type", analysis_options)
    
    # AI Settings Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Settings")
    
    # API Key input
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key:", 
        type="password",
        value=st.session_state.openai_api_key,
        help="Enter your OpenAI API key to enable AI-powered statistical interpretations"
    )
    
    # Update session state when API key changes
    if api_key_input != st.session_state.openai_api_key:
        st.session_state.openai_api_key = api_key_input
    
    # API Key status
    if st.session_state.openai_api_key:
        st.sidebar.success("✅ API Key provided")
        st.sidebar.info("AI interpretation is enabled!")
    else:
        st.sidebar.warning("⚠️ No API Key")
        st.sidebar.info("Enter API key to enable AI interpretation")
    
    # API Key help
    with st.sidebar.expander("How to get OpenAI API Key"):
        st.write("""
        1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
        2. Sign up or log in to your account
        3. Create a new API key
        4. Copy and paste it above
        
        **Note:** You need credits in your OpenAI account to use the API.
        """)
    
    st.sidebar.markdown("---")
    
    # Data input section
    st.markdown('<div class="sub-header">📁 Data Input</div>', unsafe_allow_html=True)
    
    data_input_method = st.radio(
        "Choose data input method:",
        ["Upload CSV", "Manual Data Entry", "Generate Sample Data"]
    )
    
    data = None
    
    if data_input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
    
    elif data_input_method == "Manual Data Entry":
        st.write("Enter your data (comma-separated values):")
        cols = st.text_input("Column names (comma-separated):", "Variable1,Variable2,Variable3")
        rows = st.text_area("Data rows (one row per line, comma-separated):", 
                           "1,2,3\n4,5,6\n7,8,9")
        
        if st.button("Create DataFrame"):
            try:
                col_names = [col.strip() for col in cols.split(',')]
                data_rows = []
                for row in rows.strip().split('\n'):
                    data_rows.append([val.strip() for val in row.split(',')])
                
                data = pd.DataFrame(data_rows, columns=col_names)
                # Try to convert to numeric where possible
                for col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col])
                    except:
                        pass
                st.success("DataFrame created successfully!")
            except Exception as e:
                st.error(f"Error creating DataFrame: {str(e)}")
    
    elif data_input_method == "Generate Sample Data":
        if st.button("Generate Sample Data"):
            data = generate_sample_data(selected_analysis)
            st.success("Sample data generated!")
    
    # Display data preview
    if data is not None:
        st.markdown('<div class="sub-header">📋 Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(data.head())
        st.write(f"Data shape: {data.shape}")
        
        # Variable selection
        st.markdown('<div class="sub-header">🔧 Variable Selection</div>', unsafe_allow_html=True)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = data.columns.tolist()
        
        # Variable selection based on analysis type
        dependent_var = None
        independent_var = None
        dependent_vars = []
        selected_vars = []
        independent_vars = []
        
        if selected_analysis in ["T-Test", "F-Test"]:
            dependent_var = st.selectbox("Select Dependent Variable:", numeric_cols) if numeric_cols else None
            independent_var = st.selectbox("Select Independent Variable (Group):", categorical_cols) if categorical_cols else None
        elif selected_analysis in ["ANOVA", "MANOVA"]:
            dependent_vars = st.multiselect("Select Dependent Variables:", numeric_cols)
            independent_var = st.selectbox("Select Independent Variable (Group):", categorical_cols) if categorical_cols else None
        elif selected_analysis == "Correlation Analysis":
            selected_vars = st.multiselect("Select Variables for Correlation:", numeric_cols)
        elif selected_analysis == "Regression Analysis":
            dependent_var = st.selectbox("Select Dependent Variable:", numeric_cols) if numeric_cols else None
            independent_vars = st.multiselect("Select Independent Variables:", numeric_cols)
        elif selected_analysis in ["Reliability Analysis", "Factor Analysis", "Principal Component Analysis", "Cluster Analysis"]:
            selected_vars = st.multiselect("Select Variables:", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))] if numeric_cols else [])
        else:
            selected_vars = st.multiselect("Select Variables:", all_cols, default=all_cols[:min(3, len(all_cols))] if all_cols else [])
        
        # Analysis execution
        if st.button("🔍 Run Statistical Analysis"):
            with st.spinner("Running analysis..."):
                
                # Display descriptive statistics
                display_descriptive_stats(data)
                
                # Display demographic analysis
                display_demographic_analysis(data)
                
                # Run specific analysis
                st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
                
                if selected_analysis == "Frequency Analysis":
                    run_frequency_analysis(data, selected_vars)
                elif selected_analysis == "Chi-Square Test":
                    run_chi_square_test(data, categorical_cols)
                elif selected_analysis == "T-Test":
                    run_t_test(data, dependent_var, independent_var)
                elif selected_analysis == "F-Test":
                    run_f_test(data, dependent_var, independent_var)
                elif selected_analysis == "ANOVA":
                    run_anova_analysis(data, dependent_vars, independent_var)
                elif selected_analysis == "MANOVA":
                    run_manova_analysis(data, dependent_vars, independent_var)
                elif selected_analysis == "Correlation Analysis":
                    run_correlation_analysis(data, selected_vars)
                elif selected_analysis == "Reliability Analysis":
                    run_reliability_analysis(data, selected_vars)
                elif selected_analysis == "Factor Analysis":
                    run_factor_analysis(data, numeric_cols)
                elif selected_analysis == "Regression Analysis":
                    run_regression_analysis(data, dependent_var, independent_vars)
                elif selected_analysis == "Principal Component Analysis":
                    run_pca_analysis(data, numeric_cols)
                elif selected_analysis == "Cluster Analysis":
                    run_cluster_analysis(data, numeric_cols)
                else:
                    st.info(f"{selected_analysis} analysis is under development. Please select another analysis type.")

def run_frequency_analysis(data, selected_vars):
    """Run frequency analysis"""
    results_text = "FREQUENCY ANALYSIS RESULTS:\n\n"
    
    for var in selected_vars:
        if data[var].dtype == 'object' or data[var].dtype.name == 'category':
            st.write(f"**Frequency Analysis for {var}:**")
            
            freq_table = data[var].value_counts().reset_index()
            freq_table.columns = [var, 'Frequency']
            freq_table['Percentage'] = (freq_table['Frequency'] / len(data) * 100).round(2)
            
            # Add to results text for AI interpretation
            results_text += f"Variable: {var}\n"
            results_text += f"Total observations: {len(data)}\n"
            for _, row in freq_table.iterrows():
                results_text += f"  {row[var]}: {row['Frequency']} cases ({row['Percentage']}%)\n"
            results_text += "\n"
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(freq_table)
            with col2:
                fig = px.pie(freq_table, values='Frequency', names=var, 
                           title=f'{var} Distribution')
                st.plotly_chart(fig, use_container_width=True)
    
    # AI Analysis - Automatic
    st.markdown("### 🧠 AI Statistical Interpretation")
    with st.spinner("Generating AI interpretation..."):
        ai_interpretation = get_chatgpt_interpretation("Frequency Analysis", results_text)
        st.markdown(ai_interpretation)

def run_chi_square_test(data, categorical_cols):
    """Run chi-square test"""
    if len(categorical_cols) >= 2:
        var1 = categorical_cols[0]
        var2 = categorical_cols[1]
        
        # Create contingency table
        contingency_table = pd.crosstab(data[var1], data[var2])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Prepare results text for AI interpretation
        results_text = f"""CHI-SQUARE TEST RESULTS:

Variables tested: {var1} vs {var2}
Sample size: {len(data)}

Contingency Table:
{contingency_table.to_string()}

Test Statistics:
- Chi-square statistic: {chi2:.4f}
- p-value: {p_value:.4f}
- Degrees of freedom: {dof}
- Critical value (α=0.05): {stats.chi2.ppf(0.95, dof):.4f}

Expected frequencies:
{pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns).to_string()}

Statistical conclusion: {'Significant association' if p_value < 0.05 else 'No significant association'} (α=0.05)
"""
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Contingency Table:**")
            st.dataframe(contingency_table)
        with col2:
            st.write("**Test Results:**")
            st.write(f"Chi-square statistic: {chi2:.4f}")
            st.write(f"p-value: {p_value:.4f}")
            st.write(f"Degrees of freedom: {dof}")
            
            if p_value < 0.05:
                st.success("Result: Significant association (p < 0.05)")
            else:
                st.info("Result: No significant association (p ≥ 0.05)")
        
        # Visualization
        fig = px.imshow(contingency_table, text_auto=True, aspect="auto",
                       title=f"Contingency Table Heatmap: {var1} vs {var2}")
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Analysis - Automatic
        st.markdown("### 🧠 AI Statistical Interpretation")
        with st.spinner("Generating AI interpretation..."):
            ai_interpretation = get_chatgpt_interpretation("Chi-Square Test", results_text)
            st.markdown(ai_interpretation)

def run_t_test(data, dependent_var, independent_var):
    """Run t-test"""
    groups = data[independent_var].unique()
    if len(groups) == 2:
        group1_data = data[data[independent_var] == groups[0]][dependent_var]
        group2_data = data[data[independent_var] == groups[1]][dependent_var]
        
        # Perform t-test
        t_stat, p_value = ttest_ind(group1_data, group2_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                             (len(group2_data) - 1) * group2_data.var()) / 
                            (len(group1_data) + len(group2_data) - 2))
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
        
        # Prepare results text for AI interpretation
        results_text = f"""T-TEST RESULTS:

Variables:
- Dependent variable: {dependent_var}
- Independent variable: {independent_var}

Group Statistics:
{groups[0]}: n={len(group1_data)}, Mean={group1_data.mean():.3f}, SD={group1_data.std():.3f}
{groups[1]}: n={len(group2_data)}, Mean={group2_data.mean():.3f}, SD={group2_data.std():.3f}

Test Statistics:
- t-statistic: {t_stat:.4f}
- p-value: {p_value:.4f}
- Degrees of freedom: {len(group1_data) + len(group2_data) - 2}
- Mean difference: {group1_data.mean() - group2_data.mean():.3f}
- Cohen's d (effect size): {cohens_d:.3f}

Effect Size Interpretation:
- Small: d = 0.2, Medium: d = 0.5, Large: d = 0.8
- Current effect size: {'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'}

Statistical conclusion: {'Significant difference' if p_value < 0.05 else 'No significant difference'} (α=0.05)
"""
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Group Statistics:**")
            stats_df = pd.DataFrame({
                'Group': groups,
                'Mean': [group1_data.mean(), group2_data.mean()],
                'Std Dev': [group1_data.std(), group2_data.std()],
                'Count': [len(group1_data), len(group2_data)]
            })
            st.dataframe(stats_df)
            
            st.write("**Effect Size:**")
            st.write(f"Cohen's d: {cohens_d:.4f}")
        
        with col2:
            st.write("**T-Test Results:**")
            st.write(f"t-statistic: {t_stat:.4f}")
            st.write(f"p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                st.success("Result: Significant difference (p < 0.05)")
            else:
                st.info("Result: No significant difference (p ≥ 0.05)")
        
        # Visualization
        fig = px.box(data, x=independent_var, y=dependent_var,
                    title=f'{dependent_var} by {independent_var}')
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Analysis - Automatic
        st.markdown("### 🧠 AI Statistical Interpretation")
        with st.spinner("Generating AI interpretation..."):
            ai_interpretation = get_chatgpt_interpretation("T-Test", results_text)
            st.markdown(ai_interpretation)

def run_f_test(data, dependent_var, independent_var):
    """Run F-test for equality of variances"""
    groups = data[independent_var].unique()
    if len(groups) == 2:
        group1_data = data[data[independent_var] == groups[0]][dependent_var].dropna()
        group2_data = data[data[independent_var] == groups[1]][dependent_var].dropna()
        
        # Calculate F-statistic (ratio of variances)
        var1 = group1_data.var(ddof=1)
        var2 = group2_data.var(ddof=1)
        f_stat = var1 / var2 if var2 != 0 else float('inf')
        
        # Degrees of freedom
        df1 = len(group1_data) - 1
        df2 = len(group2_data) - 1
        
        # Calculate p-value using F-distribution
        from scipy.stats import f
        p_value = 2 * min(f.cdf(f_stat, df1, df2), 1 - f.cdf(f_stat, df1, df2))
        
        # Prepare results text for AI interpretation
        results_text = f"""F-TEST FOR EQUALITY OF VARIANCES:

Variables:
- Dependent variable: {dependent_var}
- Independent variable (groups): {independent_var}

Group Statistics:
{groups[0]}: n={len(group1_data)}, Variance={var1:.4f}, SD={group1_data.std():.4f}
{groups[1]}: n={len(group2_data)}, Variance={var2:.4f}, SD={group2_data.std():.4f}

Test Statistics:
- F-statistic: {f_stat:.4f}
- p-value: {p_value:.4f}
- Degrees of freedom: ({df1}, {df2})
- Variance ratio: {var1/var2 if var2 != 0 else 'undefined':.4f}

Statistical conclusion: {'Variances are significantly different' if p_value < 0.05 else 'No significant difference in variances'} (α=0.05)
"""
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**F-Test Results:**")
            st.write(f"F-statistic: {f_stat:.4f}")
            st.write(f"p-value: {p_value:.4f}")
            st.write(f"Degrees of freedom: ({df1}, {df2})")
            
            # Variance comparison
            variance_df = pd.DataFrame({
                'Group': groups,
                'Count': [len(group1_data), len(group2_data)],
                'Variance': [var1, var2],
                'Std Dev': [group1_data.std(), group2_data.std()]
            })
            st.dataframe(variance_df.round(4))
            
            if p_value < 0.05:
                st.warning("⚠️ Variances are significantly different (p < 0.05)")
                st.info("Consider using Welch's t-test instead of Student's t-test")
            else:
                st.success("✅ No significant difference in variances (p ≥ 0.05)")
        
        with col2:
            # Box plot to visualize variance differences
            fig = px.box(data, x=independent_var, y=dependent_var,
                        title=f'Variance Comparison: {dependent_var} by {independent_var}')
            st.plotly_chart(fig, use_container_width=True)
        
        # AI Analysis - Automatic
        st.markdown("### 🧠 AI Statistical Interpretation")
        with st.spinner("Generating AI interpretation..."):
            ai_interpretation = get_chatgpt_interpretation("F-Test", results_text)
            st.markdown(ai_interpretation)

def run_anova_analysis(data, dependent_vars, independent_var):
    """Run ANOVA analysis"""
    if len(dependent_vars) >= 1 and independent_var:
        results_text = f"""ANOVA ANALYSIS RESULTS:

Independent Variable (Factor): {independent_var}
Dependent Variable(s): {', '.join(dependent_vars)}
Sample size: {len(data)}

"""
        
        groups = data[independent_var].unique()
        results_text += f"Groups: {', '.join(map(str, groups))} (n={len(groups)})\n\n"
        
        for dep_var in dependent_vars:
            st.write(f"**ANOVA Results for {dep_var}:**")
            
            # Prepare data for ANOVA
            group_data = []
            group_stats = []
            
            for group in groups:
                group_values = data[data[independent_var] == group][dep_var].dropna()
                group_data.append(group_values)
                group_stats.append({
                    'Group': group,
                    'Count': len(group_values),
                    'Mean': group_values.mean(),
                    'Std Dev': group_values.std(),
                    'Min': group_values.min(),
                    'Max': group_values.max()
                })
            
            # Perform one-way ANOVA
            f_stat, p_value = f_oneway(*group_data)
            
            # Calculate effect size (eta-squared)
            # SS_between = sum of squared differences between group means and grand mean
            grand_mean = data[dep_var].mean()
            ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in group_data)
            ss_total = sum((data[dep_var] - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Add to results text
            results_text += f"""
ANOVA for {dep_var}:
- F-statistic: {f_stat:.4f}
- p-value: {p_value:.4f}
- Effect size (η²): {eta_squared:.4f}
- Degrees of freedom: Between groups = {len(groups)-1}, Within groups = {len(data)-len(groups)}

Group Statistics:
"""
            for stat in group_stats:
                results_text += f"  {stat['Group']}: n={stat['Count']}, M={stat['Mean']:.3f}, SD={stat['Std Dev']:.3f}\n"
            
            results_text += f"Statistical conclusion: {'Significant group differences' if p_value < 0.05 else 'No significant group differences'} (α=0.05)\n"
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Group statistics table
                stats_df = pd.DataFrame(group_stats)
                st.dataframe(stats_df.round(3))
                
                st.write("**ANOVA Results:**")
                st.write(f"F-statistic: {f_stat:.4f}")
                st.write(f"p-value: {p_value:.4f}")
                st.write(f"Effect size (η²): {eta_squared:.4f}")
                
                if p_value < 0.05:
                    st.success("Result: Significant group differences (p < 0.05)")
                    
                    # Post-hoc analysis suggestion
                    if len(groups) > 2:
                        st.info("💡 Consider post-hoc tests (e.g., Tukey's HSD) to identify which groups differ significantly.")
                else:
                    st.info("Result: No significant group differences (p ≥ 0.05)")
            
            with col2:
                # Box plot
                fig = px.box(data, x=independent_var, y=dep_var,
                           title=f'{dep_var} by {independent_var}')
                st.plotly_chart(fig, use_container_width=True)
            
            # Violin plot for better distribution visualization
            if len(groups) <= 6:  # Only show for reasonable number of groups
                fig_violin = px.violin(data, x=independent_var, y=dep_var,
                                     title=f'{dep_var} Distribution by {independent_var}',
                                     box=True)
                st.plotly_chart(fig_violin, use_container_width=True)
            
            # ANOVA assumptions checking
            if st.checkbox(f"Check ANOVA Assumptions for {dep_var}", key=f"assumptions_{dep_var}"):
                st.write("**ANOVA Assumptions Check:**")
                
                # Normality test for each group (Shapiro-Wilk)
                from scipy.stats import shapiro
                
                assumption_results = "ANOVA Assumptions Check:\n\n"
                
                st.write("1. **Normality Test (Shapiro-Wilk) for each group:**")
                normality_results = []
                for i, group in enumerate(groups):
                    group_values = group_data[i]
                    if len(group_values) >= 3:  # Minimum for Shapiro-Wilk
                        stat, p = shapiro(group_values)
                        normality_results.append({
                            'Group': group,
                            'Statistic': stat,
                            'p-value': p,
                            'Normal': 'Yes' if p > 0.05 else 'No'
                        })
                        assumption_results += f"  {group}: W={stat:.4f}, p={p:.4f} ({'Normal' if p > 0.05 else 'Not Normal'})\n"
                
                norm_df = pd.DataFrame(normality_results)
                st.dataframe(norm_df.round(4))
                
                # Homogeneity of variance (Levene's test)
                from scipy.stats import levene
                lev_stat, lev_p = levene(*group_data)
                
                st.write("2. **Homogeneity of Variance (Levene's Test):**")
                st.write(f"Levene's statistic: {lev_stat:.4f}")
                st.write(f"p-value: {lev_p:.4f}")
                
                assumption_results += f"\nHomogeneity of Variance (Levene's Test):\n"
                assumption_results += f"  Statistic: {lev_stat:.4f}, p-value: {lev_p:.4f}\n"
                assumption_results += f"  Equal variances: {'Yes' if lev_p > 0.05 else 'No'}\n"
                
                if lev_p > 0.05:
                    st.success("✅ Equal variances assumption met (p > 0.05)")
                else:
                    st.warning("⚠️ Equal variances assumption violated (p ≤ 0.05)")
                    st.info("Consider Welch's ANOVA or transformation of data")
                
                results_text += f"\n{assumption_results}\n"
        
        # AI Analysis - Automatic
        st.markdown("### 🧠 AI Statistical Interpretation")
        with st.spinner("Generating AI interpretation..."):
            ai_interpretation = get_chatgpt_interpretation("ANOVA", results_text)
            st.markdown(ai_interpretation)

def run_manova_analysis(data, dependent_vars, independent_var):
    """Run MANOVA analysis"""
    if len(dependent_vars) >= 2 and independent_var:
        st.write("**MANOVA (Multivariate Analysis of Variance):**")
        
        # Prepare data
        groups = data[independent_var].unique()
        
        # For demonstration, we'll show the concept and use separate ANOVAs
        st.info("💡 MANOVA tests multiple dependent variables simultaneously")
        
        results_text = f"""MANOVA ANALYSIS RESULTS:

Independent Variable (Factor): {independent_var}
Dependent Variables: {', '.join(dependent_vars)}
Groups: {', '.join(map(str, groups))}
Sample size: {len(data)}

Individual ANOVA Results:
"""
        
        # Run individual ANOVAs for each DV
        multivariate_results = []
        
        for dep_var in dependent_vars:
            # Prepare data for ANOVA
            group_data = []
            for group in groups:
                group_values = data[data[independent_var] == group][dep_var].dropna()
                group_data.append(group_values)
            
            # Perform ANOVA
            f_stat, p_value = f_oneway(*group_data)
            
            multivariate_results.append({
                'Dependent Variable': dep_var,
                'F-statistic': f_stat,
                'p-value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
            
            results_text += f"  {dep_var}: F={f_stat:.4f}, p={p_value:.4f}\n"
        
        # Display results table
        results_df = pd.DataFrame(multivariate_results)
        st.dataframe(results_df.round(4))
        
        # Correlation matrix of dependent variables
        st.write("**Correlation Matrix of Dependent Variables:**")
        corr_matrix = data[dependent_vars].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Correlation Matrix of DVs",
                       color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
        
        # Multivariate visualization
        if len(dependent_vars) >= 2:
            fig_scatter = px.scatter_matrix(data, dimensions=dependent_vars, 
                                          color=independent_var,
                                          title="Multivariate Scatter Plot Matrix")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        results_text += f"""
Dependent Variable Correlations:
{corr_matrix.round(3).to_string()}

MANOVA Interpretation:
- Tests whether groups differ on the combination of dependent variables
- More powerful than separate ANOVAs when DVs are correlated
- Controls for Type I error inflation across multiple tests
"""
        
        # AI Analysis Button
        ai_button_text = "🤖 AI Statistical Analysis" if st.session_state.openai_api_key else "🤖 AI Analysis (Demo Mode)"
        ai_button_help = "Get AI-powered interpretation" if st.session_state.openai_api_key else "Demo mode - enter API key for full AI analysis"
        
        if st.button(ai_button_text, key="manova_ai", help=ai_button_help):
            with st.spinner("Generating AI interpretation..."):
                ai_interpretation = get_chatgpt_interpretation("MANOVA", results_text)
                st.markdown("### 🧠 AI Statistical Interpretation")
                st.markdown(ai_interpretation)

def run_correlation_analysis(data, selected_vars):
    """Run correlation analysis"""
    if len(selected_vars) >= 2:
        corr_data = data[selected_vars].select_dtypes(include=[np.number])
        
        if len(corr_data.columns) >= 2:
            corr_matrix = corr_data.corr()
            
            # Calculate p-values for correlations
            n = len(corr_data)
            p_values = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
            
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    if i != j:
                        r, p = pearsonr(corr_data.iloc[:, i], corr_data.iloc[:, j])
                        p_values.iloc[i, j] = p
                    else:
                        p_values.iloc[i, j] = 0
            
            # Prepare results text for AI interpretation
            results_text = f"""CORRELATION ANALYSIS RESULTS:

Variables analyzed: {', '.join(corr_data.columns)}
Sample size: {n}

Correlation Matrix:
{corr_matrix.round(3).to_string()}

P-values Matrix:
{p_values.round(3).to_string()}

Significant Correlations (p < 0.05):
"""
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    var1, var2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    r_val = corr_matrix.iloc[i, j]
                    p_val = float(p_values.iloc[i, j])
                    
                    if p_val < 0.05:
                        strength = "Strong" if abs(r_val) > 0.7 else "Moderate" if abs(r_val) > 0.3 else "Weak"
                        direction = "positive" if r_val > 0 else "negative"
                        results_text += f"- {var1} vs {var2}: r = {r_val:.3f}, p = {p_val:.3f} ({strength} {direction})\n"
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Correlation Matrix:**")
                st.dataframe(corr_matrix.round(3))
                
                st.write("**P-values:**")
                st.dataframe(p_values.round(3))
            
            with col2:
                # Heatmap
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               title="Correlation Matrix Heatmap",
                               color_continuous_scale="RdBu_r",
                               zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot matrix
            if len(corr_data.columns) <= 5:  # Limit for readability
                fig = px.scatter_matrix(corr_data, title="Scatter Plot Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            # AI Analysis Button
            ai_button_text = "🤖 AI Statistical Analysis" if st.session_state.openai_api_key else "🤖 AI Analysis (Demo Mode)"
            ai_button_help = "Get AI-powered interpretation" if st.session_state.openai_api_key else "Demo mode - enter API key for full AI analysis"
            
            if st.button(ai_button_text, key="corr_ai", help=ai_button_help):
                with st.spinner("Generating AI interpretation..."):
                    ai_interpretation = get_chatgpt_interpretation("Correlation Analysis", results_text)
                    st.markdown("### 🧠 AI Statistical Interpretation")
                    st.markdown(ai_interpretation)

def run_reliability_analysis(data, selected_vars):
    """Run reliability analysis (Cronbach's Alpha)"""
    if len(selected_vars) >= 2:
        st.write("**Reliability Analysis (Cronbach's Alpha):**")
        
        # Select only numeric variables
        numeric_data = data[selected_vars].select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_data.columns) >= 2:
            # Calculate Cronbach's Alpha
            def cronbach_alpha(df):
                # Calculate the correlation matrix
                corr_matrix = df.corr()
                n_items = len(df.columns)
                
                # Average inter-item correlation
                avg_corr = (corr_matrix.sum().sum() - n_items) / (n_items * (n_items - 1))
                
                # Cronbach's Alpha
                alpha = (n_items * avg_corr) / (1 + (n_items - 1) * avg_corr)
                return alpha, avg_corr
            
            alpha, avg_corr = cronbach_alpha(numeric_data)
            
            # Item-total correlations
            item_total_corrs = []
            for col in numeric_data.columns:
                other_items = numeric_data.drop(columns=[col])
                total_score = other_items.sum(axis=1)
                corr_with_total = numeric_data[col].corr(total_score)
                item_total_corrs.append({
                    'Item': col,
                    'Item-Total Correlation': corr_with_total,
                    'Alpha if Deleted': cronbach_alpha(other_items)[0] if len(other_items.columns) > 1 else np.nan
                })
            
            # Prepare results text
            results_text = f"""RELIABILITY ANALYSIS RESULTS:

Scale: {', '.join(selected_vars)}
Number of items: {len(numeric_data.columns)}
Sample size: {len(numeric_data)}

Reliability Statistics:
- Cronbach's Alpha: {alpha:.4f}
- Average inter-item correlation: {avg_corr:.4f}

Reliability Interpretation:
- Excellent: α ≥ 0.9
- Good: 0.8 ≤ α < 0.9
- Acceptable: 0.7 ≤ α < 0.8
- Questionable: 0.6 ≤ α < 0.7
- Poor: α < 0.6

Current scale reliability: {
'Excellent' if alpha >= 0.9 else
'Good' if alpha >= 0.8 else
'Acceptable' if alpha >= 0.7 else
'Questionable' if alpha >= 0.6 else
'Poor'
}

Item-Total Statistics:
"""
            
            for item in item_total_corrs:
                results_text += f"  {item['Item']}: r={item['Item-Total Correlation']:.3f}, α if deleted={item['Alpha if Deleted']:.3f}\n"
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Reliability Statistics:**")
                st.metric("Cronbach's Alpha", f"{alpha:.4f}")
                st.metric("Average Inter-item Correlation", f"{avg_corr:.4f}")
                
                # Reliability interpretation
                if alpha >= 0.9:
                    st.success("✅ Excellent reliability")
                elif alpha >= 0.8:
                    st.success("✅ Good reliability")
                elif alpha >= 0.7:
                    st.info("✅ Acceptable reliability")
                elif alpha >= 0.6:
                    st.warning("⚠️ Questionable reliability")
                else:
                    st.error("❌ Poor reliability")
            
            with col2:
                # Item-total correlations
                item_df = pd.DataFrame(item_total_corrs)
                st.write("**Item-Total Statistics:**")
                st.dataframe(item_df.round(3))
            
            # Correlation matrix
            st.write("**Inter-item Correlation Matrix:**")
            fig = px.imshow(numeric_data.corr(), text_auto=True, aspect="auto",
                           title="Inter-item Correlations",
                           color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Analysis Button
            ai_button_text = "🤖 AI Statistical Analysis" if st.session_state.openai_api_key else "🤖 AI Analysis (Demo Mode)"
            ai_button_help = "Get AI-powered interpretation" if st.session_state.openai_api_key else "Demo mode - enter API key for full AI analysis"
            
            if st.button(ai_button_text, key="reliability_ai", help=ai_button_help):
                with st.spinner("Generating AI interpretation..."):
                    ai_interpretation = get_chatgpt_interpretation("Reliability Analysis", results_text)
                    st.markdown("### 🧠 AI Statistical Interpretation")
                    st.markdown(ai_interpretation)

def run_factor_analysis(data, numeric_cols):
    """Run Factor Analysis"""
    if len(numeric_cols) >= 3:
        st.write("**Factor Analysis:**")
        
        factor_data = data[numeric_cols].dropna()
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(factor_data)
        
        # Number of factors
        n_factors = st.slider("Number of factors:", 1, min(len(numeric_cols)-1, 8), 2)
        
        # Perform Factor Analysis
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa_result = fa.fit_transform(scaled_data)
        
        # Calculate communalities
        communalities = np.sum(fa.components_**2, axis=0)
        
        # Prepare results text
        results_text = f"""FACTOR ANALYSIS RESULTS:

Variables analyzed: {', '.join(numeric_cols)}
Sample size: {len(factor_data)}
Number of factors extracted: {n_factors}
Extraction method: Maximum Likelihood

Factor Loadings:
"""
        
        # Factor loadings
        loadings_df = pd.DataFrame(
            fa.components_.T,
            columns=[f'Factor {i+1}' for i in range(n_factors)],
            index=numeric_cols
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Factor Loadings:**")
            st.dataframe(loadings_df.round(3))
            
            # Communalities
            comm_df = pd.DataFrame({
                'Variable': numeric_cols,
                'Communality': communalities
            })
            st.write("**Communalities:**")
            st.dataframe(comm_df.round(3))
        
        with col2:
            # Factor loadings heatmap
            fig = px.imshow(loadings_df.T, 
                           title="Factor Loadings Heatmap",
                           color_continuous_scale="RdBu_r",
                           aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        # Factor scores scatter plot
        if n_factors >= 2:
            factor_scores_df = pd.DataFrame(fa_result[:, :2], columns=['Factor 1', 'Factor 2'])
            fig_scores = px.scatter(factor_scores_df, x='Factor 1', y='Factor 2',
                                  title='Factor Scores Plot')
            st.plotly_chart(fig_scores, use_container_width=True)
        
        # Add to results text
        results_text += f"\n{loadings_df.round(3).to_string()}\n\nCommunalities:\n"
        for i, var in enumerate(numeric_cols):
            results_text += f"  {var}: {communalities[i]:.3f}\n"
        
        results_text += f"""
Factor Analysis Interpretation:
- Loadings > 0.5 indicate strong association with factor
- Communalities show proportion of variance explained by factors
- Higher communalities indicate better factor solution fit
"""
        
        # AI Analysis Button
        ai_button_text = "🤖 AI Statistical Analysis" if st.session_state.openai_api_key else "🤖 AI Analysis (Demo Mode)"
        ai_button_help = "Get AI-powered interpretation" if st.session_state.openai_api_key else "Demo mode - enter API key for full AI analysis"
        
        if st.button(ai_button_text, key="factor_ai", help=ai_button_help):
            with st.spinner("Generating AI interpretation..."):
                ai_interpretation = get_chatgpt_interpretation("Factor Analysis", results_text)
                st.markdown("### 🧠 AI Statistical Interpretation")
                st.markdown(ai_interpretation)

def run_regression_analysis(data, dependent_var, independent_vars):
    """Run regression analysis"""
    if len(independent_vars) >= 1:
        X = data[independent_vars]
        y = data[dependent_var]
        
        # Remove any missing values
        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[independent_vars]
        y_clean = combined_data[dependent_var]
        
        # Fit regression model
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        # Predictions
        y_pred = model.predict(X_clean)
        r2 = r2_score(y_clean, y_pred)
        
        # Calculate adjusted R-squared
        n = len(y_clean)
        p = len(independent_vars)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Calculate residuals
        residuals = y_clean - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Prepare results text for AI interpretation
        results_text = f"""REGRESSION ANALYSIS RESULTS:

Model: {dependent_var} = f({', '.join(independent_vars)})
Sample size: {n}

Model Fit Statistics:
- R-squared: {r2:.4f}
- Adjusted R-squared: {adj_r2:.4f}
- RMSE: {rmse:.4f}
- Intercept: {model.intercept_:.4f}

Regression Coefficients:
"""
        
        for i, var in enumerate(independent_vars):
            results_text += f"- {var}: {model.coef_[i]:.4f}\n"
        
        results_text += f"""
Model Equation:
{dependent_var} = {model.intercept_:.4f}"""
        
        for i, var in enumerate(independent_vars):
            sign = " + " if model.coef_[i] >= 0 else " - "
            results_text += f"{sign}{abs(model.coef_[i]):.4f}*{var}"
        
        results_text += f"""

Residual Statistics:
- Mean residual: {residuals.mean():.4f}
- Std residual: {residuals.std():.4f}
- Min residual: {residuals.min():.4f}
- Max residual: {residuals.max():.4f}
"""
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Regression Results:**")
            st.write(f"R-squared: {r2:.4f}")
            st.write(f"Adjusted R-squared: {adj_r2:.4f}")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"Intercept: {model.intercept_:.4f}")
            
            coef_df = pd.DataFrame({
                'Variable': independent_vars,
                'Coefficient': model.coef_
            })
            st.dataframe(coef_df)
        
        with col2:
            # Actual vs Predicted plot
            fig = px.scatter(x=y_clean, y=y_pred, 
                           title="Actual vs Predicted Values",
                           labels={'x': 'Actual', 'y': 'Predicted'})
            # Add diagonal line
            min_val = min(y_clean.min(), y_pred.min())
            max_val = max(y_clean.max(), y_pred.max())
            fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                         line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        fig_residuals = px.scatter(x=y_pred, y=residuals,
                                 title="Residuals vs Predicted Values",
                                 labels={'x': 'Predicted', 'y': 'Residuals'})
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_residuals, use_container_width=True)
        
        # AI Analysis Button
        ai_button_text = "🤖 AI Statistical Analysis" if st.session_state.openai_api_key else "🤖 AI Analysis (Demo Mode)"
        ai_button_help = "Get AI-powered interpretation" if st.session_state.openai_api_key else "Demo mode - enter API key for full AI analysis"
        
        if st.button(ai_button_text, key="reg_ai", help=ai_button_help):
            with st.spinner("Generating AI interpretation..."):
                ai_interpretation = get_chatgpt_interpretation("Regression Analysis", results_text)
                st.markdown("### 🧠 AI Statistical Interpretation")
                st.markdown(ai_interpretation)

def run_pca_analysis(data, numeric_cols):
    """Run Principal Component Analysis"""
    if len(numeric_cols) >= 2:
        pca_data = data[numeric_cols].dropna()
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Prepare results text for AI interpretation
        results_text = f"""PRINCIPAL COMPONENT ANALYSIS RESULTS:

Variables analyzed: {', '.join(numeric_cols)}
Sample size: {len(pca_data)}
Number of components: {len(pca.explained_variance_ratio_)}

Explained Variance by Component:
"""
        
        cumulative_var = 0
        for i, var_ratio in enumerate(pca.explained_variance_ratio_):
            cumulative_var += var_ratio
            results_text += f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%) - Cumulative: {cumulative_var:.4f} ({cumulative_var*100:.2f}%)\n"
        
        results_text += f"""
Components explaining 80% variance: {sum(1 for x in np.cumsum(pca.explained_variance_ratio_) if x < 0.8) + 1}

Component Loadings (First 3 PCs):
"""
        
        for i in range(min(3, len(pca.components_))):
            results_text += f"\nPC{i+1} loadings:\n"
            for j, var in enumerate(numeric_cols):
                results_text += f"  {var}: {pca.components_[i][j]:.4f}\n"
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**PCA Results:**")
            
            # Explained variance
            explained_var_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                'Explained Variance Ratio': pca.explained_variance_ratio_,
                'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
            })
            st.dataframe(explained_var_df)
        
        with col2:
            # Scree plot
            fig = px.line(x=range(1, len(pca.explained_variance_ratio_) + 1),
                         y=pca.explained_variance_ratio_,
                         title="Scree Plot",
                         labels={'x': 'Component', 'y': 'Explained Variance Ratio'})
            fig.add_scatter(x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                           y=pca.explained_variance_ratio_,
                           mode='markers')
            st.plotly_chart(fig, use_container_width=True)
        
        # Component loadings heatmap
        if len(pca.components_) >= 2:
            loadings_df = pd.DataFrame(
                pca.components_[:min(5, len(pca.components_))].T,
                columns=[f'PC{i+1}' for i in range(min(5, len(pca.components_)))],
                index=numeric_cols
            )
            
            fig_loadings = px.imshow(loadings_df.T, 
                                   title="Component Loadings Heatmap",
                                   color_continuous_scale="RdBu_r",
                                   aspect="auto")
            st.plotly_chart(fig_loadings, use_container_width=True)
        
        # Biplot if 2 or more components
        if len(pca.components_) >= 2:
            pca_df = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
            fig = px.scatter(pca_df, x='PC1', y='PC2', title='PCA Biplot (PC1 vs PC2)')
            st.plotly_chart(fig, use_container_width=True)
        
        # AI Analysis Button
        ai_button_text = "🤖 AI Statistical Analysis" if st.session_state.openai_api_key else "🤖 AI Analysis (Demo Mode)"
        ai_button_help = "Get AI-powered interpretation" if st.session_state.openai_api_key else "Demo mode - enter API key for full AI analysis"
        
        if st.button(ai_button_text, key="pca_ai", help=ai_button_help):
            with st.spinner("Generating AI interpretation..."):
                ai_interpretation = get_chatgpt_interpretation("Principal Component Analysis", results_text)
                st.markdown("### 🧠 AI Statistical Interpretation")
                st.markdown(ai_interpretation)

def run_cluster_analysis(data, numeric_cols):
    """Run cluster analysis"""
    if len(numeric_cols) >= 2:
        cluster_data = data[numeric_cols].dropna()
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # K-means clustering
        n_clusters = st.slider("Number of clusters:", 2, 10, 3)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_
        
        # Prepare results text for AI interpretation
        results_text = f"""CLUSTER ANALYSIS RESULTS:

Variables used: {', '.join(numeric_cols)}
Sample size: {len(cluster_data)}
Number of clusters: {n_clusters}
Algorithm: K-means clustering

Cluster Quality Metrics:
- Silhouette Score: {silhouette_avg:.4f} (Range: -1 to 1, higher is better)
- Within-cluster Sum of Squares (Inertia): {inertia:.4f}

Cluster Distribution:
"""
        
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        for i, count in enumerate(cluster_counts):
            percentage = (count / len(cluster_labels)) * 100
            results_text += f"Cluster {i}: {count} observations ({percentage:.1f}%)\n"
        
        results_text += f"""
Cluster Centers (Standardized):
"""
        
        for i, center in enumerate(kmeans.cluster_centers_):
            results_text += f"\nCluster {i} center:\n"
            for j, var in enumerate(numeric_cols):
                results_text += f"  {var}: {center[j]:.4f}\n"
        
        # Convert back to original scale for interpretation
        original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        results_text += f"""
Cluster Centers (Original Scale):
"""
        
        for i, center in enumerate(original_centers):
            results_text += f"\nCluster {i} center:\n"
            for j, var in enumerate(numeric_cols):
                results_text += f"  {var}: {center[j]:.4f}\n"
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Cluster Analysis Results:**")
            st.write(f"Silhouette Score: {silhouette_avg:.4f}")
            st.write(f"Inertia: {inertia:.4f}")
            
            # Cluster centers (original scale)
            centers_df = pd.DataFrame(original_centers, 
                                    columns=numeric_cols,
                                    index=[f'Cluster {i}' for i in range(n_clusters)])
            st.dataframe(centers_df.round(3))
        
        with col2:
            # Cluster distribution
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                        title="Cluster Distribution",
                        labels={'x': 'Cluster', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot with clusters (if 2+ variables)
        if len(numeric_cols) >= 2:
            plot_data = cluster_data.copy()
            plot_data['Cluster'] = cluster_labels
            
            fig = px.scatter(plot_data, x=numeric_cols[0], y=numeric_cols[1],
                           color='Cluster', title="Cluster Visualization",
                           color_discrete_sequence=px.colors.qualitative.Set1)
            
            # Add cluster centers
            centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
            fig.add_scatter(x=centers_original[:, 0], y=centers_original[:, 1],
                          mode='markers', marker=dict(symbol='x', size=15, color='black'),
                          name='Cluster Centers')
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Elbow method for optimal clusters
        if st.checkbox("Show Elbow Method Analysis"):
            inertias = []
            k_range = range(1, 11)
            
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42)
                kmeans_temp.fit(scaled_data)
                inertias.append(kmeans_temp.inertia_)
            
            fig_elbow = px.line(x=list(k_range), y=inertias,
                              title="Elbow Method for Optimal k",
                              labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'})
            fig_elbow.add_scatter(x=list(k_range), y=inertias, mode='markers')
            st.plotly_chart(fig_elbow, use_container_width=True)
        
        # AI Analysis Button
        ai_button_text = "🤖 AI Statistical Analysis" if st.session_state.openai_api_key else "🤖 AI Analysis (Demo Mode)"
        ai_button_help = "Get AI-powered interpretation" if st.session_state.openai_api_key else "Demo mode - enter API key for full AI analysis"
        
        if st.button(ai_button_text, key="cluster_ai", help=ai_button_help):
            with st.spinner("Generating AI interpretation..."):
                ai_interpretation = get_chatgpt_interpretation("Cluster Analysis", results_text)
                st.markdown("### 🧠 AI Statistical Interpretation")
                st.markdown(ai_interpretation)

if __name__ == "__main__":
    main()
